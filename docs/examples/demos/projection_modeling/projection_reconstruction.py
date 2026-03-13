"""Projection reconstruction CLI: geometric and wave-optical inverse algorithms.

Usage::

    # Reconstruct from all 29 projection angles
    python projection_reconstruction.py geometric --data-dir ./data
    python projection_reconstruction.py wave      --data-dir ./data

    # Reconstruct from a single +/-theta pair
    python projection_reconstruction.py geometric-two-projections --angle 30 --data-dir ./data
    python projection_reconstruction.py wave-two-projections      --angle 30 --data-dir ./data

Each subcommand reads from and writes to the shared OME-Zarr store
at ``<data-dir>/projection_modeling.zarr``.

Forward models
--------------
**Geometric**: Forward = Siddon projection.  No optical blur.
Reconstructs from projections of the unblurred ``object`` column.

**Wave-optical**: Forward = 3D OTF convolution + Siddon projection.
By the Fourier-slice theorem, projecting an OTF-blurred volume at
angle theta is equivalent to applying the central slice of the 3D OTF
at angle theta to the 2D Fourier transform of the projection.  Using
the full 3D OTF in the forward model naturally accounts for the
angle-dependent resolution and defocus.  Reconstructs from projections
of the blurred+noisy ``rawimage`` column.

Both algorithms solve (H^T H + lambda I) x = H^T y via CG-Tikhonov.
Siddon projection and backprojection use sparse matmul on GPU (or CPU).
OTF convolution uses PyTorch FFT on GPU when available.
"""

from pathlib import Path

import click
import numpy as np
import torch
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from siddon import SiddonOperator, cg_tikhonov

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

# ---------------------------------------------------------------------------
# Constants (must match projection_modeling.py)
# ---------------------------------------------------------------------------
VOXEL_SIZE = 0.05  # um
ZYX_SHAPE = (256, 256, 256)
SAMPLE_TYPES = ["point", "lines", "shepplogan"]
CHANNEL_NAMES = ["Fluorescence", "Phase"]

# Imaging
WAVELENGTH_ILLUMINATION = 0.500
WAVELENGTH_EMISSION = 0.520
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
MEDIA_INDEX = 1.33
Z_PADDING = 0
BLACK_LEVEL = 100  # detector offset (counts)

# Reconstruction defaults
REG_STRENGTH = 1e-3
N_ITER = 50
PROJECTION_ANGLES = list(range(-70, 75, 5))  # 29 angles

# GPU device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_metrics(recon, ground_truth):
    """Return MSE and PSNR between optimally-scaled reconstruction and ground truth.

    Finds the least-squares optimal gain ``a`` that minimizes
    ``||a * recon - ground_truth||^2``.  This removes the arbitrary
    scale factor introduced by detector gain and forward-model units.
    """
    dot_rg = float(np.sum(recon * ground_truth))
    dot_rr = float(np.sum(recon * recon))
    alpha = dot_rg / dot_rr if dot_rr > 0 else 1.0
    recon_scaled = alpha * recon
    mse = float(np.mean((recon_scaled - ground_truth) ** 2))
    gt_range = float(ground_truth.max() - ground_truth.min())
    psnr = 10 * np.log10(gt_range**2 / mse) if mse > 0 else float("inf")
    return mse, float(psnr)


def _ensure_position(plate, row, col, fov, shape):
    """Create a position if it does not exist, or return the existing one."""
    path = f"{row}/{col}/{fov}"
    try:
        position = plate.create_position(row, col, fov)
        position.create_zeros(
            name="0",
            shape=shape,
            chunks=(1, 1, 1, 256, 256),
            dtype=np.float32,
            transform=[TransformationMeta(type="scale", scale=[1, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])],
        )
    except Exception:
        pass  # position already exists
    return plate[path]


def _reconstruct_channel(
    siddon_op,
    source_volume,
    reg_strength,
    n_iter,
    forward_blur=None,
    adjoint_blur=None,
    use_ramp_filter=False,
):
    """CG-Tikhonov reconstruction of a single channel from Siddon projections.

    Parameters
    ----------
    siddon_op : SiddonOperator
        Precomputed sparse projection operator for all angles.
    source_volume : np.ndarray, shape (Z, Y, X)
        Volume to project as measurements (object or rawimage channel).
    reg_strength : float
        Tikhonov regularization lambda.
    n_iter : int
        Number of CG iterations.
    forward_blur : callable or None
        If given, applied to the volume before Siddon projection (wave model).
        Signature: torch.Tensor -> torch.Tensor.
    adjoint_blur : callable or None
        If given, applied after Siddon backprojection (wave adjoint).
    use_ramp_filter : bool
        If True, apply Ram-Lak ramp filter to each projection before
        backprojection.  This preconditions the normal operator H*H so
        that all spatial frequencies converge at a similar rate in CG,
        reducing Gibbs ringing near sharp edges.

    Returns
    -------
    recon : np.ndarray
        Reconstructed 3D volume.
    """
    device = siddon_op.device

    # Move source volume to device and compute measurements
    source_vol_t = torch.tensor(source_volume, dtype=torch.float32, device=device)
    measurements = siddon_op.project_all(source_vol_t)
    del source_vol_t

    def forward(vol):
        v = vol
        if forward_blur is not None:
            v = forward_blur(v)
        return siddon_op.project_all(v)

    def adjoint(projs):
        bp = siddon_op.backproject_all(projs, ramp_filter=use_ramp_filter)
        if adjoint_blur is not None:
            bp = adjoint_blur(bp)
        return bp

    recon = cg_tikhonov(forward, adjoint, measurements, ZYX_SHAPE, reg_strength, n_iter, device)
    return recon.cpu().numpy()


def _make_blur_ops(otf_tensor):
    """Build GPU-accelerated forward/adjoint blur from a 3D OTF tensor.

    Forward: FFT-multiply by OTF (convolution).
    Adjoint: FFT-multiply by conjugate OTF (correlation).

    By the Fourier-slice theorem, convolving with the 3D OTF before
    Siddon projection is equivalent to applying the central slice of
    the 3D OTF at each projection angle as a 2D transfer function.
    The 3D approach correctly couples defocus and projection geometry.
    """
    otf_conj = torch.conj(otf_tensor)

    def forward_blur(vol):
        return torch.fft.ifftn(torch.fft.fftn(vol) * otf_tensor).real

    def adjoint_blur(vol):
        return torch.fft.ifftn(torch.fft.fftn(vol) * otf_conj).real

    return forward_blur, adjoint_blur


def _compute_transfer_functions():
    """Compute fluorescence OTF and phase TF, return blur ops per channel."""
    click.echo(f"Device: {DEVICE}")

    click.echo("Computing fluorescence OTF...")
    fluorescence_otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape=ZYX_SHAPE,
        yx_pixel_size=VOXEL_SIZE,
        z_pixel_size=VOXEL_SIZE,
        wavelength_emission=WAVELENGTH_EMISSION,
        z_padding=Z_PADDING,
        index_of_refraction_media=MEDIA_INDEX,
        numerical_aperture_detection=NA_DETECTION,
    ).to(DEVICE)

    click.echo("Computing phase transfer function...")
    real_tf, _imag_tf = phase_thick_3d.calculate_transfer_function(
        zyx_shape=ZYX_SHAPE,
        yx_pixel_size=VOXEL_SIZE,
        z_pixel_size=VOXEL_SIZE,
        wavelength_illumination=WAVELENGTH_ILLUMINATION,
        z_padding=Z_PADDING,
        index_of_refraction_media=MEDIA_INDEX,
        numerical_aperture_illumination=NA_ILLUMINATION,
        numerical_aperture_detection=NA_DETECTION,
    )
    real_tf = real_tf.to(DEVICE)

    fluor_fwd, fluor_adj = _make_blur_ops(fluorescence_otf)
    phase_fwd, phase_adj = _make_blur_ops(real_tf)

    return {
        "Fluorescence": (fluor_fwd, fluor_adj),
        "Phase": (phase_fwd, phase_adj),
    }


def _run_reconstruction(
    store_path, source_col, target_col, angles, reg, niter, blur_ops=None, use_ramp_filter=False
):
    """Core reconstruction loop shared by all four subcommands.

    Parameters
    ----------
    store_path : Path
        Path to the OME-Zarr store.
    source_col : str
        Column to read source volumes from ("object" or "rawimage").
    target_col : str
        Column to write reconstructions to.
    angles : list of float
        Projection angles.
    reg : float
        Regularization strength.
    niter : int
        CG iterations.
    blur_ops : dict or None
        If given, maps channel name to (forward_blur, adjoint_blur) callables.
    use_ramp_filter : bool
        If True, apply ramp filter to projections before backprojection
        (preconditions CG for more uniform frequency convergence).
    """
    # Build the Siddon operator once for all samples and channels
    click.echo(f"\nBuilding Siddon sparse matrices for {len(angles)} angles on {DEVICE}...")
    siddon_op = SiddonOperator(ZYX_SHAPE, angles, VOXEL_SIZE, DEVICE)
    click.echo("  Done.")

    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for sample in SAMPLE_TYPES:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Reconstructing {sample} → {target_col}")
            click.echo(f"  Angles: {angles}, ramp_filter={use_ramp_filter}")
            click.echo(f"{'=' * 60}")

            _ensure_position(plate, sample, target_col, "0", (1, len(CHANNEL_NAMES), *ZYX_SHAPE))

            src_pos = plate[f"{sample}/{source_col}/0"]
            obj_pos = plate[f"{sample}/object/0"]
            tgt_pos = plate[f"{sample}/{target_col}/0"]

            for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                click.echo(f"\n  Channel: {ch_name}")
                source_vol = np.array(src_pos["0"][0, c_idx], dtype=np.float32)
                ground_truth = np.array(obj_pos["0"][0, c_idx])

                # Subtract detector offset from rawimage (standard calibration step)
                if source_col == "rawimage":
                    source_vol = source_vol - BLACK_LEVEL

                fwd_blur, adj_blur = (None, None)
                if blur_ops is not None:
                    fwd_blur, adj_blur = blur_ops[ch_name]

                click.echo(f"  CG-Tikhonov: {niter} iters, lambda={reg}, {len(angles)} angles")
                recon = _reconstruct_channel(
                    siddon_op, source_vol, reg, niter, fwd_blur, adj_blur, use_ramp_filter
                )

                mse, psnr = _compute_metrics(recon, ground_truth)
                click.echo(f"  MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

                tgt_pos["0"][0, c_idx] = recon
                tgt_pos.zattrs[f"{ch_name}_mse"] = mse
                tgt_pos.zattrs[f"{ch_name}_psnr"] = psnr

            tgt_pos.zattrs["angles"] = angles
            tgt_pos.zattrs["reg_strength"] = reg
            tgt_pos.zattrs["n_iter"] = niter
            tgt_pos.zattrs["ramp_filter"] = use_ramp_filter
            click.echo(f"  Wrote {sample}/{target_col}/0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
CONTEXT = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT)
@click.option("--data-dir", type=click.Path(), default="./data", show_default=True, help="Directory containing the zarr store.")
@click.pass_context
def cli(ctx, data_dir):
    """Projection reconstruction: recover 3D volumes from tilted projections.

    Four subcommands: two forward models (geometric, wave-optical)
    crossed with two angle sets (all angles, limited +/-theta pair).

    \b
    python projection_reconstruction.py geometric         --data-dir ./data
    python projection_reconstruction.py wave              --data-dir ./data
    python projection_reconstruction.py geometric-two-projections --angle 30
    python projection_reconstruction.py wave-two-projections      --angle 30
    """
    ctx.ensure_object(dict)
    ctx.obj["store_path"] = Path(data_dir) / "projection_modeling.zarr"


# ---- geometric (all angles) -----------------------------------------------
@cli.command()
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.option("--ramp-filter", is_flag=True, default=False, help="Apply ramp filter to precondition CG (reduces edge ringing).")
@click.pass_context
def geometric(ctx, reg, niter, ramp_filter):
    """Limited-angle tomography without blur (Siddon ray-tracing only).

    Reads the ``object`` column (unblurred ground truth), computes
    Siddon projections at all 29 angles as measurements, and solves
    (H^T H + lambda I) x = H^T y.  Writes to the ``recongeo`` column.

    With --ramp-filter, the adjoint applies a Ram-Lak ramp filter to
    each projection before backprojection, preconditioning the normal
    operator for more uniform frequency convergence.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    _run_reconstruction(
        store_path, "object", "recongeo", PROJECTION_ANGLES, reg, niter, use_ramp_filter=ramp_filter
    )
    click.echo("\nDone.")


# ---- wave (all angles) ----------------------------------------------------
@cli.command()
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.option("--ramp-filter", is_flag=True, default=False, help="Apply ramp filter to precondition CG.")
@click.pass_context
def wave(ctx, reg, niter, ramp_filter):
    """Reconstruct from all projections using OTF blur + Siddon (wave model).

    The forward model convolves the volume with the 3D microscope OTF
    (fluorescence OTF or phase TF), then projects via Siddon.  By the
    Fourier-slice theorem, this is equivalent to applying the central
    slice of the 3D OTF at each projection angle as a 2D transfer
    function.  OTF convolution runs on GPU via PyTorch FFT.

    Reads the ``rawimage`` column (blurred + noisy).  Compares against
    the unblurred ``object``.  Writes to the ``reconwave`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    blur_ops = _compute_transfer_functions()
    _run_reconstruction(
        store_path, "rawimage", "reconwave", PROJECTION_ANGLES, reg, niter, blur_ops, use_ramp_filter=ramp_filter
    )
    click.echo("\nDone.")


# ---- geometric-two-projections (single +/-theta pair) ---------------------
@cli.command("geometric-two-projections")
@click.option("--angle", type=int, required=True, help="Half-angle theta; uses +/-theta projection pair.")
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def geometric_two_projections(ctx, angle, reg, niter):
    """Reconstruct from a single +/-theta pair using Siddon (no OTF).

    Uses only two projections at +angle and -angle degrees.
    Writes to the ``recongeo2`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    angles = [-angle, +angle]
    _run_reconstruction(store_path, "object", "recongeo2", angles, reg, niter)
    click.echo("\nDone.")


# ---- wave-two-projections (single +/-theta pair) -------------------------
@cli.command("wave-two-projections")
@click.option("--angle", type=int, required=True, help="Half-angle theta; uses +/-theta projection pair.")
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def wave_two_projections(ctx, angle, reg, niter):
    """Reconstruct from a single +/-theta pair using OTF blur + Siddon.

    Uses only two projections at +angle and -angle degrees.
    The 3D OTF in the forward model captures angle-dependent
    resolution via the Fourier-slice theorem.  Writes to the
    ``reconwave2`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    blur_ops = _compute_transfer_functions()
    angles = [-angle, +angle]
    _run_reconstruction(store_path, "rawimage", "reconwave2", angles, reg, niter, blur_ops)
    click.echo("\nDone.")


if __name__ == "__main__":
    cli()

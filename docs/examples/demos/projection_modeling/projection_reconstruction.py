"""Projection reconstruction CLI: geometric and wave-optical inverse algorithms.

Usage::

    # Reconstruct from all 29 projection angles
    python projection_reconstruction.py geometric --data-dir ./data
    python projection_reconstruction.py wave      --data-dir ./data

    # Reconstruct from a single +/-theta pair
    python projection_reconstruction.py geometric-limited --angle 30 --data-dir ./data
    python projection_reconstruction.py wave-limited      --angle 30 --data-dir ./data

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
OTF convolution uses PyTorch FFT on GPU when available.
"""

from pathlib import Path

import click
import numpy as np
import torch
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from siddon import cg_tikhonov, siddon_backproject, siddon_project

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
    source_volume,
    angles,
    reg_strength,
    n_iter,
    forward_blur=None,
    adjoint_blur=None,
):
    """CG-Tikhonov reconstruction of a single channel from Siddon projections.

    Parameters
    ----------
    source_volume : np.ndarray, shape (Z, Y, X)
        Volume to project as measurements (object or rawimage channel).
    angles : list of float
        Projection angles in degrees.
    reg_strength : float
        Tikhonov regularization lambda.
    n_iter : int
        Number of CG iterations.
    forward_blur : callable or None
        If given, applied to the volume before Siddon projection (wave model).
        Signature: np.ndarray -> np.ndarray.
    adjoint_blur : callable or None
        If given, applied after Siddon backprojection (wave adjoint).

    Returns
    -------
    recon : np.ndarray
        Reconstructed 3D volume.
    """
    # Compute measurements: project the source volume
    # No padding — each projection keeps its native width.
    # siddon_backproject expects the same width that siddon_project produced;
    # center-padding would shift the data and break the adjoint.
    measurements = []
    for angle in angles:
        proj = siddon_project(source_volume, angle, VOXEL_SIZE, mode="sum")
        measurements.append(proj)

    def forward(vol):
        if forward_blur is not None:
            vol = forward_blur(vol)
        return [siddon_project(vol, a, VOXEL_SIZE, "sum") for a in angles]

    def adjoint(projs):
        bp = np.zeros(ZYX_SHAPE, dtype=np.float32)
        for proj, angle in zip(projs, angles):
            bp += siddon_backproject(proj, angle, ZYX_SHAPE, VOXEL_SIZE)
        if adjoint_blur is not None:
            bp = adjoint_blur(bp)
        return bp

    return cg_tikhonov(forward, adjoint, measurements, ZYX_SHAPE, reg_strength, n_iter)


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

    def forward_blur(vol_np):
        vol = torch.tensor(vol_np, dtype=torch.float32, device=DEVICE)
        return torch.fft.ifftn(torch.fft.fftn(vol) * otf_tensor).real.cpu().numpy()

    def adjoint_blur(vol_np):
        vol = torch.tensor(vol_np, dtype=torch.float32, device=DEVICE)
        return torch.fft.ifftn(torch.fft.fftn(vol) * otf_conj).real.cpu().numpy()

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


def _run_reconstruction(store_path, source_col, target_col, angles, reg, niter, blur_ops=None):
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
    """
    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for sample in SAMPLE_TYPES:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Reconstructing {sample} → {target_col}")
            click.echo(f"  Angles: {angles}")
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
                recon = _reconstruct_channel(source_vol, angles, reg, niter, fwd_blur, adj_blur)

                mse, psnr = _compute_metrics(recon, ground_truth)
                click.echo(f"  MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

                tgt_pos["0"][0, c_idx] = recon
                tgt_pos.zattrs[f"{ch_name}_mse"] = mse
                tgt_pos.zattrs[f"{ch_name}_psnr"] = psnr

            tgt_pos.zattrs["angles"] = angles
            tgt_pos.zattrs["reg_strength"] = reg
            tgt_pos.zattrs["n_iter"] = niter
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
    python projection_reconstruction.py geometric-limited --angle 30
    python projection_reconstruction.py wave-limited      --angle 30
    """
    ctx.ensure_object(dict)
    ctx.obj["store_path"] = Path(data_dir) / "projection_modeling.zarr"


# ---- geometric (all angles) -----------------------------------------------
@cli.command()
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def geometric(ctx, reg, niter):
    """Reconstruct from all projections using Siddon ray-tracing (no OTF).

    Reads the ``object`` column (unblurred ground truth), computes
    Siddon projections at all 29 angles as measurements, and solves
    (H^T H + lambda I) x = H^T y.  Writes to the ``recongeo`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    _run_reconstruction(store_path, "object", "recongeo", PROJECTION_ANGLES, reg, niter)
    click.echo("\nDone.")


# ---- wave (all angles) ----------------------------------------------------
@cli.command()
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def wave(ctx, reg, niter):
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
    _run_reconstruction(store_path, "rawimage", "reconwave", PROJECTION_ANGLES, reg, niter, blur_ops)
    click.echo("\nDone.")


# ---- geometric-limited (single +/-theta pair) -----------------------------
@cli.command("geometric-limited")
@click.option("--angle", type=int, required=True, help="Half-angle theta; uses +/-theta projection pair.")
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def geometric_limited(ctx, angle, reg, niter):
    """Reconstruct from a single +/-theta pair using Siddon (no OTF).

    Uses only two projections at +angle and -angle degrees.
    Writes to the ``recongeoL`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    angles = [-angle, +angle]
    _run_reconstruction(store_path, "object", "recongeoL", angles, reg, niter)
    click.echo("\nDone.")


# ---- wave-limited (single +/-theta pair) ----------------------------------
@cli.command("wave-limited")
@click.option("--angle", type=int, required=True, help="Half-angle theta; uses +/-theta projection pair.")
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
@click.pass_context
def wave_limited(ctx, angle, reg, niter):
    """Reconstruct from a single +/-theta pair using OTF blur + Siddon.

    Uses only two projections at +angle and -angle degrees.
    The 3D OTF in the forward model captures angle-dependent
    resolution via the Fourier-slice theorem.  Writes to the
    ``reconwaveL`` column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    blur_ops = _compute_transfer_functions()
    angles = [-angle, +angle]
    _run_reconstruction(store_path, "rawimage", "reconwaveL", angles, reg, niter, blur_ops)
    click.echo("\nDone.")


if __name__ == "__main__":
    cli()

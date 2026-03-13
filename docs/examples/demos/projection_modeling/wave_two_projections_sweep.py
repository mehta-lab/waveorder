"""Two-projection wave-optical reconstruction sweep: PSNR vs half-angle theta.

For each half-angle theta in [5, 10, 15, ..., 70], reconstructs all three
phantoms from a +/-theta projection pair using OTF blur + Siddon CG-Tikhonov.
Both fluorescence and phase channels are reconstructed.

Produces:

1. PSNR vs theta plot (all three samples, both channels).
2. Per-sample, per-channel detailed figure at the optimal theta (by fluorescence):
   projections at +/-theta, reconstruction, object, difference, FFT XZ slices.

Usage::

    cd docs/examples/demos/projection_modeling/
    uv run python wave_two_projections_sweep.py --data-dir ./data --out-dir ./plots

"""

from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from iohub.ngff import open_ome_zarr
from siddon import SiddonOperator, cg_tikhonov

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

matplotlib.use("Agg")
plt.style.use("dark_background")

# ---------------------------------------------------------------------------
# Constants (match projection_modeling.py / projection_reconstruction.py)
# ---------------------------------------------------------------------------
VOXEL_SIZE = 0.05
ZYX_SHAPE = (256, 256, 256)
SAMPLE_TYPES = ["point", "lines", "shepplogan"]
CHANNEL_NAMES = ["Fluorescence", "Phase"]

WAVELENGTH_ILLUMINATION = 0.500
WAVELENGTH_EMISSION = 0.520
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
MEDIA_INDEX = 1.33
Z_PADDING = 0
BLACK_LEVEL = 100

REG_STRENGTH = 1e-3
N_ITER = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THETA_RANGE = list(range(5, 75, 5))  # 5 to 70 inclusive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _optimal_gain(recon, gt):
    dot_rg = np.sum(recon * gt)
    dot_rr = np.sum(recon * recon)
    return dot_rg / dot_rr if dot_rr > 0 else 1.0


def _psnr(recon, gt):
    mse = float(np.mean((recon - gt) ** 2))
    gt_range = float(gt.max() - gt.min())
    return 10 * np.log10(gt_range**2 / mse) if mse > 0 else float("inf")


def _log_ft(vol):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(vol))))


def _ortho(vol):
    nz, ny, nx = vol.shape
    return vol[nz // 2], vol[:, ny // 2, :], vol[:, :, nx // 2]


def _style_ax(ax, title=None, ylabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


# ---------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------
def _compute_transfer_functions():
    """Compute fluorescence OTF and phase TF on GPU. Return dict of blur ops."""
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

    def _make_blur_ops(otf_tensor):
        otf_conj = torch.conj(otf_tensor)

        def forward_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * otf_tensor).real

        def adjoint_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * otf_conj).real

        return forward_blur, adjoint_blur

    fluor_fwd, fluor_adj = _make_blur_ops(fluorescence_otf)
    phase_fwd, phase_adj = _make_blur_ops(real_tf)

    return {
        "Fluorescence": (fluor_fwd, fluor_adj),
        "Phase": (phase_fwd, phase_adj),
    }


# ---------------------------------------------------------------------------
# Reconstruct single channel with wave model
# ---------------------------------------------------------------------------
def _reconstruct_wave(siddon_op, source_vol_np, reg, niter, forward_blur, adjoint_blur):
    """CG-Tikhonov reconstruction with OTF blur + Siddon forward model.

    The source volume is the already-blurred rawimage (minus BLACK_LEVEL).
    Measurements are Siddon projections of that volume — no additional blur.
    The CG forward model applies blur + project to the *unknown*, matching
    the physical image formation: OTF convolution then projection.
    """
    device = siddon_op.device
    source_t = torch.tensor(source_vol_np, dtype=torch.float32, device=device)

    # Measurements: project the already-blurred rawimage (no double-blur)
    measurements = siddon_op.project_all(source_t)
    del source_t

    # Subtract per-projection mean: with two views the common DC
    # background carries no angular diversity.
    measurements = [p - p.mean() for p in measurements]

    def forward(vol):
        return siddon_op.project_all(forward_blur(vol))

    def adjoint(projs):
        bp = siddon_op.backproject_all(projs, ramp_filter=True)
        return adjoint_blur(bp)

    recon = cg_tikhonov(forward, adjoint, measurements, ZYX_SHAPE, reg, niter, device)
    return recon.cpu().numpy(), measurements


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(store_path, reg, niter, blur_ops):
    """Run wave-two-projections reconstruction at every theta, return PSNR table.

    Returns
    -------
    results : dict
        {sample: {channel: {"thetas": [...], "psnrs": [...], "best_theta": int, "best_psnr": float}}}
    objects : dict
        {sample: {channel: np.ndarray}} ground-truth volumes.
    rawimages : dict
        {sample: {channel: np.ndarray}} blurred+noisy source volumes.
    """
    results = {
        s: {ch: {"thetas": [], "psnrs": []} for ch in CHANNEL_NAMES}
        for s in SAMPLE_TYPES
    }
    objects = {}
    rawimages = {}

    with open_ome_zarr(str(store_path), mode="r") as plate:
        for sample in SAMPLE_TYPES:
            objects[sample] = {}
            rawimages[sample] = {}
            for c_idx, ch in enumerate(CHANNEL_NAMES):
                objects[sample][ch] = np.array(
                    plate[f"{sample}/object/0"]["0"][0, c_idx], dtype=np.float32
                )
                raw = np.array(
                    plate[f"{sample}/rawimage/0"]["0"][0, c_idx], dtype=np.float32
                )
                rawimages[sample][ch] = raw - BLACK_LEVEL

    for theta in THETA_RANGE:
        angles = [-theta, theta]
        click.echo(f"\nTheta = {theta} deg (angles: {angles})")
        siddon_op = SiddonOperator(ZYX_SHAPE, angles, VOXEL_SIZE, DEVICE)

        for sample in SAMPLE_TYPES:
            for ch in CHANNEL_NAMES:
                fwd_blur, adj_blur = blur_ops[ch]
                source = rawimages[sample][ch]
                gt = objects[sample][ch]

                recon, _ = _reconstruct_wave(siddon_op, source, reg, niter, fwd_blur, adj_blur)
                alpha = _optimal_gain(recon, gt)
                psnr_val = _psnr(alpha * recon, gt)
                results[sample][ch]["thetas"].append(theta)
                results[sample][ch]["psnrs"].append(psnr_val)
                click.echo(f"  {sample}/{ch}: PSNR = {psnr_val:.2f} dB (gain = {alpha:.6f})")

        del siddon_op
        torch.cuda.empty_cache()

    # Identify best theta per sample (by fluorescence PSNR)
    for sample in SAMPLE_TYPES:
        for ch in CHANNEL_NAMES:
            psnrs = results[sample][ch]["psnrs"]
            best_idx = int(np.argmax(psnrs))
            results[sample][ch]["best_theta"] = results[sample][ch]["thetas"][best_idx]
            results[sample][ch]["best_psnr"] = psnrs[best_idx]
        best_t = results[sample]["Fluorescence"]["best_theta"]
        best_p = results[sample]["Fluorescence"]["best_psnr"]
        click.echo(f"\n{sample}: best theta (Fluor) = {best_t} deg, PSNR = {best_p:.2f} dB")

    return results, objects, rawimages


# ---------------------------------------------------------------------------
# Plot: PSNR vs theta
# ---------------------------------------------------------------------------
def plot_psnr_vs_theta(results, out_dir):
    """Line plot: PSNR vs half-angle theta for all samples and channels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    markers = {"point": "o", "lines": "s", "shepplogan": "^"}

    for ax, ch in zip(axes, CHANNEL_NAMES):
        for sample in SAMPLE_TYPES:
            thetas = results[sample][ch]["thetas"]
            psnrs = results[sample][ch]["psnrs"]
            best_idx = int(np.argmax(psnrs))
            ax.plot(thetas, psnrs, marker=markers[sample], label=sample, linewidth=1.5, markersize=5)
            ax.plot(
                thetas[best_idx],
                psnrs[best_idx],
                marker="*",
                markersize=14,
                color=ax.get_lines()[-1].get_color(),
                zorder=5,
            )
        ax.set_xlabel("Half-angle theta (degrees)", fontsize=12)
        ax.set_title(ch, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(THETA_RANGE)

    axes[0].set_ylabel("PSNR (dB)", fontsize=12)
    fig.suptitle("Two-View Wave-Optical Reconstruction: PSNR vs Projection Angle", fontsize=14, y=1.02)

    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "psnr_vs_theta_wave_two_projections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Plot: Detailed figure at optimal theta
# ---------------------------------------------------------------------------
def plot_optimal_reconstruction(sample, ch, gt, source, best_theta, reg, niter, blur_ops, out_dir):
    """Projections at +/-theta, gain-calibrated reconstruction, object, FFTs.

    Layout: 3 columns (Projections, Reconstruction, Object) × 4 rows
    (XY, XZ, YZ, FFT XZ) — same as geometric two-projection figures.
    """
    fwd_blur, adj_blur = blur_ops[ch]
    angles = [-best_theta, best_theta]
    siddon_op = SiddonOperator(ZYX_SHAPE, angles, VOXEL_SIZE, DEVICE)
    recon, measurements = _reconstruct_wave(siddon_op, source, reg, niter, fwd_blur, adj_blur)
    alpha = _optimal_gain(recon, gt)
    rec_s = alpha * recon
    psnr_val = _psnr(rec_s, gt)

    proj_neg = measurements[0].cpu().numpy()
    proj_pos = measurements[1].cpu().numpy()

    # Clip projection display at 1st percentile to reveal structure
    # in low-contrast channels (e.g. phase)
    all_proj = np.concatenate([proj_neg.ravel(), proj_pos.ravel()])
    proj_vmin = float(np.percentile(all_proj, 1))
    proj_vmax = float(all_proj.max())

    obj_vmin, obj_vmax = float(gt.min()), float(gt.max())
    rec_vmin, rec_vmax = float(rec_s.min()), float(rec_s.max())

    ft_rec = _log_ft(rec_s)
    ft_obj = _log_ft(gt)
    ft_max = max(float(ft_rec.max()), float(ft_obj.max()))

    rec_sl = _ortho(rec_s)
    obj_sl = _ortho(gt)
    ft_rec_sl = _ortho(ft_rec)
    ft_obj_sl = _ortho(ft_obj)
    views = ["XY (z=center)", "XZ (y=center)", "YZ (x=center)"]

    # Physical extent in micrometers for axis labeling
    nz, ny, nx = ZYX_SHAPE
    phys_z = nz * VOXEL_SIZE
    phys_y = ny * VOXEL_SIZE
    phys_x = nx * VOXEL_SIZE
    ext_xy = [0, phys_x, phys_y, 0]
    ext_xz = [0, phys_x, phys_z, 0]
    ext_yz = [0, phys_y, phys_z, 0]
    vol_extents = [ext_xy, ext_xz, ext_yz]
    axis_labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

    fig, axes = plt.subplots(4, 3, figsize=(14, 17))
    fig.suptitle(
        f"{sample} / {ch} — Two-View Wave-Optical Reconstruction at +/-{best_theta} deg "
        f"(PSNR {psnr_val:.1f} dB, gain={alpha:.6f})",
        fontsize=13,
        y=0.99,
    )

    col_titles = ["Projections", "Reconstruction", "Object"]

    # Row 0 (XY): projection at +theta
    axes[0, 0].imshow(proj_pos, cmap="gray", vmin=proj_vmin, vmax=proj_vmax, aspect="equal")
    _style_ax(axes[0, 0], title=col_titles[0], ylabel=f"+{best_theta} deg")
    axes[0, 1].imshow(rec_sl[0], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_xy)
    _style_ax(axes[0, 1], title=col_titles[1], ylabel=views[0])
    axes[0, 2].imshow(obj_sl[0], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_xy)
    _style_ax(axes[0, 2], title=col_titles[2])

    # Row 1 (XZ): projection at -theta
    axes[1, 0].imshow(proj_neg, cmap="gray", vmin=proj_vmin, vmax=proj_vmax, aspect="equal")
    _style_ax(axes[1, 0], ylabel=f"-{best_theta} deg")
    axes[1, 1].imshow(rec_sl[1], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_xz)
    _style_ax(axes[1, 1], ylabel=views[1])
    axes[1, 2].imshow(obj_sl[1], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_xz)
    _style_ax(axes[1, 2])

    # Row 2 (YZ): blank projection slot
    axes[2, 0].axis("off")
    axes[2, 1].imshow(rec_sl[2], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_yz)
    _style_ax(axes[2, 1], ylabel=views[2])
    axes[2, 2].imshow(obj_sl[2], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_yz)
    _style_ax(axes[2, 2])

    # Add scale ticks (um) to reconstruction and object columns
    for row in range(3):
        ext = vol_extents[row]
        xlbl, ylbl = axis_labels[row]
        for col in [1, 2]:
            ax = axes[row, col]
            ax.set_xticks([0, ext[1] / 2, ext[1]])
            ax.set_xticklabels(["0", f"{ext[1]/2:.1f}", f"{ext[1]:.1f}"], fontsize=7)
            ax.set_yticks([0, ext[3] if ext[3] > 0 else ext[2]])
            ax.set_yticklabels(["0", f"{max(ext[2], ext[3]):.1f}"], fontsize=7)
            if row == 2:
                ax.set_xlabel(f"{xlbl} (um)", fontsize=8)

    # Row 3: FFT XZ slices (ky=0)
    axes[3, 0].axis("off")
    axes[3, 1].imshow(ft_rec_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 1], title="FFT Reconstruction")
    axes[3, 2].imshow(ft_obj_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 2], title="FFT Object")
    axes[3, 1].set_ylabel("FFT XZ (ky=0)", fontsize=10)

    plt.tight_layout()
    out = Path(out_dir) / sample
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_{ch}_wave_two_projections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved {path}")

    del siddon_op
    torch.cuda.empty_cache()
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
CONTEXT = {"help_option_names": ["-h", "--help"]}


@click.command(context_settings=CONTEXT)
@click.option("--data-dir", default="./data", show_default=True, help="Directory containing the zarr store.")
@click.option("--out-dir", default="./plots", show_default=True, help="Output directory for figures.")
@click.option("--reg", type=float, default=REG_STRENGTH, show_default=True, help="Tikhonov regularization lambda.")
@click.option("--niter", type=int, default=N_ITER, show_default=True, help="CG iterations.")
def main(data_dir, out_dir, reg, niter):
    """Sweep theta from 5 to 70 deg: wave-two-projections reconstruction PSNR vs angle."""
    store_path = Path(data_dir) / "projection_modeling.zarr"
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    click.echo(f"Device: {DEVICE}")
    click.echo(f"Sweeping theta = {THETA_RANGE[0]} to {THETA_RANGE[-1]} deg")
    click.echo(f"CG: {niter} iters, lambda = {reg}")

    # Compute transfer functions once
    blur_ops = _compute_transfer_functions()

    # Phase 1: sweep all thetas
    results, objects, rawimages = run_sweep(store_path, reg, niter, blur_ops)

    # Phase 2: PSNR vs theta plot
    plot_psnr_vs_theta(results, out_dir)

    # Phase 3: detailed figures at optimal theta (by fluorescence PSNR)
    for sample in SAMPLE_TYPES:
        best_theta = results[sample]["Fluorescence"]["best_theta"]
        click.echo(f"\n{sample}: plotting optimal reconstruction at +/-{best_theta} deg")
        for ch in CHANNEL_NAMES:
            plot_optimal_reconstruction(
                sample, ch, objects[sample][ch], rawimages[sample][ch],
                best_theta, reg, niter, blur_ops, out_dir,
            )

    # Print summary table
    click.echo("\n" + "=" * 70)
    click.echo("Summary: PSNR vs theta (wave-two-projections)")
    click.echo("=" * 70)
    for ch in CHANNEL_NAMES:
        click.echo(f"\n{ch}:")
        header = f"{'Theta':>6}"
        for s in SAMPLE_TYPES:
            header += f"  {s:>12}"
        click.echo(header)
        for i, theta in enumerate(THETA_RANGE):
            row = f"{theta:>6}"
            for s in SAMPLE_TYPES:
                row += f"  {results[s][ch]['psnrs'][i]:>12.2f}"
            click.echo(row)

    click.echo("\nOptimal angles (by Fluorescence PSNR):")
    for s in SAMPLE_TYPES:
        fluor = results[s]["Fluorescence"]
        phase = results[s]["Phase"]
        click.echo(
            f"  {s}: +/-{fluor['best_theta']} deg -> "
            f"Fluor {fluor['best_psnr']:.2f} dB, "
            f"Phase {phase['psnrs'][results[s]['Fluorescence']['thetas'].index(fluor['best_theta'])]:.2f} dB"
        )

    click.echo("\nDone.")


if __name__ == "__main__":
    main()

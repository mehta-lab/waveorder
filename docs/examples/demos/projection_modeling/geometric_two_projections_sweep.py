"""Two-projection geometric reconstruction sweep: PSNR vs half-angle theta.

For each half-angle theta in [5, 10, 15, ..., 70], reconstructs all three
phantoms from a ±theta projection pair using Siddon-only CG-Tikhonov.
Produces:

1. PSNR vs theta plot (all three samples on one figure).
2. Per-sample detailed figure at the optimal theta: projections at ±theta,
   reconstruction, object, difference, and XZ Fourier cross-sections.

Usage::

    cd docs/examples/demos/projection_modeling/
    uv run python geometric_two_projections_sweep.py --data-dir ./data --out-dir ./plots

"""

from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from iohub.ngff import open_ome_zarr

from waveorder.projection import SiddonOperator, cg_tikhonov

matplotlib.use("Agg")
plt.style.use("dark_background")

# ---------------------------------------------------------------------------
# Constants (match projection_modeling.py / projection_reconstruction.py)
# ---------------------------------------------------------------------------
VOXEL_SIZE = 0.05
ZYX_SHAPE = (256, 256, 256)
SAMPLE_TYPES = ["point", "lines", "shepplogan"]
CHANNEL_NAMES = ["Fluorescence", "Phase"]

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


def _reconstruct_single(siddon_op, source_vol_np, reg, niter):
    """CG-Tikhonov reconstruction from precomputed SiddonOperator.

    Subtracts per-projection mean before solving: with only two views
    the common DC background carries no angular diversity.
    """
    device = siddon_op.device
    source_t = torch.tensor(source_vol_np, dtype=torch.float32, device=device)
    measurements = siddon_op.project_all(source_t)
    del source_t

    measurements = [p - p.mean() for p in measurements]

    def forward(vol):
        return siddon_op.project_all(vol)

    def adjoint(projs):
        return siddon_op.backproject_all(projs, ramp_filter=True)

    recon = cg_tikhonov(forward, adjoint, measurements, ZYX_SHAPE, reg, niter, device)
    return recon.cpu().numpy(), measurements


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(store_path, reg, niter):
    """Run geometric-two-projections reconstruction at every theta, return PSNR table.

    Returns
    -------
    results : dict
        {sample: {"thetas": [...], "psnrs": [...], "best_theta": int, "best_psnr": float}}
    objects : dict
        {sample: np.ndarray} ground-truth volumes (channel 0).
    """
    results = {s: {"thetas": [], "psnrs": []} for s in SAMPLE_TYPES}
    objects = {}

    with open_ome_zarr(str(store_path), mode="r") as plate:
        for sample in SAMPLE_TYPES:
            obj = np.array(plate[f"{sample}/object/0"]["0"][0, 0], dtype=np.float32)
            objects[sample] = obj

    for theta in THETA_RANGE:
        angles = [-theta, theta]
        click.echo(f"\nTheta = {theta} deg (angles: {angles})")
        siddon_op = SiddonOperator(ZYX_SHAPE, angles, VOXEL_SIZE, DEVICE)

        for sample in SAMPLE_TYPES:
            obj = objects[sample]
            recon, _ = _reconstruct_single(siddon_op, obj, reg, niter)
            alpha = _optimal_gain(recon, obj)
            psnr_val = _psnr(alpha * recon, obj)
            results[sample]["thetas"].append(theta)
            results[sample]["psnrs"].append(psnr_val)
            click.echo(f"  {sample}: PSNR = {psnr_val:.2f} dB (gain = {alpha:.4f})")

        del siddon_op
        torch.cuda.empty_cache()

    # Identify best theta per sample
    for sample in SAMPLE_TYPES:
        psnrs = results[sample]["psnrs"]
        best_idx = int(np.argmax(psnrs))
        results[sample]["best_theta"] = results[sample]["thetas"][best_idx]
        results[sample]["best_psnr"] = psnrs[best_idx]
        click.echo(
            f"\n{sample}: best theta = {results[sample]['best_theta']} deg, PSNR = {results[sample]['best_psnr']:.2f} dB"
        )

    return results, objects


# ---------------------------------------------------------------------------
# Plot: PSNR vs theta
# ---------------------------------------------------------------------------
def plot_psnr_vs_theta(results, out_dir):
    """Line plot: PSNR vs half-angle theta for all three samples."""
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = {"point": "o", "lines": "s", "shepplogan": "^"}
    for sample in SAMPLE_TYPES:
        thetas = results[sample]["thetas"]
        psnrs = results[sample]["psnrs"]
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
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Two-View Geometric Reconstruction: PSNR vs Projection Angle", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THETA_RANGE)

    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "psnr_vs_theta_geometric_two_projections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Plot: Detailed figure at optimal theta
# ---------------------------------------------------------------------------
def plot_optimal_reconstruction(sample, obj, best_theta, reg, niter, out_dir):
    """Object, projections at +/-theta, reconstruction, and FFT.

    Layout: 4 columns (Object, Projections, Reconstruction, FFT) × 3 rows
    (XY, XZ, YZ).
    """
    angles = [-best_theta, best_theta]
    siddon_op = SiddonOperator(ZYX_SHAPE, angles, VOXEL_SIZE, DEVICE)
    recon, measurements = _reconstruct_single(siddon_op, obj, reg, niter)
    alpha = _optimal_gain(recon, obj)
    rec_s = alpha * recon
    psnr_val = _psnr(rec_s, obj)

    proj_neg = measurements[0].cpu().numpy()
    proj_pos = measurements[1].cpu().numpy()

    # Clip projection display at 1st percentile to reveal structure
    all_proj = np.concatenate([proj_neg.ravel(), proj_pos.ravel()])
    proj_vmin = float(np.percentile(all_proj, 1))
    proj_vmax = float(all_proj.max())

    obj_vmin, obj_vmax = float(obj.min()), float(obj.max())
    rec_vmin, rec_vmax = float(rec_s.min()), float(rec_s.max())

    ft_rec = _log_ft(rec_s)
    ft_obj = _log_ft(obj)
    ft_max = max(float(ft_rec.max()), float(ft_obj.max()))

    rec_sl = _ortho(rec_s)
    obj_sl = _ortho(obj)
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

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        f"{sample} — Two-View Geometric Reconstruction at +/-{best_theta} deg "
        f"(PSNR {psnr_val:.1f} dB, gain={alpha:.4f})",
        fontsize=13,
        y=0.99,
    )

    col_titles = ["Object", "Projections", "Reconstruction", "FFT kz-kx"]

    # Row 0 (XY): object + projection at +theta + recon + FFT object
    axes[0, 0].imshow(obj_sl[0], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_xy)
    _style_ax(axes[0, 0], title=col_titles[0], ylabel=views[0])
    axes[0, 1].imshow(proj_pos, cmap="gray", vmin=proj_vmin, vmax=proj_vmax, aspect="equal")
    _style_ax(axes[0, 1], title=col_titles[1], ylabel=f"+{best_theta} deg")
    axes[0, 2].imshow(rec_sl[0], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_xy)
    _style_ax(axes[0, 2], title=col_titles[2])
    axes[0, 3].imshow(ft_obj_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[0, 3], title=col_titles[3], ylabel="Object")

    # Row 1 (XZ): object + projection at -theta + recon + FFT recon
    axes[1, 0].imshow(obj_sl[1], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_xz)
    _style_ax(axes[1, 0], ylabel=views[1])
    axes[1, 1].imshow(proj_neg, cmap="gray", vmin=proj_vmin, vmax=proj_vmax, aspect="equal")
    _style_ax(axes[1, 1], ylabel=f"-{best_theta} deg")
    axes[1, 2].imshow(rec_sl[1], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_xz)
    _style_ax(axes[1, 2])
    axes[1, 3].imshow(ft_rec_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[1, 3], ylabel="Recon")

    # Row 2 (YZ): object + blank + recon + blank
    axes[2, 0].imshow(obj_sl[2], cmap="inferno", vmin=obj_vmin, vmax=obj_vmax, aspect="equal", extent=ext_yz)
    _style_ax(axes[2, 0], ylabel=views[2])
    axes[2, 1].axis("off")
    axes[2, 2].imshow(rec_sl[2], cmap="inferno", vmin=rec_vmin, vmax=rec_vmax, aspect="equal", extent=ext_yz)
    _style_ax(axes[2, 2])
    axes[2, 3].axis("off")

    # Add scale ticks (um) to object and reconstruction columns
    for row in range(3):
        ext = vol_extents[row]
        xlbl, ylbl = axis_labels[row]
        for col in [0, 2]:
            ax = axes[row, col]
            ax.set_xticks([0, ext[1] / 2, ext[1]])
            ax.set_xticklabels(["0", f"{ext[1] / 2:.1f}", f"{ext[1]:.1f}"], fontsize=7)
            ax.set_yticks([0, ext[3] if ext[3] > 0 else ext[2]])
            ax.set_yticklabels(["0", f"{max(ext[2], ext[3]):.1f}"], fontsize=7)
            if row == 2:
                ax.set_xlabel(f"{xlbl} (\u00b5m)", fontsize=8)

    # Label projection column (col 1)
    axes[1, 1].set_xlabel(r"$X_d$ (\u00b5m)", fontsize=8)

    # Label FFT column (col 3)
    for row in [0, 1]:
        axes[row, 3].set_xlabel(r"$k_x$", fontsize=9)
        axes[row, 3].set_ylabel(r"$k_z$", fontsize=9)

    plt.tight_layout()
    out = Path(out_dir) / sample
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_geometric_two_projections.png"
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
    """Sweep theta from 5 to 70 deg: geometric-two-projections reconstruction PSNR vs angle."""
    store_path = Path(data_dir) / "projection_modeling.zarr"
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}")

    click.echo(f"Device: {DEVICE}")
    click.echo(f"Sweeping theta = {THETA_RANGE[0]} to {THETA_RANGE[-1]} deg")
    click.echo(f"CG: {niter} iters, lambda = {reg}")

    # Phase 1: sweep all thetas
    results, objects = run_sweep(store_path, reg, niter)

    # Phase 2: PSNR vs theta plot
    plot_psnr_vs_theta(results, out_dir)

    # Phase 3: detailed figures at optimal theta
    for sample in SAMPLE_TYPES:
        best_theta = results[sample]["best_theta"]
        click.echo(f"\n{sample}: plotting optimal reconstruction at ±{best_theta} deg")
        plot_optimal_reconstruction(sample, objects[sample], best_theta, reg, niter, out_dir)

    # Print summary table
    click.echo("\n" + "=" * 60)
    click.echo("Summary: PSNR vs theta (geometric-two-projections, Fluorescence)")
    click.echo("=" * 60)
    header = f"{'Theta':>6}"
    for s in SAMPLE_TYPES:
        header += f"  {s:>12}"
    click.echo(header)
    for i, theta in enumerate(THETA_RANGE):
        row = f"{theta:>6}"
        for s in SAMPLE_TYPES:
            row += f"  {results[s]['psnrs'][i]:>12.2f}"
        click.echo(row)

    click.echo("\nOptimal angles:")
    for s in SAMPLE_TYPES:
        click.echo(f"  {s}: ±{results[s]['best_theta']} deg → {results[s]['best_psnr']:.2f} dB")

    click.echo("\nDone.")


if __name__ == "__main__":
    main()

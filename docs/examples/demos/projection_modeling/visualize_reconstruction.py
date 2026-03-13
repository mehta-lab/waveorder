"""Visualize projection modeling results.

Figure types:

1. **Forward simulation** (per sample, per channel): object, rawimage,
   and three representative projections (0 deg, +15 deg, -15 deg).
2. **Geometric reconstruction** (per sample, channel-independent):
   sinogram, reconstruction, object, difference, and Fourier XZ slices.
3. **Wave reconstruction** (per sample, per channel): sinogram,
   reconstruction, object, diff vs OTF-blurred object, diff vs object,
   and Fourier XZ slices.

All volume panels show three orthogonal center slices (XY, XZ, YZ).
FFT row shows kz-kx (XZ) slices to reveal the missing cone.
Dark theme: black background, white labels.

Usage::

    uv run python visualize_reconstruction.py plot   --sample all
    uv run python visualize_reconstruction.py plot   --sample point --channel 0
    uv run python visualize_reconstruction.py napari --sample point
"""

from pathlib import Path

import click
import numpy as np
from iohub.ngff import open_ome_zarr

# ---------------------------------------------------------------------------
# Constants (match projection_modeling.py / projection_reconstruction.py)
# ---------------------------------------------------------------------------
SAMPLES = ["point", "lines", "shepplogan"]
CHANNEL_NAMES = ["Fluorescence", "Phase"]
VOXEL_SIZE = 0.05
ZYX_SHAPE = (256, 256, 256)
PROJECTION_ANGLES = list(range(-70, 75, 5))
BLACK_LEVEL = 100

WAVELENGTH_ILLUMINATION = 0.500
WAVELENGTH_EMISSION = 0.520
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
MEDIA_INDEX = 1.33
Z_PADDING = 0

CONTEXT = {"help_option_names": ["-h", "--help"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _optimal_gain(recon, gt):
    """Least-squares gain minimizing ||a*recon - gt||^2."""
    dot_rg = np.sum(recon * gt)
    dot_rr = np.sum(recon * recon)
    return dot_rg / dot_rr if dot_rr > 0 else 1.0


def _log_ft(vol):
    """Centered log-magnitude 3D Fourier spectrum."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(vol))))


def _psnr(recon, gt):
    mse = float(np.mean((recon - gt) ** 2))
    gt_range = float(gt.max() - gt.min())
    return 10 * np.log10(gt_range**2 / mse) if mse > 0 else float("inf")


def _ortho(vol):
    """Return (XY, XZ, YZ) center slices of a 3D volume."""
    nz, ny, nx = vol.shape
    return vol[nz // 2], vol[:, ny // 2, :], vol[:, :, nx // 2]


def _compute_projection_stack(volume, angles, voxel_size):
    """Compute full 2D Siddon sum-projections at all angles.

    Returns (n_angles, ny, max_width) array, center-padded to uniform width.
    """
    from siddon import siddon_project

    projs = []
    max_w = 0
    for angle in angles:
        proj = siddon_project(volume, angle, voxel_size, mode="sum")
        projs.append(proj)
        max_w = max(max_w, proj.shape[1])

    ny = volume.shape[1]
    stack = np.zeros((len(angles), ny, max_w), dtype=np.float32)
    for i, proj in enumerate(projs):
        w = proj.shape[1]
        pad_l = (max_w - w) // 2
        stack[i, :, pad_l : pad_l + w] = proj
    return stack


def _stored_sinogram(plate, sample, c_idx):
    """Extract sinogram from stored projection stack (center-Y line per angle)."""
    proj = np.array(plate[f"{sample}/projections/0"]["0"][0, c_idx], dtype=np.float32)
    cy = proj.shape[1] // 2
    return proj[:, cy, :]


def _compute_otf(c_idx):
    """Compute 3D transfer function for the given channel.

    Returns (otf_tensor, device).
    """
    import torch
    from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if c_idx == 0:
        otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
            zyx_shape=ZYX_SHAPE,
            yx_pixel_size=VOXEL_SIZE,
            z_pixel_size=VOXEL_SIZE,
            wavelength_emission=WAVELENGTH_EMISSION,
            z_padding=Z_PADDING,
            index_of_refraction_media=MEDIA_INDEX,
            numerical_aperture_detection=NA_DETECTION,
        ).to(device)
    else:
        real_tf, _ = phase_thick_3d.calculate_transfer_function(
            zyx_shape=ZYX_SHAPE,
            yx_pixel_size=VOXEL_SIZE,
            z_pixel_size=VOXEL_SIZE,
            wavelength_illumination=WAVELENGTH_ILLUMINATION,
            z_padding=Z_PADDING,
            index_of_refraction_media=MEDIA_INDEX,
            numerical_aperture_illumination=NA_ILLUMINATION,
            numerical_aperture_detection=NA_DETECTION,
        )
        otf = real_tf.to(device)

    return otf, device


def _apply_otf(volume, otf, device):
    """Convolve a 3D volume with an OTF via FFT. Returns numpy array."""
    import torch

    vol_t = torch.tensor(volume, dtype=torch.float32, device=device)
    blurred = torch.fft.ifftn(torch.fft.fftn(vol_t) * otf).real.cpu().numpy()
    return blurred


def _style_ax(ax, title=None, ylabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


# ---------------------------------------------------------------------------
# Figure: Forward simulation
# ---------------------------------------------------------------------------
def _plot_forward(plate, sample, c_idx, out_dir):
    """Object, rawimage, and three representative projections (0, +15, -15 deg)."""
    import matplotlib.pyplot as plt

    ch = CHANNEL_NAMES[c_idx]
    obj = np.array(plate[f"{sample}/object/0"]["0"][0, c_idx], dtype=np.float32)
    raw = np.array(plate[f"{sample}/rawimage/0"]["0"][0, c_idx], dtype=np.float32)
    proj_stack = np.array(
        plate[f"{sample}/projections/0"]["0"][0, c_idx], dtype=np.float32
    )

    obj_sl = _ortho(obj)
    raw_sl = _ortho(raw)
    views = ["XY (z=center)", "XZ (y=center)", "YZ (x=center)"]

    # Projection indices for 0, +15, -15 degrees
    proj_angles_show = [0, 15, -15]
    proj_indices = [PROJECTION_ANGLES.index(a) for a in proj_angles_show]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"{sample} / {ch} — Forward Simulation", fontsize=14, y=0.98)

    for row in range(3):
        axes[row, 0].imshow(obj_sl[row], cmap="gray", aspect="equal")
        _style_ax(axes[row, 0], title="Object" if row == 0 else None, ylabel=views[row])
        axes[row, 1].imshow(raw_sl[row], cmap="gray", aspect="equal")
        _style_ax(axes[row, 1], title="Rawimage (blurred + noise)" if row == 0 else None)

        # Rightmost column: projections at 0, +15, -15 degrees
        idx = proj_indices[row]
        angle = proj_angles_show[row]
        axes[row, 2].imshow(proj_stack[idx], cmap="gray", aspect="equal")
        _style_ax(
            axes[row, 2],
            title="Projections" if row == 0 else None,
            ylabel=f"{angle} deg",
        )

    plt.tight_layout()
    out = Path(out_dir) / sample
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_{ch}_forward.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure: Geometric reconstruction (channel-independent)
# ---------------------------------------------------------------------------
def _plot_geometric(plate, sample, out_dir, stack_cache):
    """Sinogram orthogonal views, reconstruction, object, difference, FFTs.

    Geometric reconstruction is identical for both channels, so this
    generates one figure per sample using channel 0.

    Column 0 shows three orthogonal slices through the projection stack
    (angle, Y, X), matching the volume views:
      Row 0: projection at 0 deg — (Y, X)
      Row 1: sinogram at Y=center — (angle, X)
      Row 2: sinogram at X=center — (angle, Y)
    """
    import matplotlib.pyplot as plt

    c_idx = 0  # both channels reconstruct identically
    obj = np.array(plate[f"{sample}/object/0"]["0"][0, c_idx], dtype=np.float32)
    rec = np.array(plate[f"{sample}/recongeo/0"]["0"][0, c_idx], dtype=np.float32)

    alpha = _optimal_gain(rec, obj)
    rec_s = alpha * rec
    psnr_val = _psnr(rec_s, obj)

    # Full projection stack: (n_angles, ny, nx_padded)
    if sample not in stack_cache:
        click.echo(f"  Computing projection stack for {sample} (Siddon, 29 angles)...")
        stack_cache[sample] = _compute_projection_stack(obj, PROJECTION_ANGLES, VOXEL_SIZE)
    stack = stack_cache[sample]

    n_angles, ny_s, nx_s = stack.shape
    ca = n_angles // 2  # center angle index (0 deg)
    cy_s = ny_s // 2
    cx_s = nx_s // 2

    diff = rec_s - obj
    vmin, vmax = float(obj.min()), float(obj.max())
    dlim = max(float(np.abs(diff).max()) * 0.5, 1e-10)

    ft_rec = _log_ft(rec_s)
    ft_obj = _log_ft(obj)
    ft_max = max(float(ft_rec.max()), float(ft_obj.max()))

    obj_sl = _ortho(obj)
    rec_sl = _ortho(rec_s)
    diff_sl = _ortho(diff)
    ft_rec_sl = _ortho(ft_rec)
    ft_obj_sl = _ortho(ft_obj)
    views = ["XY (z=center)", "XZ (y=center)", "YZ (x=center)"]

    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    fig.suptitle(
        f"{sample} — Limited-Angle Tomography "
        f"(PSNR {psnr_val:.1f} dB, gain={alpha:.4f})",
        fontsize=13,
        y=0.99,
    )

    col_titles = ["Measurements", "Reconstruction", "Object", "Rec - Object"]

    # Row 0 (XY): projection at 0 deg + volume slices
    axes[0, 0].imshow(stack[ca], cmap="gray", aspect="equal")
    _style_ax(axes[0, 0], title=col_titles[0], ylabel=f"Proj at {PROJECTION_ANGLES[ca]} deg")
    axes[0, 1].imshow(rec_sl[0], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[0, 1], title=col_titles[1], ylabel=views[0])
    axes[0, 2].imshow(obj_sl[0], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[0, 2], title=col_titles[2])
    axes[0, 3].imshow(diff_sl[0], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    _style_ax(axes[0, 3], title=col_titles[3])

    # Row 1 (XZ): sinogram at Y=center (angle vs X)
    axes[1, 0].imshow(
        stack[:, cy_s, :],
        cmap="gray",
        aspect="auto",
        extent=[0, nx_s, PROJECTION_ANGLES[-1], PROJECTION_ANGLES[0]],
    )
    _style_ax(axes[1, 0], ylabel="Sino (angle, X)\nat Y=center")
    axes[1, 1].imshow(rec_sl[1], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[1, 1], ylabel=views[1])
    axes[1, 2].imshow(obj_sl[1], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[1, 2])
    axes[1, 3].imshow(diff_sl[1], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    _style_ax(axes[1, 3])

    # Row 2 (YZ): sinogram at X=center (angle vs Y)
    axes[2, 0].imshow(
        stack[:, :, cx_s],
        cmap="gray",
        aspect="auto",
        extent=[0, ny_s, PROJECTION_ANGLES[-1], PROJECTION_ANGLES[0]],
    )
    _style_ax(axes[2, 0], ylabel="Sino (angle, Y)\nat X=center")
    axes[2, 1].imshow(rec_sl[2], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[2, 1], ylabel=views[2])
    axes[2, 2].imshow(obj_sl[2], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[2, 2])
    axes[2, 3].imshow(diff_sl[2], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    _style_ax(axes[2, 3])

    # Row 3: FFT XZ slices (ky=0)
    axes[3, 0].axis("off")
    axes[3, 1].imshow(ft_rec_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 1], title="FFT Reconstruction")
    axes[3, 2].imshow(ft_obj_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 2], title="FFT Object")
    axes[3, 3].axis("off")
    axes[3, 1].set_ylabel("FFT XZ (ky=0)", fontsize=10)

    plt.tight_layout()
    out = Path(out_dir) / sample
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_geometric.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure: Wave-optical reconstruction
# ---------------------------------------------------------------------------
def _plot_wave(plate, sample, c_idx, out_dir, otf, device):
    """Sinogram, reconstruction, object, diffs (vs blur and vs object), FFTs."""
    import matplotlib.pyplot as plt

    ch = CHANNEL_NAMES[c_idx]
    obj = np.array(plate[f"{sample}/object/0"]["0"][0, c_idx], dtype=np.float32)
    raw = np.array(plate[f"{sample}/rawimage/0"]["0"][0, c_idx], dtype=np.float32)
    rec = np.array(plate[f"{sample}/reconwave/0"]["0"][0, c_idx], dtype=np.float32)

    alpha = _optimal_gain(rec, obj)
    rec_s = alpha * rec
    psnr_val = _psnr(rec_s, obj)

    blurred = _apply_otf(obj, otf, device)
    sino = _stored_sinogram(plate, sample, c_idx)

    diff_blur = rec_s - blurred
    diff_obj = rec_s - obj
    vmin, vmax = float(obj.min()), float(obj.max())
    dlim = max(
        float(np.abs(diff_blur).max()) * 0.5,
        float(np.abs(diff_obj).max()) * 0.5,
        1e-10,
    )

    ft_rec = _log_ft(rec_s)
    ft_obj = _log_ft(obj)
    ft_blur = _log_ft(blurred)
    ft_raw = _log_ft(raw - float(raw.min()))
    ft_max = max(float(ft_rec.max()), float(ft_obj.max()), float(ft_blur.max()))

    rec_sl = _ortho(rec_s)
    obj_sl = _ortho(obj)
    db_sl = _ortho(diff_blur)
    do_sl = _ortho(diff_obj)
    ft_rec_sl = _ortho(ft_rec)
    ft_obj_sl = _ortho(ft_obj)
    ft_blur_sl = _ortho(ft_blur)
    ft_raw_sl = _ortho(ft_raw)
    views = ["XY (z=center)", "XZ (y=center)", "YZ (x=center)"]

    fig, axes = plt.subplots(4, 5, figsize=(22, 17))
    fig.suptitle(
        f"{sample} / {ch} — Wave-Optical Reconstruction "
        f"(PSNR {psnr_val:.1f} dB, gain={alpha:.6f})",
        fontsize=13,
        y=0.99,
    )

    col_titles = [
        "Measurements",
        "Reconstruction",
        "Object",
        "Rec - OTF*Obj",
        "Rec - Object",
    ]

    # Row 0 (XY): sinogram + volume slices
    axes[0, 0].imshow(
        sino,
        cmap="gray",
        aspect="auto",
        extent=[0, sino.shape[1], PROJECTION_ANGLES[-1], PROJECTION_ANGLES[0]],
    )
    axes[0, 0].set_ylabel("Angle (deg)", fontsize=9)
    axes[0, 0].set_title(col_titles[0], fontsize=10)
    axes[0, 1].imshow(rec_sl[0], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[0, 1], title=col_titles[1])
    axes[0, 2].imshow(obj_sl[0], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    _style_ax(axes[0, 2], title=col_titles[2])
    axes[0, 3].imshow(db_sl[0], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    _style_ax(axes[0, 3], title=col_titles[3])
    axes[0, 4].imshow(do_sl[0], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    _style_ax(axes[0, 4], title=col_titles[4])

    # Rows 1-2 (XZ, YZ)
    for row in range(1, 3):
        axes[row, 0].axis("off")
        axes[row, 1].imshow(rec_sl[row], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        _style_ax(axes[row, 1], ylabel=views[row])
        axes[row, 2].imshow(obj_sl[row], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        _style_ax(axes[row, 2])
        axes[row, 3].imshow(db_sl[row], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
        _style_ax(axes[row, 3])
        axes[row, 4].imshow(do_sl[row], cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
        _style_ax(axes[row, 4])
    axes[0, 1].set_ylabel(views[0], fontsize=10)

    # Row 3: FFT XZ slices (ky=0)
    axes[3, 0].imshow(ft_raw_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 0], title="FFT Rawimage")
    axes[3, 1].imshow(ft_rec_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 1], title="FFT Reconstruction")
    axes[3, 2].imshow(ft_obj_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 2], title="FFT Object")
    axes[3, 3].imshow(ft_blur_sl[1], cmap="inferno", vmin=0, vmax=ft_max, aspect="equal")
    _style_ax(axes[3, 3], title="FFT OTF*Object")
    axes[3, 4].axis("off")
    axes[3, 0].set_ylabel("FFT XZ (ky=0)", fontsize=10)

    plt.tight_layout()
    out = Path(out_dir) / sample
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_{ch}_wave.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.group(context_settings=CONTEXT)
def cli():
    """Visualize projection modeling results."""


@cli.command()
@click.option("--sample", default="all", show_default=True, help="Sample name or 'all'.")
@click.option("--channel", default=-1, show_default=True, help="Channel index (0/1) or -1 for both.")
@click.option("--data-dir", default="./data", show_default=True)
@click.option("--out-dir", default="./plots", show_default=True)
def plot(sample, channel, data_dir, out_dir):
    """Generate forward-simulation, geometric, and wave reconstruction figures."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    plt.style.use("dark_background")

    store_path = Path(data_dir) / "projection_modeling.zarr"
    samples = SAMPLES if sample == "all" else [sample]
    channels = [0, 1] if channel < 0 else [channel]

    # Pre-compute OTFs for wave figures
    click.echo("Computing transfer functions...")
    otf_cache = {}
    for c in channels:
        click.echo(f"  {CHANNEL_NAMES[c]}...")
        otf_cache[c] = _compute_otf(c)

    stack_cache = {}  # sample -> projection stack (geometric, shared across channels)

    with open_ome_zarr(str(store_path), mode="r") as plate:
        for s in samples:
            # Forward and wave: per channel
            for c in channels:
                ch = CHANNEL_NAMES[c]
                click.echo(f"\n{'=' * 50}")
                click.echo(f"{s} / {ch}")
                click.echo(f"{'=' * 50}")

                _plot_forward(plate, s, c, out_dir)
                otf, device = otf_cache[c]
                _plot_wave(plate, s, c, out_dir, otf, device)

            # Geometric: once per sample (channel-independent)
            click.echo(f"\n  {s} — geometric (channel-independent)")
            _plot_geometric(plate, s, out_dir, stack_cache)

    click.echo("\nDone.")


@cli.command()
@click.option("--sample", default="point", show_default=True)
@click.option("--channel", default=0, show_default=True, help="0=Fluorescence, 1=Phase")
@click.option("--data-dir", default="./data", show_default=True)
def napari(sample, channel, data_dir):
    """Open napari with ground truth, geometric, and wave reconstructions."""
    import napari as nap

    store_path = Path(data_dir) / "projection_modeling.zarr"
    ch_name = CHANNEL_NAMES[channel]

    with open_ome_zarr(str(store_path), mode="r") as plate:
        gt = np.array(plate[f"{sample}/object/0"]["0"][0, channel], dtype=np.float32)
        rawimage = np.array(plate[f"{sample}/rawimage/0"]["0"][0, channel], dtype=np.float32)
        recongeo = np.array(plate[f"{sample}/recongeo/0"]["0"][0, channel], dtype=np.float32)
        reconwave = np.array(plate[f"{sample}/reconwave/0"]["0"][0, channel], dtype=np.float32)

    alpha_geo = _optimal_gain(recongeo, gt)
    alpha_wave = _optimal_gain(reconwave, gt)
    geo = alpha_geo * recongeo
    wave = alpha_wave * reconwave

    diff_geo = geo - gt
    diff_wave = wave - gt
    diff_lim = max(np.abs(diff_geo).max(), np.abs(diff_wave).max()) * 0.5

    ft_gt = _log_ft(gt)
    ft_geo = _log_ft(geo)
    ft_wave = _log_ft(wave)
    ft_max = max(ft_gt.max(), ft_geo.max(), ft_wave.max())

    viewer = nap.Viewer(title=f"{sample} / {ch_name} — reconstruction comparison")
    viewer.add_image(gt, name="ground truth", colormap="gray", visible=True)
    viewer.add_image(geo, name="geometric recon", colormap="gray", visible=False)
    viewer.add_image(wave, name="wave recon", colormap="gray", visible=False)
    viewer.add_image(rawimage, name="rawimage", colormap="gray", visible=False)
    viewer.add_image(
        diff_geo,
        name="diff: geo - gt",
        colormap="RdBu",
        visible=False,
        contrast_limits=(-diff_lim, diff_lim),
    )
    viewer.add_image(
        diff_wave,
        name="diff: wave - gt",
        colormap="RdBu",
        visible=False,
        contrast_limits=(-diff_lim, diff_lim),
    )
    viewer.add_image(ft_gt, name="FFT: gt", colormap="inferno", visible=False, contrast_limits=(0, ft_max))
    viewer.add_image(ft_geo, name="FFT: geo", colormap="inferno", visible=False, contrast_limits=(0, ft_max))
    viewer.add_image(ft_wave, name="FFT: wave", colormap="inferno", visible=False, contrast_limits=(0, ft_max))
    viewer.dims.current_step = (gt.shape[0] // 2,) + tuple(viewer.dims.current_step[1:])
    click.echo(f"Gain — geo: {alpha_geo:.6f}, wave: {alpha_wave:.6f}")

    nap.run()


if __name__ == "__main__":
    cli()

"""Projection modeling CLI: simulate phantoms, images, and projections.

Usage::

    python projection_modeling.py object   --data-dir ./data
    python projection_modeling.py image    --data-dir ./data
    python projection_modeling.py project  --data-dir ./data

Each subcommand builds on the previous stage stored in the OME-Zarr store
at ``<data-dir>/projection_modeling.zarr``.
"""

import shutil
from pathlib import Path

import click
import numpy as np
import torch
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from scipy.ndimage import gaussian_filter
from siddon import siddon_project

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
# Object
BEAD_RADIUS = 0.25  # um
BEAD_INDEX = 1.52
MEDIA_INDEX = 1.33
LINE_RADIUS = 0.25  # um
BRACKET_HALF_EXTENT = 2.0  # um
BRACKET_X_OFFSET = 2.0  # um
BRACKET_CAP_LENGTH = 0.6  # um
TORUS_MAJOR_RADIUS = 0.8  # um

# Volume
VOLUME_EXTENT = 12.8  # um  (256 voxels at 50 nm)
VOXEL_SIZE = 0.05  # um

# Imaging
WAVELENGTH_ILLUMINATION = 0.500  # um
WAVELENGTH_EMISSION = 0.520  # um
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
Z_PADDING = 0
BLACK_LEVEL = 100  # counts
PEAK_INTENSITY = 1024  # counts

# Derived
N = int(VOLUME_EXTENT / VOXEL_SIZE)  # 256
ZYX_SHAPE = (N, N, N)
SAMPLE_TYPES = ["point", "lines", "shepplogan"]
CHANNEL_NAMES = ["Fluorescence", "Phase"]
PROJECTION_ANGLES = list(range(-70, 75, 5))  # -70 to +70, step 5


# ---------------------------------------------------------------------------
# Phantom generators
# ---------------------------------------------------------------------------
def generate_isolated_bead(zyx_shape, voxel_size, bead_radius):
    """Single sphere at volume center. Returns binary volume (0/1)."""
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(zz**2 + yy**2 + xx**2)
    return (dist <= bead_radius).astype(np.float32)


def generate_line_pattern(
    zyx_shape, voxel_size, line_radius, bracket_half_extent, bracket_x_offset, bracket_cap_length, torus_major_radius
):
    """[o] pattern from bracket-shaped cylinders and a torus ring."""
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    volume = np.zeros(zyx_shape, dtype=np.float32)

    def _add_cylinder_y(vol, x_center, y_min, y_max):
        y_clamped = np.clip(yy, y_min, y_max)
        dist = np.sqrt((xx - x_center) ** 2 + zz**2 + (yy - y_clamped) ** 2)
        vol[dist <= line_radius] = 1.0

    def _add_cylinder_x(vol, y_center, x_min, x_max):
        x_clamped = np.clip(xx, x_min, x_max)
        dist = np.sqrt((yy - y_center) ** 2 + zz**2 + (xx - x_clamped) ** 2)
        vol[dist <= line_radius] = 1.0

    # '[' bracket
    x_left = -bracket_x_offset
    _add_cylinder_y(volume, x_left, -bracket_half_extent, bracket_half_extent)
    _add_cylinder_x(volume, +bracket_half_extent, x_left, x_left + bracket_cap_length)
    _add_cylinder_x(volume, -bracket_half_extent, x_left, x_left + bracket_cap_length)

    # ']' bracket
    x_right = +bracket_x_offset
    _add_cylinder_y(volume, x_right, -bracket_half_extent, bracket_half_extent)
    _add_cylinder_x(volume, +bracket_half_extent, x_right - bracket_cap_length, x_right)
    _add_cylinder_x(volume, -bracket_half_extent, x_right - bracket_cap_length, x_right)

    # 'o' torus
    rho = np.sqrt(xx**2 + yy**2)
    dist_torus = np.sqrt((rho - torus_major_radius) ** 2 + zz**2)
    volume[dist_torus <= line_radius] = 1.0

    blur_sigma = 1.0
    volume = gaussian_filter(volume, sigma=blur_sigma)
    if volume.max() > 0:
        volume /= volume.max()
    return volume


def generate_shepp_logan_3d(zyx_shape):
    """3D Shepp-Logan phantom. Returns density array normalized to [0, 1]."""
    ellipsoids = [
        (1.0, 0.0, 0.0, 0.0, 0.69, 0.92, 0.81, 0),
        (-0.8, 0.0, -0.0184, 0.0, 0.6624, 0.8740, 0.78, 0),
        (-0.2, 0.22, 0.0, 0.0, 0.11, 0.31, 0.22, -18),
        (-0.2, -0.22, 0.0, 0.0, 0.16, 0.41, 0.28, 18),
        (0.1, 0.0, 0.35, 0.0, 0.21, 0.25, 0.41, 0),
        (0.1, 0.0, 0.1, 0.0, 0.046, 0.046, 0.05, 0),
        (0.1, 0.0, -0.1, 0.0, 0.046, 0.046, 0.05, 0),
        (0.1, -0.08, -0.605, 0.0, 0.046, 0.023, 0.05, 0),
        (0.1, 0.0, -0.605, 0.0, 0.023, 0.023, 0.02, 0),
        (0.1, 0.06, -0.605, 0.0, 0.046, 0.023, 0.02, 0),
    ]
    nz, ny, nx = zyx_shape
    phantom = np.zeros(zyx_shape, dtype=np.float32)
    z = np.linspace(-1, 1, nz)
    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    for density, cx, cy, cz, sa, sb, sc, phi_deg in ellipsoids:
        phi = np.radians(phi_deg)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        xp, yp, zp = xx - cx, yy - cy, zz - cz
        xr = xp * cos_phi + yp * sin_phi
        yr = -xp * sin_phi + yp * cos_phi
        inside = (xr / sa) ** 2 + (yr / sb) ** 2 + (zp / sc) ** 2 <= 1.0
        phantom[inside] += density

    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom /= phantom.max()
    return phantom


def phantom_to_fluorescence_and_phase(volume, voxel_size, bead_index, media_index, wavelength_illumination):
    """Convert density volume to fluorescence and phase tensors."""
    fluorescence = torch.tensor(volume, dtype=torch.float32)
    wavelength_medium = wavelength_illumination / media_index
    delta_n = (bead_index - media_index) * volume
    phase = torch.tensor(delta_n * voxel_size / wavelength_medium, dtype=torch.float32)
    return fluorescence, phase


def _generate_all_phantoms():
    """Return dict of {sample_name: volume} for the three test phantoms."""
    phantoms = {}
    click.echo("Generating isolated bead...")
    phantoms["point"] = generate_isolated_bead(ZYX_SHAPE, VOXEL_SIZE, BEAD_RADIUS)
    click.echo("Generating line [o] pattern...")
    phantoms["lines"] = generate_line_pattern(
        ZYX_SHAPE, VOXEL_SIZE, LINE_RADIUS, BRACKET_HALF_EXTENT, BRACKET_X_OFFSET, BRACKET_CAP_LENGTH, TORUS_MAJOR_RADIUS
    )
    click.echo("Generating Shepp-Logan phantom...")
    phantoms["shepplogan"] = generate_shepp_logan_3d(ZYX_SHAPE)
    for name, vol in phantoms.items():
        click.echo(f"  {name}: shape={vol.shape}, range=[{vol.min():.3f}, {vol.max():.3f}]")
    return phantoms


def _scale_and_noise(vol):
    """Scale to [BLACK_LEVEL, PEAK_INTENSITY] and apply Poisson noise."""
    v = vol.numpy() if isinstance(vol, torch.Tensor) else vol
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = BLACK_LEVEL + (PEAK_INTENSITY - BLACK_LEVEL) * (v - v_min) / (v_max - v_min)
    else:
        v = np.full_like(v, BLACK_LEVEL)
    return np.random.poisson(np.clip(v, 0, None)).astype(np.float32)


def _pad_projection(proj, target_width):
    """Center-pad a 2D projection to target_width along axis 1."""
    if proj.shape[1] >= target_width:
        return proj
    pad_left = (target_width - proj.shape[1]) // 2
    pad_right = target_width - proj.shape[1] - pad_left
    return np.pad(proj, ((0, 0), (pad_left, pad_right)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
CONTEXT = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT)
@click.option("--data-dir", type=click.Path(), default="./data", show_default=True, help="Directory for the zarr store.")
@click.pass_context
def cli(ctx, data_dir):
    """Projection modeling: generate phantoms, simulate images, and compute projections.

    Three subcommands run in sequence.  Each reads the output of the
    previous stage from the shared OME-Zarr store at
    ``<data-dir>/projection_modeling.zarr``.

    \b
    python projection_modeling.py object   --data-dir ./data
    python projection_modeling.py image    --data-dir ./data
    python projection_modeling.py project  --data-dir ./data
    """
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)
    ctx.obj["store_path"] = Path(data_dir) / "projection_modeling.zarr"


# ---- object ---------------------------------------------------------------
@cli.command()
@click.pass_context
def object(ctx):
    """Generate 3D test phantoms and write ground-truth densities.

    Creates three phantoms (point bead, [o] lines, Shepp-Logan),
    converts each to fluorescence density and phase, and writes them
    to the ``object`` column of the OME-Zarr store.

    Overwrites any existing store.
    """
    store_path = ctx.obj["store_path"]
    data_dir = ctx.obj["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Remove old store
    if store_path.exists():
        shutil.rmtree(store_path)
        click.echo(f"Removed existing {store_path}")

    phantoms = _generate_all_phantoms()

    click.echo(f"\nCreating OME-Zarr v3 store at {store_path} ...")
    plate = open_ome_zarr(
        str(store_path),
        layout="hcs",
        mode="w-",
        channel_names=CHANNEL_NAMES,
        version="0.5",
    )
    for sample in SAMPLE_TYPES:
        position = plate.create_position(sample, "object", "0")
        position.create_zeros(
            name="0",
            shape=(1, len(CHANNEL_NAMES), *ZYX_SHAPE),
            chunks=(1, 1, 1, 256, 256),
            dtype=np.float32,
            transform=[TransformationMeta(type="scale", scale=[1, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])],
        )
    plate.close()

    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for name, volume in phantoms.items():
            fluorescence, phase = phantom_to_fluorescence_and_phase(
                volume, VOXEL_SIZE, BEAD_INDEX, MEDIA_INDEX, WAVELENGTH_ILLUMINATION
            )
            pos = plate[f"{name}/object/0"]
            pos["0"][0, 0] = fluorescence.numpy()
            pos["0"][0, 1] = phase.numpy()
            click.echo(f"  Wrote {name}/object/0")

    click.echo("Done.")


# ---- image ----------------------------------------------------------------
@cli.command()
@click.pass_context
def image(ctx):
    """Blur phantoms through microscope transfer functions and add noise.

    Reads the ``object`` column, applies the fluorescence OTF and
    phase transfer function, scales each volume to the detector range
    [100, 1024], adds Poisson noise, and writes to the ``rawimage``
    column.
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}. Run 'object' first.")

    click.echo("Computing fluorescence OTF...")
    fluorescence_otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape=ZYX_SHAPE,
        yx_pixel_size=VOXEL_SIZE,
        z_pixel_size=VOXEL_SIZE,
        wavelength_emission=WAVELENGTH_EMISSION,
        z_padding=Z_PADDING,
        index_of_refraction_media=MEDIA_INDEX,
        numerical_aperture_detection=NA_DETECTION,
    )
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

    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for sample in SAMPLE_TYPES:
            click.echo(f"\nProcessing {sample}...")

            # Create rawimage position
            position = plate.create_position(sample, "rawimage", "0")
            position.create_zeros(
                name="0",
                shape=(1, len(CHANNEL_NAMES), *ZYX_SHAPE),
                chunks=(1, 1, 1, 256, 256),
                dtype=np.float32,
                transform=[TransformationMeta(type="scale", scale=[1, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])],
            )

            # Read object
            obj_pos = plate[f"{sample}/object/0"]
            fluorescence = torch.tensor(np.array(obj_pos["0"][0, 0]), dtype=torch.float32)
            phase = torch.tensor(np.array(obj_pos["0"][0, 1]), dtype=torch.float32)

            click.echo("  Blurring fluorescence...")
            fluor_blurred = isotropic_fluorescent_thick_3d.apply_transfer_function(fluorescence, fluorescence_otf, Z_PADDING)

            click.echo("  Blurring phase...")
            phase_blurred = phase_thick_3d.apply_transfer_function(phase, real_tf, Z_PADDING, brightness=1e3)

            click.echo("  Scaling + Poisson noise...")
            img_pos = plate[f"{sample}/rawimage/0"]
            img_pos["0"][0, 0] = _scale_and_noise(fluor_blurred)
            img_pos["0"][0, 1] = _scale_and_noise(phase_blurred)
            click.echo(f"  Wrote {sample}/rawimage/0")

    click.echo("Done.")


# ---- project --------------------------------------------------------------
@cli.command()
@click.pass_context
def project(ctx):
    """Compute Siddon mean-projections of simulated images at -70 to +70 degrees.

    Reads the ``rawimage`` column, projects each volume at 29 angles
    (-70 to +70 in 5-degree steps) using Siddon's ray-tracing
    algorithm, divides by the ray path length to obtain the mean, and
    writes the projection stacks to the ``projections`` column.

    The mean projection preserves the dynamic range of the 3D volume.
    Each stack has shape (1, C, N_angles, Y, X_padded) in TCZYX
    convention; the Z-axis scale encodes the angular step (5 degrees).
    """
    store_path = ctx.obj["store_path"]
    if not store_path.exists():
        raise click.UsageError(f"Store not found: {store_path}. Run 'object' and 'image' first.")

    angles = PROJECTION_ANGLES
    n_angles = len(angles)

    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for sample in SAMPLE_TYPES:
            click.echo(f"\nProjecting {sample}...")

            img_pos = plate[f"{sample}/rawimage/0"]
            fluor_vol = np.array(img_pos["0"][0, 0])
            phase_vol = np.array(img_pos["0"][0, 1])

            # First pass: compute mean projections and find max lateral width
            fluor_projs = []
            phase_projs = []
            max_width = 0
            nz_vol = fluor_vol.shape[0]
            nx_vol = fluor_vol.shape[2]
            for angle in angles:
                fp = siddon_project(fluor_vol, angle, VOXEL_SIZE, mode="sum")
                pp = siddon_project(phase_vol, angle, VOXEL_SIZE, mode="sum")
                # Normalize sum to mean: divide by physical path length (um)
                # siddon_project returns sum weighted by intersection length,
                # so dividing by total path gives the mean voxel value.
                theta = np.radians(angle)
                cos_t, sin_t = abs(np.cos(theta)), abs(np.sin(theta))
                path_length = (nz_vol * cos_t + nx_vol * sin_t) * VOXEL_SIZE
                if path_length > 0:
                    fp = fp / path_length
                    pp = pp / path_length
                fluor_projs.append(fp)
                phase_projs.append(pp)
                max_width = max(max_width, fp.shape[1], pp.shape[1])

            # Second pass: pad to uniform width and stack
            ny_proj = fluor_projs[0].shape[0]
            fluor_stack = np.zeros((n_angles, ny_proj, max_width), dtype=np.float32)
            phase_stack = np.zeros((n_angles, ny_proj, max_width), dtype=np.float32)
            for i in range(n_angles):
                fluor_stack[i] = _pad_projection(fluor_projs[i], max_width)
                phase_stack[i] = _pad_projection(phase_projs[i], max_width)

            click.echo(f"  Stack shape: ({n_angles}, {ny_proj}, {max_width}), angles {angles[0]} to {angles[-1]} deg")

            # Create projections position with angular Z-scale
            position = plate.create_position(sample, "projections", "0")
            position.create_zeros(
                name="0",
                shape=(1, len(CHANNEL_NAMES), n_angles, ny_proj, max_width),
                chunks=(1, 1, 1, ny_proj, max_width),
                dtype=np.float32,
                transform=[TransformationMeta(type="scale", scale=[1, 1, 5.0, VOXEL_SIZE, VOXEL_SIZE])],
            )

            pos = plate[f"{sample}/projections/0"]
            pos["0"][0, 0] = fluor_stack
            pos["0"][0, 1] = phase_stack
            pos.zattrs["projection_angles_deg"] = angles
            pos.zattrs["angle_step_deg"] = 5
            pos.zattrs["angle_start_deg"] = angles[0]
            click.echo(f"  Wrote {sample}/projections/0")

    click.echo("Done.")


if __name__ == "__main__":
    cli()

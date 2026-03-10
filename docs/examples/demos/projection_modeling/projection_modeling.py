# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Projection Modeling: Three Test Phantoms and Forward Simulation

This notebook generates three distinct 3D phantoms, converts each to
fluorescence density and phase density, and blurs them through the
microscope transfer functions.

Phantoms:
1. Isolated bead — a single 0.5 um sphere
2. Line `[o]` pattern — thin cylinders and a torus ring
3. Shepp-Logan — complex extended object
"""

# %%
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.ndimage import gaussian_filter

from siddon import siddon_project
from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

# %% [markdown]
"""
## Configurable Parameters
"""

# %%
# --- Object parameters ---
bead_radius = 0.25  # um (0.5 um diameter)
bead_index = 1.52  # refractive index of structures
media_index = 1.33  # water
line_radius = 0.25  # um (0.5 um line thickness → radius)
pattern_extent = 2.5  # um, half-extent of [o] pattern around center
line_spacing = 2.0  # um, spacing between bracket line positions

# --- Volume parameters ---
volume_extent_x = 10.0  # um
volume_extent_y = 10.0  # um
volume_extent_z = 10.0  # um
voxel_size = 0.05  # um (50 nm isotropic sampling)

# --- Imaging parameters ---
wavelength_illumination = 0.500  # um
wavelength_emission = 0.520  # um (Stokes-shifted fluorescence)
na_detection = 1.0
na_illumination = 0.5
z_padding = 0

# --- Projection parameters ---
projection_angles = [-60, 0, 60]  # degrees from z-axis (rotation around Y)

# --- Derived parameters ---
nz = int(volume_extent_z / voxel_size)
ny = int(volume_extent_y / voxel_size)
nx = int(volume_extent_x / voxel_size)
zyx_shape = (nz, ny, nx)

print(f"Volume shape: {zyx_shape} ({nz * ny * nx / 1e6:.1f}M voxels)")
print(f"Physical size: {volume_extent_z} x {volume_extent_y} x {volume_extent_x} um")

# %% [markdown]
"""
## Phantom Generators
"""


# %%
def generate_isolated_bead(zyx_shape, voxel_size, bead_radius):
    """Single sphere at volume center.

    Parameters
    ----------
    zyx_shape : tuple of int
        (nz, ny, nx) volume shape.
    voxel_size : float
        Isotropic voxel spacing in um.
    bead_radius : float
        Sphere radius in um.

    Returns
    -------
    volume : np.ndarray
        Binary volume with 1 inside the sphere, 0 outside.
    """
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(zz**2 + yy**2 + xx**2)
    volume = (dist <= bead_radius).astype(np.float32)
    return volume


def generate_line_pattern(zyx_shape, voxel_size, line_radius, pattern_extent, line_spacing):
    """[o] pattern from cylinders (brackets) and a torus (ring).

    Brackets `[` and `]` are vertical line segments (cylinders along Y)
    at x = +/-pattern_extent/2, spanning y in [-pattern_extent/2, +pattern_extent/2],
    centered at z = 0.

    The `o` is a torus in the central XY plane with major radius 0.8 um
    and tube radius = line_radius.

    Parameters
    ----------
    zyx_shape : tuple of int
        (nz, ny, nx) volume shape.
    voxel_size : float
        Isotropic voxel spacing in um.
    line_radius : float
        Half-thickness of lines in um.
    pattern_extent : float
        Half-extent of the pattern in um.
    line_spacing : float
        Not used directly; reserved for future bracket segment spacing.

    Returns
    -------
    volume : np.ndarray
        Smooth volume (0-1) with the [o] pattern.
    """
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    volume = np.zeros(zyx_shape, dtype=np.float32)

    # '[' bracket: cylinder along Y at x = -pattern_extent/2, z = 0
    x_left = -pattern_extent / 2
    y_mask_left = (yy >= -pattern_extent / 2) & (yy <= pattern_extent / 2)
    dist_left = np.sqrt((xx - x_left) ** 2 + zz**2)
    volume[(dist_left <= line_radius) & y_mask_left] = 1.0

    # ']' bracket: cylinder along Y at x = +pattern_extent/2, z = 0
    x_right = pattern_extent / 2
    y_mask_right = (yy >= -pattern_extent / 2) & (yy <= pattern_extent / 2)
    dist_right = np.sqrt((xx - x_right) ** 2 + zz**2)
    volume[(dist_right <= line_radius) & y_mask_right] = 1.0

    # 'o' ring: torus with major radius R in XY plane, tube radius = line_radius
    torus_major_radius = 0.8  # um
    # Distance from the ring centerline (circle of radius R in XY at z=0):
    # d = sqrt((sqrt(x^2 + y^2) - R)^2 + z^2)
    rho = np.sqrt(xx**2 + yy**2)
    dist_torus = np.sqrt((rho - torus_major_radius) ** 2 + zz**2)
    volume[dist_torus <= line_radius] = 1.0

    # Smooth edges
    blur_sigma = 1.0  # voxels
    volume = gaussian_filter(volume, sigma=blur_sigma)
    if volume.max() > 0:
        volume = volume / volume.max()

    return volume


def generate_shepp_logan_3d(zyx_shape):
    """3D Shepp-Logan phantom from the standard 10-ellipsoid parameterization.

    Parameters
    ----------
    zyx_shape : tuple of int
        (nz, ny, nx) volume shape.

    Returns
    -------
    phantom : np.ndarray
        Density array normalized to [0, 1].
    """
    # (density, cx, cy, cz, semi_a, semi_b, semi_c, phi_deg)
    # Coordinates and semi-axes in normalized units [-1, 1]
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
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        xp = xx - cx
        yp = yy - cy
        zp = zz - cz

        xr = xp * cos_phi + yp * sin_phi
        yr = -xp * sin_phi + yp * cos_phi

        inside = (xr / sa) ** 2 + (yr / sb) ** 2 + (zp / sc) ** 2 <= 1.0
        phantom[inside] += density

    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom = phantom / phantom.max()

    return phantom


def phantom_to_fluorescence_and_phase(volume, voxel_size, bead_index, media_index, wavelength_illumination):
    """Convert a density volume to fluorescence and phase representations.

    Parameters
    ----------
    volume : np.ndarray
        Density values in [0, 1].
    voxel_size : float
        Isotropic voxel spacing in um.
    bead_index : float
        Refractive index of the structure.
    media_index : float
        Refractive index of the surrounding medium.
    wavelength_illumination : float
        Illumination wavelength in um.

    Returns
    -------
    fluorescence : torch.Tensor
        Fluorescence density (0-1).
    phase : torch.Tensor
        Phase in cycles per voxel.
    """
    fluorescence = torch.tensor(volume, dtype=torch.float32)
    wavelength_medium = wavelength_illumination / media_index
    delta_n = (bead_index - media_index) * volume
    phase = torch.tensor(delta_n * voxel_size / wavelength_medium, dtype=torch.float32)
    return fluorescence, phase


# %% [markdown]
"""
## Generate Phantoms
"""

# %%
phantoms = {}

print("Generating isolated bead...")
phantoms["isolated_bead"] = generate_isolated_bead(zyx_shape, voxel_size, bead_radius)

print("Generating line [o] pattern...")
phantoms["line_pattern"] = generate_line_pattern(zyx_shape, voxel_size, line_radius, pattern_extent, line_spacing)

print("Generating Shepp-Logan phantom...")
phantoms["shepp_logan"] = generate_shepp_logan_3d(zyx_shape)

for name, vol in phantoms.items():
    print(f"  {name}: shape={vol.shape}, range=[{vol.min():.3f}, {vol.max():.3f}]")

# %% [markdown]
"""
## Forward Simulation

Compute fluorescence OTF and phase transfer function once,
then apply to each phantom.
"""

# %%
print("Computing fluorescence OTF...")
fluorescence_otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    wavelength_emission=wavelength_emission,
    z_padding=z_padding,
    index_of_refraction_media=media_index,
    numerical_aperture_detection=na_detection,
)
print(f"Fluorescence OTF shape: {fluorescence_otf.shape}")

print("Computing phase transfer functions...")
real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    wavelength_illumination=wavelength_illumination,
    z_padding=z_padding,
    index_of_refraction_media=media_index,
    numerical_aperture_illumination=na_illumination,
    numerical_aperture_detection=na_detection,
)
print(f"Phase TF shape: {real_tf.shape}")

# %% [markdown]
"""
## Convert, Blur, and Save
"""

# %%
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

store = zarr.open(str(data_dir / "projection_modeling.zarr"), mode="w")

# Keep blurred volumes for projection computation
blurred_volumes = {}

for name, volume in phantoms.items():
    print(f"\nProcessing {name}...")

    fluorescence, phase = phantom_to_fluorescence_and_phase(
        volume, voxel_size, bead_index, media_index, wavelength_illumination
    )

    print("  Blurring fluorescence...")
    fluorescence_blurred = isotropic_fluorescent_thick_3d.apply_transfer_function(
        fluorescence, fluorescence_otf, z_padding
    )

    print("  Blurring phase...")
    phase_blurred = phase_thick_3d.apply_transfer_function(phase, real_tf, z_padding, brightness=1e3)

    store[f"{name}/fluorescence_density"] = fluorescence.numpy()
    store[f"{name}/phase_density"] = phase.numpy()
    store[f"{name}/fluorescence_blurred"] = fluorescence_blurred.numpy()
    store[f"{name}/phase_blurred"] = phase_blurred.numpy()

    blurred_volumes[name] = {
        "fluorescence_density": fluorescence.numpy(),
        "phase_density": phase.numpy(),
        "fluorescence_blurred": fluorescence_blurred.numpy(),
        "phase_blurred": phase_blurred.numpy(),
    }

# %% [markdown]
"""
## Projections via Siddon's Algorithm

Compute average and max projections of each volume at 0° and ±60°.
At 0° the projection reduces to a simple axis sum/max along Z.
At oblique angles, Siddon's ray-tracing algorithm computes exact
voxel intersection lengths for accurate line integrals.
"""

# %%
for name in phantoms:
    print(f"\nProjecting {name}...")
    for vol_type, vol_data in blurred_volumes[name].items():
        for angle in projection_angles:
            for mode in ("avg", "max"):
                siddon_mode = "sum" if mode == "avg" else "max"
                proj = siddon_project(vol_data, angle, voxel_size, mode=siddon_mode)
                if mode == "avg":
                    # Normalize sum projection to average by dividing by
                    # the path length through the volume at this angle
                    theta = np.radians(angle)
                    if abs(np.cos(theta)) > 1e-10:
                        path_length = nz * voxel_size / abs(np.cos(theta))
                    else:
                        path_length = nx * voxel_size / abs(np.sin(theta))
                    proj = proj / path_length
                arr_name = f"{name}/proj_{vol_type}_{mode}_{angle:+d}deg"
                store[arr_name] = proj
                print(f"  {arr_name}: shape={proj.shape}")

# Store metadata
store.attrs["voxel_size_um"] = voxel_size
store.attrs["zyx_shape"] = list(zyx_shape)
store.attrs["wavelength_illumination_um"] = wavelength_illumination
store.attrs["wavelength_emission_um"] = wavelength_emission
store.attrs["na_detection"] = na_detection
store.attrs["na_illumination"] = na_illumination
store.attrs["media_index"] = media_index
store.attrs["bead_index"] = bead_index
store.attrs["projection_angles"] = projection_angles

print(f"\nSaved zarr store: {data_dir / 'projection_modeling.zarr'}")
print(f"Groups: {list(store.group_keys())}")
for group_name in store.group_keys():
    group = store[group_name]
    for arr_name in group.array_keys():
        arr = group[arr_name]
        print(f"  {group_name}/{arr_name}: shape={arr.shape}, dtype={arr.dtype}")

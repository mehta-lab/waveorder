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
# Projection Modeling: Forward Simulation and Multi-Angle Projections

This notebook demonstrates:
1. Generating a 3D phantom with beads arranged in a `[o]` pattern
2. Generating a miniaturized 3D Shepp-Logan phantom (absorbing specimen)
3. Simulating fluorescence and phase images of both objects
4. Computing mean and max projections at multiple angles relative to the z-axis
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy.ndimage import gaussian_filter, rotate

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d

# %% [markdown]
"""
## Configurable Parameters
"""

# %%
# --- Object parameters ---
bead_radius = 0.25  # um (0.5 um diameter)
bead_index = 1.52  # refractive index of beads
media_index = 1.33  # water
bead_spacing = 1.0  # um, center-to-center spacing in [o] pattern
pattern_extent = 2.5  # um, half-extent of [o] pattern around center

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
projection_angles = [0, 15, 30, 45, 60, 75, 90]  # degrees from z-axis

# --- Derived parameters ---
nz = int(volume_extent_z / voxel_size)
ny = int(volume_extent_y / voxel_size)
nx = int(volume_extent_x / voxel_size)
zyx_shape = (nz, ny, nx)

print(f"Volume shape: {zyx_shape} ({nz * ny * nx / 1e6:.1f}M voxels)")
print(f"Physical size: {volume_extent_z} x {volume_extent_y} x {volume_extent_x} um")

# %% [markdown]
"""
## Part 1: Bead `[o]` Phantom

The pattern `[o]` consists of:
- `[` : vertical column of beads on the left
- `o` : ring of beads in the center
- `]` : vertical column of beads on the right

All beads are placed in the central XY plane.
"""


# %%
def generate_bead_pattern_phantom(
    zyx_shape,
    voxel_size,
    bead_radius,
    bead_spacing,
    pattern_extent,
    bead_index,
    media_index,
    wavelength_illumination,
):
    """Generate a 3D volume with beads arranged in a [o] pattern.

    Returns
    -------
    zyx_fluorescence : torch.Tensor
        Fluorescence density (0-1)
    zyx_phase : torch.Tensor
        Phase in cycles per voxel
    bead_centers : list of (z, y, x) tuples in um
    """
    nz, ny, nx = zyx_shape

    # Physical coordinates centered on volume (um)
    z_coords = (np.arange(nz) - nz // 2) * voxel_size
    y_coords = (np.arange(ny) - ny // 2) * voxel_size
    x_coords = (np.arange(nx) - nx // 2) * voxel_size

    bead_centers = []

    # '[' bracket: vertical column at x = -pattern_extent/2
    x_bracket_left = -pattern_extent / 2
    y_positions = np.arange(-pattern_extent / 2, pattern_extent / 2 + bead_spacing / 2, bead_spacing)
    for y_pos in y_positions:
        bead_centers.append((0.0, y_pos, x_bracket_left))

    # ']' bracket: vertical column at x = +pattern_extent/2
    x_bracket_right = pattern_extent / 2
    for y_pos in y_positions:
        bead_centers.append((0.0, y_pos, x_bracket_right))

    # 'o' circle: ring of beads in the center
    circle_radius = 0.8  # um
    circumference = 2 * np.pi * circle_radius
    n_beads_circle = max(6, int(circumference / bead_spacing))
    for i in range(n_beads_circle):
        theta = 2 * np.pi * i / n_beads_circle
        bead_centers.append((0.0, circle_radius * np.sin(theta), circle_radius * np.cos(theta)))

    print(f"Total beads: {len(bead_centers)}")

    # Paint beads into volume
    volume = np.zeros(zyx_shape, dtype=np.float32)
    bead_radius_voxels = bead_radius / voxel_size

    for z0, y0, x0 in bead_centers:
        # Convert physical center to voxel indices
        iz = int(round(z0 / voxel_size)) + nz // 2
        iy = int(round(y0 / voxel_size)) + ny // 2
        ix = int(round(x0 / voxel_size)) + nx // 2

        # Bounding box in voxels
        r_vox = int(np.ceil(bead_radius_voxels)) + 1
        for dz in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dx in range(-r_vox, r_vox + 1):
                    zz, yy, xx = iz + dz, iy + dy, ix + dx
                    if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                        dist = np.sqrt(dz**2 + dy**2 + dx**2)
                        if dist <= bead_radius_voxels:
                            volume[zz, yy, xx] = 1.0

    # Smooth edges with small Gaussian blur
    blur_sigma = 1.0  # voxels
    volume = gaussian_filter(volume, sigma=blur_sigma)
    volume = volume / volume.max() if volume.max() > 0 else volume

    zyx_fluorescence = torch.tensor(volume, dtype=torch.float32)

    # Convert to phase (cycles per voxel)
    wavelength_medium = wavelength_illumination / media_index
    delta_n = (bead_index - media_index) * volume
    zyx_phase = torch.tensor(delta_n * voxel_size / wavelength_medium, dtype=torch.float32)

    return zyx_fluorescence, zyx_phase, bead_centers


# %%
zyx_bead_fluorescence, zyx_bead_phase, bead_centers = generate_bead_pattern_phantom(
    zyx_shape,
    voxel_size,
    bead_radius,
    bead_spacing,
    pattern_extent,
    bead_index,
    media_index,
    wavelength_illumination,
)

# Quick view of central slices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
z_center = nz // 2

axes[0].imshow(zyx_bead_fluorescence[z_center].numpy(), cmap="gray", origin="lower")
axes[0].set_title("Fluorescence density (XY, z=center)")

axes[1].imshow(zyx_bead_fluorescence[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[1].set_title("Fluorescence density (XZ, y=center)")

axes[2].imshow(zyx_bead_phase[z_center].numpy(), cmap="RdBu_r", origin="lower")
axes[2].set_title("Phase [cycles/voxel] (XY, z=center)")

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Part 2: Shepp-Logan Phantom (Absorbing Specimen)

A miniaturized 3D Shepp-Logan phantom scaled to fit the simulation volume.
The density values represent absorption coefficient.
"""


# %%
def generate_shepp_logan_3d(zyx_shape, voxel_size, volume_extents):
    """Generate a 3D Shepp-Logan phantom.

    Uses the standard 10-ellipsoid parameterization scaled to fit
    within the specified volume. Returns absorption density.

    Parameters
    ----------
    zyx_shape : tuple
        (nz, ny, nx) shape of the output volume
    voxel_size : float
        Isotropic voxel size in um
    volume_extents : tuple
        (ez, ey, ex) physical extents in um

    Returns
    -------
    phantom : np.ndarray
        3D absorption density array
    """
    # Standard Shepp-Logan ellipsoid parameters
    # (density, center_x, center_y, center_z, semi_a, semi_b, semi_c, phi_deg)
    # Coordinates and semi-axes are in normalized units [-1, 1]
    ellipsoids = [
        (1.0, 0.0, 0.0, 0.0, 0.69, 0.92, 0.81, 0),       # outer skull
        (-0.8, 0.0, -0.0184, 0.0, 0.6624, 0.8740, 0.78, 0),  # inner skull
        (-0.2, 0.22, 0.0, 0.0, 0.11, 0.31, 0.22, -18),    # left eye
        (-0.2, -0.22, 0.0, 0.0, 0.16, 0.41, 0.28, 18),    # right eye
        (0.1, 0.0, 0.35, 0.0, 0.21, 0.25, 0.41, 0),       # nose
        (0.1, 0.0, 0.1, 0.0, 0.046, 0.046, 0.05, 0),      # mouth center
        (0.1, 0.0, -0.1, 0.0, 0.046, 0.046, 0.05, 0),     # chin
        (0.1, -0.08, -0.605, 0.0, 0.046, 0.023, 0.05, 0), # left ear
        (0.1, 0.0, -0.605, 0.0, 0.023, 0.023, 0.02, 0),   # right ear
        (0.1, 0.06, -0.605, 0.0, 0.046, 0.023, 0.02, 0),  # top
    ]

    nz, ny, nx = zyx_shape
    phantom = np.zeros(zyx_shape, dtype=np.float32)

    # Create normalized coordinate grids [-1, 1]
    z = np.linspace(-1, 1, nz)
    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    for density, cx, cy, cz, sa, sb, sc, phi_deg in ellipsoids:
        phi = np.radians(phi_deg)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Translate
        xp = xx - cx
        yp = yy - cy
        zp = zz - cz

        # Rotate in XY plane
        xr = xp * cos_phi + yp * sin_phi
        yr = -xp * sin_phi + yp * cos_phi
        zr = zp

        # Test if inside ellipsoid
        inside = (xr / sa) ** 2 + (yr / sb) ** 2 + (zr / sc) ** 2 <= 1.0
        phantom[inside] += density

    # Normalize to [0, 1] range
    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom = phantom / phantom.max()

    return phantom


# %%
shepp_logan = generate_shepp_logan_3d(zyx_shape, voxel_size, (volume_extent_z, volume_extent_y, volume_extent_x))

# Convert to tensors
zyx_shepp_fluorescence = torch.tensor(shepp_logan, dtype=torch.float32)

# For phase: treat absorption density as refractive index variation
# Scale so max delta_n is comparable to bead phantom
shepp_delta_n = shepp_logan * (bead_index - media_index)
wavelength_medium = wavelength_illumination / media_index
zyx_shepp_phase = torch.tensor(shepp_delta_n * voxel_size / wavelength_medium, dtype=torch.float32)

# Quick view
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
z_center = nz // 2

axes[0].imshow(shepp_logan[z_center], cmap="gray", origin="lower")
axes[0].set_title("Shepp-Logan absorption (XY, z=center)")

axes[1].imshow(shepp_logan[:, ny // 2], cmap="gray", origin="lower", aspect=1)
axes[1].set_title("Shepp-Logan absorption (XZ, y=center)")

axes[2].imshow(zyx_shepp_phase[z_center].numpy(), cmap="RdBu_r", origin="lower")
axes[2].set_title("Shepp-Logan phase [cycles/voxel] (XY)")

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Part 3: Forward Simulation — Fluorescence
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

# %%
print("Simulating fluorescence: bead phantom...")
zyx_bead_fluor_data = isotropic_fluorescent_thick_3d.apply_transfer_function(
    zyx_bead_fluorescence, fluorescence_otf, z_padding
)

print("Simulating fluorescence: Shepp-Logan phantom...")
zyx_shepp_fluor_data = isotropic_fluorescent_thick_3d.apply_transfer_function(
    zyx_shepp_fluorescence, fluorescence_otf, z_padding
)

# %% [markdown]
"""
## Part 4: Forward Simulation — Phase (Transmission)
"""

# %%
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

# %%
print("Simulating phase: bead phantom...")
zyx_bead_phase_data = phase_thick_3d.apply_transfer_function(
    zyx_bead_phase, real_tf, z_padding, brightness=1e3
)

print("Simulating phase: Shepp-Logan phantom...")
zyx_shepp_phase_data = phase_thick_3d.apply_transfer_function(
    zyx_shepp_phase, real_tf, z_padding, brightness=1e3
)

# %% [markdown]
"""
## Save all arrays as zarr stores

Ground-truth phantoms and simulated (blurred) volumes are saved to `./data/`.
"""

# %%
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

store = zarr.open(str(data_dir / "projection_modeling.zarr"), mode="w")

# Bead phantom arrays
store["bead/fluorescence_density"] = zyx_bead_fluorescence.numpy()
store["bead/phase_density"] = zyx_bead_phase.numpy()
store["bead/fluorescence_blurred"] = zyx_bead_fluor_data.numpy()
store["bead/phase_blurred"] = zyx_bead_phase_data.numpy()

# Shepp-Logan phantom arrays
store["shepp_logan/fluorescence_density"] = zyx_shepp_fluorescence.numpy()
store["shepp_logan/phase_density"] = zyx_shepp_phase.numpy()
store["shepp_logan/fluorescence_blurred"] = zyx_shepp_fluor_data.numpy()
store["shepp_logan/phase_blurred"] = zyx_shepp_phase_data.numpy()

# Store metadata as attributes
store.attrs["voxel_size_um"] = voxel_size
store.attrs["zyx_shape"] = list(zyx_shape)
store.attrs["wavelength_illumination_um"] = wavelength_illumination
store.attrs["wavelength_emission_um"] = wavelength_emission
store.attrs["na_detection"] = na_detection
store.attrs["na_illumination"] = na_illumination
store.attrs["media_index"] = media_index
store.attrs["bead_index"] = bead_index

print(f"Saved zarr store: {data_dir / 'projection_modeling.zarr'}")
print(f"Groups: {list(store.group_keys())}")
for group_name in store.group_keys():
    group = store[group_name]
    for arr_name in group.array_keys():
        arr = group[arr_name]
        print(f"  {group_name}/{arr_name}: shape={arr.shape}, dtype={arr.dtype}")

# %% [markdown]
"""
## Part 5: Visualize Forward Simulations

Compare ground truth phantoms with simulated fluorescence and phase images.
"""

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
z_center = nz // 2

# Bead phantom
axes[0, 0].imshow(zyx_bead_fluorescence[z_center].numpy(), cmap="gray", origin="lower")
axes[0, 0].set_title("Bead phantom (ground truth)")
axes[0, 1].imshow(zyx_bead_fluor_data[z_center].numpy(), cmap="gray", origin="lower")
axes[0, 1].set_title("Bead fluorescence simulation")
axes[0, 2].imshow(zyx_bead_phase_data[z_center].numpy(), cmap="gray", origin="lower")
axes[0, 2].set_title("Bead phase simulation")

# Shepp-Logan
axes[1, 0].imshow(zyx_shepp_fluorescence[z_center].numpy(), cmap="gray", origin="lower")
axes[1, 0].set_title("Shepp-Logan phantom (ground truth)")
axes[1, 1].imshow(zyx_shepp_fluor_data[z_center].numpy(), cmap="gray", origin="lower")
axes[1, 1].set_title("Shepp-Logan fluorescence simulation")
axes[1, 2].imshow(zyx_shepp_phase_data[z_center].numpy(), cmap="gray", origin="lower")
axes[1, 2].set_title("Shepp-Logan phase simulation")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Central XY slices: Ground Truth vs Simulated Data", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# XZ cross-sections
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(zyx_bead_fluorescence[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[0, 0].set_title("Bead phantom (XZ)")
axes[0, 1].imshow(zyx_bead_fluor_data[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[0, 1].set_title("Bead fluorescence (XZ)")
axes[0, 2].imshow(zyx_bead_phase_data[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[0, 2].set_title("Bead phase (XZ)")

axes[1, 0].imshow(zyx_shepp_fluorescence[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[1, 0].set_title("Shepp-Logan phantom (XZ)")
axes[1, 1].imshow(zyx_shepp_fluor_data[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[1, 1].set_title("Shepp-Logan fluorescence (XZ)")
axes[1, 2].imshow(zyx_shepp_phase_data[:, ny // 2].numpy(), cmap="gray", origin="lower", aspect=1)
axes[1, 2].set_title("Shepp-Logan phase (XZ)")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("XZ cross-sections through center", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Part 6: Multi-Angle Projections

Compute mean and max projections of simulated volumes at different
angles relative to the z-axis, by rotating around the Y-axis.
"""


# %%
def compute_projections(volume, angles, rotation_axis="y"):
    """Compute mean and max projections at multiple angles.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    angles : list of float
        Tilt angles in degrees from the z-axis
    rotation_axis : str
        "y" rotates in the ZX plane, "x" rotates in the ZY plane

    Returns
    -------
    dict : {angle: {"mean": 2d_array, "max": 2d_array}}
    """
    results = {}
    for angle in angles:
        if angle == 0:
            rotated = volume
        else:
            if rotation_axis == "y":
                rotated = rotate(volume, angle, axes=(0, 2), reshape=False, order=1)
            elif rotation_axis == "x":
                rotated = rotate(volume, angle, axes=(0, 1), reshape=False, order=1)
            else:
                raise ValueError(f"Unknown rotation_axis: {rotation_axis}")

        results[angle] = {
            "mean": rotated.mean(axis=0),
            "max": rotated.max(axis=0),
        }
    return results


# %%
# Convert tensors to numpy for projection computation
volumes = {
    "Bead phantom": zyx_bead_fluorescence.numpy(),
    "Bead fluorescence": zyx_bead_fluor_data.numpy(),
    "Bead phase": zyx_bead_phase_data.numpy(),
    "Shepp-Logan phantom": zyx_shepp_fluorescence.numpy(),
    "Shepp-Logan fluorescence": zyx_shepp_fluor_data.numpy(),
    "Shepp-Logan phase": zyx_shepp_phase_data.numpy(),
}

all_projections = {}
for name, vol in volumes.items():
    print(f"Computing projections: {name}...")
    all_projections[name] = compute_projections(vol, projection_angles)

# %% [markdown]
"""
## Part 7: Visualize Projections — Bead Phantom
"""

# %%
# Bead phantom: mean projections at all angles
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Bead phantom"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Bead phantom"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Bead Phantom — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Bead fluorescence simulation: mean and max projections
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Bead fluorescence"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Bead fluorescence"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Bead Fluorescence Simulation — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Bead phase simulation: mean and max projections
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Bead phase"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Bead phase"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Bead Phase Simulation — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Part 8: Visualize Projections — Shepp-Logan Phantom
"""

# %%
# Shepp-Logan phantom: mean and max projections
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Shepp-Logan phantom"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Shepp-Logan phantom"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Shepp-Logan Phantom — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Shepp-Logan fluorescence simulation: mean and max projections
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Shepp-Logan fluorescence"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Shepp-Logan fluorescence"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Shepp-Logan Fluorescence Simulation — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Shepp-Logan phase simulation: mean and max projections
fig, axes = plt.subplots(2, len(projection_angles), figsize=(3 * len(projection_angles), 6))

for i, angle in enumerate(projection_angles):
    axes[0, i].imshow(all_projections["Shepp-Logan phase"][angle]["mean"], cmap="gray", origin="lower")
    axes[0, i].set_title(f"{angle}\u00b0")
    axes[0, i].axis("off")

    axes[1, i].imshow(all_projections["Shepp-Logan phase"][angle]["max"], cmap="gray", origin="lower")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Mean proj.", fontsize=12)
axes[1, 0].set_ylabel("Max proj.", fontsize=12)
plt.suptitle("Shepp-Logan Phase Simulation — Projections at Different Angles", fontsize=14)
plt.tight_layout()
plt.show()

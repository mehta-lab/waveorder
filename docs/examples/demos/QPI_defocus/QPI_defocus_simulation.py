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
# Quantitative Phase Imaging from Defocus Demo

This notebook demonstrates forward simulation and reconstruction for Quantitative Phase Imaging (QPI) from defocus.
The simulation and reconstruction are based on partially coherent optical diffraction tomography (ODT):

J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative 3D tomographic imaging
for partially coherent light microscopy," Opt. Express 25, 15699-15712 (2017)
"""

# %% [markdown]
"""
## Setup and Imports
First, let's install the latest version of waveorder from the main branch
"""

# %%
import sys
import subprocess

# Install latest waveorder from main branch
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/mehta-lab/waveorder.git@main",
    ]
)

# %%
import numpy as np
from pathlib import Path
from platformdirs import user_data_dir
import matplotlib.pyplot as plt
import torch

from waveorder import util
from waveorder.models import phase_thick_3d
from waveorder.visuals import jupyter_visuals

# %% [markdown]
"""
## Forward Simulation Parameters
"""

# %%
# Parameters (all lengths in micrometers)
simulation_arguments = {
    "zyx_shape": (100, 256, 256),  # 3D shape of the volume
    "yx_pixel_size": 6.5 / 63,  # Lateral pixel size
    "z_pixel_size": 0.25,  # Axial pixel size
    "index_of_refraction_media": 1.3,  # Refractive index of medium
}

phantom_arguments = {
    "index_of_refraction_sample": 1.50,  # Refractive index of sample
    "sphere_radius": 5,  # Radius of test sphere in microns
}

transfer_function_arguments = {
    "z_padding": 0,  # Padding in z direction
    "wavelength_illumination": 0.532,  # Wavelength in microns
    "numerical_aperture_illumination": 0.9,  # Illumination NA
    "numerical_aperture_detection": 1.2,  # Detection NA
}

# %% [markdown]
"""
## Generate Test Phantom
"""

# %%
# Create a phantom

# 3D Star target
star, _, _ = util.generate_star_target(
    yx_shape=simulation_arguments["zyx_shape"][1:3]
)
yx_phase = star * (
    phantom_arguments["index_of_refraction_sample"]
    - simulation_arguments["index_of_refraction_media"]
)  # phase in radians
# Initialize zyx_phase with zeros
zyx_phase = torch.zeros(simulation_arguments["zyx_shape"])

# Copy yx_phase into the central 10 z slices
z_center = simulation_arguments["zyx_shape"][0] // 2
z_start = z_center - 5
z_end = z_center + 6
zyx_phase[z_start:z_end] = yx_phase

# Bead target
# zyx_phase = phase_thick_3d.generate_test_phantom(
#     **simulation_arguments, **phantom_arguments
# )


# Show 5 z-slices, five apart, centered on the central z slice
z_slices = np.arange(z_center - 10, z_center + 11, 5)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, z in enumerate(z_slices):
    axes[i].imshow(zyx_phase[z], cmap="gray", origin="lower")
    axes[i].set_title(f"z = {z - z_center}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Calculate Transfer Functions
"""

# %%
# Calculate real and imaginary parts of the transfer function
(
    real_component_transfer_function,
    imaginary_component_transfer_function,
) = phase_thick_3d.calculate_transfer_function(
    **simulation_arguments, **transfer_function_arguments
)

# Magnitude and phase of the real component of the transfer function
tf_real_magnitude = np.fft.ifftshift(real_component_transfer_function.abs())
tf_real_phase = np.fft.ifftshift(real_component_transfer_function.angle())

# Visualize transfer functions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, z in enumerate(z_slices):
    axes[0, i].imshow(
        tf_real_magnitude[z],
        cmap="gray",
        origin="lower",
    )
    axes[0, i].set_title(f"Magnitude of TF, z = {z - z_center}")
    axes[0, i].axis("off")
    axes[1, i].imshow(
        tf_real_phase[z],
        cmap="gray",
        origin="lower",
    )
    axes[1, i].set_title(f"Phase of TF, z = {z - z_center}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Forward Simulation
"""

# %%
# Simulate defocus data
zyx_data = phase_thick_3d.apply_transfer_function(
    zyx_phase,
    real_component_transfer_function,
    transfer_function_arguments["z_padding"],
    brightness=1e3,
)
zyx_data_norm = (zyx_data - zyx_data.min()) / (zyx_data.max() - zyx_data.min())
# Visualize simulated data
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, z in enumerate(z_slices):
    axes[i].imshow(
        zyx_data_norm[z], cmap="gray", origin="lower", vmin=0, vmax=1
    )
    axes[i].set_title(f"Data, z = {z - z_center}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Phase Reconstruction
"""

# %%
# Reconstruct phase
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data,
    real_component_transfer_function,
    imaginary_component_transfer_function,
    transfer_function_arguments["z_padding"],
    reconstruction_algorithm="Tikhonov",
    regularization_strength=1e-5,
)

# Visualize reconstruction compared to ground truth
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

# Normalize data and reconstruction between 0 and 1
zyx_data_norm = (zyx_data - zyx_data.min()) / (zyx_data.max() - zyx_data.min())
zyx_recon_norm = (zyx_recon - zyx_recon.min()) / (
    zyx_recon.max() - zyx_recon.min()
)

for i, z in enumerate(z_slices):
    axes[0, i].imshow(zyx_phase[z], cmap="gray", origin="lower")
    axes[0, i].set_title(f"Ground Truth, z = {z - z_center}")
    axes[0, i].axis("off")
    axes[1, i].imshow(
        zyx_data_norm[z], cmap="gray", origin="lower", vmin=0, vmax=1
    )
    axes[1, i].set_title(f"Data, z = {z - z_center}")
    axes[1, i].axis("off")
    axes[2, i].imshow(
        zyx_recon_norm[z], cmap="gray", origin="lower", vmin=0, vmax=1
    )
    axes[2, i].set_title(f"Reconstruction, z = {z - z_center}")
    axes[2, i].axis("off")
plt.tight_layout()
plt.show()

# %%

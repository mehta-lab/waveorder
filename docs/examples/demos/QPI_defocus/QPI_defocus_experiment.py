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

This notebook demonstrates reconstruction for Quantitative Phase Imaging (QPI) from defocus.
The reconstruction is based on partially coherent optical diffraction tomography (ODT):

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
!pip install iohub==0.2.0

# %%
import matplotlib.pyplot as plt
import torch

from waveorder.models import phase_thick_3d
import requests
import zipfile
import io
from iohub import open_ome_zarr

# %% Load the data
# Download the file from Zenodo
url = "https://zenodo.org/record/8386856/files/recOrder_session.zip"
response = requests.get(url)
response.raise_for_status()

# Unzip the downloaded file
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("./recOrder_session")

# %% Load data into an array
zarr_path = "./recOrder_session/recOrder_session/phase_snap_0/raw_data.zarr"
with open_ome_zarr(zarr_path) as input_plate:
    zyx_data = input_plate["0/0/0/0"][0, 0]

# %% specify reconstruction parameters (all lengths in micrometers)
transfer_function_arguments = {
    "yx_pixel_size": 6.5 / 20,  # Lateral pixel size
    "z_pixel_size": 2.0,  # Axial pixel size
    "index_of_refraction_media": 1.0,  # Refractive index of medium
    "z_padding": 0,  # Padding in z direction
    "wavelength_illumination": 0.532,  # Wavelength in microns
    "numerical_aperture_illumination": 0.4,  # Illumination NA
    "numerical_aperture_detection": 0.55,  # Detection NA
}
# %% Calculate the transfer function
(
    real_potential_transfer_function,
    imag_potential_transfer_function,
) = phase_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_data.shape, **transfer_function_arguments
)

# %% Reconstruct
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    torch.Tensor(zyx_data),
    real_potential_transfer_function,
    imag_potential_transfer_function,
    transfer_function_arguments["z_padding"],
    regularization_parameter=0.01,
)
# %% Compared data and reconstruction
z_slices = [1, 3, 5, 7, 9]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, z in enumerate(z_slices):
    axes[0, i].imshow(zyx_data[z], cmap="gray", origin="lower")
    axes[0, i].set_title(f"Brightfield Data, z = {z}")
    axes[0, i].axis("off")
    axes[1, i].imshow(zyx_recon[z], cmap="gray", origin="lower")
    axes[1, i].set_title(f"Phase Reconstruction, z = {z}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

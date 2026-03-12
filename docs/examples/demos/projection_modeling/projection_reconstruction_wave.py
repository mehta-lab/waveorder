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
# Wave-Optical Projection Reconstruction

Reconstruct 3D volumes from pairs of tilted sum projections at +/-x degrees
using a forward model that includes the fluorescence OTF (convolution) followed
by Siddon projection. The adjoint backprojects, then correlates with the OTF.

CG-Tikhonov solves (H^T H + lambda I) x = H^T y, simultaneously deconvolving
and reconstructing.

Input: `fluorescence_blurred` from `projection_modeling.zarr`.
Comparison target: `fluorescence_density` (original unblurred density).
"""

# %%
from pathlib import Path

import numpy as np
import torch
import zarr
from siddon import cg_tikhonov, siddon_backproject, siddon_project

from waveorder.models import isotropic_fluorescent_thick_3d

# %% [markdown]
"""
## Parameters
"""

# %%
angles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
reg_strength = 1e-3
n_iter = 50
phantom_names = ["isolated_bead", "line_pattern"]

data_dir = Path("./data")
input_store = zarr.open(str(data_dir / "projection_modeling.zarr"), mode="r")

voxel_size = float(input_store.attrs["voxel_size_um"])
zyx_shape = tuple(input_store.attrs["zyx_shape"])
wavelength_emission = float(input_store.attrs["wavelength_emission_um"])
na_detection = float(input_store.attrs["na_detection"])
media_index = float(input_store.attrs["media_index"])

output_store = zarr.open(str(data_dir / "projection_reconstruction_wave.zarr"), mode="w")
output_store.attrs["voxel_size_um"] = voxel_size
output_store.attrs["zyx_shape"] = list(zyx_shape)
output_store.attrs["angles"] = angles
output_store.attrs["reg_strength"] = reg_strength
output_store.attrs["n_iter"] = n_iter

# %% [markdown]
"""
## Compute OTF
"""

# %%
print("Computing fluorescence OTF...")
z_padding = 0
otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    wavelength_emission=wavelength_emission,
    z_padding=z_padding,
    index_of_refraction_media=media_index,
    numerical_aperture_detection=na_detection,
)
print(f"OTF shape: {otf.shape}")

otf_conj = torch.conj(otf)


def apply_otf(vol_np):
    """Convolve a 3D volume with the fluorescence OTF."""
    vol_t = torch.tensor(vol_np, dtype=torch.float32)
    result = torch.fft.ifftn(torch.fft.fftn(vol_t) * otf).real
    return result.numpy()


def apply_otf_adjoint(vol_np):
    """Correlate a 3D volume with the OTF (conjugate multiply in Fourier space)."""
    vol_t = torch.tensor(vol_np, dtype=torch.float32)
    result = torch.fft.ifftn(torch.fft.fftn(vol_t) * otf_conj).real
    return result.numpy()


# %% [markdown]
"""
## Reconstruction Loop
"""

# %%
for phantom_name in phantom_names:
    print(f"\n{'=' * 60}")
    print(f"Phantom: {phantom_name}")
    print(f"{'=' * 60}")

    volume_blurred = np.array(input_store[f"{phantom_name}/fluorescence_blurred"])
    # Subtract the constant background added by apply_transfer_function
    volume_blurred = volume_blurred - volume_blurred.min()
    volume_gt = np.array(input_store[f"{phantom_name}/fluorescence_density"])
    output_store[f"{phantom_name}/ground_truth"] = volume_gt

    for angle in angles:
        print(f"\n  Angle pair: +/-{angle} deg")

        # Forward projections from blurred volume
        proj_plus = siddon_project(volume_blurred, +angle, voxel_size, "sum")
        proj_minus = siddon_project(volume_blurred, -angle, voxel_size, "sum")
        print(f"    Projection shapes: {proj_plus.shape}, {proj_minus.shape}")

        max_width = max(proj_plus.shape[1], proj_minus.shape[1])

        def _pad_to(arr, width):
            if arr.shape[1] >= width:
                return arr
            pad_left = (width - arr.shape[1]) // 2
            pad_right = width - arr.shape[1] - pad_left
            return np.pad(arr, ((0, 0), (pad_left, pad_right)))

        proj_plus = _pad_to(proj_plus, max_width)
        proj_minus = _pad_to(proj_minus, max_width)

        def forward(vol, _angle=angle, _vs=voxel_size, _w=max_width):
            blurred = apply_otf(vol)
            p_p = siddon_project(blurred, +_angle, _vs, "sum")
            p_m = siddon_project(blurred, -_angle, _vs, "sum")
            return [_pad_to(p_p, _w), _pad_to(p_m, _w)]

        def adjoint(projs, _angle=angle, _shape=zyx_shape, _vs=voxel_size):
            bp = siddon_backproject(projs[0], +_angle, _shape, _vs) + siddon_backproject(projs[1], -_angle, _shape, _vs)
            return apply_otf_adjoint(bp)

        print(f"    Running CG ({n_iter} iterations)...")
        recon = cg_tikhonov(forward, adjoint, [proj_plus, proj_minus], zyx_shape, reg_strength, n_iter)

        # Metrics against unblurred ground truth
        mse = np.mean((recon - volume_gt) ** 2)
        gt_range = volume_gt.max() - volume_gt.min()
        psnr = 10 * np.log10(gt_range**2 / mse) if mse > 0 else float("inf")
        print(f"    MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

        arr_name = f"{phantom_name}/recon_{angle:02d}deg"
        output_store[arr_name] = recon
        output_store[arr_name].attrs["angle_deg"] = angle
        output_store[arr_name].attrs["mse"] = float(mse)
        output_store[arr_name].attrs["psnr"] = float(psnr)

# %%
print(f"\nSaved: {data_dir / 'projection_reconstruction_wave.zarr'}")
for group_name in output_store.group_keys():
    group = output_store[group_name]
    for arr_name in group.array_keys():
        arr = group[arr_name]
        print(f"  {group_name}/{arr_name}: shape={arr.shape}")

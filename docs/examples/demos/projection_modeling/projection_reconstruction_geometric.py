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
# Geometric Projection Reconstruction

Reconstruct 3D volumes from pairs of tilted sum projections at +/-x degrees
using pure Siddon ray-tracing (no OTF). CG-Tikhonov solves the normal
equations (H^T H + lambda I) x = H^T y.

Input: `fluorescence_density` from `projection_modeling.zarr` (unblurred).
"""

# %%
from pathlib import Path

import numpy as np
import zarr
from siddon import cg_tikhonov, siddon_backproject, siddon_project

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

output_store = zarr.open(str(data_dir / "projection_reconstruction_geometric.zarr"), mode="w")
output_store.attrs["voxel_size_um"] = voxel_size
output_store.attrs["zyx_shape"] = list(zyx_shape)
output_store.attrs["angles"] = angles
output_store.attrs["reg_strength"] = reg_strength
output_store.attrs["n_iter"] = n_iter

# %% [markdown]
"""
## Reconstruction Loop
"""

# %%
for phantom_name in phantom_names:
    print(f"\n{'=' * 60}")
    print(f"Phantom: {phantom_name}")
    print(f"{'=' * 60}")

    volume_gt = np.array(input_store[f"{phantom_name}/fluorescence_density"])
    output_store[f"{phantom_name}/ground_truth"] = volume_gt

    for angle in angles:
        print(f"\n  Angle pair: +/-{angle} deg")

        # Forward projections from ground truth
        proj_plus = siddon_project(volume_gt, +angle, voxel_size, "sum")
        proj_minus = siddon_project(volume_gt, -angle, voxel_size, "sum")
        print(f"    Projection shapes: {proj_plus.shape}, {proj_minus.shape}")

        # Pad projections to the same lateral width (max of +/- angles)
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
            p_p = siddon_project(vol, +_angle, _vs, "sum")
            p_m = siddon_project(vol, -_angle, _vs, "sum")
            return [_pad_to(p_p, _w), _pad_to(p_m, _w)]

        def adjoint(projs, _angle=angle, _shape=zyx_shape, _vs=voxel_size):
            return siddon_backproject(projs[0], +_angle, _shape, _vs) + siddon_backproject(
                projs[1], -_angle, _shape, _vs
            )

        print(f"    Running CG ({n_iter} iterations)...")
        recon = cg_tikhonov(forward, adjoint, [proj_plus, proj_minus], zyx_shape, reg_strength, n_iter)

        # Metrics
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
print(f"\nSaved: {data_dir / 'projection_reconstruction_geometric.zarr'}")
for group_name in output_store.group_keys():
    group = output_store[group_name]
    for arr_name in group.array_keys():
        arr = group[arr_name]
        print(f"  {group_name}/{arr_name}: shape={arr.shape}")

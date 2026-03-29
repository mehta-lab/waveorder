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
# Projection Imaging: Forward Simulation and Reconstruction

This notebook demonstrates forward simulation and reconstruction for
projection-based microscopy. A 3D line-pattern phantom (the [o] target)
is blurred through fluorescence and phase transfer functions, projected
at oblique angles via Siddon ray-tracing, and reconstructed with
CG-Tikhonov inversion.

The forward model composes optical blur (3D OTF convolution) with
geometric projection: H(x) = Siddon(OTF * x). Reconstruction solves
(H^T H + lambda I) x = H^T y.
"""

# %% [markdown]
"""
## Setup and Imports
"""

# %%
import subprocess
import sys

subprocess.check_call(
    [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "waveorder @ git+https://github.com/mehta-lab/waveorder.git@projection-modeling",
    ]
)

# %%
import numpy as np
import torch

from waveorder.models import (
    isotropic_fluorescent_thick_3d,
    phase_thick_3d,
    projection_no_blur,
)
from waveorder.projection import SiddonOperator, cg_tikhonov, siddon_project
from waveorder.visuals import jupyter_visuals

# %% [markdown]
"""
## Simulation Parameters
"""

# %%
# Volume geometry
zyx_shape = (128, 128, 128)
voxel_size = 0.1  # um (isotropic)

# Optical parameters
wavelength_illumination = 0.500  # um
wavelength_emission = 0.520  # um
na_detection = 1.0
na_illumination = 0.5
index_of_refraction_media = 1.33
z_padding = 0

# Sample properties
bead_index = 1.52

# Detector model
black_level = 100  # counts
peak_intensity = 1024  # counts

# Projection angles for the demo
demo_angles = [0, -30, 30]  # degrees

# Reconstruction parameters
recon_angles = [-30, 30]  # +/- pair
reg_strength = 1e-3
n_iter = 30

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# %% [markdown]
"""
## Generate [o] Line-Pattern Phantom

The "lines" phantom from `projection_no_blur.generate_test_phantom` creates
a 3D volume with line pairs at varying spacings that spell [o].
We convert density to fluorescence (proportional to density) and phase
(proportional to optical path length).
"""

# %%
# Generate 3D phantom
volume = projection_no_blur.generate_test_phantom(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    phantom_type="lines",
    sphere_radius=0.25,
)
volume_np = volume.numpy()

# Convert to fluorescence and phase channels
fluorescence = torch.tensor(volume_np, dtype=torch.float32)

wavelength_medium = wavelength_illumination / index_of_refraction_media
delta_n = (bead_index - index_of_refraction_media) * volume_np
phase = torch.tensor(delta_n * voxel_size / wavelength_medium, dtype=torch.float32)

# Visualize center slices of the phantom
nz = zyx_shape[0]
phantom_slices = np.array([
    fluorescence[nz // 2].numpy(),
    fluorescence[:, nz // 2, :].numpy(),
    fluorescence[:, :, nz // 2].numpy(),
    phase[nz // 2].numpy(),
    phase[:, nz // 2, :].numpy(),
    phase[:, :, nz // 2].numpy(),
])
jupyter_visuals.plot_multicolumn(
    phantom_slices,
    num_col=3,
    size=4,
    set_title=True,
    titles=[
        "Fluorescence XY", "Fluorescence XZ", "Fluorescence YZ",
        "Phase XY", "Phase XZ", "Phase YZ",
    ],
    colormap="inferno",
)

# %% [markdown]
"""
## Compute Transfer Functions
"""

# %%
# Fluorescence OTF
fluorescence_otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    wavelength_emission=wavelength_emission,
    z_padding=z_padding,
    index_of_refraction_media=index_of_refraction_media,
    numerical_aperture_detection=na_detection,
)

# Phase transfer function (real component)
real_tf, _imag_tf = phase_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_shape,
    yx_pixel_size=voxel_size,
    z_pixel_size=voxel_size,
    wavelength_illumination=wavelength_illumination,
    z_padding=z_padding,
    index_of_refraction_media=index_of_refraction_media,
    numerical_aperture_illumination=na_illumination,
    numerical_aperture_detection=na_detection,
)

print(f"Fluorescence OTF shape: {fluorescence_otf.shape}")
print(f"Phase TF shape: {real_tf.shape}")

# %% [markdown]
"""
## Forward Simulation: Blur and Noise

Apply the transfer functions to simulate microscope image formation,
then scale to detector range and add Poisson noise.
"""

# %%
# Apply blur
fluor_blurred = isotropic_fluorescent_thick_3d.apply_transfer_function(
    fluorescence, fluorescence_otf, z_padding
)
phase_blurred = phase_thick_3d.apply_transfer_function(
    phase, real_tf, z_padding, brightness=1e3
)


def scale_and_noise(vol):
    """Scale to [black_level, peak_intensity] and apply Poisson noise."""
    v = vol.numpy() if isinstance(vol, torch.Tensor) else vol
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = black_level + (peak_intensity - black_level) * (v - v_min) / (v_max - v_min)
    else:
        v = np.full_like(v, black_level)
    return np.random.poisson(np.clip(v, 0, None)).astype(np.float32)


fluor_noisy = scale_and_noise(fluor_blurred)
phase_noisy = scale_and_noise(phase_blurred)

# Visualize blurred + noisy volumes
noisy_slices = np.array([
    fluor_noisy[nz // 2],
    fluor_noisy[:, nz // 2, :],
    fluor_noisy[:, :, nz // 2],
    phase_noisy[nz // 2],
    phase_noisy[:, nz // 2, :],
    phase_noisy[:, :, nz // 2],
])
jupyter_visuals.plot_multicolumn(
    noisy_slices,
    num_col=3,
    size=4,
    set_title=True,
    titles=[
        "Fluorescence XY", "Fluorescence XZ", "Fluorescence YZ",
        "Phase XY", "Phase XZ", "Phase YZ",
    ],
    colormap="inferno",
)

# %% [markdown]
"""
## Compute Projections

Project the blurred volumes at 0, -30, and +30 degrees using
Siddon ray-tracing (sum projection divided by ray path length).
"""

# %%
proj_images = []
proj_titles = []
max_width = 0
for angle in demo_angles:
    fp = siddon_project(fluor_noisy, angle, voxel_size, mode="sum")
    pp = siddon_project(phase_noisy, angle, voxel_size, mode="sum")

    # Normalize by ray path length
    theta_rad = np.radians(angle)
    cos_t, sin_t = abs(np.cos(theta_rad)), abs(np.sin(theta_rad))
    path_length = (zyx_shape[0] * cos_t + zyx_shape[2] * sin_t) * voxel_size
    if path_length > 0:
        fp = fp / path_length
        pp = pp / path_length

    proj_images.extend([fp, pp])
    proj_titles.extend([f"Fluorescence {angle}\u00b0", f"Phase {angle}\u00b0"])
    max_width = max(max_width, fp.shape[1], pp.shape[1])

# Pad projections to uniform width for display
proj_padded = []
for img in proj_images:
    pad_left = (max_width - img.shape[1]) // 2
    pad_right = max_width - img.shape[1] - pad_left
    proj_padded.append(np.pad(img, ((0, 0), (pad_left, pad_right))))

jupyter_visuals.plot_multicolumn(
    np.array(proj_padded),
    num_col=2,
    size=4,
    set_title=True,
    titles=proj_titles,
)

# %% [markdown]
"""
## Reconstruction from Two Projections

Reconstruct the 3D volume from the +/-30 degree projection pair
using CG-Tikhonov with the wave-optical forward model
(OTF convolution + Siddon projection).
"""

# %%
siddon_op = SiddonOperator(zyx_shape, recon_angles, voxel_size, device)

# Move OTFs to device
fluor_otf_dev = fluorescence_otf.to(device)
phase_tf_dev = real_tf.to(device)
fluor_otf_conj = torch.conj(fluor_otf_dev)
phase_tf_conj = torch.conj(phase_tf_dev)

results = {}
channels = {
    "Fluorescence": (fluor_noisy, fluorescence, fluor_otf_dev, fluor_otf_conj),
    "Phase": (phase_noisy, phase, phase_tf_dev, phase_tf_conj),
}

for ch_name, (noisy_vol, gt_tensor, otf_dev, otf_conj) in channels.items():
    print(f"\nReconstructing {ch_name}...")

    # Measurements: project the blurred volume
    source = torch.tensor(noisy_vol - black_level, dtype=torch.float32, device=device)
    measurements = siddon_op.project_all(source)
    measurements = [p - p.mean() for p in measurements]
    del source

    # Forward model: blur then project
    def make_forward_adjoint(otf, otf_c):
        def forward(vol):
            blurred = torch.fft.ifftn(torch.fft.fftn(vol) * otf).real
            return siddon_op.project_all(blurred)

        def adjoint(projs):
            bp = siddon_op.backproject_all(projs, ramp_filter=True)
            return torch.fft.ifftn(torch.fft.fftn(bp) * otf_c).real

        return forward, adjoint

    fwd, adj = make_forward_adjoint(otf_dev, otf_conj)
    recon = cg_tikhonov(fwd, adj, measurements, zyx_shape, reg_strength, n_iter, device)
    results[ch_name] = recon.cpu().numpy()

    # Optimal gain and PSNR
    gt = gt_tensor.numpy()
    r = results[ch_name]
    dot_rg = np.sum(r * gt)
    dot_rr = np.sum(r * r)
    alpha = dot_rg / dot_rr if dot_rr > 0 else 1.0
    mse = float(np.mean((alpha * r - gt) ** 2))
    gt_range = float(gt.max() - gt.min())
    psnr = 10 * np.log10(gt_range**2 / mse) if mse > 0 else float("inf")
    print(f"  {ch_name}: PSNR = {psnr:.2f} dB (optimal gain = {alpha:.4f})")
    results[f"{ch_name}_alpha"] = alpha

# %% [markdown]
"""
## Visualize Reconstruction vs Ground Truth

Compare center XY and XZ slices of the reconstruction (scaled by
optimal gain) with the ground-truth phantom.
"""

# %%
recon_images = []
recon_titles = []
for ch_name in ["Fluorescence", "Phase"]:
    gt = channels[ch_name][1].numpy()
    recon_scaled = results[f"{ch_name}_alpha"] * results[ch_name]

    recon_images.extend([
        gt[nz // 2],
        recon_scaled[nz // 2],
        gt[:, nz // 2, :],
        recon_scaled[:, nz // 2, :],
    ])
    recon_titles.extend([
        f"{ch_name} Object XY",
        f"{ch_name} Recon XY",
        f"{ch_name} Object XZ",
        f"{ch_name} Recon XZ",
    ])

jupyter_visuals.plot_multicolumn(
    np.array(recon_images),
    num_col=4,
    size=4,
    set_title=True,
    titles=recon_titles,
    colormap="inferno",
)

# %%

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
# 2D QLIPP Simulation and Reconstruction Demo

This notebook demonstrates forward simulation and reconstruction for Quantitative Label-free Imaging with Phase and Polarization (QLIPP).
The simulation and reconstruction are based on the QLIPP paper ([here](https://elifesciences.org/articles/55502)):

S.-M. Guo, L.-H. Yeh, J. Folkesson, I. E. Ivanov, A. P. Krishnan, M. G. Keefe, E. Hashemi,
D. Shin, B. B. Chhun, N. H. Cho, M. D. Leonetti, M. H. Han, T. J. Nowakowski, S. B. Mehta,
"Revealing architectural order with quantitative label-free imaging and deep learning,"
eLife 9:e55502 (2020).
"""

# %% [markdown]
"""
## Setup and Imports
First, let's install the latest version of waveorder from the main branch
"""

# %%
import subprocess
import sys

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift
from platformdirs import user_data_dir

from waveorder import optics, util, waveorder_simulator
from waveorder.visuals import jupyter_visuals

# %% [markdown]
"""
## Forward Simulation
Here we simulate QLIPP measurements of a Siemens star pattern with uniform phase, uniform retardance, and radial orientation.
"""

# %%
# Key parameters
N = 256  # number of pixel in y dimension
M = 256  # number of pixel in x dimension
mag = 40  # magnification
ps = 6.5 / mag  # effective pixel size
lambda_illu = 0.532  # wavelength
n_media = 1  # refractive index in the media
NA_obj = 0.55  # objective NA
NA_illu = 0.4  # illumination NA (condenser)
NA_illu_in = 0.4  # illumination NA (phase contrast inner ring)
z_defocus = (np.r_[:5] - 2) * 1.757  # a set of defocus plane
chi = 0.03 * 2 * np.pi  # swing of Polscope analyzer

# %% [markdown]
"""
### Generate Sample: Siemens Star Pattern
"""

# %%
# Sample : star with uniform phase, uniform retardance, and radial orientation
star, theta, _ = util.generate_star_target((N, M))
star = star.numpy()
theta = theta.numpy()
jupyter_visuals.plot_multicolumn(np.array([star, theta]), num_col=2, size=5)

# %%
# Assign uniform phase, uniform retardance, and radial slow axes to the star pattern
phase_value = 1  # average phase in radians (optical path length)
phi_s = star * (phase_value + 0.15)  # slower OPL across target
phi_f = star * (phase_value - 0.15)  # faster OPL across target
mu_s = np.zeros((N, M))  # absorption
mu_f = mu_s.copy()
t_eigen = np.zeros((2, N, M), complex)  # complex specimen transmission
t_eigen[0] = np.exp(-mu_s + 1j * phi_s)
t_eigen[1] = np.exp(-mu_f + 1j * phi_f)
sa = theta % np.pi  # slow axes.

jupyter_visuals.plot_multicolumn(
    np.array([phi_s, phi_f, mu_s, sa]),
    num_col=2,
    size=5,
    set_title=True,
    titles=["Phase (slow)", "Phase (fast)", "absorption", "slow axis"],
    origin="lower",
)

# %% [markdown]
"""
### Forward Model Setup
"""

# %%
# Source pupil
# Subsample source pattern for speed
xx, yy, fxx, fyy = util.gen_coordinate((N, M), ps)
radial_frequencies = np.sqrt(fxx**2 + fyy**2)
Source_cont = optics.generate_pupil(radial_frequencies, NA_illu, lambda_illu).numpy()
Source_discrete = optics.Source_subsample(Source_cont, lambda_illu * fxx, lambda_illu * fyy, subsampled_NA=0.1)
plt.figure(figsize=(10, 10))
plt.imshow(fftshift(Source_discrete), cmap="gray")
plt.show()

# %%
# Initialize microscope simulator
simulator = waveorder_simulator.waveorder_microscopy_simulator(
    (N, M),
    lambda_illu,
    ps,
    NA_obj,
    NA_illu,
    z_defocus,
    chi,
    n_media=n_media,
    illu_mode="Arbitrary",
    Source=Source_discrete,
)

# Compute image volumes and Stokes volumes
I_meas, Stokes_out = simulator.simulate_waveorder_measurements(t_eigen, sa, multiprocess=False)

# Add noise to the measurement
photon_count = 14000
ext_ratio = 10000
const_bg = photon_count / (0.5 * (1 - np.cos(chi))) / ext_ratio
I_meas_noise = (np.random.poisson(I_meas / np.max(I_meas) * photon_count + const_bg)).astype("float64")

# Save simulation
temp_dirpath = Path(user_data_dir("QLIPP_simulation"))
temp_dirpath.mkdir(parents=True, exist_ok=True)
output_file = temp_dirpath / "2D_QLIPP_simulation.npz"
np.savez(
    output_file,
    I_meas=I_meas_noise,
    Stokes_out=Stokes_out,
    lambda_illu=lambda_illu,
    n_media=n_media,
    NA_obj=NA_obj,
    NA_illu=NA_illu,
    ps=ps,
    Source_cont=Source_cont,
    z_defocus=z_defocus,
    chi=chi,
)

# %% [markdown]
"""
## Reconstruction
Now we'll reconstruct the simulated data to recover the sample properties.
"""

# %%
from waveorder import waveorder_reconstructor

# Load simulated data
array_loaded = np.load(output_file)
list_of_array_names = sorted(array_loaded)

for array_name in list_of_array_names:
    globals()[array_name] = array_loaded[array_name]

print("Loaded arrays:", list_of_array_names)

# %% [markdown]
"""
### Initial Reconstruction of Stokes Parameters and Anisotropy
"""

# %%
_, N, M, L = I_meas.shape
cali = False
bg_option = "global"

setup = waveorder_reconstructor.waveorder_microscopy(
    (N, M),
    lambda_illu,
    ps,
    NA_obj,
    NA_illu,
    z_defocus,
    chi,
    n_media=n_media,
    phase_deconv="2D",
    bire_in_plane_deconv="2D",
    illu_mode="BF",
)

S_image_recon = setup.Stokes_recon(I_meas)
S_image_tm = setup.Stokes_transform(S_image_recon)
Recon_para = setup.Polarization_recon(S_image_tm)  # Without accounting for diffraction

jupyter_visuals.plot_multicolumn(
    np.array(
        [
            Recon_para[0, :, :, L // 2],
            Recon_para[1, :, :, L // 2],
            Recon_para[2, :, :, L // 2],
            Recon_para[3, :, :, L // 2],
        ]
    ),
    num_col=2,
    size=5,
    set_title=True,
    titles=["Retardance", "2D orientation", "Brightfield", "Depolarization"],
    origin="lower",
)

jupyter_visuals.plot_hsv(
    [Recon_para[1, :, :, L // 2], Recon_para[0, :, :, L // 2]],
    max_val=1,
    origin="lower",
    size=10,
)

# %% [markdown]
"""
### 2D Retardance and Orientation Reconstruction with S₁ and S₂
"""

# %%
# Diffraction aware reconstruction assuming slowly varying transmission
S1_stack = S_image_recon[1].copy() / S_image_recon[0].mean()
S2_stack = S_image_recon[2].copy() / S_image_recon[0].mean()

# Tikhonov regularization
retardance, azimuth = setup.Birefringence_recon_2D(S1_stack, S2_stack, method="Tikhonov", reg_br=1e-2)

jupyter_visuals.plot_multicolumn(
    np.array([retardance, azimuth]),
    num_col=2,
    size=10,
    set_title=True,
    titles=["Reconstructed retardance", "Reconstructed orientation"],
    origin="lower",
)
jupyter_visuals.plot_hsv([azimuth, retardance], size=10, origin="lower")

# %%
# TV-regularized birefringence deconvolution
retardance_TV, azimuth_TV = setup.Birefringence_recon_2D(
    S1_stack,
    S2_stack,
    method="TV",
    reg_br=1e-1,
    rho=1e-5,
    lambda_br=1e-3,
    itr=20,
    verbose=True,
)

jupyter_visuals.plot_multicolumn(
    np.array([retardance_TV, azimuth_TV]),
    num_col=2,
    size=10,
    set_title=True,
    titles=["Reconstructed retardance (TV)", "Reconstructed orientation (TV)"],
    origin="lower",
)
jupyter_visuals.plot_hsv([azimuth_TV, retardance_TV], size=10, origin="lower")

# %%

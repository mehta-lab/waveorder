#!/usr/bin/env python
# coding: utf-8

#####################################################################################################
# 2D QLIPP reconstruction                                                                       #
# This simulation is based on the QLIPP paper ([here](https://elifesciences.org/articles/55502)):   #
# ``` S.-M. Guo, L.-H. Yeh, J. Folkesson, I. E. Ivanov, A. P. Krishnan, M. G. Keefe, E. Hashemi,    #
# D. Shin, B. B. Chhun, N. H. Cho, M. D. Leonetti, M. H. Han, T. J. Nowakowski, S. B. Mehta ,       #
# "Revealing architectural order with quantitative label-free imaging and deep learning,"           #
#  eLife 9:e55502 (2020).```                                                                        #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from waveorder import (
    waveorder_reconstructor,
    visual,
)


# ### Load simulated data
# Load simulations


file_name = "./2D_QLIPP_simulation.npz"

array_loaded = np.load(file_name)
list_of_array_names = sorted(array_loaded)

for array_name in list_of_array_names:
    globals()[array_name] = array_loaded[array_name]

print(list_of_array_names)

# ### Reconstruction of Stokes parameters and anisotropy
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
Recon_para = setup.Polarization_recon(
    S_image_tm
)  # Without accounting for diffraction

visual.plot_multicolumn(
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

visual.plot_hsv(
    [Recon_para[1, :, :, L // 2], Recon_para[0, :, :, L // 2]],
    max_val=1,
    origin="lower",
    size=10,
)
plt.show()

# ## 2D retardance and orientation reconstruction with $S_1$ and $S_2$
# Diffraction aware reconstruction assuming slowly varying transmission.
S1_stack = S_image_recon[1].copy() / S_image_recon[0].mean()
S2_stack = S_image_recon[2].copy() / S_image_recon[0].mean()

retardance, azimuth = setup.Birefringence_recon_2D(
    S1_stack, S2_stack, method="Tikhonov", reg_br=1e-3
)

visual.plot_multicolumn(
    np.array([retardance, azimuth]),
    num_col=2,
    size=10,
    set_title=True,
    titles=["Reconstructed retardance", "Reconstructed orientation"],
    origin="lower",
)
visual.plot_hsv([azimuth, retardance], size=10, origin="lower")
plt.show()


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

visual.plot_multicolumn(
    np.array([retardance_TV, azimuth_TV]),
    num_col=2,
    size=10,
    set_title=True,
    titles=["Reconstructed retardance", "Reconstructed orientation"],
    origin="lower",
)
visual.plot_hsv([azimuth_TV, retardance_TV], size=10, origin="lower")
plt.show()


# Commenting for now...phase recon has changed its API

# # ## 2D Phase reconstruction with $S_0$
# # Tikhonov regularizer
# reg_u = 1e-3
# reg_p = 1e-3
# S0_stack = S_image_recon[0].copy()
# mu_sample, phi_sample = setup.Phase_recon(
#     S0_stack, method="Tikhonov", reg_u=reg_u, reg_p=reg_p
# )
# wo.plot_multicolumn(
#     np.array([mu_sample, phi_sample]),
#     num_col=2,
#     size=10,
#     set_title=True,
#     titles=["Reconstructed absorption", "Reconstructed phase"],
#     origin="lower",
# )
# plt.show()


# # TV-regularized phase reconstruction
# lambda_u = 3e-3
# lambda_p = 1e-3
# S0_stack = S_image_recon[0].copy()

# mu_sample_TV, phi_sample_TV = setup.Phase_recon(
#     S0_stack, method="TV", lambda_u=lambda_u, lambda_p=lambda_p, itr=10, rho=1
# )
# wo.plot_multicolumn(
#     np.array([mu_sample_TV, phi_sample_TV]),
#     num_col=2,
#     size=10,
#     set_title=True,
#     titles=["Reconstructed absorption", "Reconstructed phase"],
#     origin="lower",
# )
# plt.show()

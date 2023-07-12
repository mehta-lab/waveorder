####################################################################
# The reconstruction of 2D uPTI                                    #
# This reconstruction is based on the uPTI paper                   #
# (https://www.biorxiv.org/content/10.1101/2020.12.15.422951v1)    #
#  ```L.-H. Yeh, I. E. Ivanov, B. B. Chhun, S.-M. Guo, E. Hashemi, #
#  J. R. Byrum, J. A. Pérez-Bermejo, H. Wang, Y. Yu,               #
#  P. G. Kazansky, B. R. Conklin, M. H. Han, and S. B. Mehta,      #
#  "uPTI: uniaxial permittivity tensor imaging of intrinsic        #
#  density and anisotropy," bioRxiv 2020.12.15.422951 (2020).```   #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift

from waveorder import (
    optics,
    waveorder_reconstructor,
    visual,
)

## Initialization
## Load simulated images and parameters

file_name = "./PTI_simulation_data_NA_det_147_NA_illu_140_2D_spoke_discrete_no_1528_ne_1553_no_noise_Born.npz"

array_loaded = np.load(file_name)
list_of_array_names = sorted(array_loaded)

for array_name in list_of_array_names:
    globals()[array_name] = array_loaded[array_name]

print(list_of_array_names)

L = I_meas.shape[-1]
I_meas = np.transpose(I_meas[:, :, :, :, L // 2], (0, 2, 3, 1))
z_defocus = np.array([0])

I_meas = I_meas[1:]
print(I_meas.shape)


### Initiate the reconstruction

_, N, M, _ = I_meas.shape
cali = False
bg_option = "global"
use_gpu = False
gpu_id = 0

# chi = pi/2

A_matrix = 0.5 * np.array([[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]])


setup = waveorder_reconstructor.waveorder_microscopy(
    (N, M),
    lambda_illu,
    ps,
    NA_obj,
    NA_illu,
    z_defocus,
    chi,
    n_media=n_media,
    cali=cali,
    phase_deconv="2D",
    A_matrix=A_matrix,
    inc_recon="2D-vec-WOTF",
    illu_mode="Arbitrary",
    Source=Source_cont,
    Source_PolState=Source_PolState,
    use_gpu=use_gpu,
    gpu_id=gpu_id,
)

## Visualize 2  D transfer functions as a function of illumination pattern

# illumination patterns used
visual.plot_multicolumn(
    fftshift(Source_cont, axes=(1, 2)), origin="lower", num_col=5, size=5
)
plt.show()

## Reconstruct Stokes images and visualize them as a function of illumination pattern

S_image_recon = setup.Stokes_recon(I_meas)

S_image_tm = np.zeros_like(S_image_recon)

S_bg_mean_0 = np.mean(S_image_recon[0, :, :, :], axis=(0, 1))[
    np.newaxis, np.newaxis, :
]
S_bg_mean_1 = np.mean(S_image_recon[1, :, :, :], axis=(0, 1))[
    np.newaxis, np.newaxis, :
]
S_bg_mean_2 = np.mean(S_image_recon[2, :, :, :], axis=(0, 1))[
    np.newaxis, np.newaxis, :
]


S_image_tm[0] = S_image_recon[0] / S_bg_mean_0 - 1
S_image_tm[1] = (
    S_image_recon[1] / S_bg_mean_0
    - S_bg_mean_1 * S_image_recon[0] / S_bg_mean_0**2
)
S_image_tm[2] = (
    S_image_recon[2] / S_bg_mean_0
    - S_bg_mean_2 * S_image_recon[0] / S_bg_mean_0**2
)

## 2D uPTI reconstruction

### Compute the components of the scattering potential tensor

reg_inc = np.array([1, 1, 1, 1, 1, 1, 1]) * 1e-1
reg_ret_pr = 1e-2
f_tensor = setup.scattering_potential_tensor_recon_2D_vec(
    S_image_tm, reg_inc=reg_inc, cupy_det=True
)

visual.plot_multicolumn(
    f_tensor,
    num_col=4,
    origin="lower",
    size=5,
    set_title=True,
    titles=[
        r"$f_{0r}$",
        r"$f_{0i}$",
        r"$f_{1c}$",
        r"$f_{1s}$",
        r"$f_{2c}$",
        r"$f_{2s}$",
        r"$f_{3}$",
    ],
)
plt.show()

### Estimate principal retardance, orientation, inclination, and optic axis from the scattering potential tensor
(
    retardance_pr,
    azimuth,
    theta,
    mat_map,
) = setup.scattering_potential_tensor_to_3D_orientation(
    f_tensor,
    S_image_tm,
    material_type="unknown",
    reg_ret_pr=reg_ret_pr,
    itr=20,
    step_size=0.1,
    fast_gpu_mode=True,
)
plt.show()

# scaling to the physical properties of the material

# optic sign probability
p_mat_map = optics.optic_sign_probability(mat_map, mat_map_thres=0.1)

# absorption and phase
phase = optics.phase_inc_correction(f_tensor[0], retardance_pr[0], theta[0])
absorption = f_tensor[1].copy()
phase_nm, absorption_nm, retardance_pr_nm = [
    optics.unit_conversion_from_scattering_potential_to_permittivity(
        SP_array, lambda_illu, n_media=n_media, imaging_mode=img_mode
    )
    for img_mode, SP_array in zip(
        ["2D", "2D", "2D-ret"], [phase, absorption, retardance_pr]
    )
]

# # clean up GPU memory leftorver

# import gc
# import cupy as cp

# gc.collect()
# cp.get_default_memory_pool().free_all_blocks()

## Visualize reconstructed physical properties of simulated sample
### Phase, principal retardance, azimuth, inclination, and optic sign

abs_min = -0.5
abs_max = 0.5
phase_min = -1
phase_max = 1
ret_min = 0
ret_max = 1.5
p_min = 0.4
p_max = 0.6


fig, ax = plt.subplots(2, 3, figsize=(30, 20))

sub_ax = ax[0, 0].imshow(
    absorption_nm, cmap="gray", origin="lower", vmin=abs_min, vmax=abs_max
)
ax[0, 0].set_title("absorption")
plt.colorbar(sub_ax, ax=ax[0, 0])

sub_ax = ax[0, 1].imshow(
    phase_nm, cmap="gray", origin="lower", vmin=phase_min, vmax=phase_max
)
ax[0, 1].set_title("phase")
plt.colorbar(sub_ax, ax=ax[0, 1])

sub_ax = ax[0, 2].imshow(
    np.abs(retardance_pr_nm[0]),
    cmap="gray",
    origin="lower",
    vmin=ret_min,
    vmax=ret_max,
)
ax[0, 2].set_title("principal retardance (+)")
plt.colorbar(sub_ax, ax=ax[0, 2])

sub_ax = ax[1, 0].imshow(
    p_mat_map, cmap="gray", origin="lower", vmin=p_min, vmax=p_max
)
ax[1, 0].set_title("optic sign probability")
plt.colorbar(sub_ax, ax=ax[1, 0])

sub_ax = ax[1, 1].imshow(
    azimuth[0], origin="lower", cmap="gray", vmin=0, vmax=np.pi
)
ax[1, 1].set_title("in-plane orientation (+)")

sub_ax = ax[1, 2].imshow(
    theta[0], origin="lower", cmap="gray", vmin=0, vmax=np.pi
)
ax[1, 2].set_title("inclination (+)")

plt.show()

### Render 3D orientation with 3D colorsphere (azimuth and inclination)
# rendering with 3D color

ret_min_color = 0
ret_max_color = 1.5

orientation_3D_image = np.transpose(
    np.array(
        [
            azimuth[0] / 2 / np.pi,
            theta[0],
            (
                np.clip(
                    np.abs(retardance_pr_nm[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (1, 2, 0),
)
orientation_3D_image_RGB = visual.orientation_3D_to_rgb(
    orientation_3D_image, interp_belt=20 / 180 * np.pi, sat_factor=1
)

plt.figure(figsize=(5, 5))
plt.imshow(orientation_3D_image_RGB, origin="lower")
plt.figure(figsize=(3, 3))
visual.orientation_3D_colorwheel(
    wheelsize=256, circ_size=50, interp_belt=20 / 180 * np.pi, sat_factor=1
)
plt.show()

### Render 3D orientation with 2 channels (in-plane orientation and out-of-plane tilt)
# in-plane orientation
from matplotlib.colors import hsv_to_rgb


ret_min_color = 0
ret_max_color = 1.5


I_hsv = np.transpose(
    np.array(
        [
            (azimuth[0]) % np.pi / np.pi,
            np.ones_like(retardance_pr_nm[0]),
            (
                np.clip(
                    np.abs(retardance_pr_nm[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (1, 2, 0),
)
in_plane_orientation = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(5, 5))
plt.imshow(in_plane_orientation, origin="lower")
plt.figure(figsize=(3, 3))
visual.orientation_2D_colorwheel()
plt.show()

# out-of-plane tilt

threshold_inc = np.pi / 90

I_hsv = np.transpose(
    np.array(
        [
            (
                -np.maximum(0, np.abs(theta[0] - np.pi / 2) - threshold_inc)
                + np.pi / 2
                + threshold_inc
            )
            / np.pi,
            np.ones_like(retardance_pr_nm[0]),
            (
                np.clip(
                    np.abs(retardance_pr_nm[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (1, 2, 0),
)
out_of_plane_tilt = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(5, 5))
plt.imshow(out_of_plane_tilt, origin="lower")
plt.show()

## Evaluation of reconstructed 3D orientation
### 3D orientation overlaid as lines on retardance image.


spacing = 4
plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
visual.plot3DVectorField(
    np.abs(retardance_pr_nm[0]),
    azimuth[0],
    theta[0],
    anisotropy=0.4 * np.abs(retardance_pr_nm[0]),
    cmapImage="gray",
    clim=[ret_min, ret_max],
    aspect=1,
    spacing=spacing,
    window=spacing,
    linelength=spacing,
    linewidth=1,
    cmapAzimuth="hsv",
    alpha=0.8,
)
plt.show()
### Angular histogram of computed 3D orientation
# Angular histogram of 3D orientation
ret_mask = np.abs(retardance_pr_nm[0]).copy()
ret_mask[ret_mask < 0.5] = 0

plt.figure(figsize=(10, 10))
plt.imshow(ret_mask, cmap="gray", origin="lower")
visual.orientation_3D_hist(
    azimuth[0].flatten(),
    theta[0].flatten(),
    ret_mask.flatten(),
    bins=36,
    num_col=1,
    size=10,
    contour_level=100,
    hist_cmap="gnuplot2",
    top_hemi=True,
)
plt.show()

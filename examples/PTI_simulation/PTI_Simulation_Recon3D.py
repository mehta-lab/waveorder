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
I_meas = I_meas[1:]
print(I_meas.shape)

### Setup reconstructor class
_, _, N, M, L = I_meas.shape
cali = False
bg_option = "global"
use_gpu = False
gpu_id = 0


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
    A_matrix=A_matrix,
    inc_recon="3D",
    illu_mode="Arbitrary",
    Source=Source_cont,
    Source_PolState=Source_PolState,
    pad_z=5,
    use_gpu=use_gpu,
    gpu_id=gpu_id,
)


### Illumination patterns used
visual.plot_multicolumn(
    fftshift(Source_cont, axes=(1, 2)), origin="lower", num_col=5, size=5
)
plt.show()

## Reconstruct Stokes images and visualize them as a function of illumination pattern
S_image_recon = setup.Stokes_recon(I_meas)

S_bg_mean_0 = np.mean(S_image_recon[0, :, :, :, :], axis=(1, 2, 3))[
    :, np.newaxis, np.newaxis, np.newaxis
]
S_bg_mean_1 = np.mean(S_image_recon[1, :, :, :, :], axis=(1, 2, 3))[
    :, np.newaxis, np.newaxis, np.newaxis
]
S_bg_mean_2 = np.mean(S_image_recon[2, :, :, :, :], axis=(1, 2, 3))[
    :, np.newaxis, np.newaxis, np.newaxis
]

S_image_tm = np.zeros_like(S_image_recon)
S_image_tm[0] = S_image_recon[0] / S_bg_mean_0 - 1
S_image_tm[1] = (
    S_image_recon[1] / S_bg_mean_0
    - S_bg_mean_1 * S_image_recon[0] / S_bg_mean_0**2
)
S_image_tm[2] = (
    S_image_recon[2] / S_bg_mean_0
    - S_bg_mean_2 * S_image_recon[0] / S_bg_mean_0**2
)

## 3D uPTI reconstruction
### 3D volumes of the components of scattering potential tensor

# Regularization parameters.

# reg_inc for 3D data
# reg_inc = np.array([1, 1, 5e1, 5e1, 2.5e1, 2.5e1, 5e1])*1
#####################


# reg_inc for 2D data
reg_inc = np.array([1, 1, 5e1, 5e1, 5e1, 5e1, 5e1]) * 1
#####################

reg_ret_pr = 1e-2
f_tensor = setup.scattering_potential_tensor_recon_3D_vec(
    S_image_tm, reg_inc=reg_inc, cupy_det=True
)

visual.plot_multicolumn(
    f_tensor[..., L // 2],
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

### 3D volumes of principal retardance, 3D orientation, and optic sign

# reconstruct 3D anisotropy (principal retardance, 3D orientation, optic sign probability)
# material type:
# "positive" -> only solution of positively uniaxial material
# "negative" -> only solution of negatively uniaxial meterial
# "unknown" -> both solutions of positively and negatively uniaxial material + optic sign estimation

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
p_mat_map = optics.optic_sign_probability(mat_map, mat_map_thres=0.2)


# absorption and phase
phase = optics.phase_inc_correction(f_tensor[0], retardance_pr[0], theta[0])
absorption = f_tensor[1].copy()
phase_PT, absorption_PT, retardance_pr_PT = [
    optics.unit_conversion_from_scattering_potential_to_permittivity(
        SP_array, lambda_illu, n_media=n_media, imaging_mode="3D"
    )
    for SP_array in [phase, absorption, retardance_pr]
]

# clean up GPU memory leftorver

# import gc
# import cupy as cp

# gc.collect()
# cp.get_default_memory_pool().free_all_blocks()

## Visualize reconstructed physical properties of simulated sample
### Reconstructed phase, absorption, principal retardance, azimuth, and inclination assuming (+) and (-) optic sign

# browse the reconstructed physical properties
visual.plot_multicolumn(
    np.stack(
        [
            phase_PT[..., L // 2],
            retardance_pr_PT[0, :, :, L // 2],
            azimuth[0, :, :, L // 2],
            theta[0, :, :, L // 2],
            absorption_PT[..., L // 2],
            retardance_pr_PT[1, :, :, L // 2],
            azimuth[1, :, :, L // 2],
            theta[1, :, :, L // 2],
        ]
    ),
    num_col=4,
    origin="lower",
    set_title=True,
    size=5,
    titles=[
        r"phase",
        r"principal retardance (+)",
        r"$\omega$ (+)",
        r"$\theta$ (+)",
        r"absorption",
        r"principal retardance (-)",
        r"$\omega$ (-)",
        r"$\theta$ (-)",
    ],
)
plt.show()

### Phase, principal retardance, color-coded 3D orientation (azimuth and inclination), and optic signtruction
## display parameters for 3D dataset ##
# z_layer = 25
# y_layer = 100
# x_layer = 100
# phase_min = -0.02
# phase_max = 0.02
# ret_min = 0
# ret_max = 0.007
# p_min = 0.4
# p_max = 0.6
# abs_min = -0.02
# abs_max = 0.02
######################################

## display parameters for 2D dataset ##
z_layer = L // 2
y_layer = M // 2
x_layer = N // 2
phase_min = -0.012
phase_max = 0.012
ret_min = 0
ret_max = 0.002
p_min = 0.4
p_max = 0.6
abs_min = -0.01
abs_max = 0.01
######################################


fig, ax = plt.subplots(6, 2, figsize=(5, 15))
sub_ax = ax[0, 0].imshow(
    absorption_PT[:, :, z_layer],
    cmap="gray",
    origin="lower",
    vmin=abs_min,
    vmax=abs_max,
)
ax[0, 0].set_title("absorption (xy)")
plt.colorbar(sub_ax, ax=ax[0, 0])

sub_ax = ax[0, 1].imshow(
    np.transpose(absorption_PT[y_layer, :, :]),
    cmap="gray",
    origin="lower",
    vmin=abs_min,
    vmax=abs_max,
    aspect=psz / ps,
)
ax[0, 1].set_title("absorption (xz)")
plt.colorbar(sub_ax, ax=ax[0, 1])

sub_ax = ax[1, 0].imshow(
    phase_PT[:, :, z_layer],
    cmap="gray",
    origin="lower",
    vmin=phase_min,
    vmax=phase_max,
)
ax[1, 0].set_title("phase (xy)")
plt.colorbar(sub_ax, ax=ax[1, 0])

sub_ax = ax[1, 1].imshow(
    np.transpose(phase_PT[y_layer, :, :]),
    cmap="gray",
    origin="lower",
    vmin=phase_min,
    vmax=phase_max,
    aspect=psz / ps,
)
ax[1, 1].set_title("absorption (xz)")
plt.colorbar(sub_ax, ax=ax[1, 1])

sub_ax = ax[2, 0].imshow(
    np.abs(retardance_pr_PT[0, :, :, z_layer]),
    cmap="gray",
    origin="lower",
    vmin=ret_min,
    vmax=ret_max,
)
ax[2, 0].set_title("principal retardance (+) (xy)")
plt.colorbar(sub_ax, ax=ax[2, 0])

sub_ax = ax[2, 1].imshow(
    np.transpose(np.abs(retardance_pr_PT[0, y_layer, :, :])),
    cmap="gray",
    origin="lower",
    vmin=ret_min,
    vmax=ret_max,
    aspect=psz / ps,
)
ax[2, 1].set_title("principal retardance (+) (xz)")
plt.colorbar(sub_ax, ax=ax[2, 1])

sub_ax = ax[3, 0].imshow(
    np.abs(p_mat_map[:, :, z_layer]),
    cmap="gray",
    origin="lower",
    vmin=p_min,
    vmax=p_max,
)
ax[3, 0].set_title("optic sign probability (xy)")
plt.colorbar(sub_ax, ax=ax[3, 0])

sub_ax = ax[3, 1].imshow(
    np.transpose(np.abs(p_mat_map[y_layer, :, :])),
    cmap="gray",
    origin="lower",
    vmin=p_min,
    vmax=p_max,
    aspect=psz / ps,
)
ax[3, 1].set_title("optic sign probability (xz)")
plt.colorbar(sub_ax, ax=ax[3, 1])

sub_ax = ax[4, 0].imshow(
    azimuth[0, :, :, z_layer], cmap="gray", origin="lower", vmin=0, vmax=np.pi
)
ax[4, 0].set_title("in-plane orientation (+) (xy)")

sub_ax = ax[4, 1].imshow(
    np.transpose(azimuth[0, y_layer, :, :]),
    cmap="gray",
    origin="lower",
    vmin=0,
    vmax=np.pi,
    aspect=psz / ps,
)
ax[4, 1].set_title("in-plane orientation (+) (xz)")

sub_ax = ax[5, 0].imshow(
    theta[0, :, :, z_layer], cmap="gray", origin="lower", vmin=0, vmax=np.pi
)
ax[5, 0].set_title("inclination (+) (xy)")

sub_ax = ax[5, 1].imshow(
    np.transpose(theta[0, y_layer, :, :]),
    cmap="gray",
    origin="lower",
    vmin=0,
    vmax=np.pi,
    aspect=psz / ps,
)
ax[5, 1].set_title("inclination (+) (xz)")
plt.show()

### Render 3D orientation with 3D colorsphere (azimuth and inclination)
# create color-coded orientation images

## display parameters for 3D dataset ##
# ret_min_color = 0
# ret_max_color = 0.007
######################################


## display parameters for 2D dataset ##
ret_min_color = 0
ret_max_color = 0.002
######################################

orientation_3D_image = np.transpose(
    np.array(
        [
            azimuth[0] / 2 / np.pi,
            theta[0],
            (
                np.clip(
                    np.abs(retardance_pr_PT[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (3, 1, 2, 0),
)
orientation_3D_image_RGB = visual.orientation_3D_to_rgb(
    orientation_3D_image, interp_belt=20 / 180 * np.pi, sat_factor=1
)


plt.figure(figsize=(10, 10))
plt.imshow(orientation_3D_image_RGB[z_layer], origin="lower")
plt.figure(figsize=(10, 10))
plt.imshow(
    orientation_3D_image_RGB[:, y_layer], origin="lower", aspect=psz / ps
)
# plot the top view of 3D orientation colorsphere
plt.figure(figsize=(3, 3))
visual.orientation_3D_colorwheel(
    wheelsize=128,
    circ_size=50,
    interp_belt=20 / 180 * np.pi,
    sat_factor=1,
    discretize=True,
)
plt.show()


### Render 3D orientation with 2 channels (in-plane orientation and out-of-plane tilt)

# in-plane orientation
from matplotlib.colors import hsv_to_rgb

I_hsv = np.transpose(
    np.array(
        [
            (azimuth[0]) % np.pi / np.pi,
            np.ones_like(retardance_pr_PT[0]),
            (
                np.clip(
                    np.abs(retardance_pr_PT[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (3, 1, 2, 0),
)
in_plane_orientation = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(10, 10))
plt.imshow(in_plane_orientation[z_layer], origin="lower")
plt.figure(figsize=(10, 10))
plt.imshow(in_plane_orientation[:, y_layer], origin="lower", aspect=psz / ps)
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
            np.ones_like(retardance_pr_PT[0]),
            (
                np.clip(
                    np.abs(retardance_pr_PT[0]), ret_min_color, ret_max_color
                )
                - ret_min_color
            )
            / (ret_max_color - ret_min_color),
        ]
    ),
    (3, 1, 2, 0),
)
out_of_plane_tilt = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(10, 10))
plt.imshow(out_of_plane_tilt[z_layer], origin="lower")
plt.figure(figsize=(10, 10))
plt.imshow(out_of_plane_tilt[:, y_layer], origin="lower", aspect=psz / ps)
plt.show()

## Evaluation of reconstructed 3D orientation
### 3D orientation overlaid as lines on retardance image.
vect = np.zeros((3,) + azimuth.shape)
vect[0] = np.sin(theta) * np.cos(azimuth)
vect[1] = np.sin(theta) * np.sin(azimuth)
vect[2] = np.cos(theta)

theta_x = np.arccos(vect[0])
azimuth_x = np.arctan2(vect[2], vect[1])

theta_y = np.arccos(vect[1])
azimuth_y = np.arctan2(vect[2], vect[0])

z_step = psz

### 3D data parameter ###
# spacing = 4
# z_layer =  25
# x_layer = 100
# y_layer = 100
# linelength_scale = 10
#########################


### select slices to plot ###
spacing = 4
z_layer = L // 2
x_layer = N // 2
y_layer = M // 2
linelength_scale = 20
#########################


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
visual.plot3DVectorField(
    np.abs(retardance_pr_PT[0, :, :, z_layer]),
    azimuth[0, :, :, z_layer],
    theta[0, :, :, z_layer],
    anisotropy=0.4 * np.abs(retardance_pr_PT[0, :, :, z_layer]),
    cmapImage="gray",
    clim=[ret_min, ret_max],
    aspect=1,
    spacing=spacing,
    window=spacing,
    linelength=spacing * linelength_scale,
    linewidth=1,
    cmapAzimuth="hsv",
    alpha=0.4,
    subplot_ax=ax[0, 0],
)
ax[0, 0].set_title(f"XY section (z= {z_layer})")

visual.plot3DVectorField(
    np.transpose(np.abs(retardance_pr_PT[0, :, x_layer, :])),
    np.transpose(azimuth_x[0, :, x_layer, :]),
    np.transpose(theta_x[0, :, x_layer, :]),
    anisotropy=0.4 * np.transpose(np.abs(retardance_pr_PT[0, :, x_layer, :])),
    cmapImage="gray",
    clim=[ret_min, ret_max],
    aspect=z_step / ps,
    spacing=spacing,
    window=spacing,
    linelength=spacing * linelength_scale,
    linewidth=1,
    cmapAzimuth="hsv",
    alpha=0.4,
    subplot_ax=ax[0, 1],
)
ax[0, 1].set_title(f"YZ section (x = {x_layer})")

visual.plot3DVectorField(
    np.transpose(np.abs(retardance_pr_PT[0, y_layer, :, :])),
    np.transpose(azimuth_y[0, y_layer, :, :]),
    np.transpose(theta_y[0, y_layer, :, :]),
    anisotropy=0.4 * np.transpose(np.abs(retardance_pr_PT[0, y_layer, :, :])),
    cmapImage="gray",
    clim=[ret_min, ret_max],
    aspect=z_step / ps,
    spacing=spacing,
    window=spacing,
    linelength=spacing * linelength_scale,
    linewidth=1,
    cmapAzimuth="hsv",
    alpha=0.4,
    subplot_ax=ax[1, 0],
)
ax[1, 0].set_title(f"XZ section (y = {y_layer})")

ax[1, 1].remove()
plt.show()


### Angular histogram of computed 3D orientation

ret_mask = np.abs(retardance_pr_PT[0]).copy()

## threshold parameters for 3D dataset ##
# ret_mask[ret_mask<0.15]=0
######################################


## threshold parameters for 2D dataset ##
ret_mask[ret_mask < 0.00125] = 0
######################################

plt.figure(figsize=(10, 10))
plt.imshow(ret_mask[:, :, z_layer], cmap="gray", origin="lower")
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

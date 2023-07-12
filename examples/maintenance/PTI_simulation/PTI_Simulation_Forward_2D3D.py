####################################################################
# Forward simulation of uniaxial permittivity tensor imaging (uPTI)#
# This simulation is based on the uPTI paper                       #
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
    waveorder_simulator,
    visual,
    util,
)

#####################################################################
# Initialization - imaging system and sample                        #
# Assumptions                                                       #
# 1. In our simulations, the specimen is always assumed to be 3D    #
#    in order to model volumetric scattering.                       #
# 2. A 2D specimen, i.e., a specimen thinner than the depth of focus#
#    of the microscope, is simulated by setting a single layer of   #
#    the volume to sample properties. For a 2D specimen, all layers #
#    other than sample layer are assumed to have the isotropic      #
#    permittivity of the surrounding medium.                        #
# 3. This forward simulation generates 3D intensity stacks acquired #
#    with 9 patterns of asymmetric illumination and 4 polarization  #
#    states for both a 2D thin specimen or a 3D specimen.           #
# 4. 3D reconstruction notebook can be used to reconstruct the 3D   #
#    uPT of the 2D or 3D specimen given their intensity stacks.     #
#    The 3D reconstruction of 2D specimen will report isotropic     #
#    permittivity of medium outside of the depth of focus of the    #
#    microscope.                                                    #
# 5. 2D reconstruction notebook computes 2D uPT from a slice of     #
#    the intensity stacks and it only works for the 2D thin         #
#    specimen.                                                      #
#####################################################################


### Parameters of sample

sample_type = "2D"  # 2D or 3D
N = 50  # number of pixel in y dimension
M = 50  # number of pixel in x dimension
L = 26  # number of layers in z dimension
z_layer = L // 2  # default z-slice to display in XY views.
y_layer = M // 2  # default y-slice to display in XZ views.

### Parameters of imaging system

n_media = 1.518  # refractive index in the media
mag = 63  # magnification
ps = 6 / mag  # effective pixel size
z_pixel_size = 0.125 / 2  # axial pixel size
lambda_illu = 0.532  # wavelength
NA_obj = 1.47  # objective NA
NA_illu = 1.4  # illumination NA
chi = 0.25 * 2 * np.pi  # swing of the microscope
z_defocus = (np.r_[:L] - L // 2) * z_pixel_size  # defocus position
use_gpu = False  # option to use gpu
gpu_id = 0  # gpu to be used

### Sample pattern

if sample_type == "3D":
    # 3D spoke pattern whose spokes are inclined 60 degrees relative to the z-axis. The principal retardance varies with depth and 3D orientation of permittivity tensor is aligned with the structural orientation of the spoke.
    blur_size = 1 * ps
    target, azimuth, inclination = util.genStarTarget_3D(
        (N, M, L),
        ps,
        z_pixel_size,
        blur_size,
        inc_upper_bound=np.pi / 8,
        inc_range=np.pi / 64,
    )
    inclination = np.round(inclination / np.pi * 8) / 8 * np.pi
    azimuth = np.round(azimuth / np.pi / 2 * 16) / 16 * np.pi * 2
elif sample_type == "2D":
    ## 2D spoke pattern, azimuth aligned with spokes, and the inclination set to 60 degrees ##
    target, azimuth, _ = util.generate_star_target(
        (N, M), blur_px=1 * ps, margin=10
    )
    target = target.numpy()
    azimuth = azimuth.numpy()
    inclination = np.ones_like(target) * np.pi / 3
    azimuth = azimuth % (np.pi * 2)
    azimuth = np.round(azimuth / np.pi / 2 * 16) / 16 * np.pi * 2

    # pad zero in the z direction outside of the focal plane to mimic thin specimens
    target = np.pad(
        target[:, :, np.newaxis],
        ((0, 0), (0, 0), (L // 2, L // 2 - 1)),
        mode="constant",
    )
    azimuth = np.pad(
        azimuth[:, :, np.newaxis],
        ((0, 0), (0, 0), (L // 2, L // 2 - 1)),
        mode="constant",
    )
    inclination = np.pad(
        inclination[:, :, np.newaxis],
        ((0, 0), (0, 0), (L // 2, L // 2 - 1)),
        mode="constant",
    )
else:
    print("sample_type needs to be 2D or 3D.")


# principal retardance, radial azimuth, and constant inclination to the pattern

n_o = n_media + 0.01
n_e = n_media + 0.035

inclination = inclination
azimuth = (azimuth) % (2 * np.pi)


no_map = np.zeros((N, M, L))
no_map[target > 0] = target[target > 0] * (n_o - n_media)
no_map += n_media
no_map_copy = no_map.copy()

ne_map = np.zeros((N, M, L))
ne_map[target > 0] = target[target > 0] * (n_e - n_media)
ne_map += n_media
ne_map_copy = ne_map.copy()

biref_map = ne_map_copy - no_map_copy
## assign material on the top to be reverse optic sign ##
# no_map[:,:,L//2:] = ne_map_copy[:,:,L//2:].copy()
# ne_map[:,:,L//2:] = no_map_copy[:,:,L//2:].copy()
#########################################################

## assign material on the right to be reverse optic sign ##
# no_map[:,M//2:,:] = ne_map_copy[:,M//2:,:].copy()
# ne_map[:,M//2:,:] = no_map_copy[:,M//2:,:].copy()
###########################################################

### Visualize sample properties

#### XY sections
visual.plot_multicolumn(
    [
        target[:, :, z_layer],
        azimuth[:, :, z_layer] % (2 * np.pi),
        inclination[:, :, z_layer],
    ],
    origin="lower",
    size=5,
    num_col=3,
    titles=[f"target, z={z_layer}", "azimuth", "inclination"],
    set_title=True,
)
#### XZ sections
visual.plot_multicolumn(
    [
        np.transpose(target[y_layer, :, :]),
        np.transpose(azimuth[y_layer, :, :]) % (2 * np.pi),
        np.transpose(inclination[y_layer, :, :]),
    ],
    origin="lower",
    size=5,
    num_col=3,
    titles=["retardance", "azimuth", "inclination"],
    set_title=True,
)
plt.show()

#### Principal retardance and 3D orientation in color

orientation_3D_image = np.transpose(
    np.array(
        [
            azimuth % (2 * np.pi) / 2 / np.pi,
            inclination,
            np.clip(
                (ne_map - no_map)
                * z_pixel_size
                * 2
                * np.pi
                / lambda_illu
                / np.pi
                / 2
                * lambda_illu
                * 1e3,
                0,
                1.5,
            )
            / 1.5,
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
plt.imshow(orientation_3D_image_RGB[:, y_layer], origin="lower")
plt.figure(figsize=(3, 3))
visual.orientation_3D_colorwheel(
    wheelsize=128,
    circ_size=50,
    interp_belt=20 / 180 * np.pi,
    sat_factor=1,
    discretize=True,
)
plt.show()

#### Angular histogram of 3D orientation
visual.orientation_3D_hist(
    azimuth.flatten(),
    inclination.flatten(),
    np.abs(target).flatten(),
    bins=36,
    num_col=1,
    size=10,
    contour_level=100,
    hist_cmap="gnuplot2",
    top_hemi=True,
)
plt.show()


## Compute permittivity and scattering potential tensors
### assign the permittivity tensor according to the sample properties

epsilon_mean = (ne_map**2 + no_map**2) / 2
epsilon_del = (ne_map**2 - no_map**2) / 2

epsilon_tensor = np.zeros((3, 3, N, M, L))
epsilon_tensor[0, 0] = epsilon_mean - epsilon_del * (
    np.cos(inclination) ** 2 - np.sin(inclination) ** 2 * np.cos(2 * azimuth)
)
epsilon_tensor[0, 1] = (
    epsilon_del * np.sin(inclination) ** 2 * np.sin(2 * azimuth)
)
epsilon_tensor[0, 2] = epsilon_del * np.sin(2 * inclination) * np.cos(azimuth)

epsilon_tensor[1, 0] = (
    epsilon_del * np.sin(inclination) ** 2 * np.sin(2 * azimuth)
)
epsilon_tensor[1, 1] = epsilon_mean - epsilon_del * (
    np.cos(inclination) ** 2 + np.sin(inclination) ** 2 * np.cos(2 * azimuth)
)
epsilon_tensor[1, 2] = epsilon_del * np.sin(2 * inclination) * np.sin(azimuth)

epsilon_tensor[2, 0] = epsilon_del * np.sin(2 * inclination) * np.cos(azimuth)
epsilon_tensor[2, 1] = epsilon_del * np.sin(2 * inclination) * np.sin(azimuth)
epsilon_tensor[2, 2] = epsilon_mean + epsilon_del * np.cos(2 * inclination)


visual.plot_multicolumn(
    [
        epsilon_tensor[0, 0, :, :, z_layer],
        epsilon_tensor[0, 1, :, :, z_layer],
        epsilon_tensor[0, 2, :, :, z_layer],
        epsilon_tensor[1, 0, :, :, z_layer],
        epsilon_tensor[1, 1, :, :, z_layer],
        epsilon_tensor[1, 2, :, :, z_layer],
        epsilon_tensor[2, 0, :, :, z_layer],
        epsilon_tensor[2, 1, :, :, z_layer],
        epsilon_tensor[2, 2, :, :, z_layer],
    ],
    origin="lower",
    num_col=3,
    titles=[
        r"$\epsilon_{xx}$",
        r"$\epsilon_{xy}$",
        r"$\epsilon_{xz}$",
        r"$\epsilon_{yx}$",
        r"$\epsilon_{yy}$",
        r"$\epsilon_{yz}$",
        r"$\epsilon_{zx}$",
        r"$\epsilon_{zy}$",
        r"$\epsilon_{zz}$",
    ],
    size=5,
    set_title=True,
)
plt.show()
#####################################################################
# Compute components of scattering potential tensor                 #
# ($f_{0r}, f_{0i}, f_{1c}, f_{1s}, f_{2c},f_{2s}, f_{3}$)          #
#####################################################################

# the corresponding components of the scattering potential tensor

del_f_component = np.zeros((7, N, M, L))
del_f_component[0] = np.real(
    ((2 * np.pi / lambda_illu) ** 2)
    * (n_media**2 - epsilon_mean + epsilon_del * np.cos(inclination) ** 2)
)
del_f_component[1] = np.imag(
    ((2 * np.pi / lambda_illu) ** 2)
    * (n_media**2 - epsilon_mean + epsilon_del * np.cos(inclination) ** 2)
)
del_f_component[2] = (
    -((2 * np.pi / lambda_illu) ** 2)
    * epsilon_del
    * np.sin(inclination) ** 2
    * np.cos(2 * azimuth)
)
del_f_component[3] = (
    -((2 * np.pi / lambda_illu) ** 2)
    * epsilon_del
    * np.sin(inclination) ** 2
    * np.sin(2 * azimuth)
)
del_f_component[4] = (
    -((2 * np.pi / lambda_illu) ** 2)
    * epsilon_del
    * np.sin(2 * inclination)
    * np.cos(azimuth)
)
del_f_component[5] = (
    -((2 * np.pi / lambda_illu) ** 2)
    * epsilon_del
    * np.sin(2 * inclination)
    * np.sin(azimuth)
)
del_f_component[6] = (
    ((2 * np.pi / lambda_illu) ** 2)
    * epsilon_del
    * (np.sin(inclination) ** 2 - 2 * np.cos(inclination) ** 2)
)


visual.plot_multicolumn(
    [
        del_f_component[0, :, :, z_layer],
        del_f_component[1, :, :, z_layer],
        del_f_component[2, :, :, z_layer],
        del_f_component[3, :, :, z_layer],
        del_f_component[4, :, :, z_layer],
        del_f_component[5, :, :, z_layer],
        del_f_component[6, :, :, z_layer],
    ],
    origin="lower",
    num_col=4,
    titles=[
        r"$f_{0r}$",
        r"$f_{0i}$",
        r"$f_{1c}$",
        r"$f_{1s}$",
        r"$f_{2c}$",
        r"$f_{2s}$",
        r"$f_{3}$",
    ],
    size=5,
    set_title=True,
)
plt.show()

#####################################################################
# Forward model of uPTI (polarization-diverse, illumination-diverse,#
# depth-diverse acquisition)                                        #
#####################################################################


# DPC + BF illumination + PolState (sector illumination)

xx, yy, fxx, fyy = util.gen_coordinate((N, M), ps)
radial_frequencies = np.sqrt(fxx**2 + fyy**2)

Pupil_obj = optics.generate_pupil(
    radial_frequencies, NA_obj / n_media, lambda_illu / n_media
).numpy()
Source_support = optics.generate_pupil(
    radial_frequencies, NA_illu / n_media, lambda_illu / n_media
).numpy()

NAx_coord = lambda_illu / n_media * fxx
NAy_coord = lambda_illu / n_media * fyy

rotation_angle = [0, 45, 90, 135, 180, 225, 270, 315]

Source = np.zeros((len(rotation_angle) + 1, N, M))
Source_cont = np.zeros_like(Source)

Source_BF = optics.generate_pupil(
    radial_frequencies, NA_illu / n_media / 2, lambda_illu / n_media
).numpy()

Source_cont[-1] = Source_BF.copy()
Source[-1] = optics.Source_subsample(
    Source_BF, NAx_coord, NAy_coord, subsampled_NA=0.1 / n_media
)

for i in range(len(rotation_angle)):
    deg = rotation_angle[i]
    Source_temp = np.zeros((N, M))
    Source_temp2 = np.zeros((N, M))
    Source_temp[
        fyy * np.cos(np.deg2rad(deg - 22.5))
        - fxx * np.sin(np.deg2rad(deg - 22.5))
        > 1e-10
    ] = 1
    Source_temp2[
        fyy * np.cos(np.deg2rad(deg - 135 - 22.5))
        - fxx * np.sin(np.deg2rad(deg - 135 - 22.5))
        > 1e-10
    ] = 1

    Source_cont[i] = Source_temp * Source_temp2 * Source_support

    Source_discrete = optics.Source_subsample(
        Source_cont[i], NAx_coord, NAy_coord, subsampled_NA=0.1 / n_media
    )
    Source[i] = np.maximum(0, Source_discrete.copy())

Source_PolState = np.zeros((len(Source), 2), complex)

for i in range(len(Source)):
    Source_PolState[i, 0] = 1
    Source_PolState[i, 1] = 1j

#### Circularly polarized illumination patterns

visual.plot_multicolumn(
    fftshift(Source_cont, axes=(1, 2)), origin="lower", num_col=5, size=5
)
# discretized illumination patterns used in simulation (faster forward model)
visual.plot_multicolumn(
    fftshift(Source, axes=(1, 2)), origin="lower", num_col=5, size=5
)
print(Source_PolState)
plt.figure(figsize=(10, 10))
plt.imshow(fftshift(np.sum(Source, axis=0)), origin="lower")
plt.show()
print(np.sum(Source, axis=(1, 2)))

#### Initialize microscope simulator with above source pattern and uniform imaging pupil

## initiate the simulator
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
    Source=Source,
    Source_PolState=Source_PolState,
    use_gpu=use_gpu,
    gpu_id=gpu_id,
)

## Compute image volumes and Stokes volumes with vectorial simulation

# forward simulation done by vectorial SEAGLE implementation
# (when itr_max = 0, we use the first Born approximation)

(
    I_meas_SEAGLE,
    Stokes_SEAGLE,
) = simulator.simulate_3D_vectorial_measurements_SEAGLE(
    epsilon_tensor, itr_max=0, tolerance=1e-4, verbose=True
)


## situation with no noise ##
photon_count = 5000
I_meas_noise = I_meas_SEAGLE / np.mean(I_meas_SEAGLE) * photon_count
#############################


# # Add noise to the measurement ##
# photon_count = 50000
# ext_ratio    = 1000
# const_bg     = photon_count/(0.5*(1-np.cos(chi)))/ext_ratio
# I_meas_noise = (np.random.poisson(I_meas_SEAGLE/np.mean(I_meas_SEAGLE) * photon_count + const_bg)).astype('float64')
# #################################

# Save simulations

output_dir = "./"

if sample_type == "3D":
    output_file = "PTI_simulation_data_NA_det_147_NA_illu_140_3D_spoke_discrete_no_1528_ne_1553_no_noise_Born"
elif sample_type == "2D":
    output_file = "PTI_simulation_data_NA_det_147_NA_illu_140_2D_spoke_discrete_no_1528_ne_1553_no_noise_Born"
else:
    print("sample_type needs to be 2D or 3D.")

np.savez(
    output_dir + output_file,
    I_meas=I_meas_noise,
    lambda_illu=lambda_illu,
    n_media=n_media,
    NA_obj=NA_obj,
    NA_illu=NA_illu,
    ps=ps,
    psz=z_pixel_size,
    Source_cont=Source_cont,
    Source_PolState=Source_PolState,
    z_defocus=z_defocus,
    chi=chi,
)

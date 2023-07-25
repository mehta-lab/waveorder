import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from IPython import display
from scipy.ndimage import uniform_filter
from .util import *
from .optics import *
from .background_estimator import *


def intensity_mapping(img_stack):
    img_stack_out = np.zeros_like(img_stack)
    img_stack_out[0] = img_stack[0].copy()
    img_stack_out[1] = img_stack[4].copy()
    img_stack_out[2] = img_stack[3].copy()
    img_stack_out[3] = img_stack[1].copy()
    img_stack_out[4] = img_stack[2].copy()

    return img_stack_out


def instrument_matrix_and_source_calibration(I_cali_mean, handedness="RCP"):
    _, N_cali = I_cali_mean.shape

    # Source intensity
    I_tot = np.sum(I_cali_mean, axis=0)

    # Calibration matrix
    theta = np.r_[0:N_cali] / N_cali * 2 * np.pi
    C_matrix = np.array(
        [np.ones((N_cali,)), np.cos(2 * theta), np.sin(2 * theta)]
    )

    # offset calibration
    I_cali_norm = I_cali_mean / I_tot
    offset_est = np.transpose(
        np.linalg.pinv(C_matrix.transpose()).dot(
            np.transpose(I_cali_norm[0, :])
        )
    )
    alpha = np.arctan2(-offset_est[2], offset_est[1]) / 2

    # Source calibration
    C_matrix_offset = np.array(
        [
            np.ones((N_cali,)),
            np.cos(2 * (theta + alpha)),
            np.sin(2 * (theta + alpha)),
        ]
    )

    S_source = np.linalg.pinv(C_matrix_offset.transpose()).dot(
        I_tot[:, np.newaxis]
    )
    S_source_norm = S_source / S_source[0]

    Ax = np.sqrt((S_source_norm[0] + S_source_norm[1]) / 2)
    Ay = np.sqrt((S_source_norm[0] - S_source_norm[1]) / 2)
    del_phi = np.arccos(S_source_norm[2] / 2 / Ax / Ay)

    if handedness == "RCP":
        E_in = np.array([Ax, Ay * np.exp(1j * del_phi)])
    elif handedness == "LCP":
        E_in = np.array([Ax, Ay * np.exp(-1j * del_phi)])
    else:
        raise TypeError("handedness type must be 'LCP' or 'RCP'")

    # Instrument matrix calibration
    A_matrix = np.transpose(
        np.linalg.pinv(C_matrix_offset.transpose()).dot(
            np.transpose(I_cali_norm)
        )
    )

    theta_fine = np.r_[0:360] / 360 * 2 * np.pi
    C_matrix_offset_fine = np.array(
        [
            np.ones((360,)),
            np.cos(2 * (theta_fine + alpha)),
            np.sin(2 * (theta_fine + alpha)),
        ]
    )

    print("Calibrated source field:\n" + str(np.round(E_in, 4)))
    print("Calibrated instrument matrix:\n" + str(np.round(A_matrix, 4)))

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].plot(theta / np.pi * 180, np.transpose(I_cali_mean))
    ax[0, 0].legend(["$I_0$", "$I_{45}$", "$I_{90}$", "$I_{135}$"])
    ax[0, 0].set_title("Calibration curve without normalization")
    ax[0, 0].set_xlabel("Orientation of LP (deg)")
    ax[0, 0].set_ylabel("Raw intensity")

    ax[0, 1].plot(theta / np.pi * 180, I_tot)
    ax[0, 1].plot(
        theta_fine / np.pi * 180,
        np.transpose(C_matrix_offset_fine).dot(S_source),
    )
    ax[0, 1].legend(["Mean source intensity", "Fitted source intensity"])
    ax[0, 1].set_title("Source calibration curve")
    ax[0, 1].set_xlabel("Orientation of LP (deg)")
    ax[0, 1].set_ylabel("Mean intensity from 4 linear channels")

    ax[1, 0].plot(theta / np.pi * 180, np.transpose(I_cali_mean / I_tot))
    ax[1, 0].legend(["$I_0$", "$I_{45}$", "$I_{90}$", "$I_{135}$"])
    ax[1, 0].set_title("Normalized calibration curve")
    ax[1, 0].set_xlabel("Orientation of LP (deg)")
    ax[1, 0].set_ylabel("Normalized intensity")

    ax[1, 1].plot(theta / np.pi * 180, np.transpose(I_cali_norm))
    ax[1, 1].plot(
        theta_fine / np.pi * 180,
        np.transpose(A_matrix.dot(C_matrix_offset_fine)),
    )
    ax[1, 1].legend(["$I_0$", "$I_{45}$", "$I_{90}$", "$I_{135}$"])
    ax[1, 1].set_xlabel("Orientation of LP (deg)")
    ax[1, 1].set_ylabel("Normalized intensity")
    ax[1, 1].set_title("Fitted calibration curves")

    return E_in, A_matrix, np.transpose(A_matrix.dot(C_matrix_offset_fine))


def instrument_matrix_calibration(I_cali_norm, I_meas):
    _, N_cali = I_cali_norm.shape

    theta = np.r_[0:N_cali] / N_cali * 2 * np.pi
    S_matrix = np.array(
        [np.ones((N_cali,)), np.cos(2 * theta), np.sin(2 * theta)]
    )
    A_matrix = np.transpose(
        np.linalg.pinv(S_matrix.transpose()).dot(np.transpose(I_cali_norm))
    )

    if I_meas.ndim == 3:
        I_mean = np.mean(I_meas, axis=(1, 2))
    elif I_meas.ndim == 4:
        I_mean = np.mean(I_meas, axis=(1, 2, 3))

    I_tot = np.sum(I_mean)
    A_matrix_S3 = I_mean / I_tot - A_matrix[:, 0]
    I_corr = (I_tot / 4) * (A_matrix_S3) / np.mean(A_matrix[:, 0])

    print("Calibrated instrument matrix:\n" + str(np.round(A_matrix, 4)))
    print(
        "Last column of instrument matrix:\n"
        + str(np.round(A_matrix_S3.reshape((4, 1)), 4))
    )

    plt.plot(np.transpose(I_cali_norm))
    plt.plot(np.transpose(A_matrix.dot(S_matrix)))
    plt.xlabel("Orientation of LP (deg)")
    plt.ylabel("Normalized intensity")
    plt.title("Fitted calibration curves")
    plt.legend(["$I_0$", "$I_{45}$", "$I_{90}$", "$I_{135}$"])

    return A_matrix, I_corr


class waveorder_microscopy:

    """

       waveorder_microscopy contains reconstruction algorithms for label-free
       microscopy with various types of dataset:

       1) 2D/3D phase reconstruction with a single brightfield defocused stack
          (Transport of intensity equation, TIE)

       2) 2D/3D phase reconstruction with intensities of asymmetric illumination
          (differential phase contrast, DPC)

       3) 2D/3D joint phase and polarization (2D orientation) reconstruction
          with brightfield-illuminated polarization-sensitive intensities (QLIPP)

       4) 2D/3D joint phase and polarization (uniaxial permittivity tensor) reconstruction
          with asymmetrically-illuminated polarization-sensitive intensities (PTI)

       The structure of waveorder_microscopy class:

       waveorder_microscopy class is structured like an actual microscope setup shown below.
       In experiment, we need to make the following choices:
       1) the microscope objective (detection system),
       2) the microscope condenser (illumination system)
       3) whether to take defocused stack or not
       4) what polarization sensitive devices are in use

       In order for the waveorder_microscopy to model these choices properly, its constructor
       takes relevant parameters to setup individual components of a virtual microscope in the following:

                 /////////////////////
                 /////////////////////              #####################################################
                 /////////////////////              # Illumination system setup:                        #
                 /////////////////////              #                                                   #
    LCD   #################*,,,,,,,,*########       # illu_mode  ('BF', 'PH', 'Arbitrary')              #
                           //////////               # NA_illu    (if under 'BF')                        #
    RCP   ,,,,,,,,,,,,,,,,,**********,,,,,,,,       # NA_illu_in (the inner radius of the ring in 'PH') #
                          ,///////////              # Source     (if under 'Arbitrary')                 #
                          ,///////////      --------# Source_PolState (if under 'Arbitrary')            #
               @@@&%#(/**//(##%%&&&@@@@@            #                                                   #
               @@@&%#(/**//(##%%&&&@@@@@            #####################################################
    condenser  @@@&%#(/**//(##%%&&&@@@@@
               @@@&%#(/**//(##%%&&&@@@@@            #####################################################
                @@&%#(/**//(##%%&&&@@@@             # Defocus kernel initialization:                    #
                  @%#(/**//(##%%&&&@@               #                                                   #
                           ///              --------# z_defocus (an array of z positions)               #
                           //              /        # pad_z     (z-padding to avoid periodic artifacts) #
    sample @@@@@@@@@@@@@@@//@@@@@@@@@@@@@@/         #                                                   #
                         ///                        #####################################################
                       ////
                  @%#(/**//(##%%&&&@@               #######################################################
                @@&%#(/**//(##%%&&&@@@@             # Detection system setup:                             #
               @@@&%#(/**//(##%%&&&@@@@@            #                                                     #
    objective  @@@&%#(/**//(##%%&&&@@@@@            # img_dim     (image dimension)                       #
               @@@&%#(/**//(##%%&&&@@@@@    --------# lambda_illu (illumination wavelength)               #
               @@@&%#(/**//(##%%&&&@@@@@            # ps          (effective pixel size, camera ps / mag) #
                  //////////,                       # NA_obj      (objective NA)                          #
                  //////////,                       # n_media     (refractive index of immersion media)   #
                  //////////...                     #                                                     #
             .....,,,,,,,,,,.............           #######################################################
        ...........,,,,,,,,,...................
             .......,,,,,,,,.............
                      //////                        ###########################################################
                       /////                        # Polarization detection (or illumination) setup:         #
                  *******///**********              #                                                         #
                  ********************      --------# chi (swing of the LC, assuming 5-state acquisition)     #
              ############################          # A_matrix (instrument matrix of the polarization system) #
    camera    ############################          #                                                         #
              ############################          ###########################################################

       After setting up the microscope parameters, the following transfer functions
       are computed if the flags are turned on:

       1) Phase transfer functions (phase_deconv: None, '2D' or '3D'):
          enable deconvolution supported by Phase_recon and Phase_recon_3D

       2) Polarization transfer functions (bire_in_plane_deconv: None, '2D' or '3D'):
          enable deconvolution supported by Birefringence_recon_2D and Birefringence_recon_3D

       3) PTI transfer functions (inc_recon: None, '2D-vec-WOTF' or '3D'):
          enable deconvolution supported by scattering_potential_tensor_recon_2D_vec,
          scattering_potential_tensor_to_3D_orientation and scattering_potential_tensor_recon_3D_vec.
          Currently, the model assumes uniaxial symmetry of the permittivity tensor.

       Parameters
       ----------
           img_dim              : tuple
                                  shape of the computed 2D space with size of (N, M)

           lambda_illu          : float
                                  wavelength of the incident light

           ps                   : float
                                  xy pixel size of the image space

           NA_obj               : float
                                  numerical aperture of the detection objective

           NA_illu              : float
                                  numerical aperture of the illumination condenser

           z_defocus            : numpy.ndarray
                                  1D array of defocused z position corresponds to the intensity stack
                                  (matters for 2D reconstruction, the direction positive z matters for 3D reconstruction)

           chi                  : float
                                  swing of the illumination or detection polarization state (in radian)

           n_media              : float
                                  refractive index of the immersing media

           cali                 : bool
                                  'True' for the orientation convention of QLIPP data,
                                  'False' for the orientation convention of PTI data

           bg_option            : str
                                  'local' for estimating background with scipy uniform filter
                                  'local_fit' for estimating background with polynomial fit
                                  other string for normal background subtraction with the provided background

           A_matrix             : numpy.ndarray
                                  self-provided instrument matrix converting polarization-sensitive intensity images into Stokes parameters
                                  with shape of (N_channel, N_Stokes)
                                  If None is provided, the instrument matrix is determined by the QLIPP convention with swing specify by chi


          QLIPP_birefringence_only : bool
                                  'True' to skip pre-processing functions for phase/PTI reconstruction
                                  'False' to continue with pre-processing functions for phase/PTI reconstruction

           bire_in_plane_deconv : str
                                  string contains the dimension of 2D birefringence deconvolution
                                  '2D' for 2D deconvolution of 2D birefringence
                                  '3D' for 3D deconvolution of 2D birefringence

           inc_recon            : str
                                  option for constructing settings for 3D orientation reconstruction
                                  '2D-vec-WOTF' for 2D diffractive reconstruction of 3D anisotropy
                                  '3D' for 3D for diffractive reconstruction of 3D anisotropy

           phase_deconv         : str
                                  string contains the phase reconstruction dimension
                                  '2D' for 2D phase deconvolution
                                  '3D' for 3D phase deconvolution

           ph_deconv_layer      : int
                                  number of layers included for each layer of semi-3D phase reconstruction

           illu_mode            : str
                                  string to set the pattern of illumination source
                                  'BF' for brightfield illumination with source pattern specified by NA_illu
                                  'PH' for phase contrast illumination with the source pattern specify by NA_illu and NA_illu_in
                                  'Arbitrary' for self-defined source pattern of dimension (N_pattern, N, M)

           NA_illu_in           : flaot
                                  numerical aperture of the inner circle for phase contrast ring illumination

           Source               : numpy.ndarray
                                  illumination source pattern with dimension of (N_pattern, N, M)

           Source_PolState      : numpy.ndarray
                                  illumination polarization states (Ex, Ey) for each illumination pattern with dimension of (N_pattern, 2)
                                  If provided with size of (2,), a single state is used for all illumination patterns

           pad_z                : int
                                  number of z-layers to pad (reflection boundary condition) for 3D deconvolution

           use_gpu              : bool
                                  option to use gpu or not

           gpu_id               : int
                                  number refering to which gpu will be used


    """

    def __init__(
        self,
        img_dim,
        lambda_illu,
        ps,
        NA_obj,
        NA_illu,
        z_defocus,
        chi=None,
        n_media=1,
        cali=False,
        bg_option="global",
        A_matrix=None,
        QLIPP_birefringence_only=False,
        bire_in_plane_deconv=None,
        inc_recon=None,
        phase_deconv=None,
        ph_deconv_layer=5,
        illu_mode="BF",
        NA_illu_in=None,
        Source=None,
        Source_PolState=np.array([1, 1j]),
        pad_z=0,
        use_gpu=False,
        gpu_id=0,
    ):
        """

        initialize the system parameters for phase and orders microscopy

        """

        t0 = time.time()

        # GPU/CPU

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self._A_matrix_inv_gpu_array = None

        if self.use_gpu:
            globals()["cp"] = __import__("cupy")
            cp.cuda.Device(self.gpu_id).use()

        # Basic parameter
        self.N, self.M = img_dim
        self.n_media = n_media
        self.lambda_illu = lambda_illu / n_media
        self.ps = ps
        self.z_defocus = z_defocus.copy()
        if len(z_defocus) >= 2:
            self.psz = np.abs(z_defocus[0] - z_defocus[1])
            self.G_tensor_z_upsampling = np.ceil(
                self.psz / (self.lambda_illu / 2)
            )
        self.pad_z = pad_z
        self.NA_obj = NA_obj / n_media
        self.NA_illu = NA_illu / n_media
        self.N_defocus = len(z_defocus)
        self.N_defocus_3D = self.N_defocus + 2 * self.pad_z
        self.chi = chi
        self.cali = cali
        self.bg_option = bg_option
        self.phase_deconv = phase_deconv

        if QLIPP_birefringence_only == False:
            # setup microscocpe variables
            self.xx, self.yy, self.fxx, self.fyy = gen_coordinate(
                (self.N, self.M), ps
            )
            self.frr = np.sqrt(self.fxx**2 + self.fyy**2)
            self.Pupil_obj = generate_pupil(
                self.frr, self.NA_obj, self.lambda_illu
            ).numpy()
            self.Pupil_support = self.Pupil_obj.copy()

            # illumination setup

            self.illumination_setup(
                illu_mode, NA_illu_in, Source, Source_PolState
            )

            # Defocus kernel initialization

            self.Hz_det_setup(
                self.phase_deconv,
                ph_deconv_layer,
                bire_in_plane_deconv,
                inc_recon,
            )

            # select either 2D or 3D model for phase deconvolution

            self.phase_deconv_setup(self.phase_deconv)

            # instrument matrix for polarization detection

            self.instrument_matrix_setup(A_matrix)

            # select either 2D or 3D model for 2D birefringence deconvolution

            self.bire_in_plane_deconv_setup(bire_in_plane_deconv)

            # inclination reconstruction model selection

            self.inclination_recon_setup(inc_recon)

        else:
            # instrument matrix for polarization detection
            self.instrument_matrix_setup(A_matrix)

    ##############   constructor function group   ##############

    def illumination_setup(
        self, illu_mode, NA_illu_in, Source, Source_PolState
    ):
        """

        setup illumination source function for transfer function computing

        Parameters
        ----------
            illu_mode       : str
                              string to set the pattern of illumination source
                              'BF' for brightfield illumination with source pattern specified by NA_illu
                              'PH' for phase contrast illumination with the source pattern specify by NA_illu and NA_illu_in
                              'Arbitrary' for self-defined source pattern of dimension (N_pattern, N, M)

            NA_illu_in      : flaot
                              numerical aperture of the inner circle for phase contrast ring illumination

            Source          : numpy.ndarray
                              illumination source pattern with dimension of (N_pattern, N, M)

            Source_PolState : numpy.ndarray
                              illumination polarization states (Ex, Ey) for each illumination pattern with dimension of (N_pattern, 2)


        """

        if illu_mode == "BF":
            self.Source = generate_pupil(
                self.frr, self.NA_illu, self.lambda_illu
            )
            self.N_pattern = 1

        elif illu_mode == "PH":
            if NA_illu_in == None:
                raise ("No inner rim NA specified in the PH illumination mode")
            else:
                self.NA_illu_in = NA_illu_in / self.n_media
                inner_pupil = gen_Pupil(
                    self.fxx,
                    self.fyy,
                    self.NA_illu_in / self.n_media,
                    self.lambda_illu,
                )
                self.Source = gen_Pupil(
                    self.fxx, self.fyy, self.NA_illu, self.lambda_illu
                )
                self.Source -= inner_pupil

                Pupil_ring_out = gen_Pupil(
                    self.fxx,
                    self.fyy,
                    self.NA_illu / self.n_media,
                    self.lambda_illu,
                )
                Pupil_ring_in = gen_Pupil(
                    self.fxx,
                    self.fyy,
                    self.NA_illu_in / self.n_media,
                    self.lambda_illu,
                )

                self.Pupil_obj = self.Pupil_obj * np.exp(
                    (Pupil_ring_out - Pupil_ring_in)
                    * (np.log(0.7) - 1j * (np.pi / 2 - 0.0 * np.pi))
                )
                self.N_pattern = 1

        elif illu_mode == "Arbitrary":
            self.Source = Source.copy()
            if Source.ndim == 2:
                self.N_pattern = 1
            else:
                self.N_pattern = len(Source)

        self.Source_PolState = np.zeros((self.N_pattern, 2), complex)

        if Source_PolState.ndim == 1:
            for i in range(self.N_pattern):
                self.Source_PolState[i] = Source_PolState / (
                    np.sum(np.abs(Source_PolState) ** 2)
                ) ** (1 / 2)
        else:
            if len(Source_PolState) != self.N_pattern:
                raise (
                    "The length of Source_PolState needs to be either 1 or the same as N_pattern"
                )
            for i in range(self.N_pattern):
                self.Source_PolState[i] = Source_PolState[i] / (
                    np.sum(np.abs(Source_PolState[i]) ** 2)
                ) ** (1 / 2)

    def Hz_det_setup(
        self, phase_deconv, ph_deconv_layer, bire_in_plane_deconv, inc_recon
    ):
        """

        setup defocus kernels for deconvolution with the corresponding dimensions

        Parameters
        ----------
            phase_deconv         : str
                                   string contains the dimension of the phase reconstruction
                                   '2D' for 2D phase deconvolution
                                   '3D' for 3D phase deconvolution

            ph_deconv_layer      : int
                                   number of layers included for each layer of semi-3D phase reconstruction

            bire_in_plane_deconv : str
                                   string contains the dimension of 2D birefringence deconvolution
                                   '2D' for 2D deconvolution of 2D birefringence
                                   '3D' for 3D deconvolution of 2D birefringence

            inc_recon            : str
                                   option for constructing settings for 3D orientation reconstruction
                                   '2D-geometric' for 2D non-diffractive reconstruction of 3D anisotropy
                                   '2D-vec-WOTF' for 2D diffractive reconstruction of 3D anisotropy
                                   '3D' for 3D for diffractive reconstruction of 3D anisotropy

        """

        if (
            phase_deconv == "2D"
            or bire_in_plane_deconv == "2D"
            or inc_recon == "2D-vec-WOTF"
        ):
            # generate defocus kernel based on Pupil function and z_defocus
            self.Hz_det_2D = (
                generate_propagation_kernel(
                    torch.tensor(self.frr),
                    torch.tensor(self.Pupil_support),
                    self.lambda_illu,
                    torch.tensor(self.z_defocus),
                )
                .numpy()
                .transpose((1, 2, 0))
            )

        if phase_deconv == "semi-3D":
            self.ph_deconv_layer = ph_deconv_layer

            if self.z_defocus[0] - self.z_defocus[1] > 0:
                z_deconv = (
                    -(
                        np.r_[: self.ph_deconv_layer]
                        - self.ph_deconv_layer // 2
                    )
                    * self.psz
                )
            else:
                z_deconv = (
                    np.r_[: self.ph_deconv_layer] - self.ph_deconv_layer // 2
                ) * self.psz

            self.Hz_det_semi_3D = generate_propagation_kernel(
                self.fxx,
                self.fyy,
                self.Pupil_support,
                self.lambda_illu,
                z_deconv,
            )
            self.G_fun_z_semi_3D = generate_greens_function_z(
                self.fxx,
                self.fyy,
                self.Pupil_support,
                self.lambda_illu,
                z_deconv,
            )

        if (
            phase_deconv == "3D"
            or bire_in_plane_deconv == "3D"
            or inc_recon == "3D"
        ):
            # generate defocus kernel and Green's function
            if self.z_defocus[0] - self.z_defocus[1] > 0:
                z = -ifftshift(
                    (np.r_[0 : self.N_defocus_3D] - self.N_defocus_3D // 2)
                    * self.psz
                )
            else:
                z = ifftshift(
                    (np.r_[0 : self.N_defocus_3D] - self.N_defocus_3D // 2)
                    * self.psz
                )
            self.Hz_det_3D = (
                generate_propagation_kernel(
                    torch.tensor(self.frr),
                    torch.tensor(self.Pupil_support),
                    self.lambda_illu,
                    torch.tensor(z),
                )
                .numpy()
                .transpose((1, 2, 0))
            )
            self.G_fun_z_3D = (
                generate_greens_function_z(
                    torch.tensor(self.frr),
                    torch.tensor(self.Pupil_support),
                    self.lambda_illu,
                    torch.tensor(z),
                )
                .numpy()
                .transpose((1, 2, 0))
            )

    def phase_deconv_setup(self, phase_deconv):
        """

        setup transfer functions for phase deconvolution with the corresponding dimensions

        Parameters
        ----------
            phase_deconv    : str
                              string contains the dimension of the phase reconstruction
                              '2D' for 2D phase deconvolution
                              '3D' for 3D phase deconvolution

            ph_deconv_layer : int
                              number of layers included for each layer of semi-3D phase reconstruction

        """

        if phase_deconv == "2D":
            # compute 2D phase transfer function
            self.gen_WOTF()

        elif phase_deconv == "semi-3D":
            self.gen_semi_3D_WOTF()

        elif phase_deconv == "3D":
            # compute 3D phase transfer function
            self.gen_3D_WOTF()

    def bire_in_plane_deconv_setup(self, bire_in_plane_deconv):
        """

        setup transfer functions for 2D birefringence deconvolution with the corresponding dimensions

        Parameters
        ----------

            bire_in_plane_deconv : str
                                   string contains the dimension of 2D birefringence deconvolution
                                   '2D' for 2D deconvolution of 2D birefringence
                                   '3D' for 3D deconvolution of 2D birefringence

        """

        if bire_in_plane_deconv == "2D":
            # generate 2D vectorial transfer function for 2D birefringence deconvolution in 2D space
            self.gen_2D_vec_WOTF(False)

        elif bire_in_plane_deconv == "3D":
            # generate 3D vectorial transfer function for 2D birefringence deconvolution in 3D space
            self.gen_3D_vec_WOTF(False)

    def inclination_recon_setup(self, inc_recon):
        """

        setup transfer functions for PTI reconstruction

        Parameters
        ----------
            phase_deconv : str
                           string contains the phase reconstruction dimension
                           '2D' for 2D phase deconvolution
                           '3D' for 3D phase deconvolution

            inc_recon    : str
                           option for constructing settings for 3D orientation reconstruction
                           '2D-geometric' for 2D non-diffractive reconstruction of 3D anisotropy
                           '2D-vec-WOTF' for 2D diffractive reconstruction of 3D anisotropy
                           '3D' for 3D for diffractive reconstruction of 3D anisotropy

        """

        if inc_recon is not None and inc_recon != "3D":
            if inc_recon == "2D-geometric":
                wave_vec_norm_x = self.lambda_illu * self.fxx
                wave_vec_norm_y = self.lambda_illu * self.fyy
                wave_vec_norm_z = (
                    np.maximum(
                        0, 1 - wave_vec_norm_x**2 - wave_vec_norm_y**2
                    )
                ) ** (0.5)

                incident_theta = np.arctan2(
                    (wave_vec_norm_x**2 + wave_vec_norm_y**2) ** (0.5),
                    wave_vec_norm_z,
                )
                incident_phi = np.arctan2(wave_vec_norm_y, wave_vec_norm_x)

                (
                    self.geometric_inc_matrix,
                    self.geometric_inc_matrix_inv,
                ) = gen_geometric_inc_matrix(
                    incident_theta, incident_phi, self.Source
                )

            elif inc_recon == "2D-vec-WOTF":
                # generate 2D vectorial transfer function for 2D PTI
                self.gen_2D_vec_WOTF(True)

                # compute the AHA matrix for later 2D inversion
                self.inc_AHA_2D_vec = np.zeros((7, 7, self.N, self.M), complex)
                for i, j, p in itertools.product(
                    range(7), range(7), range(self.N_Stokes)
                ):
                    self.inc_AHA_2D_vec[i, j] += np.sum(
                        np.conj(self.H_dyadic_2D_OTF[p, i])
                        * self.H_dyadic_2D_OTF[p, j],
                        axis=2,
                    )

        elif inc_recon == "3D":
            # generate 3D vectorial transfer function for 3D PTI
            self.gen_3D_vec_WOTF(True)
            self.inc_AHA_3D_vec = np.zeros(
                (7, 7, self.N, self.M, self.N_defocus_3D), dtype="complex64"
            )

            # compute the AHA matrix for later 3D inversion
            for i, j, p in itertools.product(
                range(7), range(7), range(self.N_Stokes)
            ):
                self.inc_AHA_3D_vec[i, j] += np.sum(
                    np.conj(self.H_dyadic_OTF[p, i]) * self.H_dyadic_OTF[p, j],
                    axis=0,
                )

    def instrument_matrix_setup(self, A_matrix):
        """

        setup instrument matrix

        Parameters
        ----------
            A_matrix : numpy.ndarray
                       self-provided instrument matrix converting polarization-sensitive intensity images into Stokes parameters
                       with shape of (N_channel, N_Stokes) or (size_X, size_Y, N_channel, N_Stokes)
                       If None is provided, the instrument matrix is determined by the QLIPP convention with swing specify by chi


        """

        if A_matrix is None:
            self.N_channel = 5
            self.N_Stokes = 4
            self.A_matrix = 0.5 * np.array(
                [
                    [1, 0, 0, -1],
                    [1, np.sin(self.chi), 0, -np.cos(self.chi)],
                    [1, 0, np.sin(self.chi), -np.cos(self.chi)],
                    [1, -np.sin(self.chi), 0, -np.cos(self.chi)],
                    [1, 0, -np.sin(self.chi), -np.cos(self.chi)],
                ]
            )
        else:
            A_matrix_shape = A_matrix.shape
            if len(A_matrix_shape) not in (2, 4):
                raise ValueError(
                    "Instrument matrix must have shape (N_channel, N_Stokes) or (N, M, N_channel, N_Stokes)"
                )
            if len(A_matrix_shape) == 4 and A_matrix_shape[:2] != (
                self.N,
                self.M,
            ):
                raise ValueError(
                    "Instrument tensor must have shape (N, M, N_channel, N_Stokes)"
                )

            self.N_channel = A_matrix_shape[-2]
            self.N_Stokes = A_matrix_shape[-1]
            self.A_matrix = A_matrix.copy()

        self.A_matrix_inv = np.linalg.pinv(self.A_matrix)

    ##############   constructor asisting function group   ##############

    def gen_WOTF(self):
        """

        generate 2D phase transfer functions


        """

        self.Hu = np.zeros(
            (self.N, self.M, self.N_defocus * self.N_pattern), complex
        )
        self.Hp = np.zeros(
            (self.N, self.M, self.N_defocus * self.N_pattern), complex
        )

        if self.N_pattern == 1:
            for i in range(self.N_defocus):
                Hu_temp, Hp_temp = compute_weak_object_transfer_function_2d(
                    torch.tensor(self.Source),
                    torch.tensor(self.Pupil_obj * self.Hz_det_2D[:, :, i]),
                )
                self.Hu[:, :, i] = Hu_temp.numpy()
                self.Hp[:, :, i] = Hp_temp.numpy()
        else:
            for i, j in itertools.product(
                range(self.N_defocus), range(self.N_pattern)
            ):
                idx = i * self.N_pattern + j
                Hu_temp, Hp_temp = compute_weak_object_transfer_function_2d(
                    torch.tensor(self.Source[j]),
                    torch.tensor(self.Pupil_obj * self.Hz_det_2D[idx, :, :]),
                )
                self.Hu[:, :, idx] = Hu_temp.numpy()
                self.Hp[:, :, idx] = Hp_temp.numpy()

    def gen_semi_3D_WOTF(self):
        """

        generate semi-3D phase transfer functions


        """

        self.Hu = np.zeros(
            (self.N, self.M, self.ph_deconv_layer * self.N_pattern), complex
        )
        self.Hp = np.zeros(
            (self.N, self.M, self.ph_deconv_layer * self.N_pattern), complex
        )

        for i, j in itertools.product(
            range(self.ph_deconv_layer), range(self.N_pattern)
        ):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[j].copy()

            idx = i * self.N_pattern + j
            self.Hu[:, :, idx], self.Hp[:, :, idx] = WOTF_semi_3D_compute(
                Source_current,
                Source_current,
                self.Pupil_obj,
                self.Hz_det_semi_3D[:, :, i],
                self.G_fun_z_semi_3D[:, :, i]
                * 4
                * np.pi
                * 1j
                / self.lambda_illu,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )

    def gen_3D_WOTF(self):
        """

        generate 3D phase transfer functions


        """

        self.H_re = np.zeros(
            (self.N_pattern, self.N, self.M, self.N_defocus_3D),
            dtype="complex64",
        )
        self.H_im = np.zeros(
            (self.N_pattern, self.N, self.M, self.N_defocus_3D),
            dtype="complex64",
        )

        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[i].copy()
            self.H_re[i], self.H_im[i] = WOTF_3D_compute(
                Source_current.astype("float32"),
                Source_current.astype("float32"),
                self.Pupil_obj.astype("complex64"),
                self.Hz_det_3D.astype("complex64"),
                self.G_fun_z_3D.astype("complex64"),
                self.psz,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )

        self.H_re = np.squeeze(self.H_re)
        self.H_im = np.squeeze(self.H_im)

    def gen_2D_vec_WOTF(self, inc_option=False):
        """

        generate 2D vectorial transfer functions for 2D PTI

         Parameters
        ----------
        inc_option  :   boolean
                        generate OTFs for multiple patterns of illumination used for reconstructing illumination.
        """

        if inc_option == True:
            self.H_dyadic_2D_OTF = np.zeros(
                (
                    self.N_Stokes,
                    7,
                    self.N,
                    self.M,
                    self.N_defocus * self.N_pattern,
                ),
                dtype="complex64",
            )
        else:
            self.H_dyadic_2D_OTF_in_plane = np.zeros(
                (2, 2, self.N, self.M, self.N_defocus * self.N_pattern),
                dtype="complex64",
            )

        # angle-dependent electric field components due to focusing effect
        fr = (self.fxx**2 + self.fyy**2) ** (0.5)
        cos_factor = (
            1 - (self.lambda_illu**2) * (fr**2) * self.Pupil_support
        ) ** (0.5) * self.Pupil_support
        dc_idx = fr == 0
        nondc_idx = fr != 0
        E_field_factor = np.zeros((5, self.N, self.M))

        E_field_factor[0, nondc_idx] = (
            (self.fxx[nondc_idx] ** 2) * cos_factor[nondc_idx]
            + self.fyy[nondc_idx] ** 2
        ) / fr[nondc_idx] ** 2
        E_field_factor[0, dc_idx] = 1
        E_field_factor[1, nondc_idx] = (
            self.fxx[nondc_idx]
            * self.fyy[nondc_idx]
            * (cos_factor[nondc_idx] - 1)
        ) / fr[nondc_idx] ** 2
        E_field_factor[2, nondc_idx] = (
            (self.fyy[nondc_idx] ** 2) * cos_factor[nondc_idx]
            + self.fxx[nondc_idx] ** 2
        ) / fr[nondc_idx] ** 2
        E_field_factor[2, dc_idx] = 1
        E_field_factor[3, nondc_idx] = -self.lambda_illu * self.fxx[nondc_idx]
        E_field_factor[4, nondc_idx] = -self.lambda_illu * self.fyy[nondc_idx]

        # generate dyadic Green's tensor
        G_fun_z = (
            generate_greens_function_z(
                torch.tensor(self.frr),
                torch.tensor(self.Pupil_support),
                self.lambda_illu,
                torch.tensor(self.z_defocus),
            )
            .numpy()
            .transpose((1, 2, 0))
        )
        G_tensor_z = gen_dyadic_Greens_tensor_z(
            self.fxx, self.fyy, G_fun_z, self.Pupil_support, self.lambda_illu
        )

        # compute transfer functions
        OTF_compute = lambda x, y, z, w: WOTF_semi_3D_compute(
            x,
            y,
            self.Pupil_obj,
            w,
            z,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
        )

        for i, j in itertools.product(
            range(self.N_defocus), range(self.N_pattern)
        ):
            if self.N_pattern == 1:
                Source_current = self.Source.numpy().copy()
            else:
                Source_current = self.Source[j].copy()

            idx = i * self.N_pattern + j

            # focusing electric field components
            Ex_field = (
                self.Source_PolState[j, 0] * E_field_factor[0]
                + self.Source_PolState[j, 1] * E_field_factor[1]
            )
            Ey_field = (
                self.Source_PolState[j, 0] * E_field_factor[1]
                + self.Source_PolState[j, 1] * E_field_factor[2]
            )
            Ez_field = (
                self.Source_PolState[j, 0] * E_field_factor[3]
                + self.Source_PolState[j, 1] * E_field_factor[4]
            )

            IF_ExEx = np.abs(Ex_field) ** 2
            IF_ExEy = Ex_field * np.conj(Ey_field)
            IF_ExEz = Ex_field * np.conj(Ez_field)
            IF_EyEy = np.abs(Ey_field) ** 2
            IF_EyEz = Ey_field * np.conj(Ez_field)

            Source_norm = Source_current * (IF_ExEx + IF_EyEy)

            # intermediate transfer functions
            ExEx_Gxx_re, ExEx_Gxx_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEx,
                G_tensor_z[0, 0, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            ExEy_Gxy_re, ExEy_Gxy_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy,
                G_tensor_z[0, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEx_Gyx_re, EyEx_Gyx_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy.conj(),
                G_tensor_z[0, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEy_Gyy_re, EyEy_Gyy_im = OTF_compute(
                Source_norm,
                Source_current * IF_EyEy,
                G_tensor_z[1, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            ExEx_Gxy_re, ExEx_Gxy_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEx,
                G_tensor_z[0, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            ExEy_Gxx_re, ExEy_Gxx_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy,
                G_tensor_z[0, 0, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEx_Gyy_re, EyEx_Gyy_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy.conj(),
                G_tensor_z[1, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEy_Gyx_re, EyEy_Gyx_im = OTF_compute(
                Source_norm,
                Source_current * IF_EyEy,
                G_tensor_z[0, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            ExEx_Gyy_re, ExEx_Gyy_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEx,
                G_tensor_z[1, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEy_Gxx_re, EyEy_Gxx_im = OTF_compute(
                Source_norm,
                Source_current * IF_EyEy,
                G_tensor_z[0, 0, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            EyEx_Gxx_re, EyEx_Gxx_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy.conj(),
                G_tensor_z[0, 0, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #
            ExEy_Gyy_re, ExEy_Gyy_im = OTF_compute(
                Source_norm,
                Source_current * IF_ExEy,
                G_tensor_z[1, 1, :, :, i],
                self.Hz_det_2D[:, :, i],
            )  #

            if inc_option == True:
                ExEz_Gxz_re, ExEz_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEz,
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEz_Gyz_re, EyEz_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEz,
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEx_Gxz_re, ExEx_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEx,
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEz_Gxx_re, ExEz_Gxx_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEz,
                    G_tensor_z[0, 0, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEx_Gyz_re, EyEx_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy.conj(),
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEz_Gyx_re, EyEz_Gyx_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEz,
                    G_tensor_z[0, 1, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEy_Gxz_re, ExEy_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy,
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEz_Gxy_re, ExEz_Gxy_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEz,
                    G_tensor_z[0, 1, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEy_Gyz_re, EyEy_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEy,
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEz_Gyy_re, EyEz_Gyy_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEz,
                    G_tensor_z[1, 1, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEz_Gyz_re, ExEz_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEz,
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEz_Gxz_re, EyEz_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEz,
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEx_Gxz_re, EyEx_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy.conj(),
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEz_Gxx_re, EyEz_Gxx_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEz,
                    G_tensor_z[0, 0, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEy_Gyz_re, ExEy_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy,
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEz_Gyy_re, ExEz_Gyy_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEz,
                    G_tensor_z[1, 1, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                EyEy_Gxz_re, EyEy_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_EyEy,
                    G_tensor_z[0, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )
                ExEx_Gyz_re, ExEx_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEx,
                    G_tensor_z[1, 2, :, :, i],
                    self.Hz_det_2D[:, :, i],
                )

                # 2D vectorial transfer functions
                self.H_dyadic_2D_OTF[0, 0, :, :, idx] = (
                    ExEx_Gxx_re
                    + ExEy_Gxy_re
                    + ExEz_Gxz_re
                    + EyEx_Gyx_re
                    + EyEy_Gyy_re
                    + EyEz_Gyz_re
                )
                self.H_dyadic_2D_OTF[0, 1, :, :, idx] = (
                    ExEx_Gxx_im
                    + ExEy_Gxy_im
                    + ExEz_Gxz_im
                    + EyEx_Gyx_im
                    + EyEy_Gyy_im
                    + EyEz_Gyz_im
                )
                self.H_dyadic_2D_OTF[0, 2, :, :, idx] = (
                    ExEx_Gxx_re - ExEy_Gxy_re + EyEx_Gyx_re - EyEy_Gyy_re
                )
                self.H_dyadic_2D_OTF[0, 3, :, :, idx] = (
                    ExEx_Gxy_re + ExEy_Gxx_re + EyEx_Gyy_re + EyEy_Gyx_re
                )
                self.H_dyadic_2D_OTF[0, 4, :, :, idx] = (
                    ExEx_Gxz_re + ExEz_Gxx_re + EyEx_Gyz_re + EyEz_Gyx_re
                )
                self.H_dyadic_2D_OTF[0, 5, :, :, idx] = (
                    ExEy_Gxz_re + ExEz_Gxy_re + EyEy_Gyz_re + EyEz_Gyy_re
                )
                self.H_dyadic_2D_OTF[0, 6, :, :, idx] = (
                    ExEz_Gxz_re + EyEz_Gyz_re
                )

                self.H_dyadic_2D_OTF[1, 0, :, :, idx] = (
                    ExEx_Gxx_re
                    + ExEy_Gxy_re
                    + ExEz_Gxz_re
                    - EyEx_Gyx_re
                    - EyEy_Gyy_re
                    - EyEz_Gyz_re
                )
                self.H_dyadic_2D_OTF[1, 1, :, :, idx] = (
                    ExEx_Gxx_im
                    + ExEy_Gxy_im
                    + ExEz_Gxz_im
                    - EyEx_Gyx_im
                    - EyEy_Gyy_im
                    - EyEz_Gyz_im
                )
                self.H_dyadic_2D_OTF[1, 2, :, :, idx] = (
                    ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
                )
                self.H_dyadic_2D_OTF[1, 3, :, :, idx] = (
                    ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
                )
                self.H_dyadic_2D_OTF[1, 4, :, :, idx] = (
                    ExEx_Gxz_re + ExEz_Gxx_re - EyEx_Gyz_re - EyEz_Gyx_re
                )
                self.H_dyadic_2D_OTF[1, 5, :, :, idx] = (
                    ExEy_Gxz_re + ExEz_Gxy_re - EyEy_Gyz_re - EyEz_Gyy_re
                )
                self.H_dyadic_2D_OTF[1, 6, :, :, idx] = (
                    ExEz_Gxz_re - EyEz_Gyz_re
                )

                self.H_dyadic_2D_OTF[2, 0, :, :, idx] = (
                    ExEx_Gxy_re
                    + ExEy_Gyy_re
                    + ExEz_Gyz_re
                    + EyEx_Gxx_re
                    + EyEy_Gyx_re
                    + EyEz_Gxz_re
                )
                self.H_dyadic_2D_OTF[2, 1, :, :, idx] = (
                    ExEx_Gxy_im
                    + ExEy_Gyy_im
                    + ExEz_Gyz_im
                    + EyEx_Gxx_im
                    + EyEy_Gyx_im
                    + EyEz_Gxz_im
                )
                self.H_dyadic_2D_OTF[2, 2, :, :, idx] = (
                    ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
                )
                self.H_dyadic_2D_OTF[2, 3, :, :, idx] = (
                    ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
                )
                self.H_dyadic_2D_OTF[2, 4, :, :, idx] = (
                    ExEx_Gyz_re + ExEz_Gxy_re + EyEx_Gxz_re + EyEz_Gxx_re
                )
                self.H_dyadic_2D_OTF[2, 5, :, :, idx] = (
                    ExEy_Gyz_re + ExEz_Gyy_re + EyEy_Gxz_re + EyEz_Gyx_re
                )
                self.H_dyadic_2D_OTF[2, 6, :, :, idx] = (
                    ExEz_Gyz_re + EyEz_Gxz_re
                )

                # transfer functions for S3
                if self.N_Stokes == 4:  # full Stokes polarimeter
                    self.H_dyadic_2D_OTF[3, 0, :, :, idx] = (
                        -ExEx_Gxy_im
                        - ExEy_Gyy_im
                        - ExEz_Gyz_im
                        + EyEx_Gxx_im
                        + EyEy_Gyx_im
                        + EyEz_Gxz_im
                    )
                    self.H_dyadic_2D_OTF[3, 1, :, :, idx] = (
                        ExEx_Gxy_re
                        + ExEy_Gyy_re
                        + ExEz_Gyz_re
                        - EyEx_Gxx_re
                        - EyEy_Gyx_re
                        - EyEz_Gxz_re
                    )
                    self.H_dyadic_2D_OTF[3, 2, :, :, idx] = (
                        -ExEx_Gxy_im + ExEy_Gyy_im + EyEx_Gxx_im - EyEy_Gyx_im
                    )
                    self.H_dyadic_2D_OTF[3, 3, :, :, idx] = (
                        -ExEx_Gyy_im - ExEy_Gxy_im + EyEx_Gyx_im + EyEy_Gxx_im
                    )
                    self.H_dyadic_2D_OTF[3, 4, :, :, idx] = (
                        -ExEx_Gyz_im - ExEz_Gxy_im + EyEx_Gxz_im + EyEz_Gxx_im
                    )
                    self.H_dyadic_2D_OTF[3, 5, :, :, idx] = (
                        -ExEy_Gyz_im - ExEz_Gyy_im + EyEy_Gxz_im + EyEz_Gyx_im
                    )
                    self.H_dyadic_2D_OTF[3, 6, :, :, idx] = (
                        -ExEz_Gyz_im + EyEz_Gxz_im
                    )
            else:  # linear Stokes polarimeter
                self.H_dyadic_2D_OTF_in_plane[0, 0, :, :, idx] = (
                    ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
                )
                self.H_dyadic_2D_OTF_in_plane[0, 1, :, :, idx] = (
                    ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
                )
                self.H_dyadic_2D_OTF_in_plane[1, 0, :, :, idx] = (
                    ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
                )
                self.H_dyadic_2D_OTF_in_plane[1, 1, :, :, idx] = (
                    ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
                )

    def gen_3D_vec_WOTF(self, inc_option):
        """

        generate 3D vectorial transfer functions for 3D PTI


        """

        if inc_option == True:
            self.H_dyadic_OTF = np.zeros(
                (
                    self.N_Stokes,
                    7,
                    self.N_pattern,
                    self.N,
                    self.M,
                    self.N_defocus_3D,
                ),
                dtype="complex64",
            )
        else:
            self.H_dyadic_OTF_in_plane = np.zeros(
                (2, 2, self.N_pattern, self.N, self.M, self.N_defocus_3D),
                dtype="complex64",
            )

        # angle-dependent electric field components due to focusing effect
        fr = (self.fxx**2 + self.fyy**2) ** (0.5)
        cos_factor = (
            1 - (self.lambda_illu**2) * (fr**2) * self.Pupil_support
        ) ** (0.5) * self.Pupil_support
        dc_idx = fr == 0
        nondc_idx = fr != 0
        E_field_factor = np.zeros((5, self.N, self.M))

        E_field_factor[0, nondc_idx] = (
            (self.fxx[nondc_idx] ** 2) * cos_factor[nondc_idx]
            + self.fyy[nondc_idx] ** 2
        ) / fr[nondc_idx] ** 2
        E_field_factor[0, dc_idx] = 1
        E_field_factor[1, nondc_idx] = (
            self.fxx[nondc_idx]
            * self.fyy[nondc_idx]
            * (cos_factor[nondc_idx] - 1)
        ) / fr[nondc_idx] ** 2
        E_field_factor[2, nondc_idx] = (
            (self.fyy[nondc_idx] ** 2) * cos_factor[nondc_idx]
            + self.fxx[nondc_idx] ** 2
        ) / fr[nondc_idx] ** 2
        E_field_factor[2, dc_idx] = 1
        E_field_factor[3, nondc_idx] = -self.lambda_illu * self.fxx[nondc_idx]
        E_field_factor[4, nondc_idx] = -self.lambda_illu * self.fyy[nondc_idx]

        # generate dyadic Green's tensor
        N_defocus = self.G_tensor_z_upsampling * self.N_defocus_3D
        psz = self.psz / self.G_tensor_z_upsampling
        if self.z_defocus[0] - self.z_defocus[1] > 0:
            z = -ifftshift((np.r_[0:N_defocus] - N_defocus // 2) * psz)
        else:
            z = ifftshift((np.r_[0:N_defocus] - N_defocus // 2) * psz)
        G_fun_z = (
            generate_greens_function_z(
                torch.tensor(self.frr),
                torch.tensor(self.Pupil_support),
                self.lambda_illu,
                torch.tensor(z),
            )
            .numpy()
            .transpose((1, 2, 0))
        )

        G_real = fftshift(ifft2(G_fun_z, axes=(0, 1)) / self.ps**2)
        G_tensor = gen_dyadic_Greens_tensor(
            G_real, self.ps, psz, self.lambda_illu, space="Fourier"
        )
        G_tensor_z = (ifft(G_tensor, axis=4) / psz)[
            ..., :: int(self.G_tensor_z_upsampling)
        ]

        # compute transfer functions
        def OTF_compute(x, y, z):
            H_re, H_im = compute_weak_object_transfer_function_3D(
                torch.tensor(x.astype("float32")),
                torch.tensor(y.astype("complex64")),
                torch.tensor(self.Pupil_obj.astype("complex64")),
                torch.tensor(
                    self.Hz_det_3D.astype("complex64").transpose((2, 1, 0))
                ),
                torch.tensor(z.astype("complex64").transpose((2, 1, 0))),
                torch.tensor(self.psz),
            )
            return H_re.numpy().transpose((1, 2, 0)), H_im.numpy().transpose(
                (1, 2, 0)
            )

        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[i].copy()

            # focusing electric field components
            Ex_field = (
                self.Source_PolState[i, 0] * E_field_factor[0]
                + self.Source_PolState[i, 1] * E_field_factor[1]
            )
            Ey_field = (
                self.Source_PolState[i, 0] * E_field_factor[1]
                + self.Source_PolState[i, 1] * E_field_factor[2]
            )
            Ez_field = (
                self.Source_PolState[i, 0] * E_field_factor[3]
                + self.Source_PolState[i, 1] * E_field_factor[4]
            )

            IF_ExEx = np.abs(Ex_field) ** 2
            IF_ExEy = Ex_field * np.conj(Ey_field)
            IF_ExEz = Ex_field * np.conj(Ez_field)
            IF_EyEy = np.abs(Ey_field) ** 2
            IF_EyEz = Ey_field * np.conj(Ez_field)

            Source_norm = Source_current * (IF_ExEx + IF_EyEy)

            # intermediate transfer functions
            ExEx_Gxx_re, ExEx_Gxx_im = OTF_compute(
                Source_norm, Source_current * IF_ExEx, G_tensor_z[0, 0]
            )  #
            ExEy_Gxy_re, ExEy_Gxy_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy, G_tensor_z[0, 1]
            )  #
            EyEx_Gyx_re, EyEx_Gyx_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy.conj(), G_tensor_z[0, 1]
            )  #
            EyEy_Gyy_re, EyEy_Gyy_im = OTF_compute(
                Source_norm, Source_current * IF_EyEy, G_tensor_z[1, 1]
            )  #
            ExEx_Gxy_re, ExEx_Gxy_im = OTF_compute(
                Source_norm, Source_current * IF_ExEx, G_tensor_z[0, 1]
            )  #
            ExEy_Gxx_re, ExEy_Gxx_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy, G_tensor_z[0, 0]
            )  #
            EyEx_Gyy_re, EyEx_Gyy_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy.conj(), G_tensor_z[1, 1]
            )  #
            EyEy_Gyx_re, EyEy_Gyx_im = OTF_compute(
                Source_norm, Source_current * IF_EyEy, G_tensor_z[0, 1]
            )  #
            ExEy_Gyy_re, ExEy_Gyy_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy, G_tensor_z[1, 1]
            )  #
            EyEx_Gxx_re, EyEx_Gxx_im = OTF_compute(
                Source_norm, Source_current * IF_ExEy.conj(), G_tensor_z[0, 0]
            )  #
            ExEx_Gyy_re, ExEx_Gyy_im = OTF_compute(
                Source_norm, Source_current * IF_ExEx, G_tensor_z[1, 1]
            )  #
            EyEy_Gxx_re, EyEy_Gxx_im = OTF_compute(
                Source_norm, Source_current * IF_EyEy, G_tensor_z[0, 0]
            )  #

            if inc_option == True:
                ExEz_Gxz_re, ExEz_Gxz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEz, G_tensor_z[0, 2]
                )
                EyEz_Gyz_re, EyEz_Gyz_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEz, G_tensor_z[1, 2]
                )
                ExEx_Gxz_re, ExEx_Gxz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEx, G_tensor_z[0, 2]
                )
                ExEz_Gxx_re, ExEz_Gxx_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEz, G_tensor_z[0, 0]
                )
                EyEx_Gyz_re, EyEx_Gyz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy.conj(),
                    G_tensor_z[1, 2],
                )
                EyEz_Gyx_re, EyEz_Gyx_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEz, G_tensor_z[0, 1]
                )
                ExEy_Gxz_re, ExEy_Gxz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEy, G_tensor_z[0, 2]
                )
                ExEz_Gxy_re, ExEz_Gxy_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEz, G_tensor_z[0, 1]
                )
                EyEy_Gyz_re, EyEy_Gyz_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEy, G_tensor_z[1, 2]
                )
                EyEz_Gyy_re, EyEz_Gyy_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEz, G_tensor_z[1, 1]
                )
                ExEz_Gyz_re, ExEz_Gyz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEz, G_tensor_z[1, 2]
                )
                EyEz_Gxz_re, EyEz_Gxz_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEz, G_tensor_z[0, 2]
                )
                EyEx_Gxz_re, EyEx_Gxz_im = OTF_compute(
                    Source_norm,
                    Source_current * IF_ExEy.conj(),
                    G_tensor_z[0, 2],
                )
                EyEz_Gxx_re, EyEz_Gxx_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEz, G_tensor_z[0, 0]
                )
                ExEy_Gyz_re, ExEy_Gyz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEy, G_tensor_z[1, 2]
                )
                ExEz_Gyy_re, ExEz_Gyy_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEz, G_tensor_z[1, 1]
                )
                EyEy_Gxz_re, EyEy_Gxz_im = OTF_compute(
                    Source_norm, Source_current * IF_EyEy, G_tensor_z[0, 2]
                )
                ExEx_Gyz_re, ExEx_Gyz_im = OTF_compute(
                    Source_norm, Source_current * IF_ExEx, G_tensor_z[1, 2]
                )

                # 3D vectorial transfer functions
                self.H_dyadic_OTF[0, 0, i] = (
                    ExEx_Gxx_re
                    + ExEy_Gxy_re
                    + ExEz_Gxz_re
                    + EyEx_Gyx_re
                    + EyEy_Gyy_re
                    + EyEz_Gyz_re
                )
                self.H_dyadic_OTF[0, 1, i] = (
                    ExEx_Gxx_im
                    + ExEy_Gxy_im
                    + ExEz_Gxz_im
                    + EyEx_Gyx_im
                    + EyEy_Gyy_im
                    + EyEz_Gyz_im
                )
                self.H_dyadic_OTF[0, 2, i] = (
                    ExEx_Gxx_re - ExEy_Gxy_re + EyEx_Gyx_re - EyEy_Gyy_re
                )
                self.H_dyadic_OTF[0, 3, i] = (
                    ExEx_Gxy_re + ExEy_Gxx_re + EyEx_Gyy_re + EyEy_Gyx_re
                )
                self.H_dyadic_OTF[0, 4, i] = (
                    ExEx_Gxz_re + ExEz_Gxx_re + EyEx_Gyz_re + EyEz_Gyx_re
                )
                self.H_dyadic_OTF[0, 5, i] = (
                    ExEy_Gxz_re + ExEz_Gxy_re + EyEy_Gyz_re + EyEz_Gyy_re
                )
                self.H_dyadic_OTF[0, 6, i] = ExEz_Gxz_re + EyEz_Gyz_re

                self.H_dyadic_OTF[1, 0, i] = (
                    ExEx_Gxx_re
                    + ExEy_Gxy_re
                    + ExEz_Gxz_re
                    - EyEx_Gyx_re
                    - EyEy_Gyy_re
                    - EyEz_Gyz_re
                )
                self.H_dyadic_OTF[1, 1, i] = (
                    ExEx_Gxx_im
                    + ExEy_Gxy_im
                    + ExEz_Gxz_im
                    - EyEx_Gyx_im
                    - EyEy_Gyy_im
                    - EyEz_Gyz_im
                )
                self.H_dyadic_OTF[1, 2, i] = (
                    ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
                )
                self.H_dyadic_OTF[1, 3, i] = (
                    ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
                )
                self.H_dyadic_OTF[1, 4, i] = (
                    ExEx_Gxz_re + ExEz_Gxx_re - EyEx_Gyz_re - EyEz_Gyx_re
                )
                self.H_dyadic_OTF[1, 5, i] = (
                    ExEy_Gxz_re + ExEz_Gxy_re - EyEy_Gyz_re - EyEz_Gyy_re
                )
                self.H_dyadic_OTF[1, 6, i] = ExEz_Gxz_re - EyEz_Gyz_re

                self.H_dyadic_OTF[2, 0, i] = (
                    ExEx_Gxy_re
                    + ExEy_Gyy_re
                    + ExEz_Gyz_re
                    + EyEx_Gxx_re
                    + EyEy_Gyx_re
                    + EyEz_Gxz_re
                )
                self.H_dyadic_OTF[2, 1, i] = (
                    ExEx_Gxy_im
                    + ExEy_Gyy_im
                    + ExEz_Gyz_im
                    + EyEx_Gxx_im
                    + EyEy_Gyx_im
                    + EyEz_Gxz_im
                )
                self.H_dyadic_OTF[2, 2, i] = (
                    ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
                )
                self.H_dyadic_OTF[2, 3, i] = (
                    ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
                )
                self.H_dyadic_OTF[2, 4, i] = (
                    ExEx_Gyz_re + ExEz_Gxy_re + EyEx_Gxz_re + EyEz_Gxx_re
                )
                self.H_dyadic_OTF[2, 5, i] = (
                    ExEy_Gyz_re + ExEz_Gyy_re + EyEy_Gxz_re + EyEz_Gyx_re
                )
                self.H_dyadic_OTF[2, 6, i] = ExEz_Gyz_re + EyEz_Gxz_re

                # transfer functions for S3
                if self.N_Stokes == 4:  # full Stokes polarimeter
                    self.H_dyadic_OTF[3, 0, i] = (
                        -ExEx_Gxy_im
                        - ExEy_Gyy_im
                        - ExEz_Gyz_im
                        + EyEx_Gxx_im
                        + EyEy_Gyx_im
                        + EyEz_Gxz_im
                    )
                    self.H_dyadic_OTF[3, 1, i] = (
                        ExEx_Gxy_re
                        + ExEy_Gyy_re
                        + ExEz_Gyz_re
                        - EyEx_Gxx_re
                        - EyEy_Gyx_re
                        - EyEz_Gxz_re
                    )
                    self.H_dyadic_OTF[3, 2, i] = (
                        -ExEx_Gxy_im + ExEy_Gyy_im + EyEx_Gxx_im - EyEy_Gyx_im
                    )
                    self.H_dyadic_OTF[3, 3, i] = (
                        -ExEx_Gyy_im - ExEy_Gxy_im + EyEx_Gyx_im + EyEy_Gxx_im
                    )
                    self.H_dyadic_OTF[3, 4, i] = (
                        -ExEx_Gyz_im - ExEz_Gxy_im + EyEx_Gxz_im + EyEz_Gxx_im
                    )
                    self.H_dyadic_OTF[3, 5, i] = (
                        -ExEy_Gyz_im - ExEz_Gyy_im + EyEy_Gxz_im + EyEz_Gyx_im
                    )
                    self.H_dyadic_OTF[3, 6, i] = -ExEz_Gyz_im + EyEz_Gxz_im
            else:  # linear Stokes polarimeter
                self.H_dyadic_OTF_in_plane[0, 0, i] = (
                    ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
                )
                self.H_dyadic_OTF_in_plane[0, 1, i] = (
                    ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
                )
                self.H_dyadic_OTF_in_plane[1, 0, i] = (
                    ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
                )
                self.H_dyadic_OTF_in_plane[1, 1, i] = (
                    ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
                )

    ##############   polarization computing function group   ##############

    def Stokes_recon(self, I_meas):
        """

        reconstruct Stokes parameters from polarization-sensitive intensity images

        Parameters
        ----------
            I_meas        : numpy.ndarray
                            polarization-sensitive intensity images with the size of (N_channel, ..., N, M) or
                            (N_channel, ..., N, M, N_defocus)

        Returns
        -------
            S_image_recon : numpy.ndarray
                            reconstructed Stokes parameters with the size of (N_Stokes, ..., N, M), or
                            (N_Stokes, ..., N, M, N_defocus)


        """

        data_dims = I_meas.shape
        if data_dims[0] != self.N_channel:
            raise ValueError(
                f"Unsupported image data size. Provide image data is of size: {data_dims}. "
                f"Image data must be of size (N_channel, ..., N, M) or (N_channel, ..., N, M, N_defocus)"
            )
        if not (
            data_dims[-2:] != (self.N, self.M)
            or data_dims[-3:] != (self.N, self.M, self.N_defocus)
        ):
            raise ValueError(
                f"Unsupported image data size. Provide image data is of size: {data_dims}. "
                f"Image data must be of size (N_channel, ..., N, M) or (N_channel, ..., N, M, N_defocus)"
            )

        # append dummy z dimension (N_defocus=1) if input data is 2D
        single_plane = False
        if data_dims[-2:] == (self.N, self.M):
            single_plane = True
            I_meas = I_meas[..., np.newaxis]

        # reshape image data into (N, M, N_channel, ...)
        img_data = np.moveaxis(I_meas, (-3, -2), (0, 1))
        data_dims2 = img_data.shape
        img_data = np.reshape(img_data, (self.N, self.M, self.N_channel, -1))

        # compute Stokes parameters
        # A_matrix_inv is shape (N_Stokes, N_channel) or (N, M, N_Stokes, N_channel)
        # img_data is shape (N, M, N_channel, ...)
        # S_image_recon is shape (N, M, N_stokes, ...)
        # If A_matrix_inv is 2D (stokes x intensity dimensions), matmul broadcasts it to spatial dimensions.
        if self.use_gpu:
            if self._A_matrix_inv_gpu_array is None:
                self._A_matrix_inv_gpu_array = cp.array(self.A_matrix_inv)
            img_gpu_array = cp.array(img_data)
            S_image_recon = cp.asnumpy(
                cp.matmul(self._A_matrix_inv_gpu_array, img_gpu_array)
            )
        else:
            S_image_recon = np.matmul(self.A_matrix_inv, img_data)

        # reshape Stokes parameters into (N_Stokes, ..., N, M, N_defocus)
        S_image_recon = np.reshape(
            S_image_recon, (self.N, self.M, self.N_Stokes) + data_dims2[3:]
        )
        S_image_recon = np.moveaxis(S_image_recon, (0, 1), (-3, -2))

        # if input data was 2D, remove dummy z dimension
        if single_plane:
            return S_image_recon[..., 0]
        else:
            return S_image_recon

    def Stokes_transform(self, S_image_recon):
        """

        transform Stokes parameters into normalized Stokes parameters

        Parameters
        ----------
            S_image_recon : numpy.ndarray
                            reconstructed Stokes parameters with the size of (N_Stokes, ...)

        Returns
        -------
            S_transformed : numpy.ndarray
                            normalized Stokes parameters with the size of (3, ...) or (5, ...)


        """

        if self.use_gpu:
            S_image_recon = cp.array(S_image_recon)
            if self.N_Stokes == 4:  # full Stokes polarimeter
                S_transformed = cp.zeros((5,) + S_image_recon.shape[1:])
            elif self.N_Stokes == 3:  # linear Stokes polarimeter
                S_transformed = cp.zeros((3,) + S_image_recon.shape[1:])
        else:
            if self.N_Stokes == 4:
                S_transformed = np.zeros((5,) + S_image_recon.shape[1:])
            elif self.N_Stokes == 3:
                S_transformed = np.zeros((3,) + S_image_recon.shape[1:])

        S_transformed[0] = S_image_recon[0]

        if self.N_Stokes == 4:  # full Stokes polarimeter
            S_transformed[1] = S_image_recon[1] / S_image_recon[3]
            S_transformed[2] = S_image_recon[2] / S_image_recon[3]
            S_transformed[3] = S_image_recon[3]
            S_transformed[4] = (
                S_image_recon[1] ** 2
                + S_image_recon[2] ** 2
                + S_image_recon[3] ** 2
            ) ** (1 / 2) / S_image_recon[
                0
            ]  # DoP
        elif self.N_Stokes == 3:  # linear Stokes polarimeter
            S_transformed[1] = S_image_recon[1] / S_image_recon[0]
            S_transformed[2] = S_image_recon[2] / S_image_recon[0]

        if self.use_gpu:
            S_transformed = cp.asnumpy(S_transformed)

        return S_transformed

    def Polscope_bg_correction(
        self, S_image_tm, S_bg_tm, kernel_size=400, poly_order=2
    ):
        """

        QLIPP background correction algorithm

        Parameters
        ----------
            S_image_tm  : numpy.ndarray
                          normalized Stokes parameters with the size of (3, ...) or (5, ...)

            S_bg_tm     : numpy.ndarray
                          normalized background Stokes parameters

            kernel_size : int
                          size of smoothing window for background estimation in 'local' method

            poly_order  : int
                          order of polynomial fitting for background estimation in 'local_fit' method

        Returns
        -------
            S_image_tm : numpy.ndarray
                         background corrected normalized Stokes parameters with the same size as the input Stokes parameters


        """

        if self.use_gpu:
            S_image_tm = cp.array(S_image_tm)
            S_bg_tm = cp.array(S_bg_tm)

        dim = S_image_tm.ndim
        if dim == 3:
            S_image_tm[0] /= S_bg_tm[0]
            S_image_tm[1] -= S_bg_tm[1]
            S_image_tm[2] -= S_bg_tm[2]
            if self.N_Stokes == 4:  # full Stokes polarimeter
                S_image_tm[4] /= S_bg_tm[4]
        else:
            S_image_tm[0] /= S_bg_tm[0, :, :, np.newaxis]
            S_image_tm[1] -= S_bg_tm[1, :, :, np.newaxis]
            S_image_tm[2] -= S_bg_tm[2, :, :, np.newaxis]
            if self.N_Stokes == 4:
                S_image_tm[4] /= S_bg_tm[4, :, :, np.newaxis]

        if self.bg_option == "local":
            if dim == 3:
                S_image_tm[1] -= uniform_filter_2D(
                    S_image_tm[1],
                    size=kernel_size,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                )
                S_image_tm[2] -= uniform_filter_2D(
                    S_image_tm[2],
                    size=kernel_size,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                )
            else:
                if self.use_gpu:
                    S1_bg = uniform_filter_2D(
                        cp.mean(S_image_tm[1], axis=-1),
                        size=kernel_size,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )
                    S2_bg = uniform_filter_2D(
                        cp.mean(S_image_tm[2], axis=-1),
                        size=kernel_size,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )
                else:
                    S1_bg = uniform_filter_2D(
                        np.mean(S_image_tm[1], axis=-1),
                        size=kernel_size,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )
                    S2_bg = uniform_filter_2D(
                        np.mean(S_image_tm[2], axis=-1),
                        size=kernel_size,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )

                for i in range(self.N_defocus):
                    S_image_tm[1, :, :, i] -= S1_bg
                    S_image_tm[2, :, :, i] -= S2_bg

        elif self.bg_option == "local_fit":
            if self.use_gpu:
                bg_estimator = BackgroundEstimator2D_GPU(gpu_id=self.gpu_id)
                if dim != 3:
                    S1_bg = bg_estimator.get_background(
                        cp.mean(S_image_tm[1], axis=-1),
                        order=poly_order,
                        normalize=False,
                    )
                    S2_bg = bg_estimator.get_background(
                        cp.mean(S_image_tm[2], axis=-1),
                        order=poly_order,
                        normalize=False,
                    )
            else:
                bg_estimator = BackgroundEstimator2D()
                if dim != 3:
                    S1_bg = bg_estimator.get_background(
                        np.mean(S_image_tm[1], axis=-1),
                        order=poly_order,
                        normalize=False,
                    )
                    S2_bg = bg_estimator.get_background(
                        np.mean(S_image_tm[2], axis=-1),
                        order=poly_order,
                        normalize=False,
                    )

            if dim == 3:
                S_image_tm[1] -= bg_estimator.get_background(
                    S_image_tm[1], order=poly_order, normalize=False
                )
                S_image_tm[2] -= bg_estimator.get_background(
                    S_image_tm[2], order=poly_order, normalize=False
                )
            else:
                for i in range(self.N_defocus):
                    S_image_tm[1, :, :, i] -= S1_bg
                    S_image_tm[2, :, :, i] -= S2_bg

        if self.use_gpu:
            S_image_tm = cp.asnumpy(S_image_tm)

        return S_image_tm

    def Polarization_recon(self, S_image_recon):
        """

        reconstruction of polarization-related physical properties in QLIPP

        Parameters
        ----------
            S_image_recon : numpy.ndarray
                            normalized Stokes parameters with the size of (3, ...) or (5, ...)

        Returns
        -------
            Recon_para    : numpy.ndarray
                            reconstructed polarization-related physical properties
                            channel 0 is retardance
                            channel 1 is in-plane orientation
                            channel 2 is brightfield
                            channel 3 is degree of polarization


        """

        if self.use_gpu:
            S_image_recon = cp.array(S_image_recon)
            Recon_para = cp.zeros((self.N_Stokes,) + S_image_recon.shape[1:])
        else:
            Recon_para = np.zeros((self.N_Stokes,) + S_image_recon.shape[1:])

        if self.use_gpu:
            if self.N_Stokes == 4:  # full Stokes polarimeter
                ret_wrapped = cp.arctan2(
                    (S_image_recon[1] ** 2 + S_image_recon[2] ** 2) ** (1 / 2)
                    * S_image_recon[3],
                    S_image_recon[3],
                )  # retardance
            elif self.N_Stokes == 3:  # linear Stokes polarimeters
                ret_wrapped = cp.arcsin(
                    cp.minimum(
                        (S_image_recon[1] ** 2 + S_image_recon[2] ** 2)
                        ** (0.5),
                        1,
                    )
                )

            if self.cali == True:
                sa_wrapped = (
                    0.5
                    * cp.arctan2(-S_image_recon[1], -S_image_recon[2])
                    % np.pi
                )  # slow-axis
            else:
                sa_wrapped = (
                    0.5
                    * cp.arctan2(-S_image_recon[1], S_image_recon[2])
                    % np.pi
                )  # slow-axis

        else:
            if self.N_Stokes == 4:  # full Stokes polarimeter
                ret_wrapped = np.arctan2(
                    (S_image_recon[1] ** 2 + S_image_recon[2] ** 2) ** (1 / 2)
                    * S_image_recon[3],
                    S_image_recon[3],
                )  # retardance
            elif self.N_Stokes == 3:  # linear Stokes polarimeters
                ret_wrapped = np.arcsin(
                    np.minimum(
                        (S_image_recon[1] ** 2 + S_image_recon[2] ** 2)
                        ** (0.5),
                        1,
                    )
                )

            if self.cali == True:
                sa_wrapped = (
                    0.5
                    * np.arctan2(-S_image_recon[1], -S_image_recon[2])
                    % np.pi
                )  # slow-axis
            else:
                sa_wrapped = (
                    0.5
                    * np.arctan2(-S_image_recon[1], S_image_recon[2])
                    % np.pi
                )  # slow-axis

        sa_wrapped[ret_wrapped < 0] += np.pi / 2
        ret_wrapped[ret_wrapped < 0] += np.pi
        Recon_para[0] = ret_wrapped.copy()
        Recon_para[1] = sa_wrapped % np.pi
        Recon_para[2] = S_image_recon[0]  # transmittance

        if self.N_Stokes == 4:  # full Stokes polarimeter
            Recon_para[3] = S_image_recon[4]  # DoP

        if self.use_gpu:
            Recon_para = cp.asnumpy(Recon_para)

        return Recon_para

    def Birefringence_recon(self, S1_stack, S2_stack, reg=1e-3):
        # Birefringence deconvolution with slowly varying transmission approximation

        if self.use_gpu:
            Hu = cp.array(self.Hu, copy=True)
            Hp = cp.array(self.Hp, copy=True)

            AHA = [
                cp.sum(cp.abs(Hu) ** 2 + cp.abs(Hp) ** 2, axis=2) + reg,
                cp.sum(Hu * cp.conj(Hp) - cp.conj(Hu) * Hp, axis=2),
                -cp.sum(Hu * cp.conj(Hp) - cp.conj(Hu) * Hp, axis=2),
                cp.sum(cp.abs(Hu) ** 2 + cp.abs(Hp) ** 2, axis=2) + reg,
            ]

            S1_stack_f = cp.fft.fft2(cp.array(S1_stack), axes=(0, 1))
            if self.cali:
                S2_stack_f = cp.fft.fft2(-cp.array(S2_stack), axes=(0, 1))
            else:
                S2_stack_f = cp.fft.fft2(cp.array(S2_stack), axes=(0, 1))

            b_vec = [
                cp.sum(
                    -cp.conj(Hu) * S1_stack_f + cp.conj(Hp) * S2_stack_f,
                    axis=2,
                ),
                cp.sum(
                    cp.conj(Hp) * S1_stack_f + cp.conj(Hu) * S2_stack_f, axis=2
                ),
            ]

        else:
            AHA = [
                np.sum(np.abs(self.Hu) ** 2 + np.abs(self.Hp) ** 2, axis=2)
                + reg,
                np.sum(
                    self.Hu * np.conj(self.Hp) - np.conj(self.Hu) * self.Hp,
                    axis=2,
                ),
                -np.sum(
                    self.Hu * np.conj(self.Hp) - np.conj(self.Hu) * self.Hp,
                    axis=2,
                ),
                np.sum(np.abs(self.Hu) ** 2 + np.abs(self.Hp) ** 2, axis=2)
                + reg,
            ]

            S1_stack_f = fft2(S1_stack, axes=(0, 1))
            if self.cali:
                S2_stack_f = fft2(-S2_stack, axes=(0, 1))
            else:
                S2_stack_f = fft2(S2_stack, axes=(0, 1))

            b_vec = [
                np.sum(
                    -np.conj(self.Hu) * S1_stack_f
                    + np.conj(self.Hp) * S2_stack_f,
                    axis=2,
                ),
                np.sum(
                    np.conj(self.Hp) * S1_stack_f
                    + np.conj(self.Hu) * S2_stack_f,
                    axis=2,
                ),
            ]

        del_phi_s, del_phi_c = dual_variable_tikhonov_deconvolution_2d(
            AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id
        )

        Retardance = 2 * (del_phi_s**2 + del_phi_c**2) ** (1 / 2)
        slowaxis = 0.5 * np.arctan2(del_phi_s, del_phi_c) % np.pi

        return Retardance, slowaxis

    def Birefringence_recon_2D(
        self,
        S1_stack,
        S2_stack,
        method="Tikhonov",
        reg_br=1,
        rho=1e-5,
        lambda_br=1e-3,
        itr=20,
        verbose=True,
    ):
        """

        conduct 2D birefringence deconvolution from defocused or asymmetrically-illuminated set of intensity images

        Parameters
        ----------
            S1_stack   : numpy.ndarray
                         defocused or asymmetrically-illuminated set of S1 intensity images with the size of (N, M, N_pattern*N_defocus)

            S2_stack   : numpy.ndarray
                         defocused or asymmetrically-illuminated set of S1 intensity images with the size of (N, M, N_pattern*N_defocus)

            method     : str
                         denoiser for 2D birefringence deconvolution
                         'Tikhonov' for Tikhonov denoiser
                         'TV'       for TV denoiser

            reg_br     : float
                         Tikhonov regularization parameter

            lambda_br  : float
                         TV regularization parameter

            rho        : float
                         augmented Lagrange multiplier for 2D ADMM algorithm

            itr        : int
                         number of iterations for 2D ADMM algorithm

            verbose    : bool
                         option to display detailed progress of computations or not


        Returns
        -------
            retardance : numpy.ndarray
                         2D retardance (in the unit of rad) reconstruction with the size of (N, M)

            azimuth    : numpy.ndarray
                         2D orientation reconstruction with the size of (N, M)


        """

        if self.N_defocus == 1:
            S1_stack = np.reshape(S1_stack, (self.N, self.M, 1))
            S2_stack = np.reshape(S2_stack, (self.N, self.M, 1))

        H_1_1c = self.H_dyadic_2D_OTF_in_plane[0, 0]
        H_1_1s = self.H_dyadic_2D_OTF_in_plane[0, 1]
        H_2_1c = self.H_dyadic_2D_OTF_in_plane[1, 0]
        H_2_1s = self.H_dyadic_2D_OTF_in_plane[1, 1]

        S1_stack_f = fft2(S1_stack, axes=(0, 1))
        S2_stack_f = fft2(S2_stack, axes=(0, 1))

        cross_term = np.sum(
            np.conj(H_1_1c) * H_1_1s + np.conj(H_2_1c) * H_2_1s, axis=2
        )

        AHA = [
            np.sum(np.abs(H_1_1c) ** 2 + np.abs(H_2_1c) ** 2, axis=2),
            cross_term,
            np.conj(cross_term),
            np.sum(np.abs(H_1_1s) ** 2 + np.abs(H_2_1s) ** 2, axis=2),
        ]

        AHA[0] += np.mean(np.abs(AHA[0])) * reg_br
        AHA[3] += np.mean(np.abs(AHA[3])) * reg_br

        b_vec = [
            np.sum(
                np.conj(H_1_1c) * S1_stack_f + np.conj(H_2_1c) * S2_stack_f,
                axis=2,
            ),
            np.sum(
                np.conj(H_1_1s) * S1_stack_f + np.conj(H_2_1s) * S2_stack_f,
                axis=2,
            ),
        ]

        for i in range(4):
            AHA[i] = torch.tensor(AHA[i])
        for i in range(2):
            b_vec[i] = torch.tensor(b_vec[i])

        if method == "Tikhonov":
            # Deconvolution with Tikhonov regularization
            g_1c_temp, g_1s_temp = dual_variable_tikhonov_deconvolution_2d(
                AHA, b_vec
            )

        elif method == "TV":
            # ADMM deconvolution with anisotropic TV regularization

            g_1c_temp, g_1s_temp = dual_variable_admm_tv_deconv_2d(
                AHA,
                b_vec,
                rho,
                lambda_br,
                lambda_br,
                itr,
                verbose,
            )

        g_1c = g_1c_temp.numpy()
        g_1s = g_1s_temp.numpy()

        azimuth = (np.arctan2(-g_1s, -g_1c) / 2) % np.pi
        retardance = ((np.abs(g_1s) ** 2 + np.abs(g_1c) ** 2) ** (1 / 2)) / (
            2 * np.pi / self.lambda_illu
        )

        return retardance, azimuth

    def Birefringence_recon_3D(
        self,
        S1_stack,
        S2_stack,
        method="Tikhonov",
        reg_br=1,
        rho=1e-5,
        lambda_br=1e-3,
        itr=20,
        verbose=True,
    ):
        """

        conduct 3D deconvolution of 2D birefringence from defocused stack of intensity images

        Parameters
        ----------
            S1_stack         : numpy.ndarray
                               defocused stack of S1 intensity images with the size of (N, M, N_defocus)

            S2_stack         : numpy.ndarray
                               defocused stack of S2 intensity images with the size of (N, M, N_defocus)

            method           : str
                               denoiser for 3D phase reconstruction
                               'Tikhonov' for Tikhonov denoiser
                               'TV'       for TV denoiser

            reg_br           : float
                               Tikhonov regularization parameter

            rho              : float
                               augmented Lagrange multiplier for 3D ADMM algorithm

            lambda_br        : float
                               TV regularization parameter

            itr              : int
                               number of iterations for 3D ADMM algorithm

            verbose          : bool
                               option to display detailed progress of computations or not


        Returns
        -------
            retardance       : numpy.ndarray
                               3D reconstruction of retardance (in the unit of rad) with the size of (N, M, N_defocus)

            azimuth          : numpy.ndarray
                               3D reconstruction of 2D orientation with the size of (N, M, N_defocus)



        """

        if self.pad_z != 0:
            S1_pad = np.pad(
                S1_stack,
                ((0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=S1_stack.mean(),
            )
            S2_pad = np.pad(
                S2_stack,
                ((0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=S2_stack.mean(),
            )
            if self.pad_z < self.N_defocus:
                S1_pad[:, :, : self.pad_z] = (S1_stack[:, :, : self.pad_z])[
                    :, :, ::-1
                ]
                S1_pad[:, :, -self.pad_z :] = (S1_stack[:, :, -self.pad_z :])[
                    :, :, ::-1
                ]
                S2_pad[:, :, : self.pad_z] = (S2_stack[:, :, : self.pad_z])[
                    :, :, ::-1
                ]
                S2_pad[:, :, -self.pad_z :] = (S2_stack[:, :, -self.pad_z :])[
                    :, :, ::-1
                ]
            else:
                print(
                    "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                )

            S1_stack = S1_pad.copy()
            S2_stack = S2_pad.copy()

        H_1_1c = self.H_dyadic_OTF_in_plane[0, 0, 0]
        H_1_1s = self.H_dyadic_OTF_in_plane[0, 1, 0]
        H_2_1c = self.H_dyadic_OTF_in_plane[1, 0, 0]
        H_2_1s = self.H_dyadic_OTF_in_plane[1, 1, 0]

        S1_stack_f = fftn(S1_stack)
        S2_stack_f = fftn(S2_stack)

        cross_term = np.conj(H_1_1c) * H_1_1s + np.conj(H_2_1c) * H_2_1s

        AHA = [
            np.abs(H_1_1c) ** 2 + np.abs(H_2_1c) ** 2,
            cross_term,
            np.conj(cross_term),
            np.abs(H_1_1s) ** 2 + np.abs(H_2_1s) ** 2,
        ]

        AHA[0] += np.mean(np.abs(AHA[0])) * reg_br
        AHA[3] += np.mean(np.abs(AHA[3])) * reg_br

        b_vec = [
            np.conj(H_1_1c) * S1_stack_f + np.conj(H_2_1c) * S2_stack_f,
            np.conj(H_1_1s) * S1_stack_f + np.conj(H_2_1s) * S2_stack_f,
        ]

        if self.use_gpu:
            AHA = cp.array(AHA)
            b_vec = cp.array(b_vec)

        if method == "Tikhonov":
            # Deconvolution with Tikhonov regularization

            f_1c, f_1s = Dual_variable_Tikhonov_deconv_3D(
                AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id
            )

        elif method == "TV":
            # ADMM deconvolution with anisotropic TV regularization

            f_1c, f_1s = Dual_variable_ADMM_TV_deconv_3D(
                AHA,
                b_vec,
                rho,
                lambda_br,
                lambda_br,
                itr,
                verbose,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )

        azimuth = (np.arctan2(-f_1s, -f_1c) / 2) % np.pi
        retardance = (
            ((np.abs(f_1s) ** 2 + np.abs(f_1c) ** 2) ** (1 / 2))
            / (2 * np.pi / self.lambda_illu)
            * self.psz
        )

        if self.pad_z != 0:
            azimuth = azimuth[:, :, self.pad_z : -(self.pad_z)]
            retardance = retardance[:, :, self.pad_z : -(self.pad_z)]

        return retardance, azimuth

    def Inclination_recon_geometric(
        self, retardance, orientation, on_axis_idx, reg_ret_pr=1e-2
    ):
        """

        estimating 2D principal retardance and 3D orientation from off-axis retardance and orientation using geometric model

        Parameters
        ----------
            retardance    : numpy.ndarray
                            measured retardance from different pattern illuminations with the size of (N_pattern, N, M)

            orientation   : numpy.ndarray
                            measured 2D orientation from different pattern illuminations with the size of (N_pattern, N, M)

            on_axis_idx   : int
                            index of the illumination pattern corresponding to on-axis illumination

            reg_ret_pr    : float
                            regularization for computing principal retardance


        Returns
        -------
            inclination   : numpy.ndarray
                            estimated inclination angle with the size of (N, M)

            retardance_pr : numpy.ndarray
                            estimated principal retardance with the size of (N, M)

            inc_coeff     : numpy.ndarray
                            estimated inclination coefficients with the size of (6, N, M)


        """

        retardance_on_axis = retardance[:, :, on_axis_idx].copy()
        orientation_on_axis = orientation[:, :, on_axis_idx].copy()

        retardance = np.transpose(retardance, (2, 0, 1))

        N_meas = self.N_pattern * self.N_defocus

        inc_coeff = np.reshape(
            self.geometric_inc_matrix_inv.dot(
                retardance.reshape((N_meas, self.N * self.M))
            ),
            (6, self.N, self.M),
        )
        inc_coeff_sin_2theta = (inc_coeff[2] ** 2 + inc_coeff[3] ** 2) ** (0.5)
        inclination = np.arctan2(retardance_on_axis * 2, inc_coeff_sin_2theta)
        inclination = np.pi / 2 - (np.pi / 2 - inclination) * np.sign(
            inc_coeff[2] * np.cos(orientation_on_axis)
            + inc_coeff[3] * np.sin(orientation_on_axis)
        )

        retardance_pr = (
            retardance_on_axis
            * np.sin(inclination) ** 2
            / (np.sin(inclination) ** 4 + reg_ret_pr)
        )

        return inclination, retardance_pr, inc_coeff

    def scattering_potential_tensor_recon_2D_vec(
        self, S_image_recon, reg_inc=1e-1 * np.ones((7,)), cupy_det=False
    ):
        """

        Tikhonov reconstruction of 2D scattering potential tensor components with vectorial model in PTI

        Parameters
        ----------
            S_image_recon : numpy.ndarray
                            background corrected Stokes parameters normalized with S0's mean with the size of (3, N, M, N_pattern)

            reg_inc       : numpy.ndarray
                            Tikhonov regularization parameters for 7 scattering potential tensor components with the size of (7,)

            cupy_det      : bool
                            option to use the determinant algorithm from cupy package (cupy v9 has very fast determinant calculation compared to array-based determinant calculation)

        Returns
        -------
            f_tensor      : numpy.ndarray
                            2D scattering potential tensor components with the size of (7, N, M)


        """

        start_time = time.time()

        S_stack_f = fft2(S_image_recon, axes=(1, 2))

        AHA = self.inc_AHA_2D_vec.copy()

        for i in range(7):
            AHA[i, i] += np.mean(np.abs(AHA[i, i])) * reg_inc[i]

        b_vec = np.zeros((7, self.N, self.M), complex)

        for i, j in itertools.product(range(7), range(self.N_Stokes)):
            b_vec[i] += np.sum(
                np.conj(self.H_dyadic_2D_OTF[j, i]) * S_stack_f[j], axis=2
            )

        print(
            "Finished preprocess, elapsed time: %.2f"
            % (time.time() - start_time)
        )

        if self.use_gpu:
            if cupy_det:
                AHA = cp.transpose(cp.array(AHA), (2, 3, 0, 1))
                b_vec = cp.transpose(cp.array(b_vec), (1, 2, 0))

                determinant = cp.linalg.det(AHA)
                f_tensor = cp.zeros((7, self.N, self.M), dtype="float32")

                for i in range(7):
                    AHA_b_vec = AHA.copy()
                    AHA_b_vec[:, :, :, i] = b_vec.copy()
                    f_tensor[i] = cp.real(
                        cp.fft.ifftn(cp.linalg.det(AHA_b_vec) / determinant)
                    )

            else:
                AHA = cp.array(AHA)
                b_vec = cp.array(b_vec)

                determinant = array_based_7x7_det(AHA)

                f_tensor = cp.zeros((7, self.N, self.M))

                for i in range(7):
                    AHA_b_vec = AHA.copy()
                    AHA_b_vec[:, i] = b_vec.copy()
                    f_tensor[i] = cp.real(
                        cp.fft.ifft2(
                            array_based_7x7_det(AHA_b_vec) / determinant
                        )
                    )

            f_tensor = cp.asnumpy(f_tensor)

        else:
            AHA_pinv = np.linalg.pinv(np.transpose(AHA, (2, 3, 0, 1)))
            f_tensor = np.real(
                ifft2(
                    np.transpose(
                        np.squeeze(
                            np.matmul(
                                AHA_pinv,
                                np.transpose(b_vec, (1, 2, 0))[
                                    ..., np.newaxis
                                ],
                            )
                        ),
                        (2, 0, 1),
                    ),
                    axes=(1, 2),
                )
            )

        print(
            "Finished reconstruction, elapsed time: %.2f"
            % (time.time() - start_time)
        )

        return f_tensor

    def scattering_potential_tensor_recon_3D_vec(
        self, S_image_recon, reg_inc=1e-1 * np.ones((7,)), cupy_det=False
    ):
        """

        Tikhonov reconstruction of 3D scattering potential tensor components with vectorial model in PTI

        Parameters
        ----------
            S_image_recon : numpy.ndarray
                            background corrected Stokes parameters normalized with S0's mean with the size of (3, N_pattern, N, M, N_defocus)

            reg_inc       : numpy.ndarray
                            Tikhonov regularization parameters for 7 scattering potential tensor components with the size of (7,)

            cupy_det      : bool
                            option to use the determinant algorithm from cupy package (cupy v9 has very fast determinant calculation compared to array-based determinant calculation)

        Returns
        -------
            f_tensor      : numpy.ndarray
                            3D scattering potential tensor components with the size of (7, N, M, N_defocus)


        """

        start_time = time.time()

        if self.pad_z != 0:
            S_pad = np.pad(
                S_image_recon,
                ((0, 0), (0, 0), (0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=0,
            )
            if self.pad_z < self.N_defocus:
                S_pad[..., : self.pad_z] = (S_image_recon[..., : self.pad_z])[
                    :, :, ::-1
                ]
                S_pad[..., -self.pad_z :] = (
                    S_image_recon[..., -self.pad_z :]
                )[:, :, ::-1]

            else:
                print(
                    "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                )

            S_image_recon = S_pad.copy()

        S_stack_f = fftn(S_image_recon, axes=(-3, -2, -1))

        AHA = self.inc_AHA_3D_vec.copy()

        for i in range(7):
            AHA[i, i] += np.mean(np.abs(AHA[i, i])) * reg_inc[i]

        b_vec = np.zeros(
            (7, self.N, self.M, self.N_defocus_3D), dtype="complex64"
        )

        for i, j in itertools.product(range(7), range(self.N_Stokes)):
            b_vec[i] += np.sum(
                np.conj(self.H_dyadic_OTF[j, i]) * S_stack_f[j], axis=0
            )

        print(
            "Finished preprocess, elapsed time: %.2f"
            % (time.time() - start_time)
        )

        if self.use_gpu:
            if cupy_det:
                AHA = cp.transpose(cp.array(AHA), (2, 3, 4, 0, 1))
                b_vec = cp.transpose(cp.array(b_vec), (1, 2, 3, 0))

                determinant = cp.linalg.det(AHA)
                f_tensor = cp.zeros(
                    (7, self.N, self.M, self.N_defocus_3D), dtype="float32"
                )

                for i in range(7):
                    AHA_b_vec = AHA.copy()
                    AHA_b_vec[:, :, :, :, i] = b_vec.copy()
                    f_tensor[i] = cp.real(
                        cp.fft.ifftn(cp.linalg.det(AHA_b_vec) / determinant)
                    )
            else:
                AHA = cp.array(AHA)
                b_vec = cp.array(b_vec)

                determinant = array_based_7x7_det(AHA)

                f_tensor = cp.zeros(
                    (7, self.N, self.M, self.N_defocus_3D), dtype="float32"
                )

                for i in range(7):
                    AHA_b_vec = AHA.copy()
                    AHA_b_vec[:, i] = b_vec.copy()
                    f_tensor[i] = cp.real(
                        cp.fft.ifftn(
                            array_based_7x7_det(AHA_b_vec) / determinant
                        )
                    )

            f_tensor = cp.asnumpy(f_tensor)

        else:
            AHA_pinv = np.linalg.pinv(np.transpose(AHA, (2, 3, 4, 0, 1)))
            f_tensor = np.real(
                ifftn(
                    np.transpose(
                        np.squeeze(
                            np.matmul(
                                AHA_pinv,
                                np.transpose(b_vec, (1, 2, 3, 0))[
                                    ..., np.newaxis
                                ],
                            )
                        ),
                        (3, 0, 1, 2),
                    ),
                    axes=(1, 2, 3),
                )
            )

        if self.pad_z != 0:
            f_tensor = f_tensor[..., self.pad_z : -(self.pad_z)]

        print(
            "Finished reconstruction, elapsed time: %.2f"
            % (time.time() - start_time)
        )

        return f_tensor

    def scattering_potential_tensor_to_3D_orientation(
        self,
        f_tensor,
        S_image_recon=None,
        material_type="positive",
        reg_ret_pr=1e-2,
        itr=20,
        step_size=0.3,
        verbose=True,
        fast_gpu_mode=False,
    ):
        """

        estimating principal retardance, 3D orientation, optic sign from scattering potential tensor components

        Parameters
        ----------
            f_tensor      : numpy.ndarray
                            scattering potential tensor components with the size of (7, N, M) or (7, N, M, N_defocus) for 3D

            S_image_recon : numpy.ndarray
                            background corrected Stokes parameters normalized with S0's mean

            material_type : str
                            'positive' for assumption of positively uniaxial material
                            'negative' for assumption of negatively uniaxial material
                            'unknown' for triggering optic sign estimation algorithm -> return two sets of solution with a probability map of material

            reg_ret_pr    : numpy.ndarray
                            regularization parameters for principal retardance estimation

            itr           : int
                            number of iterations for the optic sign retrieval algorithm

            step_size     : float
                            scaling of the gradient step size for the optic sign retrieval algorithm

            verbose       : bool
                            option to display details of optic sign retrieval algorithm in each iteration

            fast_gpu_mode : bool
                            option to use faster gpu computation mode (all arrays in gpu, it may consume more memory)

        Returns
        -------
            retardance_pr : numpy.ndarray
                            reconstructed principal retardance with the size of (2, N, M) for 2D and (2, N, M, N_defocus) for 3D
                            channel 0: positively uniaxial solution (or return retardance_pr_p when 'positive' is specified for material_type)
                            channel 1: negatively uniaxial solution (or return retardance_pr_n when 'negative' is specified for material_type)

            azimuth       : numpy.ndarray
                            reconstructed in-plane orientation with the size of (2, N, M) for 2D and (2, N, M, N_defocus) for 3D
                            channel 0: positively uniaxial solution (or return azimuth_p when 'positive' is specified for material_type)
                            channel 1: negatively uniaxial solution (or return azimuth_n when 'negative' is specified for material_type)

            theta         : numpy.ndarray
                            reconstructed out-of-plane inclination with the size of (2, N, M) for 2D and (2, N, M, N_defocus) for 3D
                            channel 0: positively uniaxial solution (or return theta_p when 'positive' is specified for material_type)
                            channel 1: negatively uniaxial solution (or return theta_n when 'negative' is specified for material_type)

            mat_map       : numpy.ndarray
                            reconstructed material tendancy with the size of (2, N, M) for 2D and (2, N, M, N_defocus) for 3D
                            channel 0: tendancy for positively uniaxial solution
                            channel 1: tendancy for negatively uniaxial solution


        """

        if self.pad_z != 0 and material_type == "unknown":
            S_pad = np.pad(
                S_image_recon,
                ((0, 0), (0, 0), (0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=0,
            )
            f_tensor_pad = np.pad(
                f_tensor,
                ((0, 0), (0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=0,
            )
            if self.pad_z < self.N_defocus:
                S_pad[..., : self.pad_z] = (S_image_recon[..., : self.pad_z])[
                    :, :, ::-1
                ]
                S_pad[..., -self.pad_z :] = (
                    S_image_recon[..., -self.pad_z :]
                )[:, :, ::-1]
                f_tensor_pad[..., : self.pad_z] = (
                    f_tensor[..., : self.pad_z]
                )[:, :, ::-1]
                f_tensor_pad[..., -self.pad_z :] = (
                    f_tensor[..., -self.pad_z :]
                )[:, :, ::-1]

            else:
                print(
                    "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                )

            S_image_recon = S_pad.copy()
            f_tensor = f_tensor_pad.copy()

        if material_type == "positive" or "unknown":
            # Positive uniaxial material

            (
                retardance_pr_p,
                azimuth_p,
                theta_p,
            ) = scattering_potential_tensor_to_3D_orientation_PN(
                f_tensor, material_type="positive", reg_ret_pr=reg_ret_pr
            )

            if material_type == "positive":
                return retardance_pr_p, azimuth_p, theta_p

        if material_type == "negative" or "unknown":
            # Negative uniaxial material

            (
                retardance_pr_n,
                azimuth_n,
                theta_n,
            ) = scattering_potential_tensor_to_3D_orientation_PN(
                f_tensor, material_type="negative", reg_ret_pr=reg_ret_pr
            )

            if material_type == "negative":
                return retardance_pr_n, azimuth_n, theta_n

        if material_type == "unknown":
            if f_tensor.ndim == 4:
                S_stack_f = fftn(S_image_recon, axes=(-3, -2, -1))

            elif f_tensor.ndim == 3:
                S_stack_f = fft2(S_image_recon, axes=(1, 2))

            f_tensor_p = np.zeros((5,) + f_tensor.shape[1:])
            f_tensor_p[0] = (
                -retardance_pr_p
                * (np.sin(theta_p) ** 2)
                * np.cos(2 * azimuth_p)
            )
            f_tensor_p[1] = (
                -retardance_pr_p
                * (np.sin(theta_p) ** 2)
                * np.sin(2 * azimuth_p)
            )
            f_tensor_p[2] = (
                -retardance_pr_p * (np.sin(2 * theta_p)) * np.cos(azimuth_p)
            )
            f_tensor_p[3] = (
                -retardance_pr_p * (np.sin(2 * theta_p)) * np.sin(azimuth_p)
            )
            f_tensor_p[4] = retardance_pr_p * (
                np.sin(theta_p) ** 2 - 2 * np.cos(theta_p) ** 2
            )

            f_tensor_n = np.zeros((5,) + f_tensor.shape[1:])
            f_tensor_n[0] = (
                -retardance_pr_n
                * (np.sin(theta_n) ** 2)
                * np.cos(2 * azimuth_n)
            )
            f_tensor_n[1] = (
                -retardance_pr_n
                * (np.sin(theta_n) ** 2)
                * np.sin(2 * azimuth_n)
            )
            f_tensor_n[2] = (
                -retardance_pr_n * (np.sin(2 * theta_n)) * np.cos(azimuth_n)
            )
            f_tensor_n[3] = (
                -retardance_pr_n * (np.sin(2 * theta_n)) * np.sin(azimuth_n)
            )
            f_tensor_n[4] = retardance_pr_n * (
                np.sin(theta_n) ** 2 - 2 * np.cos(theta_n) ** 2
            )

            f_vec = f_tensor.copy()

            x_map = np.zeros(f_tensor.shape[1:])
            y_map = np.zeros(f_tensor.shape[1:])

            if f_tensor.ndim == 4:
                f_vec_f = fftn(f_vec, axes=(1, 2, 3))
                S_est_vec = np.zeros(
                    (
                        self.N_Stokes,
                        self.N_pattern,
                        self.N,
                        self.M,
                        self.N_defocus_3D,
                    ),
                    complex,
                )

                for p, q in itertools.product(range(self.N_Stokes), range(2)):
                    S_est_vec[p] += (
                        self.H_dyadic_OTF[p, q] * f_vec_f[np.newaxis, q]
                    )

            elif f_tensor.ndim == 3:
                f_vec_f = fft2(f_vec, axes=(1, 2))
                S_est_vec = np.zeros(
                    (
                        self.N_Stokes,
                        self.N,
                        self.M,
                        self.N_defocus * self.N_pattern,
                    ),
                    complex,
                )

                for p, q in itertools.product(range(self.N_Stokes), range(2)):
                    S_est_vec[p] += (
                        self.H_dyadic_2D_OTF[p, q]
                        * f_vec_f[q, :, :, np.newaxis]
                    )

            if self.use_gpu:
                f_tensor_p = cp.array(f_tensor_p)
                f_tensor_n = cp.array(f_tensor_n)
                f_vec = cp.array(f_vec)
                if fast_gpu_mode:
                    S_stack_f = cp.array(S_stack_f)

            # iterative optic sign estimation algorithm
            err = np.zeros(itr + 1)

            tic_time = time.time()

            if verbose:
                print("|  Iter  |  error  |  Elapsed time (sec)  |")
                f1, ax = plt.subplots(2, 2, figsize=(20, 20))

            for i in range(itr):
                if self.use_gpu:
                    x_map = cp.array(x_map)
                    y_map = cp.array(y_map)

                for j in range(5):
                    f_vec[j + 2] = (
                        x_map * f_tensor_p[j] + y_map * f_tensor_n[j]
                    )

                S_est_vec_update = S_est_vec.copy()

                if self.use_gpu:
                    if fast_gpu_mode:
                        S_est_vec_update = cp.array(S_est_vec_update)

                        if f_tensor.ndim == 4:
                            f_vec_f = cp.fft.fftn(f_vec, axes=(1, 2, 3))

                            for p, q in itertools.product(
                                range(self.N_Stokes), range(5)
                            ):
                                S_est_vec_update[p] += (
                                    cp.array(self.H_dyadic_OTF[p, q + 2])
                                    * f_vec_f[np.newaxis, q + 2]
                                )

                        elif f_tensor.ndim == 3:
                            f_vec_f = cp.fft.fft2(f_vec, axes=(1, 2))

                            for p, q in itertools.product(
                                range(self.N_Stokes), range(5)
                            ):
                                S_est_vec_update[p] += (
                                    cp.array(self.H_dyadic_2D_OTF[p, q + 2])
                                    * f_vec_f[q + 2, :, :, np.newaxis]
                                )

                    else:
                        if f_tensor.ndim == 4:
                            f_vec_f = cp.fft.fftn(f_vec, axes=(1, 2, 3))

                            for p, q in itertools.product(
                                range(self.N_Stokes), range(5)
                            ):
                                S_est_vec_update[p] += cp.asnumpy(
                                    cp.array(self.H_dyadic_OTF[p, q + 2])
                                    * f_vec_f[np.newaxis, q + 2]
                                )

                        elif f_tensor.ndim == 3:
                            f_vec_f = cp.fft.fft2(f_vec, axes=(1, 2))

                            for p, q in itertools.product(
                                range(self.N_Stokes), range(5)
                            ):
                                S_est_vec_update[p] += cp.asnumpy(
                                    cp.array(self.H_dyadic_2D_OTF[p, q + 2])
                                    * f_vec_f[q + 2, :, :, np.newaxis]
                                )

                else:
                    if f_tensor.ndim == 4:
                        f_vec_f = fftn(f_vec, axes=(1, 2, 3))

                        for p, q in itertools.product(
                            range(self.N_Stokes), range(5)
                        ):
                            S_est_vec_update[p] += (
                                self.H_dyadic_OTF[p, q + 2]
                                * f_vec_f[np.newaxis, q + 2]
                            )

                    elif f_tensor.ndim == 3:
                        f_vec_f = fft2(f_vec, axes=(1, 2))

                        for p, q in itertools.product(
                            range(self.N_Stokes), range(5)
                        ):
                            S_est_vec_update[p] += (
                                self.H_dyadic_2D_OTF[p, q + 2]
                                * f_vec_f[q + 2, :, :, np.newaxis]
                            )

                S_diff = S_stack_f - S_est_vec_update

                if fast_gpu_mode and self.use_gpu:
                    err[i + 1] = cp.asnumpy(cp.sum(cp.abs(S_diff) ** 2))
                else:
                    err[i + 1] = np.sum(np.abs(S_diff) ** 2)

                if err[i + 1] > err[i] and i > 0:
                    if self.use_gpu:
                        x_map = cp.asnumpy(x_map)
                        y_map = cp.asnumpy(y_map)
                    break

                if self.use_gpu:
                    AH_S_diff = cp.zeros((5,) + f_tensor.shape[1:], complex)

                    if f_tensor.ndim == 4:
                        for p, q in itertools.product(
                            range(5), range(self.N_Stokes)
                        ):
                            if fast_gpu_mode:
                                AH_S_diff[p] += cp.sum(
                                    cp.conj(
                                        cp.array(self.H_dyadic_OTF[q, p + 2])
                                    )
                                    * S_diff[q],
                                    axis=0,
                                )
                            else:
                                AH_S_diff[p] += cp.sum(
                                    cp.conj(
                                        cp.array(self.H_dyadic_OTF[q, p + 2])
                                    )
                                    * cp.array(S_diff[q]),
                                    axis=0,
                                )

                        grad_x_map = -cp.real(
                            cp.sum(
                                f_tensor_p
                                * cp.fft.ifftn(AH_S_diff, axes=(1, 2, 3)),
                                axis=0,
                            )
                        )
                        grad_y_map = -cp.real(
                            cp.sum(
                                f_tensor_n
                                * cp.fft.ifftn(AH_S_diff, axes=(1, 2, 3)),
                                axis=0,
                            )
                        )

                    elif f_tensor.ndim == 3:
                        for p, q in itertools.product(
                            range(5), range(self.N_Stokes)
                        ):
                            if fast_gpu_mode:
                                AH_S_diff[p] += cp.sum(
                                    cp.conj(
                                        cp.array(
                                            self.H_dyadic_2D_OTF[q, p + 2]
                                        )
                                    )
                                    * S_diff[q],
                                    axis=2,
                                )
                            else:
                                AH_S_diff[p] += cp.sum(
                                    cp.conj(
                                        cp.array(
                                            self.H_dyadic_2D_OTF[q, p + 2]
                                        )
                                    )
                                    * cp.array(S_diff[q]),
                                    axis=2,
                                )

                        grad_x_map = -cp.real(
                            cp.sum(
                                f_tensor_p
                                * cp.fft.ifft2(AH_S_diff, axes=(1, 2)),
                                axis=0,
                            )
                        )
                        grad_y_map = -cp.real(
                            cp.sum(
                                f_tensor_n
                                * cp.fft.ifft2(AH_S_diff, axes=(1, 2)),
                                axis=0,
                            )
                        )

                    x_map -= (
                        grad_x_map / cp.max(cp.abs(grad_x_map)) * step_size
                    )
                    y_map -= (
                        grad_y_map / cp.max(cp.abs(grad_y_map)) * step_size
                    )

                    x_map = cp.asnumpy(x_map)
                    y_map = cp.asnumpy(y_map)

                else:
                    AH_S_diff = np.zeros((5,) + f_tensor.shape[1:], complex)

                    if f_tensor.ndim == 4:
                        for p, q in itertools.product(
                            range(5), range(self.N_Stokes)
                        ):
                            AH_S_diff[p] += np.sum(
                                np.conj(self.H_dyadic_OTF[q, p + 2])
                                * S_diff[q],
                                axis=0,
                            )

                        grad_x_map = -np.real(
                            np.sum(
                                f_tensor_p * ifftn(AH_S_diff, axes=(1, 2, 3)),
                                axis=0,
                            )
                        )
                        grad_y_map = -np.real(
                            np.sum(
                                f_tensor_n * ifftn(AH_S_diff, axes=(1, 2, 3)),
                                axis=0,
                            )
                        )

                    elif f_tensor.ndim == 3:
                        for p, q in itertools.product(
                            range(5), range(self.N_Stokes)
                        ):
                            AH_S_diff[p] += np.sum(
                                np.conj(self.H_dyadic_2D_OTF[q, p + 2])
                                * S_diff[q],
                                axis=2,
                            )

                        grad_x_map = -np.real(
                            np.sum(
                                f_tensor_p * ifft2(AH_S_diff, axes=(1, 2)),
                                axis=0,
                            )
                        )
                        grad_y_map = -np.real(
                            np.sum(
                                f_tensor_n * ifft2(AH_S_diff, axes=(1, 2)),
                                axis=0,
                            )
                        )

                    x_map -= grad_x_map / np.max(np.abs(grad_x_map)) * 0.3
                    y_map -= grad_y_map / np.max(np.abs(grad_y_map)) * 0.3

                if verbose:
                    print(
                        "|  %d  |  %.2e  |   %.2f   |"
                        % (i + 1, err[i + 1], time.time() - tic_time)
                    )

                    if i != 0:
                        ax[0, 0].cla()
                        ax[0, 1].cla()
                        ax[1, 0].cla()
                        ax[1, 1].cla()
                    if f_tensor.ndim == 4:
                        ax[0, 0].imshow(
                            x_map[:, :, self.N_defocus_3D // 2],
                            origin="lower",
                            vmin=0,
                            vmax=2,
                        )
                        ax[0, 1].imshow(
                            np.transpose(x_map[self.N // 2, :, :]),
                            origin="lower",
                            vmin=0,
                            vmax=2,
                        )
                        ax[1, 0].imshow(
                            y_map[:, :, self.N_defocus_3D // 2],
                            origin="lower",
                            vmin=0,
                            vmax=2,
                        )
                        ax[1, 1].imshow(
                            np.transpose(y_map[self.N // 2, :, :]),
                            origin="lower",
                            vmin=0,
                            vmax=2,
                        )
                    elif f_tensor.ndim == 3:
                        ax[0, 0].imshow(x_map, origin="lower", vmin=0, vmax=2)
                        ax[0, 1].imshow(y_map, origin="lower", vmin=0, vmax=2)

                    if i != itr - 1:
                        display.display(f1)
                        display.clear_output(wait=True)
                        time.sleep(0.0001)

            retardance_pr = np.stack([retardance_pr_p, retardance_pr_n])
            azimuth = np.stack([azimuth_p, azimuth_n])
            theta = np.stack([theta_p, theta_n])
            mat_map = np.stack([x_map, y_map])
            print(
                "Finish optic sign estimation, elapsed time: %.2f"
                % (time.time() - tic_time)
            )

            if self.pad_z != 0:
                retardance_pr = retardance_pr[..., self.pad_z : -(self.pad_z)]
                azimuth = azimuth[..., self.pad_z : -(self.pad_z)]
                theta = theta[..., self.pad_z : -(self.pad_z)]
                mat_map = mat_map[..., self.pad_z : -(self.pad_z)]

            return retardance_pr, azimuth, theta, mat_map

    ##############   phase computing function group   ##############

    def Phase_recon_semi_3D(
        self,
        S0_stack,
        method="Tikhonov",
        reg_u=1e-6,
        reg_p=1e-6,
        rho=1e-5,
        lambda_u=1e-3,
        lambda_p=1e-3,
        itr=20,
        verbose=False,
    ):
        mu_sample = np.zeros((self.N, self.M, self.N_defocus))
        phi_sample = np.zeros((self.N, self.M, self.N_defocus))

        for i in range(self.N_defocus):
            if i <= self.ph_deconv_layer // 2:
                tf_start_idx = self.ph_deconv_layer // 2 - i
            else:
                tf_start_idx = 0

            obj_start_idx = np.maximum(0, i - self.ph_deconv_layer // 2)

            if self.N_defocus - i - 1 < self.ph_deconv_layer // 2:
                tf_end_idx = self.ph_deconv_layer // 2 + (self.N_defocus - i)
            else:
                tf_end_idx = self.ph_deconv_layer

            obj_end_idx = np.minimum(
                self.N_defocus,
                i + self.ph_deconv_layer - self.ph_deconv_layer // 2,
            )

            print(
                "TF_index = (%d,%d), obj_z_index=(%d,%d), consistency: %s"
                % (
                    tf_start_idx,
                    tf_end_idx,
                    obj_start_idx,
                    obj_end_idx,
                    (obj_end_idx - obj_start_idx)
                    == (tf_end_idx - tf_start_idx),
                )
            )

            if self.use_gpu:
                S0_stack_sub = self.inten_normalization(
                    cp.array(S0_stack[:, :, obj_start_idx:obj_end_idx])
                )
                Hu = cp.array(
                    self.Hu[:, :, tf_start_idx:tf_end_idx], copy=True
                )
                Hp = cp.array(
                    self.Hp[:, :, tf_start_idx:tf_end_idx], copy=True
                )

                S0_stack_f = cp.fft.fft2(S0_stack_sub, axes=(0, 1))

                AHA = [
                    cp.sum(cp.abs(Hu) ** 2, axis=2) + reg_u,
                    cp.sum(cp.conj(Hu) * Hp, axis=2),
                    cp.sum(cp.conj(Hp) * Hu, axis=2),
                    cp.sum(cp.abs(Hp) ** 2, axis=2) + reg_p,
                ]

                b_vec = [
                    cp.sum(cp.conj(Hu) * S0_stack_f, axis=2),
                    cp.sum(cp.conj(Hp) * S0_stack_f, axis=2),
                ]

            else:
                S0_stack_sub = self.inten_normalization(
                    S0_stack[:, :, obj_start_idx:obj_end_idx]
                )
                S0_stack_f = fft2(S0_stack_sub, axes=(0, 1))

                Hu = self.Hu[:, :, tf_start_idx:tf_end_idx]
                Hp = self.Hp[:, :, tf_start_idx:tf_end_idx]

                AHA = [
                    np.sum(np.abs(Hu) ** 2, axis=2) + reg_u,
                    np.sum(np.conj(Hu) * Hp, axis=2),
                    np.sum(np.conj(Hp) * Hu, axis=2),
                    np.sum(np.abs(Hp) ** 2, axis=2) + reg_p,
                ]

                b_vec = [
                    np.sum(np.conj(Hu) * S0_stack_f, axis=2),
                    np.sum(np.conj(Hp) * S0_stack_f, axis=2),
                ]

            if method == "Tikhonov":
                # Deconvolution with Tikhonov regularization

                (
                    mu_sample_temp,
                    phi_sample_temp,
                ) = dual_variable_tikhonov_deconvolution_2d(
                    AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id
                )

            elif method == "TV":
                # ADMM deconvolution with anisotropic TV regularization

                (
                    mu_sample_temp,
                    phi_sample_temp,
                ) = dual_variable_admm_tv_deconv_2d(
                    AHA,
                    b_vec,
                    rho,
                    lambda_u,
                    lambda_p,
                    itr,
                    verbose,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                )

            mu_sample[:, :, i] = mu_sample_temp.copy()
            phi_sample[:, :, i] = phi_sample_temp - phi_sample_temp.mean()

        return mu_sample, phi_sample

    def Phase_recon_3D(
        self,
        S0_stack,
        absorption_ratio=0.0,
        method="Tikhonov",
        reg_re=1e-4,
        autotune_re=False,
        reg_im=1e-4,
        rho=1e-5,
        lambda_re=1e-3,
        lambda_im=1e-3,
        itr=20,
        verbose=True,
    ):
        """

        conduct 3D phase reconstruction from defocused or asymmetrically-illuminated stack of intensity images (TIE or DPC)

        Parameters
        ----------
            S0_stack         : numpy.ndarray
                               defocused or asymmetrically-illuminated stack of S0 intensity images with the size of (N_pattern, N, M, N_defocus) or (N, M, N_defocus)

            absorption_ratio : float
                               assumption of correlation between phase and absorption (0 means absorption = phase*0, effective when N_pattern==1)

            method           : str
                               denoiser for 3D phase reconstruction
                               'Tikhonov' for Tikhonov denoiser
                               'TV'       for TV denoiser

            reg_re           : float
                               Tikhonov regularization parameter for 3D phase

            autotune_re      : bool
                               option to automatically choose Tikhonov regularization parameter for 3D phase, with search centered around reg_re

            reg_im           : float
                               Tikhonov regularization parameter for 3D absorption

            rho              : float
                               augmented Lagrange multiplier for 3D ADMM algorithm

            lambda_re        : float
                               TV regularization parameter for 3D absorption

            lambda_im        : float
                               TV regularization parameter for 3D absorption

            itr              : int
                               number of iterations for 3D ADMM algorithm

            verbose          : bool
                               option to display detailed progress of computations or not


        Returns
        -------
            scaled f_real    : numpy.ndarray
                               3D reconstruction of phase (in the unit of rad) with the size of (N, M, N_defocus)
                               if autotune_re is True, returns 3 reconstructions from different regularization parameters, size (3, N, M, N_defocus)

            scaled f_imag    : numpy.ndarray
                               3D reconstruction of absorption with the size of (N, M, N_defocus)


        """

        if self.N_pattern == 1:
            if self.pad_z == 0:
                S0_stack = inten_normalization_3D(S0_stack)
            else:
                S0_pad = np.pad(
                    S0_stack,
                    ((0, 0), (0, 0), (self.pad_z, self.pad_z)),
                    mode="constant",
                    constant_values=0,
                )
                if self.pad_z < self.N_defocus:
                    S0_pad[:, :, : self.pad_z] = (
                        S0_stack[:, :, : self.pad_z]
                    )[:, :, ::-1]
                    S0_pad[:, :, -self.pad_z :] = (
                        S0_stack[:, :, -self.pad_z :]
                    )[:, :, ::-1]
                else:
                    print(
                        "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                    )

                S0_stack = inten_normalization_3D(S0_pad)

            H_eff = self.H_re + absorption_ratio * self.H_im

            if method == "Tikhonov":
                f_real = single_variable_tikhonov_deconvolution_3D(
                    S0_stack,
                    H_eff,
                    reg_re,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                    autotune=autotune_re,
                    verbose=verbose,
                )

            elif method == "TV":
                f_real = single_variable_admm_tv_deconvolution_3D(
                    S0_stack,
                    H_eff,
                    rho,
                    reg_re,
                    lambda_re,
                    itr,
                    verbose,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                )

            if self.pad_z != 0:
                f_real = f_real[..., self.pad_z : -(self.pad_z)]

            return -f_real * self.psz / 4 / np.pi * self.lambda_illu

        else:
            if self.pad_z == 0:
                S0_stack = inten_normalization_3D(S0_stack)
            else:
                S0_pad = np.pad(
                    S0_stack,
                    ((0, 0), (0, 0), (0, 0), (self.pad_z, self.pad_z)),
                    mode="constant",
                    constant_values=0,
                )
                if self.pad_z < self.N_defocus:
                    S0_pad[..., : self.pad_z] = (S0_stack[..., : self.pad_z])[
                        ..., ::-1
                    ]
                    S0_pad[..., -self.pad_z :] = (
                        S0_stack[..., -self.pad_z :]
                    )[..., ::-1]
                else:
                    print(
                        "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                    )

                S0_stack = inten_normalization_3D(S0_pad)

            if self.use_gpu:
                H_re = cp.array(self.H_re)
                H_im = cp.array(self.H_im)

                S0_stack_f = cp.fft.fftn(
                    cp.array(S0_stack).astype("float32"), axes=(-3, -2, -1)
                )

                AHA = [
                    cp.sum(cp.abs(H_re) ** 2, axis=0) + reg_re,
                    cp.sum(cp.conj(H_re) * H_im, axis=0),
                    cp.sum(cp.conj(H_im) * H_re, axis=0),
                    cp.sum(cp.abs(H_im) ** 2, axis=0) + reg_im,
                ]

                b_vec = [
                    cp.sum(cp.conj(H_re) * S0_stack_f, axis=0),
                    cp.sum(cp.conj(H_im) * S0_stack_f, axis=0),
                ]

            else:
                S0_stack_f = fftn(S0_stack, axes=(-3, -2, -1))

                AHA = [
                    np.sum(np.abs(self.H_re) ** 2, axis=0) + reg_re,
                    np.sum(np.conj(self.H_re) * self.H_im, axis=0),
                    np.sum(np.conj(self.H_im) * self.H_re, axis=0),
                    np.sum(np.abs(self.H_im) ** 2, axis=0) + reg_im,
                ]

                b_vec = [
                    np.sum(np.conj(self.H_re) * S0_stack_f, axis=0),
                    np.sum(np.conj(self.H_im) * S0_stack_f, axis=0),
                ]

            if method == "Tikhonov":
                # Deconvolution with Tikhonov regularization

                f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(
                    AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id
                )

            elif method == "TV":
                # ADMM deconvolution with anisotropic TV regularization

                f_real, f_imag = Dual_variable_ADMM_TV_deconv_3D(
                    AHA,
                    b_vec,
                    rho,
                    lambda_re,
                    lambda_im,
                    itr,
                    verbose,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id,
                )

            if self.pad_z != 0:
                f_real = f_real[..., self.pad_z : -(self.pad_z)]
                f_imag = f_imag[..., self.pad_z : -(self.pad_z)]

            return (
                -f_real * self.psz / 4 / np.pi * self.lambda_illu,
                f_imag * self.psz / 4 / np.pi * self.lambda_illu,
            )


class fluorescence_microscopy:
    """

    fluorescence_microscopy contains methods to compute object transfer function (OTF)
    for fluorescence images:

    1) 2D/3D Deconvolution of widefield fluorescence microscopy


    Parameters
    ----------
        img_dim              : tuple
                               shape of the computed 2D space with size of (N, M, Z)

        lambda_emiss         : list
                               list of wavelength of the fluorescence emmission
                               the order of the emission wavelength should match the order of the first index of the fluorescence intensity

        ps                   : float
                               xy pixel size of the image space

        psz                  : float
                               z step size of the image space

        NA_obj               : float
                               numerical aperture of the detection objective

        n_media              : float
                               refractive index of the immersing media

        deconv_mode          : str
                               '2D-WF' refers to 2D deconvolution of the widefield fluorescence microscopy
                               '3D-WF' refers to 3D deconvolution of the widefield fluorescence microscopy

        pad_z                : int
                               number of z-layers to pad (reflection boundary condition) for 3D deconvolution

        use_gpu              : bool
                               option to use gpu or not

        gpu_id               : int
                               number refering to which gpu will be used


    """

    def __init__(
        self,
        img_dim,
        lambda_emiss,
        ps,
        psz,
        NA_obj,
        n_media=1,
        deconv_mode="3D-WF",
        pad_z=0,
        use_gpu=False,
        gpu_id=0,
    ):
        """

        initialize the system parameters for phase and orders microscopy

        """

        t0 = time.time()

        # GPU/CPU

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        if self.use_gpu:
            globals()["cp"] = __import__("cupy")
            cp.cuda.Device(self.gpu_id).use()

        # Basic parameter
        self.N, self.M, self.N_defocus = img_dim
        self.n_media = n_media
        self.lambda_emiss = np.array(lambda_emiss) / self.n_media
        self.ps = ps
        self.psz = psz
        self.pad_z = pad_z
        self.NA_obj = NA_obj / n_media
        self.N_wavelength = len(lambda_emiss)
        self.deconv_mode = deconv_mode

        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate(
            (self.N, self.M), ps
        )

        # Setup defocus kernel
        self.Hz_det_setup(deconv_mode)

        # Set up PSF and OTF for 3D deconvolution
        self.fluor_deconv_setup(deconv_mode)

    def Hz_det_setup(self, deconv_mode):
        """
        Initiate the defocus kernel

        Parameters
        ----------
            deconv_mode : str
                          '2D-WF' refers to 2D deconvolution of the widefield fluorescence microscopy
                          '3D-WF' refers to 3D deconvolution of the widefield fluorescence microscopy

        """
        self.Pupil_obj = np.zeros((self.N_wavelength, self.N, self.M))

        for i in range(self.N_wavelength):
            self.Pupil_obj[i] = gen_Pupil(
                self.fxx, self.fyy, self.NA_obj, self.lambda_emiss[i]
            )
        self.Pupil_support = self.Pupil_obj.copy()

        if deconv_mode == "3D-WF":
            self.N_defocus_3D = self.N_defocus + 2 * self.pad_z
            self.z = ifftshift(
                (np.r_[0 : self.N_defocus_3D] - self.N_defocus_3D // 2)
                * self.psz
            )

            self.Hz_det = np.zeros(
                (self.N_wavelength, self.N, self.M, self.N_defocus_3D), complex
            )

            for i in range(self.N_wavelength):
                self.Hz_det[i] = generate_propagation_kernel(
                    self.fxx,
                    self.fyy,
                    self.Pupil_support[i],
                    self.lambda_emiss[i],
                    self.z,
                )

    def fluor_deconv_setup(self, deconv_mode):
        """
        Set up the PSF and OTF for 3D deconvolution

        Parameters
        ----------
            Hz_det          : defocus kernel

        Returns
        -------

        """

        if deconv_mode == "2D-WF":
            self.PSF_WF_2D = np.abs(ifft2(self.Pupil_obj, axes=(1, 2))) ** 2
            self.OTF_WF_2D = fft2(self.PSF_WF_2D, axes=(1, 2))
            self.OTF_WF_2D /= (np.max(np.abs(self.OTF_WF_2D), axis=(1, 2)))[
                :, np.newaxis, np.newaxis
            ]

        if deconv_mode == "3D-WF":
            self.PSF_WF_3D = np.abs(ifft2(self.Hz_det, axes=(1, 2))) ** 2
            self.OTF_WF_3D = fftn(self.PSF_WF_3D, axes=(1, 2, 3))
            self.OTF_WF_3D /= (np.max(np.abs(self.OTF_WF_3D), axis=(1, 2, 3)))[
                :, np.newaxis, np.newaxis, np.newaxis
            ]

    def deconvolve_fluor_2D(self, I_fluor, bg_level, reg):
        """

        Performs deconvolution with Tikhonov regularization on raw fluorescence stack.

        Parameters
        ----------
            I_fluor         : numpy.ndarray
                              Raw fluorescence intensity stack in dimensions (N_wavelength, N, M) or (N, M)
                              the order of the first index of I_fluor should match the order of the emission wavelengths

            bg_level        : list or numpy.ndarray
                              Estimated background intensity level in dimensions (N_wavelength,)
                              the order of the bg value should match the order of the first index of I_fluor

            reg             : list or numpy.array
                              an array of Tikhonov regularization parameters in dimensions (N_wavelength,)
                              the order of the reg value should match the order of the first index of I_fluor

        Returns
        -------
            I_fluor_deconv  : numpy.ndarray
                              2D deconvolved fluoresence image in dimensions (N_wavelength, N, M)

        """

        if I_fluor.ndim == 2:
            I_fluor_process = I_fluor[np.newaxis, :, :].copy()
        elif I_fluor.ndim == 3:
            I_fluor_process = I_fluor.copy()

        I_fluor_process = I_fluor_process.astype("float")

        I_fluor_deconv = np.zeros_like(I_fluor_process)

        for i in range(self.N_wavelength):
            I_fluor_minus_bg = np.maximum(0, I_fluor_process[i] - bg_level[i])

            if self.use_gpu:
                I_fluor_f = cp.fft.fft2(
                    cp.array(I_fluor_minus_bg.astype("float32")), axes=(-2, -1)
                )
                H_eff = cp.array(self.OTF_WF_2D[i].astype("complex64"))

                I_fluor_deconv[i] = cp.asnumpy(
                    np.maximum(
                        cp.real(
                            cp.fft.ifft2(
                                I_fluor_f
                                * cp.conj(H_eff)
                                / (cp.abs(H_eff) ** 2 + reg[i]),
                                axes=(-2, -1),
                            )
                        ),
                        0,
                    )
                )
            else:
                I_fluor_f = fft2(I_fluor_minus_bg, axes=(-2, -1))
                I_fluor_deconv[i] = np.maximum(
                    np.real(
                        ifftn(
                            I_fluor_f
                            * np.conj(self.OTF_WF_2D[i])
                            / (np.abs(self.OTF_WF_2D[i]) ** 2 + reg[i]),
                            axes=(-2, -1),
                        )
                    ),
                    0,
                )

        return np.squeeze(I_fluor_deconv)

    def deconvolve_fluor_3D(
        self,
        I_fluor,
        bg_level,
        reg,
        autotune=False,
        search_range_auto=3,
        verbose=True,
    ):
        """

        Performs deconvolution with Tikhonov regularization on raw fluorescence stack.

        Parameters
        ----------
            I_fluor         : numpy.ndarray
                              Raw fluorescence intensity stack in dimensions (N_wavelength, N, M, Z) or (N, M, Z)
                              the order of the first index of I_fluor should match the order of the emission wavelengths

            bg_level        : list or numpy.ndarray
                              Estimated background intensity level in dimensions (N_wavelength,)
                              the order of the bg value should match the order of the first index of I_fluor


            reg             : list or numpy.array
                              an array of Tikhonov regularization parameters in dimensions (N_wavelength,)
                              the order of the reg value should match the order of the first index of I_fluor

            autotune        : bool
                              option to automatically choose Tikhonov regularization parameter, with search centered around reg

            search_range_auto : int
                                the search range of the regularization in terms of the order of magnitude

            verbose         : bool
                             option to display detailed progress of computations or not


        Returns
        -------
            I_fluor_deconv  : numpy.ndarray
                              3D deconvolved fluoresence stack in dimensions (N_wavelength, N, M, Z)
                              if autotune is True, returns 3 deconvolved stacks for each channel, for 3 diff

        """

        if I_fluor.ndim == 3:
            I_fluor_process = I_fluor[np.newaxis, :, :, :].copy()
        elif I_fluor.ndim == 4:
            I_fluor_process = I_fluor.copy()

        if self.pad_z != 0:
            I_fluor_pad = np.pad(
                I_fluor_process,
                ((0, 0), (0, 0), (0, 0), (self.pad_z, self.pad_z)),
                mode="constant",
                constant_values=0,
            )
            if self.pad_z < self.N_defocus:
                I_fluor_pad[:, :, :, : self.pad_z] = (
                    I_fluor_process[:, :, :, : self.pad_z]
                )[:, :, :, ::-1]
                I_fluor_pad[:, :, :, -self.pad_z :] = (
                    I_fluor_process[:, :, :, -self.pad_z :]
                )[:, :, :, ::-1]
            else:
                print(
                    "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
                )

        else:
            I_fluor_pad = I_fluor_process

        I_fluor_process = I_fluor_process.astype("float")

        if autotune:
            N, M, Z = I_fluor_process.shape[1:]
            I_fluor_deconv = np.zeros((self.N_wavelength, 3, N, M, Z))
        else:
            I_fluor_deconv = np.zeros_like(I_fluor_process)

        if verbose:
            print(
                "I_fluor_pad",
                I_fluor_pad.shape,
                "I_fluor_deconv",
                I_fluor_deconv.shape,
            )

        for i in range(self.N_wavelength):
            I_fluor_minus_bg = np.maximum(0, I_fluor_pad[i] - bg_level[i])

            I_fluor_deconv_pad = single_variable_tikhonov_deconvolution_3D(
                I_fluor_minus_bg,
                self.OTF_WF_3D[i],
                reg[i],
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
                autotune=autotune,
                verbose=verbose,
                search_range_auto=search_range_auto,
            )

            if self.pad_z != 0:
                I_fluor_deconv[i] = np.maximum(
                    I_fluor_deconv_pad[..., self.pad_z : -(self.pad_z)], 0
                )
            else:
                I_fluor_deconv[i] = np.maximum(I_fluor_deconv_pad, 0)

        return np.squeeze(I_fluor_deconv)

    def Fluor_anisotropy_recon(self, S1_stack, S2_stack):
        """

        Reconstruct fluorescence anisotropy and ensemble fluorophore orientation from normalized S1 and S2 Stokes
        parameters

        Parameters
        ----------
        S1_stack        : numpy.ndarray
                          Normalized S1 images
        S2_stack        : numpy.ndarray
                          Normalized S2 images

        Returns
        -------
            anisotropy  : numpy.ndarray
                          Fluorescence anisotropy

            orientation : numpy.ndarray
                          Ensemble fluorophore orientation, in the range [0, pi] radians

        """

        if self.use_gpu:
            S1_stack = cp.array(S1_stack)
            S2_stack = cp.array(S2_stack)

            anisotropy = cp.asnumpy(
                0.5 * cp.sqrt(S1_stack**2 + S2_stack**2)
            )
            orientation = cp.asnumpy(
                (0.5 * cp.arctan2(S2_stack, S1_stack)) % np.pi
            )

        else:
            anisotropy = 0.5 * np.sqrt(S1_stack**2 + S2_stack**2)
            orientation = (0.5 * np.arctan2(S2_stack, S1_stack)) % np.pi

        return anisotropy, orientation

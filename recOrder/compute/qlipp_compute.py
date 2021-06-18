import waveorder as wo
from waveorder.waveorder_reconstructor import waveorder_microscopy as setup
import numpy as np
import time


def initialize_reconstructor(image_dim, wavelength, swing, N_channel, anistropy_only, NA_obj, NA_illu, mag, N_slices, z_step, pad_z,
                             pixel_size, bg_option='local_fit', n_media=1.0, mode='3D', use_gpu=False, gpu_id=0):
    """
    Initialize the QLIPP reconstructor for downstream tasks

        Parameters
        ----------

            image_dim         : tuple
                                (height, width) of images in pixels

            wavelength        : int
                                wavelength of illumination in nm

            swing             : float
                                swing used for calibration in waves

            N_channel         : int
                                number of label-free channels used in acquisition

            anisotropy_only   : bool
                                True if only want to process Ret, Ori, BF.  False if phase processing

            NA_obj            : float
                                numerical aperture of the detection objective
            NA_illu           : float
                                numerical aperture of the illumination condenser

            mag               : float
                                magnification used for imaging (e.g. 20 for 20x)

            N_slices          : int
                                number of slices in the z-stack

            z_step            : float
                                z step size of the image space

            pad_z             : float
                                how many padding slices to add for phase computation

            pixel_size        : float
                                pixel size of the camera in um
            bg_option         : str
                                'local' for estimating background with scipy uniform filter
                                'local_fit' for estimating background with polynomial fit
                                'None' for no BG correction
                                'Global' for normal background subtraction with the provided background

            n_media           : float
                                refractive index of the immersing media

            use_gpu           : bool
                                option to use gpu or not

            gpu_id            : int
                                number refering to which gpu will be used


        Returns
        -------
            reconstructor     : object
                                reconstruction object initialized with reconstruction parameters

        """

    lambda_illu = wavelength / 1000
    N_defocus = N_slices
    z_defocus = -(np.r_[:N_defocus] - N_defocus // 2) * z_step
    ps = pixel_size / mag
    cali = True

    if N_channel == 4:
        chi = swing
        inst_mat = np.array([[1, 0, 0, -1],
                             [1, np.sin(2 * np.pi * chi), 0, -np.cos(2 * np.pi * chi)],
                             [1, -0.5 * np.sin(2 * np.pi * chi), np.sqrt(3) * np.cos(np.pi * chi) * np.sin(np.pi * chi), \
                              -np.cos(2 * np.pi * chi)],
                             [1, -0.5 * np.sin(2 * np.pi * chi), -np.sqrt(3) / 2 * np.sin(2 * np.pi * chi), \
                              -np.cos(2 * np.pi * chi)]])

    #         print(f'Instrument Matrix: \n\n{inst_mat}')
    else:
        chi = swing * 2 * np.pi
        inst_mat = None

    print('Initializing Reconstructor...')

    start_time = time.time()
    recon = setup(image_dim, lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi=chi,
                  n_media=n_media, cali=cali, bg_option=bg_option,
                  A_matrix=inst_mat, QLIPP_birefringence_only=anistropy_only, pad_z=pad_z,
                  phase_deconv=mode, illu_mode='BF', use_gpu=use_gpu, gpu_id=gpu_id)

    recon.N_channel = N_channel

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Initializing Reconstructor ({elapsed_time:0.2f} min)')

    return recon


def reconstruct_qlipp_stokes(data, recon, bg_stokes):
    """
    From intensity data, use the waveorder.waveorder_microscopy (recon) to build a stokes array
        if recon background correction flag is selected, will also perform backgroudn correction

    Parameters
    ----------
        data                : np.ndarray or zarr array
                              intensity data of shape: (C, Z, Y, X)

        recon               : waveorder.waveorder_microscopy object
                              initialized by initialize_reconstructor

        bg_stokes           : np.ndarray
                              stokes array representing background data

    Returns
    -------
        stokes_stack        : np.ndarray
                              array representing stokes array
                              has shape: (Z, C, Y, X), where C = 5

    """

    stokes_stack = []
    raw_stack = np.transpose(data, (1, 0, 2, 3))

    for z in range(data.shape[1]):
        stokes_data = recon.Stokes_recon(raw_stack[z])
        stokes_data = recon.Stokes_transform(stokes_data)

        if recon.bg_option != 'None':

            S_image_tm = recon.Polscope_bg_correction(stokes_data, bg_stokes)
            stokes_stack.append(S_image_tm)

        else:
            stokes_stack.append(stokes_data)

    return np.asarray(stokes_stack)


def reconstruct_qlipp_birefringence(stokes, recon):
    """
    From stokes data, use waveorder.waveorder_microscopy (recon) to build a birefringence array

    Parameters
    ----------
    stokes                  : np.ndarray or zarr array
                              stokes array generated by reconstruct_qlipp_stokes

    recon                   : waveorder.waveorder_microscopy object
                              initialized by initialize_reconstructor

    Returns
    -------
        recon_data          : np.ndarray
                              volume of shape (Z, C, Y, X) containing reconstructed birefringence data.
    """

    if len(stokes.shape) == 4:
        slices = stokes.shape[0]
        recon_data = np.zeros([slices, 4, stokes.shape[-2], stokes.shape[-1]])
    elif len(stokes.shape) == 3:
        slices = 1
        recon_data = np.zeros([1, 4, stokes.shape[-2], stokes.shape[-1]])

    for z in range(slices):
        recon_data[z, :, :, :] = recon.Polarization_recon(stokes[z] if slices != 1 else stokes)


    return np.transpose(recon_data, (1,0,2,3))


def reconstruct_qlipp_phase2D(S0, recon, method='Tikhonov', reg_p=1e-4, rho=1,
                              lambda_p=1e-4, itr=50):

    _, phase2D = recon.Phase_recon(S0, method=method, reg_p=reg_p, rho=rho, lambda_p=lambda_p, itr=itr, verbose=False)

    return phase2D


def reconstruct_qlipp_phase3D(S0, recon, method='Tikhonov', reg_re=1e-4,
                              rho=1e-3, lambda_re=1e-4, itr=50):

    phase3D = recon.Phase_recon_3D(S0, method=method, reg_re=reg_re, rho=rho, lambda_re=lambda_re,
                                   itr=itr, verbose=False)

    phase3D = np.transpose(phase3D, (2, 0, 1))

    return phase3D

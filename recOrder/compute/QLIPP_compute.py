import waveorder as wo
from waveorder.waveorder_reconstructor import waveorder_microscopy as setup
import numpy as np
import tifffile as tiff
import glob
import time


def initialize_reconstructor(image_dim, wavelength, swing, N_channel, NA_obj, NA_illu, mag, N_slices, z_step, pad_z,
                             pixel_size, bg_option='local_fit', n_media=1.0, use_gpu=False, gpu_id=0):
    """
    Compute 3D birefringence and phase from a single position

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
    N_pattern = 1
    N_defocus = N_slices + 2 * pad_z
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
                  A_matrix=inst_mat, QLIPP_birefringence_only=False, pad_z=pad_z,
                  phase_deconv='3D', illu_mode='BF', use_gpu=use_gpu, gpu_id=gpu_id)

    recon.N_channel = N_channel

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Initializing Reconstructor ({elapsed_time:0.2f} min)')

    return recon

def reconstruct_QLIPP_stokes(data, recon, bg_stokes):

    stokes_stack = []
    raw_stack = np.transpose(data[0:recon.N_channel], (1, 0, 2, 3))

    for z in range(data.shape[1]):
        stokes_data = recon.Stokes_recon(raw_stack[z])
        stokes_data = recon.Stokes_transform(stokes_data)

        if recon.bg_option != 'None':

            S_image_tm = recon.Polscope_bg_correction(stokes_data, bg_stokes)
            stokes_stack.append(S_image_tm)

        else:
            stokes_stack.append(stokes_data)

    return np.asarray(stokes_stack)

def reconstruct_QLIPP_birefringence(stokes, recon):

    if len(stokes.shape) == 4:
        slices = stokes.shape[0]
        recon_data = np.zeros([stokes.shape[0], 4, stokes.shape[-2], stokes.shape[-1]])
    elif len(stokes.shape) == 3:
        slices = 1
        recon_data = np.zeros([1, 4, stokes.shape[-2], stokes.shape[-1]])

    for z in range(slices):
        recon_data[z, :, :, :] = recon.Polarization_recon(stokes[z])

    return np.transpose(recon_data, (1,0,2,3))

def reconstruct_QLIPP_birefringence(position, recon, bg_stokes):

    start_time = time.time()
    # print('Computing Birefringence...')

    recon_data = np.zeros([position.shape[1], 4, position.shape[2], position.shape[3]])
    raw_stack = np.transpose(position[0:recon.N_channel], (1, 0, 2, 3))

    for z in range(position.shape[1]):
        stokes_data = recon.Stokes_recon(raw_stack[z])

        stokes_data = recon.Stokes_transform(stokes_data)

        if recon.bg_option != 'None':

            S_image_tm = recon.Polscope_bg_correction(stokes_data, bg_stokes)
            recon_data[z, :, :, :] = recon.Polarization_recon(S_image_tm)

        else:
            recon_data[z, :, :, :] = recon.Polarization_recon(stokes_data)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Computing Birefringence ({elapsed_time:0.2f} min)')

    return np.transpose(recon_data, (1,0,2,3))



def reconstruct_QLIPP_3D(position, bg_data, reconstructor, method='Tikhonov',
                         reg_re=1e-4, reg_im=1e-4, rho=1e-5,
                         lambda_re=1e-3, lambda_im=1e-3, itr=20):
    """

        Compute 3D birefringence and phase from a single position

        Parameters
        ----------
            position         : zarr array
                                zarr position array.  Should have dimensions [C, Z, Y, X]

            bg_path           : str
                                path to the folder containing background images

            reconstructor     : str
                                waveOrder.microscopy class to setup the reconstruction parameters

            method            : str
                                'Tikhonov' or 'TV'

            reg_re            : float
                                Tikhonov regularization parameter for 3D phase

            reg_im            : float
                                Tikhonov regularization parameter for 3D absorption

            rho               : float
                                augmented Lagrange multiplier for 3D ADMM algorithm

            lambda_re         : float
                                TV regularization parameter for 3D absorption

            lambda_im         : float
                                TV regularization parameter for 3D absorption

            itr               : int
                                number of iterations for 3D ADMM algorithm

        Returns
        -------
            retardance_stack  : numpy.ndarray
                                3D reconstruction of retardance (in the unit of nm) with the size of (Z, Y, X)

            orientation_stack : numpy.ndarray
                                3D reconstruction of orientation (in the unit of radian) with the size of (Z, Y, X)

            BF_stack          : numpy.ndarray
                                3D reconstruction of transmission with the size of (Z, Y, X)


            phase3d           : numpy.ndarray
                                3D reconstruction of phase (in the unit of radian) with the size of (Z, Y, X)


    """

    recon = reconstructor

    wavelength = recon.lambda_illu * recon.n_media * 1000

    start_time = time.time()
    print('Computing Birefringence...')

    #     bg_data = load_bg(bg_path, height, width)

    bg_stokes = recon.Stokes_recon(bg_data)
    bg_stokes = recon.Stokes_transform(bg_stokes)

    retardance_stack = np.zeros([position.shape[1], position.shape[-2], position.shape[-1]])
    orientation_stack = np.zeros([position.shape[1], position.shape[-2], position.shape[-1]])
    BF_stack = np.zeros([position.shape[1], position.shape[-2], position.shape[-1]])

    raw_stack = np.transpose(position[0:recon.N_channel], (1, 0, 2, 3))
    for z in range(position.shape[1]):
        stokes_data = recon.Stokes_recon(raw_stack[z])

        stokes_data = recon.Stokes_transform(stokes_data)

        if recon.bg_option != 'None':

            S_image_tm = recon.Polscope_bg_correction(stokes_data, bg_stokes)
            recon_data = recon.Polarization_recon(S_image_tm)
            retardance_stack[z, :, :] = recon_data[0] / (2 * np.pi) * wavelength
            orientation_stack[z, :, :] = recon_data[1]
            BF_stack[z, :, :] = recon_data[2]

        else:

            recon_data = recon.Polarization_recon(stokes_data)
            retardance_stack[z, :, :] = recon_data[0] / (2 * np.pi) * wavelength
            orientation_stack[z, :, :] = recon_data[1]
            BF_stack[z, :, :] = recon_data[2]

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Computing Birefringence ({elapsed_time:0.2f} min)')

    print('Computing 3d Phase...')

    S0_stack = np.transpose(BF_stack, (1, 2, 0))
    phase3d = recon.Phase_recon_3D(S0_stack, absorption_ratio=0.0, method=method, reg_re=reg_re, reg_im=reg_im, \
                                   rho=rho, lambda_re=lambda_re, lambda_im=lambda_im, itr=itr, verbose=True)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Reconstruction ({elapsed_time:0.2f} min)\n')
    return retardance_stack, orientation_stack, BF_stack, phase3d
from waveorder.waveorder_reconstructor import waveorder_microscopy
import numpy as np
import time


def initialize_reconstructor(pipeline, image_dim=None, wavelength_nm=None, swing=None, calibration_scheme=None,
                             NA_obj=None, NA_illu=None, mag=None, n_slices=None, z_step_um=None,
                             pad_z=0, pixel_size_um=None, bg_correction='local_fit', n_obj_media=1.0, mode='3D',
                             use_gpu=False, gpu_id=0):
    """
    Initialize the QLIPP reconstructor for downstream tasks. See tags next to parameters
    for which parameters are needed for each pipeline

    Parameters
    ----------

        pipeline          : string
                            'birefringence', 'QLIPP', 'PhaseFromBF'

        image_dim         : tuple
                            (height, width) of images in pixels

        wavelength_nm      : int
                            wavelength of illumination in nm

        swing             : float
                            swing used for calibration in waves

        calibration_scheme: str
                            '4-State' or '5-State'

        NA_obj            : float
                            numerical aperture of the detection objective

        NA_illu           : float
                            numerical aperture of the illumination condenser

        mag               : float
                            magnification used for imaging (e.g. 20 for 20x)

        n_slices          : int
                            number of slices in the z-stack

        z_step_um          : float
                            z step size of the image space

        pad_z             : float
                            how many padding slices to add for phase computation

        pixel_size_um     : float
                            pixel size of the camera in um

        bg_correction      : str
                            'local' for estimating background with scipy uniform filter
                            'local_fit' for estimating background with polynomial fit
                            'None' for no BG correction
                            'Global' for normal background subtraction with the provided background

        n_obj_media        : float
                            refractive index of the objective immersion media

        mode               : str
                            '2D' or '3D' (phase, fluorescence reconstruction only)

        use_gpu           : bool
                            option to use gpu or not

        gpu_id            : int
                            number refering to which gpu will be used


        Returns
        -------
            reconstructor     : object
                                reconstruction object initialized with reconstruction parameters

        """

    anisotropy_only = False

    if pipeline == 'QLIPP' or pipeline == 'PhaseFromBF':

        if not NA_obj:
            raise ValueError('Please specify NA_obj in function parameters')
        if not NA_illu:
            raise ValueError('Please specify NA_illu in function parameters')
        if not mag:
            raise ValueError('Please specify mag (magnification) in function parameters')
        if not wavelength_nm:
            raise ValueError('Please specify the wavelength for reconstruction')
        if not n_slices:
            raise ValueError('Please specify n_slices in function parameters')
        if not z_step_um:
            raise ValueError('Please specify z_step_um in function parameters')
        if not pixel_size_um:
            raise ValueError('Please specify NA_obj in function parameters')
        if not n_obj_media:
            raise ValueError('Please specify NA_obj in function parameters')
        if not image_dim:
            raise ValueError('Please specify image_dim in function parameters')

        if pipeline == 'QLIPP':
            if not calibration_scheme:
                raise ValueError('Please specify qlipp_scheme (calibration scheme) for QLIPP reconstruction')
            if not swing:
                raise ValueError('Please specify swing in function parameters')

    elif pipeline == 'birefringence':

        anisotropy_only = True

        if not calibration_scheme:
            raise ValueError('Please specify qlipp_scheme (calibration scheme) for QLIPP reconstruction')
        if not wavelength_nm:
            raise ValueError('Please specify the wavelength for QLIPP reconstruction')
        if not swing:
            raise ValueError('Please specify swing in function parameters')

    else:
        raise ValueError(f'Pipeline {pipeline} not understood')

    # Modify user inputs to fit waveorder input requirements
    lambda_illu = wavelength_nm / 1000 if wavelength_nm else None
    n_defocus = n_slices if n_slices else 0
    z_step_um = 0 if not z_step_um else z_step_um
    z_defocus = -(np.r_[:n_defocus] - n_defocus // 2) * z_step_um # assumes stack starts from the bottom
    ps = pixel_size_um / mag if pixel_size_um else None
    cali = True
    NA_obj = 0 if not NA_obj else NA_obj
    NA_illu = 0 if not NA_illu else NA_illu

    if calibration_scheme == '4-State':
        inst_mat = np.array([[1, 0, 0, -1],
                             [1, np.sin(2 * np.pi * swing), 0, -np.cos(2 * np.pi * swing)],
                             [1, -0.5 * np.sin(2 * np.pi * swing), np.sqrt(3) * np.cos(np.pi * swing) * np.sin(np.pi * swing),
                              -np.cos(2 * np.pi * swing)],
                             [1, -0.5 * np.sin(2 * np.pi * swing), -np.sqrt(3) / 2 * np.sin(2 * np.pi * swing),
                              -np.cos(2 * np.pi * swing)]])
        n_channel = 4

    elif calibration_scheme == '5-State':
        swing = swing * 2 * np.pi
        inst_mat = None
        n_channel = 5

    elif calibration_scheme == 'PhaseFromBF':
        inst_mat = None
        n_channel = 1
        swing = 0
    else:
        inst_mat = None
        n_channel = 1
        swing = 0

    print('Initializing Reconstructor...')
    start_time = time.time()
    print(bg_correction)
    recon = waveorder_microscopy(img_dim=image_dim,
                                 lambda_illu=lambda_illu,
                                 ps=ps,
                                 NA_obj=NA_obj,
                                 NA_illu=NA_illu,
                                 z_defocus=z_defocus,
                                 chi=swing,
                                 n_media=n_obj_media,
                                 cali=cali,
                                 bg_option=bg_correction,
                                 A_matrix=inst_mat,
                                 QLIPP_birefringence_only=anisotropy_only,
                                 pad_z=pad_z,
                                 phase_deconv=mode,
                                 illu_mode='BF',
                                 use_gpu=use_gpu,
                                 gpu_id=gpu_id)

    recon.N_channel = n_channel

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Initializing Reconstructor ({elapsed_time:0.2f} min)')

    return recon

def reconstruct_qlipp_stokes(data, recon, bg_stokes=None):
    """
    From intensity data, use the waveorder.waveorder_microscopy (recon) to build a stokes array
        if recon background correction flag is selected, will also perform backgroudn correction

    Parameters
    ----------
        data                : np.ndarray or zarr array
                              intensity data of shape: (C, Z, Y, X)

        recon               : waveorder.waveorder_microscopy object
                              initialized by initialize_reconstructor

        bg_stokes           : np.ndarray (5, Y, X) or (4, Y, X)
                              stokes array representing background data

    Returns
    -------
        stokes_stack        : np.ndarray
                              array representing stokes array
                              has shape: (C, Z, Y, X), where C = 5
                              or (C, Y, X) if not reconstructing a z-stack
    """

    stokes_data = recon.Stokes_recon(np.copy(data))
    stokes_data = recon.Stokes_transform(stokes_data)

    # Don't do background correction if BG data isn't provided
    if recon.bg_option == 'None':
        return stokes_data # C(Z)YX

    # Compute Stokes with background correction
    else:
        if len(np.shape(stokes_data)) == 4:
            s_image = recon.Polscope_bg_correction(np.transpose(stokes_data, (-4, -2, -1, -3)), bg_stokes)
            s_image = np.transpose(s_image, (0, 3, 1, 2)) # Tranpose to CZYX
        else:
            s_image = recon.Polscope_bg_correction(stokes_data, bg_stokes) # CYX

        return s_image

def reconstruct_qlipp_birefringence(stokes, recon):
    """
    From stokes data, use waveorder.waveorder_microscopy (recon) to build a birefringence array

    Parameters
    ----------
    stokes                  : np.ndarray or zarr array
                              stokes array generated by reconstruct_qlipp_stokes
                              dimensions: (C, Z, Y, X) or (C, Y, X)

    recon                   : waveorder.waveorder_microscopy object
                              initialized by initialize_reconstructor

    Returns
    -------
        recon_data          : np.ndarray
                              volume of shape (C, Z, Y, X) or (C, Y, X) containing reconstructed birefringence data.
    """

    if stokes.ndim == 4:
        stokes = np.transpose(stokes, (0, 2, 3, 1))
    elif stokes.ndim == 3:
        pass
    else:
        raise ValueError(f'Incompatible stokes dimension: {stokes.shape}')

    birefringence = recon.Polarization_recon(np.copy(stokes))

    # Return the transposed birefringence array with channel first
    return np.transpose(birefringence, (-4, -1, -3, -2)) if len(birefringence.shape) == 4 else birefringence


def reconstruct_phase2D(S0, recon, method='Tikhonov', reg_p=1e-4, rho=1,
                        lambda_p=1e-4, itr=50):
    """
    Reconstruct 2D phase from a given S0 or BF stack.

    Parameters
    ----------
    S0:             (nd-array) BF/S0 stack of dimensions (Z, Y, X)
    recon:          (waveorder_microscopy Object): initialized reconstructor object
    method:         (str) Regularization method 'Tikhonov' or 'TV'
    reg_p:          (float) Tikhonov regularization parameters
    rho:            (float) TV regularization parameter
    lambda_p:       (float) TV regularization parameter
    itr:            (int) TV Regularization number of iterations

    Returns
    -------
    phase2D:        (nd-array) Phase2D image of size (Y, X)

    """

    S0 = np.transpose(S0, (1, 2, 0))
    _, phase2D = recon.Phase_recon(np.copy(S0).astype('float'), method=method, reg_p=reg_p, rho=rho, lambda_p=lambda_p, itr=itr, verbose=False)

    return phase2D


def reconstruct_phase3D(S0, recon, method='Tikhonov', reg_re=1e-4,
                        rho=1e-3, lambda_re=1e-4, itr=50):
    """
    Reconstruct 2D phase from a given S0 or BF stack.

    Parameters
    ----------
    S0:             (nd-array) BF/S0 stack of dimensions (Z, Y, X)
    recon:          (waveorder_microscopy Object): initialized reconstructor object
    method:         (str) Regularization method 'Tikhonov' or 'TV'
    reg_p:          (float) Tikhonov regularization parameters
    rho:            (float) TV regularization parameter
    lambda_p:       (float) TV regularization parameter
    itr:            (int) TV Regularization number of iterations

    Returns
    -------
    phase3D:        (nd-array) Phase2D image of size (Z, Y, X)

    """

    S0 = np.transpose(S0, (1, 2, 0))
    phase3D = recon.Phase_recon_3D(np.copy(S0).astype('float'), method=method, reg_re=reg_re, rho=rho, lambda_re=lambda_re,
                                   itr=itr, verbose=False)

    phase3D = np.transpose(phase3D, (-1, -3, -2))

    return phase3D

class QLIPPBirefringenceCompute:
    """
    Convenience Class for computing QLIPP birefringence only.
    """

    def __init__(self, shape, scheme, wavelength, swing, n_slices, bg_option, bg_data=None):

        self.shape = shape
        self.scheme = scheme
        self.wavelength = wavelength
        self.swing = swing
        self.n_slices = n_slices
        self.bg_option = bg_option

        self.reconstructor = initialize_reconstructor(pipeline='birefringence',
                                                      image_dim=self.shape,
                                                      calibration_scheme=self.scheme,
                                                      wavelength_nm=self.wavelength,
                                                      swing=self.swing,
                                                      bg_correction=self.bg_option,
                                                      n_slices=self.n_slices)

        if bg_option != 'None':
            self.bg_stokes = reconstruct_qlipp_stokes(bg_data, self.reconstructor)
        else:
            self.bg_stokes = None

    def reconstruct(self, array):
        """
        reconstructs raw data into birefringence data

        Parameters
        ----------
        array:          (nd-array) of image shape (C, Z, Y, X) or (C, Y, X) with minimum C=4

        Returns
        -------
        birefringence:  (nd-array) birefringence array of size (2, Z, Y, X) or (2, Y, X)
                                    first channel is retardance [nm] second channel is orientation [0, pi]

        """
        stokes = reconstruct_qlipp_stokes(array, self.reconstructor, self.bg_stokes)
        birefringence = reconstruct_qlipp_birefringence(stokes, self.reconstructor)
        birefringence[0] = birefringence[0] / (2 * np.pi) * self.wavelength

        return birefringence[0:2]

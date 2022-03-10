from waveorder.waveorder_reconstructor import fluorescence_microscopy
from recOrder.io.utils import get_unimodal_threshold
import time
import numpy as np


def initialize_fluorescence_reconstructor(img_dim, wavelength_nm, pixel_size_um, z_step_um, NA_obj, magnification, mode,
                                          n_obj_media=1.0, pad_z=0, use_gpu=False, gpu_id=0):

    """
    Initialize the fluorescence_deconvolution reconstructor for downstream tasks. See tags next to parameters
    for which parameters are needed for each pipeline

    Parameters
    ----------
    img_dim:            (tuple) image dimensions of (Y, X, Z)
    wavelength_nm:      (list or int) wavelength of the fluorescence emission(s) in nanometer
    pixel_size_um:      (float) pixel size of the camera in um
    z_step_um:          (float) step size of the z-stack in um
    NA_obj:             (float) Numerical Aperture of the objective
    magnification:      (float) Effective magnification
    mode:               (str) '2D' or '3D' depending on what type of reconstruction you desire
    n_obj_media:        (float) Refractive index of objective immersion media
    pad_z:              (int) Z-Padding to account for edge artifacts
    use_gpu:            (bool) True/False
    gpu_id:             (int) GPU ID

    Returns
    -------
    reconstructor:      (waveorder.fluorescence_microscopy) Reconstructor object

    """

    if mode != '2D' and mode != '3D':
        raise ValueError(f'mode {mode} not understood.  Please specify "2D" or "3D"')

    deconv_mode = '2D-WF' if mode == '2D' else '3D-WF'

    # account for lists, singular values + convert to numpy array
    if not isinstance(wavelength_nm, list):
        if isinstance(wavelength_nm, float) or isinstance(wavelength_nm, int):
            wavelength_nm = np.asarray([wavelength_nm])
        else:
            raise ValueError('wavelength_nm must be a list of floats/ints or singular float/int')
    else:
        wavelength_nm = np.asarray(wavelength_nm)


    print('Initializing Reconstructor...')
    start_time = time.time()

    reconstructor = fluorescence_microscopy(img_dim=img_dim,
                                            lambda_emiss=wavelength_nm/1000,
                                            ps=pixel_size_um/magnification,
                                            psz=z_step_um,
                                            NA_obj=NA_obj,
                                            n_media=n_obj_media,
                                            deconv_mode=deconv_mode,
                                            pad_z=pad_z,
                                            use_gpu=use_gpu,
                                            gpu_id=gpu_id)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished Initializing Reconstructor ({elapsed_time:0.2f} min)')

    return reconstructor


#TODO: figure out robust background correction method
def calculate_background(data):
    """
    Calculate the background with uni-modal thresholding

    Parameters
    ----------
    data:           (nd-array) data array of size (N_fluor, Y, X)

    Returns
    -------
    background_vals:    (list) list of average background values of size [N_fluor]
    """

    if data.ndim == 3:
        fluors = data.shape[0]
    elif data.ndim == 2:
        fluors = 1
        data = data[np.newaxis, :, :]
    else:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Y, X) or (Y, X)')

    background_vals = []
    for i in range(fluors):
        background_vals.append(get_unimodal_threshold(data[i]))

    return background_vals


def deconvolve_fluorescence_2D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):
    """
    Deconvolve 2D fluorescence image(s).  Will loop through multiple fluorescence channels

    Parameters
    ----------
    data:               (nd-array) 2D data of size (N_fluor, Y, X) or (Y, X)
    reconstructor:      (waveorder.fluorescence_microscopy) Initialized reconstructor object
    bg_level:           (list) list of background levels of size [N_Fluor] or 1
    reg:                (list) list of regularization parameters of size [N_Fluor] or 1

    Returns
    -------
    deconvolved_data:   (nd-array) of size (N_fluor, Y, X) or (Y, X)

    """

    if data.ndim > 3:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Z, Y, X) or (Z, Y, X)')

    deconvolved_data = reconstructor.deconvolve_fluor_2D(data, bg_level, reg)

    return deconvolved_data


def deconvolve_fluorescence_3D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):
    """
    Deconvolve 3D fluorescence volume(s).  Will loop through multiple fluorescence channels

    Parameters
    ----------
    data:               (nd-array) 2D data of size (N_fluor, Z, Y, X) or (Z, Y, X)
    reconstructor:      (waveorder.fluorescence_microscopy) Initialized reconstructor object
    bg_level:           (list) list of background levels of size [N_Fluor] or 1
    reg:                (list) list of regularization parameters of size [N_Fluor] or 1

    Returns
    -------
    deconvolved_data:   (nd-array) of size (N_fluor, Z, Y, X) or (Z, Y, X)
    """

    if data.ndim == 4:
        data_process = np.transpose(data, (-4, -2, -1, -3))
    elif data.ndim == 3:
        data_process = np.transpose(data, (-2, -1, -3))
    else:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Z, Y, X) or (Z, Y, X)')

    deconvolved_data = reconstructor.deconvolve_fluor_3D(data_process, bg_level, reg)

    if deconvolved_data.ndim == 4:
        deconvolved_data = np.transpose(deconvolved_data, (0, 3, 1, 2))
    elif deconvolved_data.ndim == 3:
        deconvolved_data = np.transpose(deconvolved_data, (2, 0, 1))
    else:
        raise ValueError(f'Unexpected output dimensions: {deconvolved_data.shape}')

    return deconvolved_data

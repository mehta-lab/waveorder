from waveorder.waveorder_reconstructor import fluorescence_microscopy
from recOrder.preproc.pre_processing import get_autocontrast_limits
import time
import numpy as np


def initialize_fluorescence_reconstructor(img_dim, wavelength_nm, pixel_size_um, z_step_um, NA_obj, mode,
                                          n_obj_media=1.0, pad_z=0, use_gpu=False, gpu_id=0):

    # transform into waveorder standards
    img_dim = (img_dim[1], img_dim[2], img_dim[0])

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
                                            ps=pixel_size_um,
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

    if data.ndim == 3:
        fluors = data.shape[0]
    elif data.ndim == 2:
        fluors = 1
        data = data[np.newaxis, :, :]
    else:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Z, Y, X) or (Z, Y, X)')

    background_vals = []
    for i in range(fluors):
        min_, max_ = get_autocontrast_limits(data[i], clip=0.01)
        background_vals.append(np.average(data[i], weights=(data[i] < min_)))

    return background_vals


def deconvolve_fluorescence_2D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):

    if data.ndim > 3:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Z, Y, X) or (Z, Y, X)')

    deconvolved_data = reconstructor.deconvolve_fluor_2D(data, bg_level, reg)

    return deconvolved_data


def deconvolve_fluorescence_3D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):

    if data.ndim == 4:
        data_process = np.transpose(data, (0, 2, 3, 1))
    elif data.ndim == 3:
        data_process = np.transpose(data, (1, 2, 0))
    else:
        raise ValueError('invalid input data dimensions.  Data must be (N_fluor, Z, Y, X) or (Z, Y, X)')

    deconvolved_data = reconstructor.deconvolve_fluor_3D(data_process, bg_level, reg)

    return deconvolved_data

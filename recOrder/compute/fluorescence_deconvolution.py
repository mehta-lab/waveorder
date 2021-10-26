from waveorder.waveorder_reconstructor import fluorescence_microscopy
from recOrder.preproc.pre_processing import find_focus
import time


def initialize_fluorescence_reconstructor(img_dim, wavelength_nm, pixel_size_um, z_step_um, NA_obj, mode,
                                          n_obj_media=1.0, pad_z=0, use_gpu=False, gpu_id=0):

    if mode != '2D' and mode != '3D':
        raise ValueError(f'mode {mode} not understood.  Please specify "2D" or "3D"')

    deconv_mode = '2D-WF' if mode == '2D' else '3D-WF'

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


    background_values = []
    for volume in range(data.shape[0]):
        max, _ = find_focus(data[volume])



    pass

def deconvolve_fluorescence_2D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):

    deconvolved_data = reconstructor.deconvolve_fluor_2D(data, bg_level, reg)

    return deconvolved_data


def deconvolve_fluorescence_3D(data, reconstructor: fluorescence_microscopy, bg_level, reg=1e-4):

    deconvolved_data = reconstructor.deconvolve_fluor_3D(data, bg_level, reg)

    return deconvolved_data




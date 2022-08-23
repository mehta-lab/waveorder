from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.postproc.post_processing import post_proc_denoise
from recOrder.compute.fluorescence_compute import calculate_background, deconvolve_fluorescence_3D
from recOrder.preproc.pre_processing import preproc_denoise
from recOrder.compute.qlipp_compute import reconstruct_qlipp_stokes, reconstruct_qlipp_birefringence
from os.path import dirname, abspath
import numpy as np
import os
import zarr

def test_pre_processing(get_ometiff_data_dir, setup_data_save_folder):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/config_preprocessing_pytest.yml')

    config = ConfigReader(path_to_config, data_dir=ometiff_data, save_dir=save_folder)

    manager = PipelineManager(config)

    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    params = [['S0', 0.5, 1], ['S1', 0.5, 1], ['S2', 0.5, 1], ['S3', 0.5, 1]]
    stokes_denoise = preproc_denoise(stokes, params)

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_001']['arr_0']

    # Check Stokes
    assert (np.sum(np.abs(stokes_denoise[0, z, :, :] - array[0, 0, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[0, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[1, z, :, :] - array[0, 1, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[1, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[2, z, :, :] - array[0, 2, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[2, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[3, z, :, :] - array[0, 3, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[3, z, :, :])) ** 2 < 0.1)


def test_post_processing(get_ometiff_data_dir, setup_data_save_folder):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'test_configs/config_postprocessing_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=ometiff_data, save_dir=save_folder)

    manager = PipelineManager(config)
    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)

    birefringence = reconstruct_qlipp_birefringence(stokes, recon)
    params = ['Retardance', 0.1, 1]
    ret_denoise = post_proc_denoise(birefringence[0], params)
    ret_denoise = ret_denoise / (2*np.pi)*config.wavelength

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_001']['arr_0']

    data_decon = np.asarray([data[t, 1], data[t, 2]])
    bg_level = calculate_background(data_decon[:, z])
    fluor3D = deconvolve_fluorescence_3D(data_decon, manager.deconv_reconstructor, bg_level, reg=[1e-4, 1e-4])

    # Check Birefringence
    assert(np.sum(np.abs(ret_denoise[z] - array[0, 0, z]) ** 2) / np.sum(np.abs(ret_denoise[z])) < 0.1)

    # Check deconvolution
    assert(np.sum(np.abs(array[0, 2, z] - fluor3D[1, z]) ** 2) / np.sum(np.abs(array[0, 2, z]) ** 2) < 0.1)

    # Check Registration
    assert(np.sum(np.abs(array[0, 1, z, 100:, 100:] - fluor3D[0, z, 0:-100, 0:-100])**2)
           / np.sum(np.abs(array[0, 1, z, 100:, 100:])**2) < 0.1)
    assert(np.mean(array[0, 1, z, 0:100, 0:100]) == 0.0)

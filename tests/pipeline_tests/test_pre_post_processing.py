from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.postproc.post_processing import post_proc_denoise, translate_3D
from recOrder.preproc.pre_processing import preproc_denoise
from recOrder.compute.qlipp_compute import reconstruct_qlipp_stokes, reconstruct_qlipp_birefringence
from os.path import dirname, abspath
import numpy as np
import os
import zarr

def test_pre_processing(setup_test_data, setup_data_save_folder):
    folder, data = setup_test_data
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/config_pre_post_processing_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=data, save_dir=save_folder)

    manager = PipelineManager(config)

    manager.run()

    pos, t, z = 1, 0, 40
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    params = [['S0', 0.5, 1], ['S1', 0.5, 1], ['S2', 0.5, 1], ['S3', 0.5, 1]]
    stokes_denoise = preproc_denoise(stokes, params)

    store = zarr.open(os.path.join(save_folder, '2T_3P_81Z_231Y_498X_Kazansky_2.zarr'))
    array = store['Pos_001.zarr']['physical_data']['array']

    # Check Stokes
    assert (np.sum(np.abs(stokes_denoise[z, 0] - array[0, 0, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[z, 0])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[z, 1] - array[0, 1, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[z, 1])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[z, 2] - array[0, 2, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[z, 2])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes_denoise[z, 3] - array[0, 3, z]) ** 2) / np.sum(
        np.abs(stokes_denoise[z, 3])) ** 2 < 0.1)


def test_post_processing(setup_test_data, setup_data_save_folder):
    folder, data = setup_test_data
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'test_configs/config_pre_post_processing_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=data, save_dir=save_folder)

    manager = PipelineManager(config)
    manager.run()

    pos, t, z = 1, 0, 40
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    params = [['S0', 0.5, 1], ['S1', 0.5, 1], ['S2', 0.5, 1], ['S3', 0.5, 1]]
    stokes_denoise = preproc_denoise(stokes, params)

    birefringence = reconstruct_qlipp_birefringence(stokes_denoise, recon)
    params = ['Retardance', 5, 1]
    birefringence_denoise = post_proc_denoise(birefringence[0], params)

    store = zarr.open(os.path.join(save_folder, '2T_3P_81Z_231Y_498X_Kazansky_2.zarr'))
    array = store['Pos_001.zarr']['physical_data']['array']

    # Check Birefringence
    assert(np.sum(np.abs((birefringence_denoise[0, z]/(2 * np.pi)*config.wavelength) - array[0, 4, z]) ** 2)
           / np.sum(np.abs(birefringence_denoise[0, z]/(2 * np.pi)*config.wavelength)**2) < 0.1)

    # Check Registration
    assert(array[0, 0, z, 100:, 100:] == (birefringence_denoise[0, z]/(2 * np.pi)*config.wavelength)[0:-100,0:-100])
    assert(array[0, 0, z, 0:100, 0:100] == np.zeros_like(array[0, 0, z, 0:100, 0:100]))

from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.compute.qlipp_compute import reconstruct_qlipp_stokes, reconstruct_qlipp_birefringence
from os.path import dirname, abspath
import numpy as np
import os
import zarr

def test_old_processing(get_ometiff_data_dir, setup_data_save_folder):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/config_old_processing_pytest.yml')

    config = ConfigReader(path_to_config, data_dir=ometiff_data, save_dir=save_folder)

    manager = PipelineManager(config)

    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    params = [['S0', 0.5, 1], ['S1', 0.5, 1], ['S2', 0.5, 1], ['S3', 0.5, 1]]
    
    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_001']['arr_0']

    # Check Stokes
    assert (np.sum(np.abs(stokes[0, z, :, :] - array[0, 0, z]) ** 2) / np.sum(
        np.abs(stokes[0, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes[1, z, :, :] - array[0, 1, z]) ** 2) / np.sum(
        np.abs(stokes[1, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes[2, z, :, :] - array[0, 2, z]) ** 2) / np.sum(
        np.abs(stokes[2, z, :, :])) ** 2 < 0.1)
    assert (np.sum(np.abs(stokes[3, z, :, :] - array[0, 3, z]) ** 2) / np.sum(
        np.abs(stokes[3, z, :, :])) ** 2 < 0.1)
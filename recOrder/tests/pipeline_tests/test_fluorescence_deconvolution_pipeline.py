from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.pipelines.fluor_deconv_pipeline import FluorescenceDeconvolution
from waveorder.io.writer import WaveorderWriter
from recOrder.compute.fluorescence_compute import calculate_background, deconvolve_fluorescence_3D, \
    deconvolve_fluorescence_2D

from os.path import dirname, abspath
import numpy as np
import os
import zarr


def test_pipeline_manager_initiate(init_fluor_decon_pipeline_manager):

    save_folder, config, manager = init_fluor_decon_pipeline_manager

    assert(manager.config is not None)
    assert(manager.data is not None)
    assert(manager.data.get_num_positions()*manager.data.frames == len(manager.pt_set))
    assert(manager.pipeline is not None)
    assert(isinstance(manager.pipeline, FluorescenceDeconvolution))


def test_fluor_decon_pipeline_initiate(init_fluor_decon_pipeline_manager):

    save_folder, config, manager = init_fluor_decon_pipeline_manager

    pipeline = manager.pipeline
    assert(pipeline.config == manager.config)
    assert(pipeline.data == manager.data)
    assert(pipeline.t == manager.num_t)
    assert(pipeline.mode == '3D')
    assert(pipeline.slices == manager.data.slices)
    assert(pipeline.img_dim == (manager.data.height, manager.data.width, manager.data.slices))
    assert(pipeline.fluor_idxs == [0, 1])
    assert(pipeline.data_shape == (manager.data.frames, len(config.output_channels),
                                   manager.data.slices, manager.data.height, manager.data.width))
    assert(pipeline.chunk_size == (1, 1, 1, manager.data.height, manager.data.width))
    assert(isinstance(pipeline.writer, WaveorderWriter))
    assert(pipeline.reconstructor is not None)


def test_pipeline_manager_run(init_fluor_decon_pipeline_manager):

    save_folder, config, manager = init_fluor_decon_pipeline_manager
    manager.run()

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr'))
    array = store['Row_0']['Col_0']['Pos_000']['arr_0']

    assert (store.attrs.asdict()['Config'] == config.yaml_dict)
    assert (store['Row_0']['Col_0']['Pos_000'])
    assert (store['Row_0']['Col_1']['Pos_001'])
    assert (store['Row_0']['Col_2']['Pos_002'])
    assert (array.shape == (2, 2, 16, manager.data.height, manager.data.width))


def test_3D_reconstruction(get_zarr_data_dir, setup_data_save_folder):

    folder, zarr_data = get_zarr_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/fluor_deconv/config_fluor_3D_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=zarr_data, save_dir=save_folder)

    manager = PipelineManager(config)
    assert(manager.pipeline.mode == '3D')
    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)
    bg_level = calculate_background(data[t, 0, manager.data.slices // 2])
    recon = manager.pipeline.reconstructor

    fluor3D = deconvolve_fluorescence_3D(data[t, 0], recon, bg_level, [config.reg])
    # fluor3D = np.transpose(fluor3D, (-1, -3, -2))

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr'), 'r')
    array = store['Row_0']['Col_1']['Pos_001']['arr_0']

    # Check Shape
    assert(array.shape == (1, len(config.output_channels), 16, 128, 256))

    # Check deconvolved fluor
    assert (np.sum(np.abs(fluor3D[z] - array[0, 0, z]) ** 2) / np.sum(np.abs(fluor3D[z])**2) < 0.1)


def test_2D_reconstruction(get_zarr_data_dir, setup_data_save_folder):

    folder, zarr_data = get_zarr_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/fluor_deconv/config_fluor_2D_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=zarr_data, save_dir=save_folder)

    manager = PipelineManager(config)
    assert(manager.pipeline.slices == 1)
    assert(manager.pipeline.mode == '2D')
    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)[t, 0, z]
    data = np.expand_dims(data, axis=0)
    bg_level = calculate_background(data)
    recon = manager.pipeline.reconstructor

    fluor2D = deconvolve_fluorescence_2D(data, recon, bg_level, reg=[config.reg])
    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr'), 'r')
    array = store['Row_0']['Col_1']['Pos_001']['arr_0']

    # Check Shapes
    assert(array.shape == (1, len(config.output_channels), 1, 128, 256))

    # Check Deconvolved Fluor
    assert (np.sum(np.abs(fluor2D - array[0, 0, 0]) ** 2) / np.sum(np.abs(fluor2D)**2) < 0.1)


def test_deconvolution_and_registration(get_zarr_data_dir, setup_data_save_folder):

    folder, zarr_data = get_zarr_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))),
                                  'test_configs/fluor_deconv/config_fluor_full_registration_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=zarr_data, save_dir=save_folder)

    manager = PipelineManager(config)
    manager.run()

    pos, t, z = 1, 0, manager.data.slices // 2
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    data_decon = np.asarray([data[t, 0], data[t, 1]])
    bg_level = calculate_background(data_decon[:, z])
    fluor3D = deconvolve_fluorescence_3D(data_decon, recon, bg_level, reg=[config.reg]*2)
    # fluor3D = np.transpose(fluor3D, (-4, -1, -3, -2))

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr'), 'r')
    array = store['Row_0']['Col_1']['Pos_001']['arr_0']

    # Check Registration - Chan0
    assert(np.sum(np.abs(array[0, 0, z, 100:, 100:] - fluor3D[0, z, 0:-100, 0:-100])**2)
           / np.sum(np.abs(array[0, 0, z, 100:, 100:])**2) < 0.1)
    assert(np.mean(array[0, 0, z, 0:100, 0:100]) == 0.0)

    # Check Registration - Chan1
    assert(np.sum(np.abs(array[0, 1, z, 100:, 100:] - fluor3D[1, z, 0:-100, 0:-100])**2)
           / np.sum(np.abs(array[0, 1, z, 100:, 100:])**2) < 0.1)
    assert(np.mean(array[0, 1, z, 0:100, 0:100]) == 0.0)

    # Check Registration - Chan2
    assert(np.sum(np.abs(array[0, 2, z, 100:, 100:] - data[t, 2, z, 0:-100, 0:-100])**2)
           / np.sum(np.abs(array[0, 2, z, 100:, 100:])**2) < 0.1)
    assert(np.mean(array[0, 2, z, 0:100, 0:100]) == 0.0)

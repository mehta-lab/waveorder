from recOrder.io.config_reader import ConfigReader
from recOrder.io import MetadataReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.pipelines.qlipp_pipeline import QLIPP
from waveorder.io.writer import WaveorderWriter
from recOrder.compute.qlipp_compute import reconstruct_qlipp_stokes, reconstruct_qlipp_birefringence, \
    reconstruct_phase3D, reconstruct_phase2D
from os.path import dirname, abspath
import numpy as np
import os
import zarr

def test_pipeline_manager_initiate(init_qlipp_tiff_pipeline_manager):

    save_folder, config, manager = init_qlipp_tiff_pipeline_manager

    assert(manager.config is not None)
    assert(manager.data is not None)

    # check to make sure position mapping is correct
    list_p = []
    for i in manager.pt_set:
        if i[0] not in list_p:
            list_p.append(i[0])
    list_p.sort()
    for idx, p in enumerate(list_p):
        assert(manager.indices_map[p] == idx)

    assert(manager.data.get_num_positions()*manager.data.frames == len(manager.pt_set))
    assert(manager.pipeline is not None)
    assert(isinstance(manager.pipeline, QLIPP))

def test_qlipp_pipeline_initiate(init_qlipp_tiff_pipeline_manager):

    save_folder, config, manager = init_qlipp_tiff_pipeline_manager

    pipeline = manager.pipeline
    assert(pipeline.config == manager.config)
    assert(pipeline.data == manager.data)
    assert(pipeline.t == manager.num_t)
    assert(isinstance(pipeline.writer, WaveorderWriter))
    assert(pipeline.mode == '3D')
    assert(pipeline.no_birefringence == False)
    assert(pipeline.slices == manager.data.slices)
    assert(pipeline.img_dim == (manager.data.height, manager.data.width, manager.data.slices))
    assert(pipeline.chan_names == manager.data.channel_names)
    assert(isinstance(pipeline.calib_meta, MetadataReader))
    assert(pipeline.bg_path == manager.config.background)

    assert(pipeline.s0_idx == 0)
    assert(pipeline.s1_idx == 1)
    assert(pipeline.s2_idx == 2)
    assert(pipeline.s3_idx == 3)
    assert(pipeline.fluor_idxs == [])
    assert(pipeline.data_shape == (manager.data.frames, manager.data.channels,
                                   manager.data.slices, manager.data.height, manager.data.width))
    assert(pipeline.chunk_size == (1, 1, 1, manager.data.height, manager.data.width))
    assert(isinstance(pipeline.writer, WaveorderWriter))
    assert(pipeline.reconstructor is not None)
    assert(pipeline.bg_stokes is not None)

def test_pipeline_manager_run(init_qlipp_tiff_pipeline_manager):

    save_folder, config, manager = init_qlipp_tiff_pipeline_manager
    manager.run()

    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_000']['arr_0']

    assert(store.attrs.asdict()['Config'] == config.yaml_dict)
    assert(store['Row_0']['Col_0']['Pos_000'])
    assert(store['Row_0']['Col_1']['Pos_001'])
    assert(store['Row_0']['Col_2']['Pos_002'])
    assert(array.shape == (2, 4, 16, manager.data.height, manager.data.width))

def test_3D_reconstruction(get_ometiff_data_dir, setup_data_save_folder):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'test_configs/qlipp/config_qlipp_full_recon_pytest_tiff.yml')
    config = ConfigReader(path_to_config, data_dir=ometiff_data, save_dir=save_folder)

    manager = PipelineManager(config)
    assert(manager.pipeline.mode == '3D')
    manager.run()

    pos, t, z = 1, 0, 8
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    birefringence = reconstruct_qlipp_birefringence(stokes, recon)
    phase3D = reconstruct_phase3D(stokes[0], recon, method=config.phase_denoiser_3D, reg_re=config.Tik_reg_ph_3D,
                                  rho=config.rho_3D, lambda_re=config.TV_reg_ph_3D, itr=config.itr_3D)
    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_001']['arr_0']

    # Check Shape
    assert(array.shape == (1, len(config.output_channels), 16, 128, 256))

    # Check Stokes
    assert(np.sum(np.abs(stokes[0, z, :, :] - array[0, 4, z]) ** 2) / np.sum(np.abs(stokes[0, z, :, :])**2) < 0.1)
    assert(np.sum(np.abs(stokes[1, z, :, :] - array[0, 5, z]) ** 2) / np.sum(np.abs(stokes[1, z, :, :])**2) < 0.1)
    assert(np.sum(np.abs(stokes[2, z, :, :] - array[0, 6, z]) ** 2) / np.sum(np.abs(stokes[2, z, :, :])**2) < 0.1)
    assert(np.sum(np.abs(stokes[3, z, :, :] - array[0, 7, z]) ** 2) / np.sum(np.abs(stokes[3, z, :, :])**2) < 0.1)

    # Check Birefringence
    assert(np.sum(np.abs((birefringence[0, z]/(2 * np.pi)*config.wavelength) - array[0, 0, z]) ** 2)
           / np.sum(np.abs(birefringence[0, z]/(2 * np.pi)*config.wavelength)**2) < 0.1)
    assert (np.sum(np.abs(birefringence[1, z] - array[0, 1, z]) ** 2) / np.sum(np.abs(birefringence[1, z])**2) < 0.1)
    assert (np.sum(np.abs(birefringence[2, z] - array[0, 2, z]) ** 2) / np.sum(np.abs(birefringence[2, z])**2) < 0.1)

    # Check Phase
    assert (np.sum(np.abs(phase3D[z] - array[0, 3, z]) ** 2) / np.sum(np.abs(phase3D[z])**2) < 0.1)


def test_2D_reconstruction(get_ometiff_data_dir, setup_data_save_folder):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'test_configs/qlipp/config_qlipp_2D_pytest_tiff.yml')
    config = ConfigReader(path_to_config, data_dir=ometiff_data, save_dir=save_folder)

    manager = PipelineManager(config)
    assert(manager.pipeline.mode == '2D')
    manager.run()

    pos, t, z = 1, 0, manager.pipeline.focus_slice
    data = manager.data.get_array(pos)
    recon = manager.pipeline.reconstructor

    stokes = reconstruct_qlipp_stokes(data[t], recon, manager.pipeline.bg_stokes)
    birefringence = reconstruct_qlipp_birefringence(stokes[:, z, :, :], recon)
    phase2D = reconstruct_phase2D(stokes[0], recon, method=config.phase_denoiser_2D, reg_p=config.Tik_reg_ph_2D,
                                  rho=config.rho_2D, lambda_p=config.TV_reg_ph_2D, itr=config.itr_2D)
    store = zarr.open(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_1.zarr'), 'r')
    array = store['Row_0']['Col_0']['Pos_001']['arr_0']

    # Check Shapes
    assert(array.shape == (1, len(config.output_channels), 1, 128, 256))

    # Check Stokes
    assert(np.sum(np.abs(stokes[0, z, :, :] - array[0, 4, 0]) ** 2) / np.sum(np.abs(stokes[0, z, :, :]))**2 < 0.1)
    assert(np.sum(np.abs(stokes[1, z, :, :] - array[0, 5, 0]) ** 2) / np.sum(np.abs(stokes[1, z, :, :]))**2 < 0.1)
    assert(np.sum(np.abs(stokes[2, z, :, :] - array[0, 6, 0]) ** 2) / np.sum(np.abs(stokes[2, z, :, :]))**2 < 0.1)
    assert(np.sum(np.abs(stokes[3, z, :, :] - array[0, 7, 0]) ** 2) / np.sum(np.abs(stokes[3, z, :, :]))**2 < 0.1)

    # Check Birefringence

    assert(np.sum(np.abs((birefringence[0]/(2 * np.pi)*config.wavelength) - array[0, 0, 0]) ** 2)
           / np.sum(np.abs(birefringence[0]/(2 * np.pi)*config.wavelength)**2) < 0.1)
    assert (np.sum(np.abs(birefringence[1] - array[0, 1, 0]) ** 2) / np.sum(np.abs(birefringence[1])**2) < 0.1)
    assert (np.sum(np.abs(birefringence[2] - array[0, 2, 0]) ** 2) / np.sum(np.abs(birefringence[2])**2) < 0.1)

    # Check Phase
    assert (np.sum(np.abs(phase2D - array[0, 3, 0]) ** 2) / np.sum(np.abs(phase2D)**2) < 0.1)
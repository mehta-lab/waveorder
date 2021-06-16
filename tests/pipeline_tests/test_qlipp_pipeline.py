import pytest
from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager
from recOrder.pipelines.QLIPP_Pipeline import qlipp_pipeline
from waveorder.io.writer import WaveorderWriter
from os.path import dirname, abspath
import os
import zarr
import shutil

def test_pipeline_daemon_initiate(setup_test_data):

    folder, data = setup_test_data

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'config_qlipp_full_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=data, save_dir=folder)

    manager = PipelineManager(config)
    assert(manager.config is not None)
    assert(manager.data is not None)
    assert(manager.data.get_num_positions()*manager.data.frames == len(manager.pt_set))
    assert(manager.pipeline is not None)
    assert(isinstance(manager.pipeline, qlipp_pipeline))

    shutil.rmtree(os.path.join(folder, config.data_save_name+'.zarr'))

def test_qlipp_pipeline_initiate(setup_test_data):
    folder, data = setup_test_data

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'config_qlipp_full_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=data, save_dir=folder)

    manager = PipelineManager(config)

    pipeline = manager.pipeline
    assert(pipeline.config == manager.config)
    assert(pipeline.data == manager.data)
    assert(pipeline.config.data_save_name == manager.pipeline.name)
    assert(pipeline.t == manager.num_t)
    assert(pipeline.mode == '3D')
    assert(pipeline.phase_only == False)
    assert(pipeline.stokes_only == False)
    assert(pipeline.slices == manager.data.slices)
    assert(pipeline.img_dim == (manager.data.height, manager.data.width, manager.data.slices))
    assert(pipeline.chan_names == manager.data.channel_names)
    assert(isinstance(pipeline.calib_meta, dict))
    assert(pipeline.bg_path == manager.config.background)

    #todo: assert bg dimensions when bug is fixed in calibration

    # assert(pipeline.bg_roi == (0, 0, daemon.data.width, daemon.data.height))
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

    shutil.rmtree(os.path.join(folder, config.data_save_name+'.zarr'))

def test_pipeline_daemon_run(setup_test_data):

    folder, data = setup_test_data

    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'config_qlipp_full_pytest.yml')
    config = ConfigReader(path_to_config, data_dir=data, save_dir=folder)

    manager = PipelineManager(config)
    manager.run()

    store = zarr.open(os.path.join(folder, '2T_3P_81Z_231Y_498X_Kazansky_2.zarr'))
    array = store['Pos_000.zarr']['physical_data']['array']

    assert(store.attrs.asdict() == config.yaml_dict)
    assert(store['Pos_000.zarr'])
    assert(store['Pos_001.zarr'])
    assert(store['Pos_002.zarr'])
    assert(array.shape == (2, 4, 81, manager.data.height, manager.data.width))

    shutil.rmtree(os.path.join(folder, config.data_save_name+'.zarr'))




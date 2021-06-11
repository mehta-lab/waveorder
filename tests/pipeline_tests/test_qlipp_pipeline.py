import pytest
from ..conftest import setup_folder_qlipp_pipeline, setup_test_data
from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_daemon import PipelineDaemon
from recOrder.pipelines.QLIPP_Pipeline import qlipp_pipeline
from waveorder.io.writer import WaveorderWriter
import os
import zarr

def test_pipeline_daemon_initiate():

    # folder = setup_folder_qlipp_pipeline()
    # test = setup_test_data()

    folder = '/Users/cameron.foltz/recOrder/pytest_temp/pipeline_test/2021_06_11_recOrder_pytest_20x_04NA'

    data = os.path.join(folder, '2T_3P_81Z_231Y_498X_Kazansky_2')

    config = ConfigReader('/Users/cameron.foltz/recOrder/tests/pipeline_tests/config_full_pytest.yml',
                          data_dir=data, save_dir=folder)

    daemon = PipelineDaemon(config)
    assert(daemon.config is not None)
    assert(daemon.data is not None)
    assert(daemon.data.get_num_positions()*daemon.data.frames == len(daemon.pt_set))
    assert(daemon.pipeline is not None)
    assert(isinstance(daemon.pipeline, qlipp_pipeline))

def test_qlipp_pipeline_initiate():
    folder = '/Users/cameron.foltz/recOrder/pytest_temp/pipeline_test/2021_06_11_recOrder_pytest_20x_04NA'

    data = os.path.join(folder, '2T_3P_81Z_231Y_498X_Kazansky_2')

    config = ConfigReader('/Users/cameron.foltz/recOrder/tests/pipeline_tests/config_full_pytest.yml',
                          data_dir=data, save_dir=folder)

    daemon = PipelineDaemon(config)

    pipeline = daemon.pipeline
    assert(pipeline.config == daemon.config)
    assert(pipeline.data == daemon.data)
    assert(pipeline.config.data_save_name == daemon.pipeline.name)
    assert(pipeline.t == daemon.num_t)
    assert(pipeline.method == 'QLIPP')
    assert(pipeline.mode == '3D')
    assert(pipeline.phase_only == False)
    assert(pipeline.slices == daemon.data.slices)
    assert(pipeline.img_dim == (daemon.data.height, daemon.data.width, daemon.data.slices))
    assert(pipeline.chan_names == daemon.data.channel_names)
    assert(isinstance(pipeline.calib_meta, dict))
    assert(pipeline.bg_path == daemon.config.background)
    assert(pipeline.bg_roi == (0, 0, daemon.data.width, daemon.data.height))
    assert(pipeline.s0_idx == 0)
    assert(pipeline.s1_idx == 1)
    assert(pipeline.s2_idx == 2)
    assert(pipeline.s3_idx == 3)
    assert(pipeline.fluor_idxs == [])
    assert(pipeline.data_shape == (daemon.data.frames, daemon.data.channels,
                                   daemon.data.slices, daemon.data.height, daemon.data.width))
    assert(pipeline.chunk_size == (1, 1, 1, daemon.data.height, daemon.data.width))
    assert(isinstance(pipeline.writer, WaveorderWriter))
    assert(pipeline.reconstructor is not None)
    assert(pipeline.bg_stokes is not None)

# def test_pipeline_daemon_run():
#
#     # folder = setup_folder_qlipp_pipeline()
#     # test = setup_test_data()
#
#     folder = '/Users/cameron.foltz/recOrder/pytest_temp/pipeline_test/2021_06_11_recOrder_pytest_20x_04NA'
#
#     data = os.path.join(folder, '2T_3P_81Z_231Y_498X_Kazansky_2')
#
#     config = ConfigReader('/Users/cameron.foltz/recOrder/tests/pipeline_tests/config_full_pytest.yml',
#                           data_dir=data, save_dir=folder)
#
#     daemon = PipelineDaemon(config)
#     daemon.run()
#
#     store = zarr.open(os.path.join(folder, '2T_3P_81Z_231Y_498X_Kazansky_2.zarr'))
#
#     assert(store['Pos_000.zarr']['physical_data']['array'].shape)




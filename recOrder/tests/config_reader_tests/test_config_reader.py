from recOrder.io.config_reader import ConfigReader, DATASET, PREPROCESSING, POSTPROCESSING, PROCESSING
import os
from os.path import dirname, abspath
import yaml

def test_config_reader(get_zarr_data_dir):

    folder, zarr_data = get_zarr_data_dir
    path_to_config = os.path.join(dirname(dirname(abspath(__file__))), 'test_configs/qlipp/config_qlipp_full_pytest.yml')

    raw = yaml.full_load(open(path_to_config))
    config = ConfigReader(path_to_config, data_dir=zarr_data, save_dir=folder)

    avoid = ['data_dir', 'save_dir', 'data_save_name', 'positions', 'timepoints',
             'background_ROI', 'qlipp_birefringence_only', 'phase_denoiser_2D',
             'Tik_reg_abs_2D', 'Tik_reg_ph_2D', 'rho_2D', 'itr_2D', 'TV_reg_abs_2D',
             'TV_reg_ph_2D', 'brightfield_channel_index', 'fluorescence_channel_indices',
             'reg', 'fluorescence_background']

    for key, value in DATASET.items():
        if key not in avoid:
            assert(raw['dataset'][key] == getattr(config, key))

    assert(folder == config.save_dir)
    assert(zarr_data == config.data_dir)
    assert([raw['dataset']['positions']] == config.positions)
    assert([raw['dataset']['timepoints']] == config.timepoints)
    assert(config.background_ROI == None)

    for key, value in PREPROCESSING.items():
        for sub_key, sub_value in PREPROCESSING[key].items():
            assert(raw['pre_processing'][key][sub_key] == getattr(config.preprocessing, f'{key}_{sub_key}'))

    for key, value in PROCESSING.items():
        if key not in avoid:
            assert(raw['processing'][key] == getattr(config, key))

    assert(config.qlipp_birefringence_only == False)

    for key, value in POSTPROCESSING.items():
        for sub_key, sub_value in POSTPROCESSING[key].items():
            assert(raw['post_processing'][key][sub_key] == getattr(config.postprocessing, f'{key}_{sub_key}'))






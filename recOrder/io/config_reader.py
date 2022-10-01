import yaml
import pathlib
import os
import warnings

DATASET = {
    'method': None,
    'mode': None,
    'data_dir': None,
    'save_dir': None,
    'data_type': 'zarr',
    'data_save_name': None,
    'positions': 'all',
    'timepoints': 'all',
    'background': None,
    'background_ROI': None,
    'calibration_metadata': None
}

PROCESSING = {
    'output_channels': None,
    'qlipp_birefringence_only': False,
    'background_correction': 'None',
    'use_gpu': False,
    'gpu_id': 0,
    'wavelength': None,
    'pixel_size': None,
    'magnification': None,
    'NA_objective': None,
    'NA_condenser': None,
    'n_objective_media': 1.003,
    'brightfield_channel_index': 0,
    'z_step': None,
    'focus_zidx': None,
    'reg': 1e-4,
    'phase_denoiser_2D': 'Tikhonov',
    'Tik_reg_abs_2D': 1e-4,
    'Tik_reg_ph_2D': 1e-4,
    'rho_2D': 1,
    'itr_2D': 50,
    'TV_reg_abs_2D': 1e-4,
    'TV_reg_ph_2D': 1e-4,
    'phase_denoiser_3D': 'Tikhonov',
    'rho_3D': 1e-3,
    'itr_3D': 50,
    'Tik_reg_ph_3D': 1e-4,
    'TV_reg_ph_3D': 5e-5,
    'pad_z': 0
}

class ConfigReader():
    """
    Config Reader handles all of the requirements necessary for running the pipeline.  Default values
    are used for those that do not need to be specified.  CLI will always overried config

    """

    def __init__(self, cfg_path=None, data_dir=None, save_dir=None, method=None, mode=None, name=None, immutable=True):

        self.immutable = immutable


        if data_dir is not None:
            setattr(self, 'data_dir', data_dir)
        if save_dir is not None:
            setattr(self, 'save_dir', save_dir)
        if method is not None:
            setattr(self, 'method', method)
        if mode is not None:
            setattr(self, 'mode', mode)
        if name is not None:
            setattr(self, 'data_save_name', name)

        # initialize defaults
        for key, value in DATASET.items():
            if hasattr(self, key):
                continue
            else:
                setattr(self, key, value)

        for key, value in PROCESSING.items():
            setattr(self, key, value)

        # parse config
        if cfg_path:
            # self.allow_yaml_tuple()
            self.read_config(cfg_path, data_dir, save_dir, method, mode, name)

        # Create yaml dict to save in save_dir
        setattr(self, 'yaml_dict', self._create_yaml_dict())
        # self._save_yaml()

    # Override set attribute function to disallow changes after init
    def __setattr__(self, name, value):
        if hasattr(self, name) and self.immutable:
            raise AttributeError("Attempting to change immutable object")
        else:
            super().__setattr__(name, value)

    def _check_assertions(self, data_dir, save_dir, method, mode, name):

        # assert main fields of config
        assert 'dataset' in self.config, \
            'dataset is a required field in the config yaml file'
        assert 'processing' in self.config, \
            'processing is a required field in the config yaml file'

        if not method: assert 'method' in self.config['dataset'], \
            'Please provide method in config file or CLI argument'
        if not mode: assert 'mode' in self.config['dataset'], \
            'Please provide mode in config file or CLI argument'
        if not data_dir: assert 'data_dir' in self.config['dataset'], \
            'Please provide data_dir in config file or CLI argument'
        if not save_dir: assert 'save_dir' in self.config['dataset'], \
            'Please provide save_dir in config file or CLI argument'

        if self.config['dataset']['positions'] != 'all' and not isinstance(self.config['dataset']['positions'], int):
            assert isinstance(self.config['dataset']['positions'], list), \
                'if not single integer value or "all", positions must be list (nested lists/tuples allowed)'
        if self.config['dataset']['timepoints'] != 'all' and not isinstance(self.config['dataset']['timepoints'], int):
            assert isinstance(self.config['dataset']['timepoints'], list), \
                'if not single integer value or "all", timepoints must be list (nested lists/tuples allowed)'

        for key,value in PROCESSING.items():
            if key == 'output_channels':
                if 'Phase3D' in self.config['processing'][key] or 'Phase2D' in self.config['processing'][key]:
                    assert 'wavelength' in self.config['processing'], 'wavelength required for phase reconstruction'
                    assert 'pixel_size' in self.config['processing'], 'pixel_size required for phase reconstruction'
                    assert 'magnification' in self.config['processing'], 'magnification required for phase reconstruction'
                    assert 'NA_objective' in self.config['processing'], 'NA_objective required for phase reconstruction'
                    assert 'NA_condenser' in self.config['processing'], 'NA_condenser required for phase reconstruction'

                if 'Phase3D' in self.config['processing'][key] and 'Phase2D' in self.config['processing'][key]:
                    raise KeyError(f'Both Phase3D and Phase2D cannot be specified in {key}.  Please compute separately')

                if 'Phase3D' in self.config['processing'][key] and self.config['dataset']['mode'] == '2D':
                    raise KeyError(f'Specified mode is 2D and Phase3D was specified for reconstruction. '
                                   'Only 2D reconstructions can be performed in 2D mode')

                if 'Phase2D' in self.config['processing'][key] and self.config['dataset']['mode'] == '3D':
                    raise KeyError(f'Specified mode is 3D and Phase2D was specified for reconstruction. '
                                   'Only 3D reconstructions can be performed in 3D mode')

            elif key == 'background_correction' and self.config['dataset']['method'] == 'QLIPP':
                mode = self.config['processing'][key]
                if mode == 'None' or mode == 'local_fit' or mode == 'local_fit+':
                    pass
                else:
                    assert self.config['dataset']['background'] is not None, \
                        'path to background data must be specified for this background correction method'

    def save_yaml(self, dir_=None, name=None):
        self.immutable = False
        self.yaml_dict = self._create_yaml_dict()
        self.immutable = True

        dir_ = self.save_dir if dir_ is None else dir_
        name = f'config_{self.data_save_name}.yml' if name is None else name
        if not name.endswith('.yml'):
            name += '.yml'
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        with open(os.path.join(dir_, name), 'w') as file:
            yaml.dump(self.yaml_dict, file)
        file.close()

    def _create_yaml_dict(self):

        yaml_dict = {'dataset': {},
                     'processing': {}}

        for key, value in DATASET.items():
            yaml_dict['dataset'][key] = getattr(self, key)

        for key, value in PROCESSING.items():
            yaml_dict['processing'][key] = getattr(self, key)

        return yaml_dict

    def _use_default_name(self):
        path = pathlib.PurePath(self.data_dir)
        super().__setattr__('data_save_name', path.name)

    def _parse_dataset(self):
        for key, value in self.config['dataset'].items():
            if key in DATASET.keys():

                # This section will prioritize the command line arguments over the
                # config arguments (data_dir, save_dir, data_save_name)
                if key == 'name' and self.data_save_name:
                    continue
                elif key == 'data_dir' and self.data_dir:
                    continue
                elif key == 'save_dir' and self.save_dir:
                    continue
                elif key == 'mode' and self.mode:
                    continue
                elif key == 'method' and self.method:
                    continue

                # this section appends the rest of the dataset parameters specified
                elif key == 'positions' or key == 'timepoints':
                    if isinstance(value, str):
                        if value == 'all':
                            super().__setattr__(key, [value])
                        else:
                            raise KeyError(f'{key} value {value} not understood,\
                                       please specify a list or "all"')
                    elif isinstance(value, int):
                        super().__setattr__(key, [value])
                    elif isinstance(value, tuple):
                        if len(value) == 2:
                            super().__setattr__(key, value)
                        else:
                            raise KeyError(f'{key} value {value} is not a tuple with length of 2')
                    elif isinstance(value, list):
                        super().__setattr__(key, value)
                    else:
                        raise KeyError(f'{key} value {value} format not understood. \
                                       Must be list with nested tuple or list or "all"')
                else:
                    super().__setattr__(key, value)

            else:
                warnings.warn(f'yaml DATASET config field {key} is not recognized')

    def _parse_processing(self):
        for key, value in self.config['processing'].items():
            if key in PROCESSING.keys():
                super().__setattr__(key, value)
            else:
                warnings.warn(f'yaml PROCESSING config field {key} is not recognized')

        if 'Phase3D' not in self.output_channels and 'Phase2D' not in self.output_channels:
            super().__setattr__('qlipp_birefringence_only', True)

    def read_config(self, cfg_path, data_dir, save_dir, method, mode, name):

        if isinstance(cfg_path, str):
            self.config = yaml.full_load(open(cfg_path))
        if isinstance(cfg_path, dict):
            self.config = cfg_path

        self._check_assertions(data_dir, save_dir, method, mode, name)
        self._parse_dataset()
        self._parse_processing()
        
        if not self.data_save_name:
            self._use_default_name()

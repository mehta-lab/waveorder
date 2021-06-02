import yaml
import pathlib
import warnings

DATASET = {
    'method': None,
    'mode': None,
    'data_dir': None,
    'save_dir': None,
    'data_type': 'ometiff',
    'data_save_name': None,
    'positions': ['all'],
    'timepoints': ['all'],
    'background': None,
    'background_ROI': None,
    'calibration_metadata': None
}

PREPROCESSING = {
    'denoise': {
        'use': False,
        'channels': None,
        'threshold': None,
        'level': None
    }
}

PROCESSING = {
    'output_channels': None,
    'background_correction': 'None',
    'flatfield_correction': None,
    'use_gpu': False,
    'gpu_id': 0,
    'wavelength':None,
    'pixel_size': None,
    'magnification': None,
    'NA_objective': None,
    'NA_condenser': None,
    'n_objective_media': 1.003,
    'z_step': None,
    'focus_zidx': None,
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



POSTPROCESSING = {
    'denoise':{
        'use': False,
        'channels': None,
        'threshold': None,
        'level': None
    },

    'registration':{
        'use': False,
        'channel_idx': None,
        'shift': None
    }
}

class Object():
    pass

class ConfigReader(object):

    def __init__(self, cfg_path=None, data_dir=None, save_dir=None, method=None, mode=None, name=None):

        # initialize defaults
        self.__set_attr(self, 'preprocessing', Object())
        self.__set_attr(self, 'postprocessing', Object())

        for key, value in DATASET.items():
            self.__set_attr(self, key, value)
        for key, value in PREPROCESSING.items():
            if isinstance(value, dict):
                for key_child, value_child in PREPROCESSING[key].items():
                    self.__set_attr(self.preprocessing, f'{key}_{key_child}', value_child)
            else:
                self.__set_attr(self.preprocessing, key, value)
        for key, value in PROCESSING.items():
            self.__set_attr(self, key, value)
        for key, value in POSTPROCESSING.items():
            if isinstance(value, dict):
                for key_child, value_child in POSTPROCESSING[key].items():
                    self.__set_attr(self.postprocessing, f'{key}_{key_child}', value_child)
            else:
                self.__set_attr(self.postprocessing, key, value)

        # parse config
        if cfg_path:
            self.read_config(cfg_path, data_dir, save_dir, method, mode, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # Override set attribute function to disallow changes after init
    def __setattr__(self, name, value):
        raise AttributeError("Attempting to change immutable object")

    def __set_attr(self, object_, name, value):
        object.__setattr__(object_, name, value)

    def read_config(self, cfg_path, data_dir, save_dir, method, mode, name):


        with open(cfg_path, 'r') as file:
            config = yaml.safe_load(file)

        self.__set_attr(self, 'config', config)

        self._check_assertions(data_dir, save_dir, method, mode, name)
        self._parse_cli(data_dir, save_dir, method, mode, name)
        self._parse_dataset()
        self._parse_preprocessing()
        self._parse_processing()
        self._parse_postprocessing()

        if self.data_save_name == None:
            self._use_default_name()

    #todo: finish assertions for processing field
    def _check_assertions(self, data_dir, save_dir, method, mode, name):

        # assert main fields of config
        assert 'dataset' in self.config, \
            'dataset is a required field in the config yaml file'
        assert 'processing' in self.config, \
            'processing is a required field in the config yaml file'

        if not method: assert 'mode' in self.config['dataset'], \
            'Please provide method in config file or CLI argument'
        if not mode: assert 'mode' in self.config['dataset'], \
            'Please provide mode in config file or CLI argument'
        if not data_dir: assert 'data_dir' in self.config['dataset'], \
            'Please provide data_dir in config file or CLI argument'
        if not save_dir: assert 'save_dir' in self.config['dataset'], \
            'Please provide save_dir in config file or CLI argument'
        if not name: assert 'save_dir' in self.config['dataset'], \
            'Please provide data_save_name in config file or CLI argument'

        if 'preprocessing' in self.config:
            for key, value in PREPROCESSING.items():
                if self.config['preprocessing'][key]['use']:
                    for key_child, value_child in PREPROCESSING[key].items():
                        assert key_child in self.config['preprocessing'][key], \
                            f'User must specify {key_child} to use for {key}'

        if 'postprocessing' in self.config:
            for key, value in POSTPROCESSING.items():
                if self.config['postprocessing'][key]['use']:
                    for key_child, value_child in POSTPROCESSING[key].items():
                        assert key_child in self.config['postprocessing'][key], \
                            f'User must specify {key_child} to use for {key}'

    def _use_default_name(self):
        path = pathlib.PurePath(self.data_dir)
        self.__set_attr(self, 'data_save_name', path.name)

    def _parse_cli(self, data_dir, save_dir, method, mode, name):
        if data_dir:
            self.__set_attr(self, 'data_dir', data_dir)
        if save_dir:
            self.__set_attr(self, 'save_dir', save_dir)
        if method:
            self.__set_attr(self, 'method', method)
        if mode:
            self.__set_attr(self, 'mode', mode)
        if name:
            self.__set_attr(self, 'data_save_name', name)

    def _parse_dataset(self):
        for key, value in self.config['dataset'].items():
            if key in DATASET.keys():

                # if config has a data_save name, but a user specifies a different name in CLI,
                # skip and use the user-specified name
                if key == 'name' and self.data_save_name:
                    continue

                elif key == 'positions':
                    if isinstance(value, str) and value == 'all':
                        self.__set_attr(self, key, [value])
                    else:
                        self.__set_attr(self, key, value)

                elif key == 'timepoints':
                    if isinstance(value, str) and value == 'all':
                        self.__set_attr(self, key, [value])
                    else:
                        self.__set_attr(self, key, value)

                else:
                    self.__set_attr(self, key, value)

            else:
                warnings.warn(f'yaml DATASET config field {key} is not recognized')

    #TODO: MAKE COMPATIBLE WITH PREDEFINED LIST
    def _parse_preprocessing(self):
        for key, value in self.config['pre_processing'].items():
            if key in PREPROCESSING.keys():
                for key_child, value_child in self.config['pre_processing'][key].items():
                    if key_child in PREPROCESSING[key].keys():
                        self.__set_attr(self.preprocessing, f'{key}_{key_child}', value_child)
                    else:
                        warnings.warn(f'yaml PREPROCESSING config field {key}, {key_child} is not recognized')
            else:
                warnings.warn(f'yaml PREPROCESSING config field {key} is not recognized')

    def _parse_processing(self):
        for key, value in self.config['processing'].items():
            if key in PROCESSING.keys():
                self.__set_attr(self, key, value)
            else:
                warnings.warn(f'yaml PROCESSING config field {key} is not recognized')

    def _parse_postprocessing(self):
        for key, value in self.config['post_processing'].items():
            if key in POSTPROCESSING.keys():
                for key_child, value_child in self.config['post_processing'][key].items():
                    if key_child in POSTPROCESSING[key].keys():
                        self.__set_attr(self.postprocessing, f'{key}_{key_child}', value_child)
                    else:
                        warnings.warn(f'yaml POSTPROCESSING config field {key}, {key_child} is not recognized')
            else:
                warnings.warn(f'yaml POSTPROCESSING config field {key} is not recognized')

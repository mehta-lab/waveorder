import yaml
import pathlib
import warnings

DATASET = {
    'method': None,
    'mode': None,
    'data_dir': None,
    'save_dir': None,
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
        self.preprocessing = Object()
        self.postprocessing = Object()

        for entry in DATASET:
            setattr(self, entry[0], entry[1])
        for entry in PREPROCESSING:
            setattr(self.preprocessing, entry[0], entry[1])
        for entry in PROCESSING:
            setattr(self, entry[0], entry[1])
        for entry in POSTPROCESSING:
            setattr(self.postprocessing, entry[0], entry[1])
        # parse config
        if cfg_path:
            self.read_config(cfg_path, data_dir, save_dir, method, mode, name)

    # Override set attribute function to disallow changes after init
    def __setattr__(self, name, value):
        raise AttributeError("Attempting to change immutable object")

    def read_config(self, cfg_path, data_dir, save_dir, method, mode, name):

        with open(cfg_path) as f:
            setattr(self, 'config', yaml.load(f))

        self._check_assertions(data_dir, save_dir, method, mode, name)
        self._parse_cli(data_dir, save_dir, method, mode, name)
        self._parse_dataset(data_dir, save_dir, method, mode, name)
        self._parse_preprocessing()
        self._parse_processing()
        self._parse_postprocessing()

        if self.data_save_name == None:
            self._use_default_name()

    def _check_assertions(self):
        pass

    def _use_default_name(self):
        path = pathlib.PurePath(self.data_dir)
        setattr(self, 'data_save_name', path.name)

    def _parse_cli(self, data_dir, save_dir, method, mode, name):
        if data_dir:
            setattr(self, 'data_dir', data_dir)
        if save_dir:
            setattr(self, 'save_dir', save_dir)
        if method:
            setattr(self, 'method', method)
        if mode:
            setattr(self, 'mode', mode)
        if name:
            setattr(self, 'data_save_name', name)

    def _parse_dataset(self):
        for key, value in self.config['dataset'].items():
            if key in DATASET.keys():

                # if config has a data_save name, but a user specifies a different name in CLI,
                # skip and use the user-specified name
                if key == 'name' and self.data_save_name:
                    continue

                elif key == 'positions':
                    if isinstance(value, str) and value == 'all':
                        setattr(self, key, [value])
                    else:
                        setattr(self, key, value)

                elif key == 'timepoints':
                    if isinstance(value, str) and value == 'all':
                        setattr(self, key, [value])
                    else:
                        setattr(self, key, value)

                else:
                    setattr(self, key, value)

            else:
                warnings.warn(f'yaml DATASET config field {key} is not recognized')

    #TODO: MAKE COMPATIBLE WITH PREDEFINED LIST
    def _parse_preprocessing(self):
        for key, value in self.config['preprocessing'].items():
            if key in PREPROCESSING.keys():
                for key_child, value_child in self.config['preprocessing'][key]:
                    if key_child in key.keys():
                        setattr(self.preprocessing, f'{key}_{key_child}', value_child)
                    else:
                        warnings.warn(f'yaml PREPROCESSING config field {key}, {key_child} is not recognized')
            else:
                warnings.warn(f'yaml PREPROCESSING config field {key} is not recognized')

    def _parse_processing(self):
        for key, value in self.config['dataset'].items():
            if key in PROCESSING.keys():
                setattr(self, key, value)
            else:
                warnings.warn(f'yaml PROCESSING config field {key} is not recognized')

    def _parse_postprocessing(self):
        for key, value in self.config['postprocessing'].items():
            if key in POSTPROCESSING.keys():
                for key_child, value_child in self.config['postprocessing'][key]:
                    if key_child in key.keys():
                        setattr(self.postprocessing, f'{key}_{key_child}', value_child)
                    else:
                        warnings.warn(f'yaml POSTPROCESSING config field {key}, {key_child} is not recognized')
            else:
                warnings.warn(f'yaml POSTPROCESSING config field {key} is not recognized')

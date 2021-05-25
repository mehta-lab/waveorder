import yaml
# import glob
import os.path
from dataclasses import dataclass
# from collections.abc import Iterable
# from .imgIO import get_sub_dirs

# @dataclass(frozen=True)
class ConfigReader(object):

    def __init__(self, path=[]):
        object.__setattr__(self, 'yaml_config', None)

        # Dataset Parameters
        object.__setattr__(self, 'data_dir', None)
        object.__setattr__(self, 'data_type', None)
        object.__setattr__(self, 'processed_dir', None)
        object.__setattr__(self, 'samples', None)
        object.__setattr__(self, 'positions', 'all')
        object.__setattr__(self, 'z_slices', 'all')
        object.__setattr__(self, 'timepoints', 'all')
        object.__setattr__(self, 'background', None)
        object.__setattr__(self, 'calibration_metadata', None)
        object.__setattr__(self, 'sample_paths', None)

        # Pre-Processing
        ## Denoising
        object.__setattr__(self, 'preproc_denoise_use', False)
        object.__setattr__(self, 'preproc_denoise_channels', None)
        object.__setattr__(self, 'preproc_denoise_thresholds', None)
        object.__setattr__(self, 'preproc_denoise_levels', None)

        # Background Correction
        object.__setattr__(self, 'background_correction', 'None')
        object.__setattr__(self, 'flatfield_correction', None)

        # Output Parameters
        object.__setattr__(self, 'circularity', 'rcp')
        # GPU
        object.__setattr__(self, 'use_gpu', False)
        object.__setattr__(self, 'gpu_id', 0)

        # Phase Reconstruction Parameters
        object.__setattr__(self, 'wavelength', None)
        object.__setattr__(self, 'pixel_size', None)
        object.__setattr__(self, 'magnification', None)
        object.__setattr__(self, 'NA_objective', None)
        object.__setattr__(self, 'NA_condenser', None)
        object.__setattr__(self, 'n_objective_media', 1.003)
        object.__setattr__(self, 'z_step', None)
        object.__setattr__(self, 'focus_zidx', None)
        object.__setattr__(self, 'pad_z', 0)

        # Phase Algorithm Tuning Parameters
        object.__setattr__(self, 'phase_denoiser_2D', 'Tikhonov')
        object.__setattr__(self, 'Tik_reg_abs_2D', 1e-4)
        object.__setattr__(self, 'Tik_reg_ph_2D', 1e-4)
        object.__setattr__(self, 'rho_2D', 1)
        object.__setattr__(self, 'itr_2D', 50)
        object.__setattr__(self, 'TV_reg_abs_2D', 1e-3)
        object.__setattr__(self, 'TV_reg_ph_2D', 1e-5)
        object.__setattr__(self, 'phase_denoiser_3D', 'Tikhonov')
        object.__setattr__(self, 'rho_3D', 1e-3)
        object.__setattr__(self, 'itr_3D', 50)
        object.__setattr__(self, 'Tik_reg_ph_3D', 1e-4)
        object.__setattr__(self, 'TV_reg_ph_3D', 5e-5)

        # Post-Processing
        ## Denoising
        object.__setattr__(self, 'postproc_denoise_use', False)
        object.__setattr__(self, 'postproc_denoise_channels', None)
        object.__setattr__(self, 'postproc_denoise_thresholds', None)
        object.__setattr__(self, 'postproc_denoise_levels', None)

        object.__setattr__(self, 'postproc_registration_use', False)
        object.__setattr__(self, 'postproc_registration_channel_idx', None)
        object.__setattr__(self, 'postproc_registration_shift', None)

        # Plotting Parameters
        object.__setattr__(self, 'normalize_color_images', True)
        object.__setattr__(self, 'retardance_scaling', 1e3)
        object.__setattr__(self, 'transmission_scaling', 1e4)
        object.__setattr__(self, 'phase_2D_scaling', 1)
        object.__setattr__(self, 'absorption_2D_scaling', 1)
        object.__setattr__(self, 'phase_3D_scaling', 1)
        object.__setattr__(self, 'save_birefringence_fig', False)
        object.__setattr__(self, 'save_stokes_fig', False)
        object.__setattr__(self, 'save_polarization_fig', False)
        object.__setattr__(self, 'save_micromanager_fig', False)

        if path:
            self.read_config(path)

    # def __setattr__(self, name, value):
    #     raise AttributeError('''Can't set attribute "{0}"'''.format(name))

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def __setattr__(self, name, value):
        raise AttributeError("Attempting to change immutable object")
    # def __setattr__(self, name, value):
    #     raise AttributeError('''Can't set attribute "{0}"'''.format(name))

    def read_config(self, path):
        with open(path, 'r') as f:
            object.__setattr__(self, 'yaml_config', yaml.safe_load(f))
            # self.yaml_config = yaml.load(f)

        assert 'dataset' in self.yaml_config, \
            'dataset is a required field in the config yaml file'
        # assert 'data_dir' in self.yaml_config['dataset'], \
        #     'Please provide data_dir in config file'
        # assert 'processed_dir' in self.yaml_config['dataset'], \
        #     'Please provide processed_dir in config file'
        # assert 'samples' in self.yaml_config['dataset'], \
        #     'Please provide samples in config file'

        # self.data_dir = self.yaml_config['dataset']['data_dir']
        # object.__setattr__(self, 'data_dir', self.yaml_config['dataset']['data_dir'])
        # object.__setattr__(self, 'data_type', self.yaml_config['dataset']['data_type'])
        # object.__setattr__(self, 'processed_dir', self.yaml_config['dataset']['processed_dir'])
        # object.__setattr__(self, 'default', self.yaml_config['default'])

        for (key, value) in self.yaml_config['dataset'].items():
            # if key == 'samples':
            #     object.__setattr__(self, 'samples', value)
            if key == 'positions':
                object.__setattr__(self, 'positions', value)
            elif key == 'z_slices':
                object.__setattr__(self, 'z_slices', value)
            elif key == 'timepoints':
                object.__setattr__(self, 'timepoints', value)
            elif key == 'background':
                object.__setattr__(self, 'background', value)
            elif key == 'background_ROI':
                object.__setattr__(self, 'background_ROI', value)
            elif key == 'calibration_metadata':
                object.__setattr__(self, 'calibration_metadata', value)
            elif key not in ('data_dir', 'processed_dir', 'data_type'):
                raise NameError('Unrecognized configfile field:{}, key:{}'.format('dataset', key))

        # for sample in self.samples:
        #     paths = []
        #     paths.append(os.path.join(self.data_dir, sample))
        #     object.__setattr__(self, 'sample_paths', paths)

        if 'pre_processing' in self.yaml_config:
            for (key1, value) in self.yaml_config['pre_processing'].items():
                if 'denoise' in key1:
                    for (key, value) in self.yaml_config['pre_processing']['denoise'].items():
                        if key == 'use':
                            object.__setattr__(self, 'preproc_denoise_use', value)
                            preproc_denoise = True if value else False
                        elif key == 'channels':
                            if preproc_denoise:
                                assert value is not None, \
                                    'User must specify the channels to use for de-noising'
                            object.__setattr__(self, 'preproc_denoise_channels', value)
                        elif key == 'threshold':
                            object.__setattr__(self, 'preproc_denoise_thresholds', value)
                        elif key == 'level':
                            object.__setattr__(self, 'preproc_denoise_levels', value)

        if 'post_processing' in self.yaml_config:
            for (key1, value) in self.yaml_config['post_processing'].items():
                if 'denoise' in key1:
                    for (key, value) in self.yaml_config['post_processing']['denoise'].items():
                        if key == 'use':
                            object.__setattr__(self, 'postproc_denoise_use', value)
                            postproc_denoise = True if value else False
                        elif key == 'channels':
                            object.__setattr__(self, 'postproc_denoise_channels', value)
                            if postproc_denoise:
                                assert value is not None, \
                                    'User must specify the channels to use for de-noising'
                        elif key == 'threshold':
                            object.__setattr__(self, 'postproc_denoise_thresholds', value)
                        elif key == 'level':
                            object.__setattr__(self, 'postproc_denoise_levels', value)
                if 'registration' in key1:
                    for (key, value) in self.yaml_config['post_processing']['registration'].items():
                        if key == 'use':
                            object.__setattr__(self, 'postproc_registration_use', value)
                            postproc_registration = True if value else False
                        elif key == 'channel_idx':
                            if postproc_registration:
                                assert value is not None, \
                                    'User must specify the channel index to use for registration'

                            if isinstance(value, int):
                                value = [value]

                            object.__setattr__(self, 'postproc_registration_channel_idx', value)
                        elif key == 'shift':
                            object.__setattr__(self, 'postproc_registration_shift', value)
                            if postproc_registration:
                                assert value is not None, \
                                    'User must specify the shift to use for registration'

        if 'processing' in self.yaml_config:
            for (key, value) in self.yaml_config['processing'].items():
                if key == 'output_channels':
                    object.__setattr__(self, 'output_channels', value)
                    if 'Phase2D' in value or 'Phase_semi3D' in value  or 'Phase3D' in value:
                        phase_processing = True
                    else:
                        phase_processing = False
                elif key == 'calibration_scheme':
                    object.__setattr__(self, 'calibration_scheme', value)
                elif key == 'background_correction':
                    object.__setattr__(self, 'background_correction', value)
                elif key == 'flatfield_correction':
                    object.__setattr__(self, 'flatfield_correction', value)
                elif key == 'use_gpu':
                    object.__setattr__(self, 'use_gpu', value)
                elif key == 'gpu_id':
                    object.__setattr__(self, 'gpu_id', value)
                elif key == 'wavelength':
                    object.__setattr__(self, 'wavelength', value)
                elif key == 'pixel_size':
                    object.__setattr__(self, 'pixel_size', value)
                elif key == 'magnification':
                    object.__setattr__(self, 'magnification', value)
                elif key == 'NA_objective':
                    object.__setattr__(self, 'NA_objective', value)
                elif key == 'NA_condenser':
                    object.__setattr__(self, 'NA_condenser', value)
                elif key == 'n_objective_media':
                    object.__setattr__(self, 'n_objective_media', value)
                elif key == 'z_step':
                    object.__setattr__(self, 'z_step', value)
                elif key == 'focus_zidx':
                    object.__setattr__(self, 'focus_zidx', value)
                elif key == 'phase_denoiser_2D':
                    object.__setattr__(self, 'phase_denoiser_2D', value)
                elif key == 'Tik_reg_abs_2D':
                    object.__setattr__(self, 'Tik_reg_abs_2D', value)
                elif key == 'Tik_reg_ph_2D':
                    object.__setattr__(self, 'Tik_reg_ph_2D', value)
                elif key == 'rho_2D':
                    object.__setattr__(self, 'rho_2D', value)
                elif key == 'itr_2D':
                    object.__setattr__(self, 'itr_2D', value)
                elif key == 'TV_reg_abs_2D':
                    object.__setattr__(self, 'TV_reg_abs_2D', value)
                elif key == 'TV_reg_ph_2D':
                    object.__setattr__(self, 'TV_reg_ph_2D', value)
                elif key == 'phase_denoiser_3D':
                    object.__setattr__(self, 'phase_denoiser_3D', value)
                elif key == 'rho_3D':
                    object.__setattr__(self, 'rho_3D', value)
                elif key == 'itr_3D':
                    object.__setattr__(self, 'itr_3D', value)
                elif key == 'Tik_reg_ph_3D':
                    object.__setattr__(self, 'Tik_reg_ph_3D', value)
                elif key == 'TV_reg_ph_3D':
                    object.__setattr__(self, 'TV_reg_ph_3D', value)
                elif key == 'pad_z':
                    object.__setattr__(self, 'pad_z', value)
                else:
                    raise NameError('Unrecognized configfile field:{}, key:{}'.format('processing', key))

            if phase_processing:

                assert self.pixel_size is not None, \
                "pixel_size (camera pixel size) has to be specified to run phase reconstruction"

                assert self.magnification is not None, \
                "magnification (microscope magnification) has to be specified to run phase reconstruction"

                assert self.NA_objective is not None, \
                "NA_objective (numerical aperture of the objective) has to be specified to run phase reconstruction"

                assert self.NA_condenser is not None, \
                "NA_condenser (numerical aperture of the condenser) has to be specified to run phase reconstruction"

                assert self.n_objective_media is not None, \
                "n_objective_media (refractive index of the immersing media) has to be specified to run phase reconstruction"

                assert self.n_objective_media >= self.NA_objective and self.n_objective_media >= self.NA_condenser, \
                "n_objective_media (refractive index of the immersing media) has to be larger than the NA of the objective and condenser"

                assert self.z_slices == 'all', \
                "z_slices has to be 'all' in order to run phase reconstruction properly"


            if 'Phase2D' in self.output_channels:

                assert self.focus_zidx is not None, \
                "focus_zidx has to be specified to run 2D phase reconstruction"


        # if 'plotting' in self.yaml_config:
        #     for (key, value) in self.yaml_config['plotting'].items():
        #         if key == 'normalize_color_images':
        #             object.__setattr__(self, 'normalize_color_images', value)
        #         elif key == 'retardance_scaling':
        #             object.__setattr__(self, 'retardance_scaling', float(value))
        #         elif key == 'transmission_scaling':
        #             object.__setattr__(self, 'transmission_scaling', float(value))
        #         elif key == 'phase_2D_scaling':
        #             object.__setattr__(self, 'phase_2D_scaling', float(value))
        #         elif key == 'absorption_2D_scaling':
        #             object.__setattr__(self, 'absorption_2D_scaling', float(value))
        #         elif key == 'phase_3D_scaling':
        #             object.__setattr__(self, 'phase_3D_scaling', float(value))
        #         elif key == 'save_birefringence_fig':
        #             object.__setattr__(self, 'save_birefringence_fig', value)
        #         elif key == 'save_stokes_fig':
        #             object.__setattr__(self, 'save_stokes_fig', value)
        #         elif key == 'save_polarization_fig':
        #             object.__setattr__(self, 'save_polarization_fig', value)
        #         elif key == 'save_micromanager_fig':
        #             object.__setattr__(self, 'save_micromanager_fig', value)
        #         else:
        #             raise NameError('Unrecognized configfile field:{}, key:{}'.format('plotting', key))

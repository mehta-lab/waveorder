import yaml
# import glob
import os.path
# from collections.abc import Iterable
# from .imgIO import get_sub_dirs

class ConfigReader:

    def __init__(self, path=[]):
        self.yaml_config = None

        # Dataset Parameters
        self.data_dir = None
        self.processed_dir = None
        self.samples = None
        self.positions = 'all'
        self.z_slices = 'all'
        self.timepoints = 'all'
        self.background = None
        self.calib_data = None
        self.sample_paths = None

        # Background Correction
        self.background_correction = 'None'
        self.n_slice_local_bg = 'all'
        self.flatfield_correction = None
        self.local_fit_order = 2

        # Output Parameters
        self.separate_positions = True
        self.circularity = 'rcp'
        self.binning = 1

        # GPU
        self.use_gpu = False
        self.gpu_id = 0

        # Phase Reconstruction Parameters
        self.pixel_size = None
        self.magnification = None
        self.NA_objective = None
        self.NA_condenser = None
        self.n_objective_media = 1.003
        self.focus_zidx = None
        self.pad_z = 0

        # Phase Algorithm Tuning Parameters
        self.phase_denoiser_2D = 'Tikhonov'
        self.Tik_reg_abs_2D = 1e-4
        self.Tik_reg_ph_2D = 1e-4
        self.rho_2D = 1
        self.itr_2D = 50
        self.TV_reg_abs_2D = 1e-3
        self.TV_reg_ph_2D = 1e-5
        self.phase_denoiser_3D = 'Tikhonov'
        self.rho_3D = 1e-3
        self.itr_3D = 50
        self.Tik_reg_ph_3D = 1e-4
        self.TV_reg_ph_3D = 5e-5
        
        # Plotting Parameters
        self.normalize_color_images = True
        self.retardance_scaling = 1e3
        self.transmission_scaling = 1e4
        self.phase_2D_scaling = 1
        self.absorption_2D_scaling = 1
        self.phase_3D_scaling = 1
        self.save_birefringence_fig = False
        self.save_stokes_fig = False
        self.save_polarization_fig = False
        self.save_micromanager_fig = False

        if path:
            self.read_config(path)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read_config(self, path):
        with open(path, 'r') as f:
            self.yaml_config = yaml.load(f)

        assert 'dataset' in self.yaml_config, \
            'dataset is a required field in the config yaml file'
        assert 'data_dir' in self.yaml_config['dataset'], \
            'Please provide data_dir in config file'
        assert 'processed_dir' in self.yaml_config['dataset'], \
            'Please provide processed_dir in config file'
        assert 'samples' in self.yaml_config['dataset'], \
            'Please provide samples in config file'

        self.data_dir = self.yaml_config['dataset']['data_dir']
        self.processed_dir = self.yaml_config['dataset']['processed_dir']

        for (key, value) in self.yaml_config['dataset'].items():
            if key == 'samples':
                self.samples = value
            elif key == 'positions':
                self.positions = value
            elif key == 'ROI':
                self.ROI = value
            elif key == 'z_slices':
                self.z_slices = value
            elif key == 'timepoints':
                self.timepoints = value
            elif key == 'background':
                self.background = value
            elif key == 'path_to_calibration_data':
                self.calib_data = value
            elif key not in ('data_dir', 'processed_dir'):
                raise NameError('Unrecognized configfile field:{}, key:{}'.format('dataset', key))

        for sample in self.samples:
            paths = []
            paths.append(os.path.join(self.data_dir,sample))
            self.sample_paths = paths

        if 'processing' in self.yaml_config:

            for (key, value) in self.yaml_config['processing'].items():
                if key == 'output_channels':
                    self.output_channels = value
                    if 'Phase2D' in value or 'Phase_semi3D' in value  or 'Phase3D' in value:
                        phase_processing = True
                    else:
                        phase_processing = False
                elif key == 'circularity':
                    self.circularity = value
                elif key == 'calibration_scheme':
                    self.calibration_scheme = value
                elif key == 'background_correction':
                    self.background_correction = value
                elif key == 'flatfield_correction':
                    self.flatfield_correction = value
                elif key == 'separate_positions':
                    self.separate_positions = value
                elif key == 'n_slice_local_bg':
                    self.n_slice_local_bg = value
                elif key == 'local_fit_order':
                    self.local_fit_order = value
                elif key == 'binning':
                    self.binning = value
                elif key == 'use_gpu':
                    self.use_gpu = value
                elif key == 'gpu_id':
                    self.gpu_id = value
                elif key == 'pixel_size':
                    self.pixel_size = value
                elif key == 'magnification':
                    self.magnification = value
                elif key == 'NA_objective':
                    self.NA_objective = value
                elif key == 'NA_condenser':
                    self.NA_condenser = value
                elif key == 'n_objective_media':
                    self.n_objective_media = value
                elif key == 'focus_zidx':
                    self.focus_zidx = value
                elif key == 'phase_denoiser_2D':
                    self.phase_denoiser_2D = value
                elif key == 'Tik_reg_abs_2D':
                    self.Tik_reg_abs_2D = value
                elif key == 'Tik_reg_ph_2D':
                    self.Tik_reg_ph_2D = value
                elif key == 'rho_2D':
                    self.rho_2D = value
                elif key == 'itr_2D':
                    self.itr_2D = value
                elif key == 'TV_reg_abs_2D':
                    self.TV_reg_abs_2D = value
                elif key == 'TV_reg_ph_2D':
                    self.TV_reg_ph_2D = value
                elif key == 'phase_denoiser_3D':
                    self.phase_denoiser_3D = value
                elif key == 'rho_3D':
                    self.rho_3D = value
                elif key == 'itr_3D':
                    self.itr_3D = value
                elif key == 'Tik_reg_ph_3D':
                    self.Tik_reg_ph_3D = value
                elif key == 'TV_reg_ph_3D':
                    self.TV_reg_ph_3D = value
                elif key == 'pad_z':
                    self.pad_z = value
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

                assert self.n_slice_local_bg == 'all', \
                "n_slice_local_bg has to be 'all' in order to run phase reconstruction properly"

                assert self.z_slices == 'all', \
                "z_slices has to be 'all' in order to run phase reconstruction properly"


            if 'Phase2D' in self.output_channels:

                assert self.focus_zidx is not None, \
                "focus_zidx has to be specified to run 2D phase reconstruction"




        if 'plotting' in self.yaml_config:
            for (key, value) in self.yaml_config['plotting'].items():
                if key == 'normalize_color_images':
                    self.normalize_color_images = value
                elif key == 'retardance_scaling':
                    self.retardance_scaling = float(value)
                elif key == 'transmission_scaling':
                    self.transmission_scaling = float(value)
                elif key == 'phase_2D_scaling':
                    self.phase_2D_scaling = float(value)
                elif key == 'absorption_2D_scaling':
                    self.absorption_2D_scaling = float(value)
                elif key == 'phase_3D_scaling':
                    self.phase_3D_scaling = float(value)
                elif key == 'save_birefringence_fig':
                    self.save_birefringence_fig = value
                elif key == 'save_stokes_fig':
                    self.save_stokes_fig = value
                elif key == 'save_polarization_fig':
                    self.save_polarization_fig = value
                elif key == 'save_micromanager_fig':
                    self.save_micromanager_fig = value
                else:
                    raise NameError('Unrecognized configfile field:{}, key:{}'.format('plotting', key))
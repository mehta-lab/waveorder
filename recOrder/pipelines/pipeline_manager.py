from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
import time
from recOrder.pipelines.qlipp_pipeline import qlipp_pipeline
from recOrder.postproc.post_processing import *
from recOrder.preproc.pre_processing import *


class PipelineManager:
    """
    This will pull the necessary pipeline based off the config default.
    """

    def __init__(self, config: ConfigReader):

        start = time.time()
        print('Reading Data...')
        data = MicromanagerReader(config.data_dir, config.data_type, extract_data=True)
        end = time.time()
        print(f'Finished Reading Data ({(end - start) / 60 :0.1f} min)')

        self.config = config
        self.data = data

        self._gen_coord_set()

        if self.config.method == 'QLIPP':
            self.pipeline = qlipp_pipeline(self.config, self.data, self.config.save_dir,
                                           self.config.data_save_name, self.config.mode, self.num_t)

        elif self.config.method == 'denoise':
            raise NotImplementedError

        elif self.config.method == 'UPTI':
            raise NotImplementedError

        elif self.config.method == 'IPS':
            raise NotImplementedError

    def _get_preprocessing_params(self):
        """
        method to get pre-processing functions and parameters.
        Only supports denoising at the moment

        Returns
        -------
        denoise_params:     (list) [[channels, thresholds, levels]]

        """
        # CAN ADD OTHER PREPROC FUNCTIONS IN FUTURE
        denoise_params = []
        if self.config.preprocessing.denoise_use:
            for i in range(len(self.config.preprocessing.denoise_channels)):
                threshold = 0.1 if self.config.preprocessing.denoise_threshold is None \
                    else self.config.preprocessing.denoise_threshold[i]
                level = 1 if self.config.preprocessing.denoise_level is None \
                    else self.config.preprocessing.denoise_level[i]

                denoise_params.append([self.config.preprocessing.denoise_channels[i], threshold, level])

            return denoise_params

        else:
            return None

    def _get_postprocessing_params(self):
        """
        Method to gather parameters for post_processing functions.
        Currently only supports denoising, registration

        CAN ADD MORE IN FUTURE

        Returns
        -------
        denoise_params:         (list) [[channel, threshold, levels]]

        registration_params:    (list) [[channel index, shift]]

        """

        denoise_params = []
        if self.config.postprocessing.denoise_use:
            for i in range(len(self.config.postprocessing.denoise_channels)):
                threshold = 0.1 if self.config.postprocessing.denoise_threshold is None \
                    else self.config.postprocessing.denoise_threshold[i]
                level = 1 if self.config.postprocessing.denoise_level is None \
                    else self.config.postprocessing.denoise_level[i]

                denoise_params.append([self.config.postprocessing.denoise_channels[i], threshold, level])

        else:
            denoise_params = None

        registration_params = []
        if self.config.postprocessing.registration_use:
            for i in range(len(self.config.postprocessing.registration_channel_idx)):
                registration_params.append([self.config.postprocessing.registration_channel_idx[i],
                                            self.config.postprocessing.registration_shift[i]])
        else:
            registration_params = None

        return denoise_params, registration_params

    def _gen_coord_set(self):
        """
        Function creates a set of all position, time values to loop through for reconstruction

        Returns
        -------

        """

        self.pt_set = set()
        p_indices = set()
        t_indices = set()

        for p_entry in self.config.positions:
            if p_entry == 'all':
                for p in range(self.data.get_num_positions()):
                    p_indices.add(p)
                break
            elif isinstance(p_entry, int):
                p_indices.add(p_entry)
            elif isinstance(p_entry, list):
                for p in p_entry:
                    p_indices.add(p)
            elif isinstance(p_entry, tuple):
                for p in range(p_entry[0], p_entry[1]):
                    p_indices.add(p)
            else:
                raise ValueError(f'Did not understand entry {p_entry} in config specified positions')

        for t_entry in self.config.timepoints:
            if t_entry == 'all':
                for t in range(self.data.frames):
                    t_indices.add(t)
                break
            elif isinstance(t_entry, int):
                t_indices.add(t_entry)
            elif isinstance(t_entry, list):
                for t in t_entry:
                    t_indices.add(t)
            elif isinstance(t_entry, tuple):
                for t in range(t_entry[0],t_entry[1]):
                    t_indices.add(t)
            else:
                raise ValueError(f'Did not understand entry {t_entry} in config specified positions')

        self.num_t = len(t_indices)
        self.num_p = len(p_indices)

        for pos in p_indices:
            for time_point in t_indices:
                self.pt_set.add((pos, time_point))

    def _create_or_open_group(self, pt):
        try:
            self.pipeline.writer.create_position(pt[0])
            self.pipeline.writer.init_array(self.pipeline.data_shape,
                                            self.pipeline.chunk_size,
                                            self.pipeline.output_channels)
        except:
            self.pipeline.writer.open_position(pt[0])

    #TODO: use arbol print statements
    #TODO: Refactor Birefringence to Anisotropy
    def run(self):

        print(f'Beginning Reconstruction...')

        for pt in self.pt_set:
            start_time = time.time()

            self._create_or_open_group(pt)

            pt_data = self.data.get_array(pt[0])[pt[1]]

            stokes = self.pipeline.reconstruct_stokes_volume(pt_data)

            stokes = self.pre_processing(stokes)

            birefringence = self.pipeline.reconstruct_birefringence_volume(stokes)

            phase2D, phase3D = self.pipeline.reconstruct_phase_volume(stokes)

            birefringence, phase2D, phase3D, registered_data = self.post_processing(pt_data, phase2D, phase3D, birefringence)

            self.pipeline.write_data(pt, pt_data, stokes, birefringence, phase2D, phase3D, registered_data)

            end_time = time.time()
            print(f'Finishing Reconstructing P = {pt[0]}, T = {pt[1]} ({(end_time-start_time)/60:0.2f}) min')

    def pre_processing(self, stokes):

        denoise_params = self._get_preprocessing_params()

        return preproc_denoise(stokes, denoise_params) if denoise_params else stokes

    def post_processing(self, pt_data, phase2D, phase3D, birefringence):

        denoise_params, registration_params = self._get_postprocessing_params()

        phase2D_denoise = np.copy(phase2D)
        phase3D_denoise = np.copy(phase2D)
        birefringence_denoise = np.copy(birefringence)
        if denoise_params:
            for chan_param in denoise_params:
                if 'Retardance' in chan_param[0]:
                    birefringence_denoise[0] = post_proc_denoise(birefringence[0], chan_param)
                elif 'Orientation' in chan_param[0]:
                    birefringence_denoise[1] = post_proc_denoise(birefringence[1], chan_param)
                elif 'Brightfield' in chan_param[0]:
                    birefringence_denoise[2] = post_proc_denoise(birefringence[2], chan_param)
                elif 'Phase2D' in chan_param[0]:
                    phase2D_denoise = post_proc_denoise(phase3D, chan_param)
                elif 'Phase3D' in chan_param[0]:
                    phase3D_denoise = post_proc_denoise(phase3D, chan_param)
                else:
                    raise ValueError(f'Didnt understand post_proc denoise channel {chan_param[0]}')

        if registration_params:
            registered_stacks = []
            for param in registration_params:
                registered_stacks.append(translate_3D(pt_data[param[0]], param[1]))
        else:
            registered_stacks = None

        return birefringence_denoise, phase2D_denoise, phase3D_denoise, registered_stacks

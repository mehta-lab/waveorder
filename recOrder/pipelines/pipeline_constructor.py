from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
import time
from recOrder.pipelines.QLIPP_Pipelines import qlipp_pipeline
from recOrder.postproc.post_processing import *
from recOrder.preproc.pre_processing import *

class PipelineConstructor:
    """
    This will pull the necessary pipeline based off the config default.
    """

    #TODO: Reorganize arguments with new config reader
    #TODO: copy config to save_dir
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
            self.reconstructor = qlipp_pipeline(self.config, self.data, self.config.save_dir,
                                                self.config.data_save_name, self.config.mode)

        elif self.config.mode == 'denoise':
            raise NotImplementedError

        elif self.config.mode == 'UPTI':
            raise NotImplementedError

        elif self.config.mode == 'IPS':
            raise NotImplementedError


    def _get_preprocessing(self):
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

    def _get_postprocessing(self):
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


    #TODO: Create metadata dictionary to append to zarr attributes
    def _create_meta_dict(self):

        dict = {'Dataset': {}}

    def _gen_coord_set(self):
        """
        Function creates a set of all position, time values to loop through for reconstruction

        Returns
        -------

        """

        self.pt_set = set()
        p_indices = []
        t_indices = []

        for p_entry in self.config.positions:
            if p_entry == 'all':
                for p in range(self.data.get_num_positions()):
                    p_indices.append(p)
                break
            elif isinstance(p_entry,list):
                for p in p_entry:
                    p_indices.append(p)
            elif isinstance(p_entry, tuple):
                for p in range(p_entry[0],p_entry[1]):
                    p_indices.append(p)
            else:
                raise ValueError(f'Did not understand entry {p_entry} in config specified positions')

        for t_entry in self.config.timepoints:
            if t_entry == 'all':
                for t in range(self.data.frames):
                    t_indices.append(t)
                break
            elif isinstance(t_entry,list):
                for t in t_entry:
                    t_indices.append(t)
            elif isinstance(t_entry, tuple):
                for t in range(t_entry[0],t_entry[1]):
                    t_indices.append(p)
            else:
                raise ValueError(f'Did not understand entry {t_entry} in config specified positions')

        for pos in p_indices:
            for time_point in t_indices:
                self.pt_set.add((pos, time_point))

    def _create_or_open_group(self, pt):
        try:
            self.reconstructor.writer.create_position(pt[0])
            self.reconstructor.writer.init_array(self.reconstructor.data_shape,
                                                 self.reconstructor.chunk_size,
                                                 self.reconstructor.channels)
        except:
            self.reconstructor.writer.open_position(pt[0])


    #TODO: use arbol print statements
    def run(self):

        print(f'Beginning Reconstruction...')

        for pt in self.pt_set:
            start_time = time.time()

            self._create_or_open_group(pt)

            pt_data = self.data.get_array(pt[0])[pt[1]]

            stokes = self.reconstructor.reconstruct_stokes_volume(pt_data)
            stokes = self.pre_processing(stokes)

            birefringence = self.reconstructor.reconstruct_birefringence_volume(stokes)
            phase = self.reconstructor.reconstruct_phase_volume(stokes)

            birefringence, phase, registered_data = self.post_processing(pt_data, phase, birefringence)

            self.reconstructor.write_data(pt, pt_data, stokes, birefringence, phase, registered_data)

            end_time = time.time()
            print(f'Finishing Reconstructing P = {pt[0]}, T = {pt[1]} ({(end_time-start_time)/60:0.2f}) min')

    def pre_processing(self, stokes):

        denoise_params = self._get_preprocessing()

        return preproc_denoise(stokes, denoise_params) if denoise_params else stokes

    def post_processing(self, pt_data, phase, birefringence):

        denoise_params, registration_params = self._get_postprocessing()

        phase_denoise = np.copy(phase)
        birefringence_denoise = np.copy(birefringence)
        if denoise_params:
            for chan in denoise_params:
                if 'Retardance' in chan[0]:
                    birefringence_denoise[0] = post_proc_denoise(birefringence[0])
                elif 'Orientation' in chan[0]:
                    birefringence_denoise[1] = post_proc_denoise(birefringence[1])
                elif 'Brightfield' in chan[0]:
                    birefringence_denoise[2] = post_proc_denoise(birefringence[2])
                elif 'Phase' in chan[0]:
                    phase_denoise = post_proc_denoise(phase)
                else:
                    raise ValueError(f'Didnt understand post_proc denoise channel {chan[0]}')

        if registration_params:
            registered_stacks = []
            for param in registration_params:
                registered_stacks.append(translate_3D(pt_data[param[0]], param[1]))
        else:
            registered_stacks = None

        return birefringence_denoise, phase_denoise, registered_stacks





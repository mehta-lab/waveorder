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

    def __init__(self, method: str, mode: str, data_dir: str, save_dir: str, name: str,
                 config: ConfigReader, data_type: str = 'ometiff'):

        start = time.time()
        print('Reading Data...')
        data = MicromanagerReader(data_dir, data_type, extract_data=True)
        end = time.time()
        print(f'Finished Reading Data ({end - start / 60:0.1f} min')

        self.config = config
        self.data = data

        if method == 'QLIPP':
            self.reconstructor = qlipp_pipeline(config, data, save_dir, name, mode)

        elif mode == 'denoise':
            raise NotImplementedError

        elif mode == 'UPTI':
            raise NotImplementedError

        elif mode == 'IPS':
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
        if self.config.pre_proc_denoise_use:
            for i in range(len(self.config.preproc_denoise_channels)):
                threshold = 0.1 if self.config.preproc_denoise_thresholds is None \
                    else self.config.preproc_denoise_thresholds[i]
                level = 1 if self.config.preproc_denoise_levels is None \
                    else self.config.preproc_denoise_levels[i]

                denoise_params.append([self.config.preproc_denoise_channels[i], threshold, level])

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
        if self.config.post_proc_denoise_use:
            for i in range(len(self.config.postproc_denoise_channels)):
                threshold = 0.1 if self.config.postproc_denoise_thresholds is None \
                    else self.config.postproc_denoise_thresholds[i]
                level = 1 if self.config.postproc_denoise_levels is None \
                    else self.config.postproc_denoise_levels[i]

                denoise_params.append([self.config.postproc_denoise_channels[i], threshold, level])

        else:
            denoise_params = None

        registration_params = []
        if self.config.post_proc_registration_use:
            for i in range(len(self.config.postproc_registration_channel_idx)):
                registration_params.append([self.config.postproc_registration_channel_idx[i],
                                            self.config.post_proc_registration_shift[i]])
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

    #TODO: use arbol print statements
    def run(self):

        print(f'Beginning Reconstruction...')
        for pt in self.pt_set:

            self.reconstructor.writer.create_position(pt[0])
            self.reconstructor.writer.init_array(self.reconstructor.data_shape,
                                                 self.reconstructor.chunk_size,
                                                 self.reconstructor.channels)

            pt_data = self.data.get_array(pt[0], pt[1])

            stokes = self.reconstructor.reconstruct_stokes_volume(pt_data)
            stokes = self.pre_processing(stokes)

            birefringence = self.reconstructor.reconstruct_birefringence_volume(stokes)
            phase = self.reconstruct.reconstruct_phase_volume(stokes)

            denoised_data, registered_data = self.reconstructor.post_processing

            self.reconstructor.write_data(pt, pt_data, stokes, birefringence, phase)

    def pre_processing(self, stokes):

        denoise_params = self._get_preprocessing()

        return preproc_denoise(stokes, denoise_params) if denoise_params else stokes

    def post_processing(self, phase):

        #TODO: IMPLEMENT POSTPROC RETURN





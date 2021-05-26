from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
import time
from recOrder.pipelines.QLIPP_Pipelines import qlipp_3D_pipeline
from recOrder.postproc.post_processing import *
from recOrder.preproc.pre_proce import *

class PipelineConstructor:
    """
    This will pull the necessary pipeline based off the config default.
    """

    def __init__(self, mode: str, data_dir: str, save_dir: str, name: str, config: ConfigReader, data_type: str = 'ometiff'):

        start = time.time()
        print('Reading Data...')
        data = MicromanagerReader(data_dir, data_type, extract_data=True)
        end = time.time()
        print(f'Finished Reading Data ({end - start / 60:0.1f} min')

        self.config = config
        self.data = data

        if mode == 'QLIPP_3D':
            self.reconstructor = qlipp_3D_pipeline(config, data, save_dir, name)

        elif mode == 'QLIPP_2D':
            self.reconstructor = qlipp_2D_pipeline(config, data, save_dir, name)

        ##TODO: determine automatically which method to compute stokes (IPS, QLIPP, UPTI)
        elif mode == 'stokes':
            raise NotImplementedError

        elif mode == 'denoise':
            raise NotImplementedError

        elif mode == 'UPTI':
            raise NotImplementedError

        elif mode == 'IPS':
            raise NotImplementedError

        elif mode == 'None':
            raise NotImplementedError
            # self.reconstructor == 'Custom'

    def _get_preprocessing(self):
        pass

    def _get_postprocessing(self):
        pass

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

            self.writer.create_position(pt[0])
            self.writer.init_array(self.reconstructor.data_shape,
                                   self.reconstructor.chunk_size,
                                   self.reconstructor.channels)

            # self.pre_processing(pt)
            self.reconstructor.reconstruct_volume(pt)
            # self.reconstructor.post_processing

    def pre_processing(self):
        pass
from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
import time
from recOrder.pipelines.QLIPP_Pipelines import qlipp_3D_pipeline

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

    def run(self):
        self.reconstructor.reconstruct_all()



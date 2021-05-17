from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
from recOrder.pipelines.QLIPP_Pipelines import qlipp_3D_pipeline


class PipelineConstructor:
    """
    This will pull the necessary pipeline based off the config default.
    """

    def __init__(self, config: ConfigReader, data: MicromanagerReader, sample: str):

        if config.default == 'QLIPP_3D':
            self.reconstructor = qlipp_3D_pipeline(config, data, sample)

        elif config.default == 'UPTI':
            raise NotImplementedError

        elif config.default == 'IPS':
            raise NotImplementedError

        elif config.default == 'None':
            raise NotImplementedError
            # self.reconstructor == 'Custom'

        self.config = config

    def run(self):
        self.reconstructor.reconstruct_all()

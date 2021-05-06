from waveorder.io.reader import MicromanagerReader
import os
from recOrder.pipelines.QLIPP_Pipeline_Constructor import qlipp_pipeline_constructor
from recOrder.io.config_reader import ConfigReader

def run_pipeline(config: ConfigReader):

    if 'QLIPP' in config.default:
        for sample in config.samples:
            data = MicromanagerReader(os.path.join(config.data_dir, sample))
            pipeline = qlipp_pipeline_constructor(config, data, sample)

            pipeline.run_reconstruction()


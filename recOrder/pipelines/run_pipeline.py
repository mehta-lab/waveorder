from waveorder.io.reader import MicromanagerReader
import os
from recOrder.pipelines.pipeline_constructor import PipelineConstructor
from recOrder.io.config_reader import ConfigReader

def run_pipeline(config: ConfigReader):
    """
    This function will load the data and call the correct pipeline based off of the config.
    It will then run the pipeline based on config parameters.  Initialized by CLI.

    Parameters
    ----------
    config:     ConfigReader object as read by the runReconstruction.py

    Returns
    -------

    """

    for sample in config.samples:

        print('Reading Data...')
        # data = MicromanagerReader(os.path.join(config.data_dir, sample), config.data_type, extract_data=True)
        data = MicromanagerReader(os.path.join(config.data_dir, sample), config.data_type)
        data.frames = 121
        data.reader._create_stores(data.reader.master_ome_tiff)
        print('Finished Reading Data')

        pipeline = PipelineConstructor(config, data, sample)
        pipeline.run()
from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import load_bg
from recOrder.compute.QLIPP_compute import initialize_reconstructor, reconstruct_QLIPP_birefringence
from recOrder.pipelines.QLIPP_Pipelines import qlipp_3D_pipeline
from waveorder.util import wavelet_softThreshold
import json
import numpy as np
import time


class PipelineConstructor:

    def __init__(self, config: ConfigReader, data: MicromanagerReader, sample: str):

        if config.default == 'QLIPP_3D':
            self.reconstructor = qlipp_3D_pipeline(config, data, sample)

        elif config.default == 'UPTI':
            raise NotImplementedError

        elif config.default == 'IPS':
            raise NotImplementedError

        elif config.default == 'None':
            raise NotImplementedError
            pass
            # self.reconstructor == 'Custom'

        self.config = config
        # self.add_preprocessing(config.pre_processing)
        # self.add_postprocessing(config.post_processing)


    # def add_preprocessing(self):
    #
    #     if self.config.denoise == 'denoise':
    #
    #     pass
    #
    # def add_postprocessing(self):
    #     pass

    def run(self):
        self.reconstructor.reconstruct_all()

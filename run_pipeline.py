from .recOrder.io.config_reader import ConfigReader
from .recOrder.io.QLIPP_Pipeline_Constructor import

config = ConfigReader(path)

for dataset in config.samples:
    pass
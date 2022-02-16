import click
import os
from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager

@click.command()
@click.option('--method', required=False, type=str, help='mode of reconstruction: \
                                                      QLIPP,IPS,UPTI')
@click.option('--mode', required=False, type=str, help='mode of reconstruction: \
                                                      2D,3D,Stokes')
@click.option('--data_dir', required=False, type=click.Path(exists=True), help='path to the data')
@click.option('--save_dir', required=False, type=click.Path(), help='path to the save directory')
@click.option('--name', required=False, type=str, help='name to use for saving the data')
@click.option('--config', required=False, type=click.Path(exists=True), help='path to config yml file')
@click.option('--overwrite', required=False, type=bool,
              help='whether or not to overwrite any previous data under this name')
def parse_args(method, mode, data_dir, save_dir, name, config, overwrite=False):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, method, mode, data_dir, save_dir, name, config, overwrite):
            self.method = method
            self.mode = mode
            self.config = config
            self.data_dir = data_dir
            self.save_dir = save_dir
            self.name = name
            self.overwrite = overwrite

    return Args(method, mode, data_dir, save_dir, name, config, overwrite)

def main():

    Args = parse_args(standalone_mode=False)

    if Args.config:
        if not os.path.exists(Args.config):
            raise ValueError('Specified config path does not exist')
        else:
            config = ConfigReader(Args.config, Args.data_dir, Args.save_dir, Args.method, Args.mode, Args.name,
                                  immutable=False)
            config.save_yaml()
    else:
        config = ConfigReader(None, Args.data_dir, Args.save_dir, Args.method, Args.mode, Args.name, immutable=False)
        config.save_yaml()

    manager = PipelineManager(config, Args.overwrite)
    manager.run()

import click
import sys
import os
from recOrder.io.zarr_converter import ZarrConverter
from recOrder.io.config_reader import ConfigReader
from recOrder.pipelines.pipeline_manager import PipelineManager

# From https://stackoverflow.com/questions/50975203/display-help-if-incorrect-or-mission-option-in-click
class ShowUsageOnMissingError(click.Command):
    def __call__(self, *args, **kwargs):
        try:
            return super(ShowUsageOnMissingError, self).__call__(
                *args, standalone_mode=False, **kwargs)
        except click.MissingParameter as exc:
            exc.ctx = None
            exc.show(file=sys.stdout)
            click.echo()
            try:
                super(ShowUsageOnMissingError, self).__call__(['--help'])
            except SystemExit:
                sys.exit(exc.exit_code)

@click.command(cls=ShowUsageOnMissingError)
@click.help_option('-h', '--help')
def help():
    """\033[92mrecOrder: Computational Toolkit for Label-Free Imaging\033[0m

    recOrder has two command-line interfaces:

    \033[96mrecOrder.reconstruct \033[0m\n
    Allows you to reconstruct data through a variety of pipelines:
    Fluorescence Deconvolution, QLIPP, and Phase from Brightfield
    Please see example config files in /examples/example_configs

    \033[96mrecOrder.convert \033[0m\n
    Converts Micromanager .tif files to ome-zarr data format

    To use recOrder\'s napari plugin, use \033[96mnapari -w recOrder-napari\033[0m

    Thank you for using recOrder.
    """
    print(help.__doc__)

@click.command(cls=ShowUsageOnMissingError)
@click.help_option('-h', '--help')
@click.option('--method', required=False, type=str, help='mode of reconstruction: \
                                                      QLIPP, IPS, UPTI')
@click.option('--mode', required=False, type=str, help='mode of reconstruction: \
                                                      2D, 3D, Stokes')
@click.option('--data_dir', required=False, type=click.Path(exists=True), help='path to the data')
@click.option('--save_dir', required=False, type=click.Path(), help='path to the save directory')
@click.option('--name', required=False, type=str, help='name to use for saving the data')
@click.option('--config', required=True, type=click.Path(exists=True), help='path to config yml file')
@click.option('--overwrite', required=False, type=bool,
              help='whether or not to overwrite any previous data under this name')
def reconstruct(method, mode, data_dir, save_dir, name, config, overwrite=False):

    if config:
        if not os.path.exists(config):
            raise ValueError('Specified config path does not exist')
        else:
            config = ConfigReader(config, data_dir, save_dir, method, mode, name,
                                  immutable=False)
            config.save_yaml()
    else:
        config = ConfigReader(None, data_dir, save_dir, method, mode, name, immutable=False)
        config.save_yaml()

    manager = PipelineManager(config, overwrite)
    manager.run()

@click.command(cls=ShowUsageOnMissingError)
@click.help_option('-h', '--help')
@click.option('--input', required=True, type=click.Path(exists=True), help='path to the raw data folder containing ome.tifs')
@click.option('--output', required=True, type=str, help='full path to save the zarr store (../../Experiment.zarr')
@click.option('--data_type', required=False, type=str, help='Data type, "ometiff", "upti", "zarr"')
@click.option('--replace_pos_name', required=False, type=bool, help='whether or not to append position name to data')
@click.option('--format_hcs', required=False, type=bool, help='whether or not to format the data as an HCS "well-plate"')
def convert(input, output, data_type, replace_pos_name, format_hcs):
    converter = ZarrConverter(input, output, data_type, replace_pos_name, format_hcs)
    converter.run_conversion()
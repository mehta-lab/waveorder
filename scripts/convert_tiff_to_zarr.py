from recOrder.io.zarr_converter import ZarrConverter
import click
import glob
import os

#TODO: Make save_dir all the way to .zarr
#todo: make data_dir be the folder containing ome-tiffs
# specify --input --output

@click.command()
@click.option('--input', required=True, type=str, help='path to the raw data folder containing ome.tifs')
@click.option('--output', required=True, type=str, help='full path to save the zarr store (../../Experiment.zarr')
@click.option('--add_pos_name', required=False, type=bool, help='whether or not to append position name to data')
def parse_args(input, output, add_pos_name):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, input, output, add_pos_name = False):
            self.input = input
            self.output = output
            self.add_pos_name = add_pos_name

    return Args(input, output, add_pos_name)

def main():

    Args = parse_args(standalone_mode=False)

    converter = ZarrConverter(Args.input, Args.output, Args.add_pos_name)
    converter.run_conversion()



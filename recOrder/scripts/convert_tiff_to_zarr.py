from recOrder.io.zarr_converter import ZarrConverter
import click

@click.command()
@click.option('--input', required=True, type=click.Path(exists=True), help='path to the raw data folder containing ome.tifs')
@click.option('--output', required=True, type=str, help='full path to save the zarr store (../../Experiment.zarr')
@click.option('--data_type', required=False, type=str, help='Data type, "ometiff", "upti", "zarr"')
@click.option('--replace_pos_name', required=False, type=bool, help='whether or not to append position name to data')
@click.option('--format_hcs', required=False, type=bool, help='whether or not to format the data as an HCS "well-plate"')
def parse_args(input, output, data_type, replace_pos_name, format_hcs):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, input, output, data_type=None, replace_pos_name=False, format_hcs=False):
            self.input = input
            self.output = output
            self.data_type = data_type
            self.replace_pos_name = replace_pos_name
            self.format_hcs = format_hcs

    return Args(input, output, data_type, replace_pos_name, format_hcs)

def main():

    Args = parse_args(standalone_mode=False)

    converter = ZarrConverter(Args.input, Args.output, Args.data_type, Args.replace_pos_name, Args.format_hcs)
    converter.run_conversion()



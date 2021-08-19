from recOrder.io.zarr_converter import ZarrConverter
import click

@click.command()
@click.option('--save_dir', required=True, type=str, help='path to the save directory')
@click.option('--save_name', required=False, type=str, help='name to use for saving the data')
def parse_args(save_dir, save_name):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, save_dir, save_name=None):
            self.save_dir = save_dir
            self.save_name = save_name

    return Args(save_dir, save_name)

def main():

    Args = parse_args(standalone_mode=False)

    converter = ZarrConverter(Args.save_dir, Args.save_name)
    converter.run_conversion()



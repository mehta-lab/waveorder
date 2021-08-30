from recOrder.io.zarr_converter import ZarrConverter
import click


@click.command()
@click.option('--data_dir', required=True, type=str, help='path to the raw data directory')
@click.option('--save_dir', required=True, type=str, help='path to the save directory')
@click.option('--save_name', required=False, type=str, help='name to use for saving the data')
@click.option('--add_pos_name', required=False, type=bool, help='whether or not to append position name to data')
def parse_args(data_dir, save_dir, save_name, add_pos_name):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, data_dir, save_dir, save_name=None, add_pos_name = False):
            self.data_dir = data_dir
            self.save_dir = save_dir
            self.save_name = save_name
            self.add_pos_name = add_pos_name

    return Args(data_dir, save_dir, save_name, add_pos_name)

def main():

    Args = parse_args(standalone_mode=False)

    converter = ZarrConverter(Args.data_dir, Args.save_dir, Args.save_name, Args.add_pos_name)
    converter.run_conversion()



import click

@click.command()
def print_help_statements():

    click.echo(click.style('recOrder: Computational Toolkit for Label-Free Imaging', underline=True, fg='red'))
    click.echo('Thank you for using recOrder.  Here are some brief commands which are included in the package:\n')

    click.echo(click.style('recOrder.reconstruct', bold=True, fg='cyan'))
    click.echo('Allows you to reconstruct data through a variety of pipelines:')
    click.echo('Fluorescence Deconvolution, QLIPP, and Phase from Brightfield')
    click.echo('Please see example config files in /examples/example_configs')

    click.echo(click.style('  Instructions', italic=True, fg='magenta'))
    click.echo('    --method (str) method of reconstruction: QLIPP,IPS,UPTI')
    click.echo('    --mode (str) mode of reconstruction: 2D, 3D')
    click.echo('    --data_dir (str) path to raw data folder')
    click.echo('    --save_dir (str) path to folder where reconstructed data will be saved')
    click.echo('    --name (str) name under which to save the reconstructed data')
    click.echo('    --config (str) path to configuration file (see /examples/example_configs')
    click.echo('    --overwrite (bool) True/False whether or not to overwrite data that exists under save_dir/name')

    click.echo(click.style('\nrecOrder.convert', bold=True, fg='cyan'))
    click.echo('Allows you to convert Micromanager .tif files to ome-zarr data format')

    click.echo(click.style('  Instructions', italic=True, fg='magenta'))
    click.echo('    --input (str) path to folder containing micromanager tif files')
    click.echo('    --output (str) full path to save the ome-zarr data, i.e. /path/to/Data.zarr')
    click.echo('    --data_type (str) micromananger data-type: ometiff, singlepagetiff')
    click.echo('    --replace_pos_names (bool) True/False whether to replace zarr position names with ones listed in micro-manager metadata')
    click.echo('    --format_hcs (bool) if tiled micromanager dataset, format in ome-zarr HCS format')

def main():
    print_help_statements()
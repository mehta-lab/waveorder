import sys
from contextlib import redirect_stdout

import click

from waveorder.scripts.samples import download_and_unzip


@click.command("download-examples")
def _download_examples_cli():
    """Download example polarization data from Zenodo (10 MB, cached).

    Prints the path to the downloaded raw_data.zarr.
    """
    with redirect_stdout(sys.stderr):
        data_path, _ = download_and_unzip("target")
    click.echo(data_path)

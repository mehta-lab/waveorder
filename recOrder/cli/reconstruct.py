import os

import click

from recOrder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_cli,
)
from recOrder.cli.compute_transfer_function import (
    compute_transfer_function_cli,
)
from recOrder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from recOrder.cli.utils import get_output_paths


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def reconstruct(input_position_dirpaths, config_filepath, output_dirpath):
    """
    Reconstruct a dataset using a configuration file. This is a
    convenience function for a `compute-tf` call followed by a `apply-inv-tf`
    call.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`, then applies that transfer function
    to all positions in the list `input-position-dirpaths`, so all positions
    must have the same TCZYX shape.

    See /examples for example configuration files.

    >> recorder reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """

    # Handle transfer function path
    output_directory = os.path.dirname(output_dirpath)
    transfer_function_path = os.path.join(
        output_directory, "transfer_function.zarr"
    )

    # Compute transfer function
    compute_transfer_function_cli(
        input_position_dirpaths[0], config_filepath, transfer_function_path
    )

    # Apply inverse to each position
    output_position_dirpaths = get_output_paths(
        input_position_dirpaths, output_dirpath
    )

    for input_position_dirpath, output_position_dirpath in zip(
        input_position_dirpaths, output_position_dirpaths
    ):
        apply_inverse_transfer_function_cli(
            input_position_dirpath,
            transfer_function_path,
            config_filepath,
            output_position_dirpath,
        )

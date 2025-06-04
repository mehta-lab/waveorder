from pathlib import Path

import click

from waveorder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_cli,
)
from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli,
)
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    ram_multiplier,
    unique_id,
)


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
def reconstruct(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int,
):
    """
    Reconstruct a dataset using a configuration file. This is a
    convenience function for a `compute-tf` call followed by a `apply-inv-tf`
    call.

    Calculates the transfer function based on the shape of the position
    in the path `input-position-dirpaths`, then applies that transfer function
    to the position in the path `input-position-dirpaths`.

    See /examples for example configuration files.

    >> waveorder reconstruct -i ./input.zarr/B/1/000000 -c ./examples/birefringence.yml -o ./output.zarr
    """

    if len(input_position_dirpaths) > 1:
        raise ValueError(
            "Reconstruct on waveorder only supports a single input position directory. For parallel reconstruction, use https://github.com/czbiohub-sf/biahub."
        )

    # Handle transfer function path
    transfer_function_path = output_dirpath.parent / Path(
        "transfer_function_" + config_filepath.stem + ".zarr"
    )

    # Compute transfer function
    compute_transfer_function_cli(
        input_position_dirpaths[0],
        config_filepath,
        transfer_function_path,
    )

    # Apply inverse transfer function
    apply_inverse_transfer_function_cli(
        input_position_dirpaths[0],
        transfer_function_path,
        config_filepath,
        output_dirpath,
        num_processes,
    )

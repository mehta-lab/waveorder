import threading
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
    unique_id,
)
from waveorder.cli.printing import JM


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
@unique_id()
def _reconstruct_cli(
    input_position_dirpaths,
    config_filepath,
    output_dirpath,
    num_processes,
    unique_id,
):
    """
    Reconstruct a dataset using a configuration file. This is a
    convenience function for a `compute-tf` call followed by a `apply-inv-tf`
    call.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`, then applies that transfer function
    to all positions in the list `input-position-dirpaths`, so all positions
    must have the same TCZYX shape.

    See /examples for example configuration files.

    >> waveorder reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """
    threading.Thread(
        target=_reconstruct_cli_thread,
        args=(
            input_position_dirpaths,
            config_filepath,
            output_dirpath,
            num_processes,
            unique_id,
        ),
    ).start()


def _reconstruct_cli_thread(
    input_position_dirpaths,
    config_filepath,
    output_dirpath,
    num_processes,
    unique_id,
):
    has_errored = False
    if unique_id != "":
        JM.start_client()
        JM.do_print = False
        JM.set_shorter_timeout()
        JM.put_Job_in_list(uID=unique_id, msg="Initialization")

    # Handle transfer function path
    transfer_function_path = output_dirpath.parent / Path(
        "transfer_function_" + config_filepath.stem + ".zarr"
    )

    # Compute transfer function
    try:
        compute_transfer_function_cli(
            input_position_dirpaths[0],
            config_filepath,
            transfer_function_path,
            unique_id,
        )
    except Exception as exc:
        has_errored = True
        err = "Error: " + str("\n".join(exc.args))
        print(err)
        if unique_id != "":
            JM.put_Job_in_list(uID=unique_id, msg=err)

    # Apply inverse transfer function
    try:
        apply_inverse_transfer_function_cli(
            input_position_dirpaths,
            transfer_function_path,
            config_filepath,
            output_dirpath,
            num_processes,
            unique_id,
        )
    except Exception as exc:
        has_errored = True
        err = "Error: " + str("\n".join(exc.args))
        print(err)
        if unique_id != "":
            JM.put_Job_in_list(uID=unique_id, msg=err)

    if unique_id != "":
        if has_errored:
            JM.put_Job_in_list(
                uID=unique_id, msg="Submitted job triggered an exception"
            )
        else:
            JM.put_Job_in_list(uID=unique_id, msg="Job completed successfully")
        JM.put_Job_completion_in_list(uID=unique_id, finished=True)

from pathlib import Path
from typing import Callable

import click
import torch.multiprocessing as mp
from iohub.ngff import Plate, open_ome_zarr
from natsort import natsorted

from recOrder.cli.option_eat_all import OptionEatAll


def _validate_and_process_paths(
    ctx: click.Context, opt: click.Option, value: str
) -> list[Path]:
    # Sort and validate the input paths
    input_paths = [Path(path) for path in natsorted(value)]
    for path in input_paths:
        with open_ome_zarr(path, mode="r") as dataset:
            if isinstance(dataset, Plate):
                raise ValueError(
                    "Please supply a list of positions instead of an HCS plate. Likely fix: replace 'input.zarr' with 'input.zarr/*/*/*' or 'input.zarr/0/0/0'"
                )
    return input_paths


def _str_to_path(ctx: click.Context, opt: click.Option, value: str) -> Path:
    return Path(value)


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--input-position-dirpaths",
            "-i",
            cls=OptionEatAll,
            type=tuple,
            required=True,
            callback=_validate_and_process_paths,
            help="List of paths to input positions, each with the same TCZYX shape. Supports wildcards e.g. 'input.zarr/*/*/*'.",
        )(f)

    return decorator


def config_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-filepath",
            "-c",
            required=True,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            callback=_str_to_path,
            help="Path to YAML configuration file.",
        )(f)

    return decorator


def transfer_function_dirpath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--transfer-function-dirpath",
            "-t",
            required=True,
            type=click.Path(exists=False),
            callback=_str_to_path,
            help="Path to transfer function .zarr.",
        )(f)

    return decorator


def output_dirpath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-dirpath",
            "-o",
            required=True,
            type=click.Path(exists=False),
            callback=_str_to_path,
            help="Path to output directory.",
        )(f)

    return decorator


# TODO: this setting will have to be collected from SLURM?
def processes_option(default: int = None) -> Callable:
    def check_processes_option(ctx, param, value):
        max_processes = mp.cpu_count()
        if value > max_processes:
            raise click.BadParameter(
                f"Maximum number of processes is {max_processes}"
            )
        return value

    def decorator(f: Callable) -> Callable:
        return click.option(
            "--num_processes",
            "-j",
            default=default or mp.cpu_count(),
            type=int,
            help="Number of processes to run in parallel.",
            callback=check_processes_option,
        )(f)

    return decorator

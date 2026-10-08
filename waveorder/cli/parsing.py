from pathlib import Path
from typing import Callable

import click

from waveorder.cli.option_eat_all import OptionEatAll


def _validate_and_process_paths(ctx: click.Context, opt: click.Option, value: str) -> list[Path]:
    # Deferred imports: iohub and natsort are heavy (pull in torch via zarr/numpy chain).
    # Only needed when the command actually runs, not for --help.
    from iohub.ngff import Plate, open_ome_zarr
    from natsort import natsorted

    # Sort and validate the input paths, expanding plates into lists of positions
    input_paths = [Path(path) for path in natsorted(value)]
    # Filter out non-directories (e.g., zarr.json files from glob expansion)
    input_paths = [path for path in input_paths if path.is_dir()]
    for path in input_paths:
        with open_ome_zarr(path, mode="r") as dataset:
            if isinstance(dataset, Plate):
                plate_path = input_paths.pop()
                for position in dataset.positions():
                    input_paths.append(plate_path / position[0])

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


def threads_option(default: int = None) -> Callable:
    """CLI option for number of threads to run in parallel.

    Accepts --num-threads (canonical) or --num-processes / --num_processes
    (deprecated aliases, kept for backward compatibility with existing scripts).
    """
    import os

    def check_threads_option(ctx, param, value):
        if value is None:
            return default or 1
        max_threads = os.cpu_count() or 1
        if value > max_threads:
            raise click.BadParameter(f"Maximum number of threads is {max_threads}")
        return value

    def deprecated_processes_callback(ctx, param, value):
        if value is not None:
            click.echo(
                "Warning: --num-processes / --num_processes is deprecated. "
                "Use --num-threads instead.",
                err=True,
            )
            ctx.params["num_threads"] = check_threads_option(ctx, param, value)
        return value

    def decorator(f: Callable) -> Callable:
        f = click.option(
            "--num-threads",
            "-j",
            default=default or 1,
            type=int,
            help="Number of threads to run in parallel.",
            callback=check_threads_option,
        )(f)
        f = click.option(
            "--num-processes",
            "--num_processes",
            default=None,
            type=int,
            is_eager=True,
            expose_value=False,
            hidden=True,
            help="Deprecated: use --num-threads instead.",
            callback=deprecated_processes_callback,
        )(f)
        return f

    return decorator


# Keep old name as alias for backward compatibility with any direct Python imports
processes_option = threads_option


def write_config_scale_to_output() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--write-config-scale-to-output",
            is_flag=True,
            default=False,
            help="Write the reconstruction config's pixel sizes to the output zarr instead of copying from the input.",
        )(f)

    return decorator


def unique_id() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--unique-id",
            "-uid",
            default="",
            required=False,
            type=str,
            help="Unique ID.",
        )(f)

    return decorator

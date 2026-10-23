from pathlib import Path
from typing import Annotated

import typer

INPUT_PATHS_HELP = "List of paths to input positions. Supports wildcards e.g. 'input.zarr/*/*/*'."


def _validate_and_process_paths(value: list[Path]) -> list[Path]:
    # Deferred imports keep command help independent of the scientific stack.
    from iohub.cli import expand_position_dirpaths
    from natsort import natsorted

    # Expands plates and quoted globs, skips file matches (e.g. zarr.json), and
    # raises BadParameter when nothing is left. Sorting first keeps the input
    # order, which sets the output keys for single-position stores.
    return expand_position_dirpaths([str(path) for path in natsorted(value)])


InputPositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--input-position-dirpaths",
        "-i",
        callback=_validate_and_process_paths,
        help=INPUT_PATHS_HELP,
    ),
]
ConfigFilepath = Annotated[
    Path,
    typer.Option(
        "--config-filepath",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to YAML configuration file.",
    ),
]
TransferFunctionDirpath = Annotated[
    Path,
    typer.Option(
        "--transfer-function-dirpath",
        "-t",
        help="Path to transfer function .zarr.",
    ),
]
OutputDirpath = Annotated[
    Path,
    typer.Option("--output-dirpath", "-o", help="Path to output directory."),
]
WriteConfigScaleToOutput = Annotated[
    bool,
    typer.Option(
        "--write-config-scale-to-output",
        help=("Write the reconstruction config's pixel sizes to the output zarr instead of copying from the input."),
    ),
]
UniqueId = Annotated[
    str,
    typer.Option("--unique-id", "-uid", help="Unique ID."),
]


def check_processes_option(value: int) -> int:
    # Deferred: torch.multiprocessing pulls in all of torch.
    import torch.multiprocessing as mp

    max_processes = mp.cpu_count()
    if value > max_processes:
        raise typer.BadParameter(f"Maximum number of processes is {max_processes}")
    return value


ProcessesOption = Annotated[
    int,
    typer.Option(
        "--num_processes",
        "-j",
        callback=check_processes_option,
        help="Number of processes to run in parallel.",
    ),
]
ResumeOption = Annotated[
    bool,
    typer.Option(
        "--resume",
        help=(
            "Skip timepoints that an interrupted previous run into the same output already finished. "
            "Changed reconstruction settings or a recomputed transfer function are detected and "
            "recomputed; changed input data is not. Needs an OME-Zarr v0.5 output."
        ),
    ),
]

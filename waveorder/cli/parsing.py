from pathlib import Path
from typing import Annotated

import typer


INPUT_PATHS_HELP = (
    "List of paths to input positions, each with the same TCZYX shape. "
    "Supports wildcards e.g. 'input.zarr/*/*/*'."
)


def _validate_and_process_paths(value: list[Path]) -> list[Path]:
    # Deferred imports keep command help independent of the scientific stack.
    from iohub.ngff import Plate, open_ome_zarr
    from natsort import natsorted

    input_paths = [Path(path) for path in natsorted(value) if Path(path).is_dir()]
    expanded_paths = []
    for path in input_paths:
        with open_ome_zarr(path, mode="r") as dataset:
            if isinstance(dataset, Plate):
                expanded_paths.extend(path / Path(*position_key) for position_key, _ in dataset.positions())
            else:
                expanded_paths.append(path)
    return expanded_paths


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
        help=(
            "Write the reconstruction config's pixel sizes to the output zarr "
            "instead of copying from the input."
        ),
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

import os
from pathlib import Path

import click
import numpy as np
import xarray as xr
from iohub import read_images
from iohub.convert import TIFFConverter
from iohub.fov import BaseFOVMapping
from iohub.ngff import open_ome_zarr
from iohub.ngff.nodes import NGFFNode, Plate
from iohub.reader import _infer_format, sizeof_fmt


def resolve_time_indices(time_indices, num_timepoints: int) -> list[int]:
    """Resolve time_indices config value to a concrete list of ints."""
    if time_indices == "all":
        return list(range(num_timepoints))
    elif isinstance(time_indices, list):
        return time_indices
    else:
        return [int(time_indices)]


def generate_valid_position_key(index: int) -> tuple[str, str, str]:
    """Generate a valid HCS position key for single-position stores.

    Args:
        index: Position index (0-based)

    Returns:
        Tuple of (row, column, field) with alphanumeric characters only
    """
    row = chr(65 + (index // 10))  # A, B, C, etc.
    column = str((index % 10) + 1)  # 1, 2, 3, etc.
    field = "0"  # Always 0 for single positions
    return (row, column, field)


def is_single_position_store(position_path: Path) -> bool:
    """Check if a position path is from a single-position store (not HCS plate).

    Args:
        position_path: Path to the position directory

    Returns:
        True if it's a single-position store, False if it's part of an HCS plate
    """
    try:
        # Try to open as HCS plate 3 levels up
        open_ome_zarr(position_path.parent.parent.parent, mode="r")
        return False  # Successfully opened as plate
    except (RuntimeError, FileNotFoundError):
        return True  # Not a plate structure


def apply_inverse_to_zyx_and_save(
    func,
    input_data: xr.DataArray,
    output_path: Path,
    input_channel_names: list[str],
    t_idx: int = 0,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Load a zyx array from an xarray DataArray, apply a transformation and save the result to file.

    Parameters
    ----------
    func : callable
        Model function: xr.DataArray CZYX in, xr.DataArray CZYX out.
    input_data : xr.DataArray
        5D TCZYX input data.
    output_path : Path
        Path to the output position.
    input_channel_names : list[str]
        Channel names to select from input_data.
    t_idx : int
        Time index to process.
    verbose : bool
        Print progress messages per time point.
    **kwargs
        Additional arguments passed to func.
    """
    if verbose:
        click.echo(f"Reconstructing t={t_idx}")

    # Extract CZYX xarray slice
    czyx_slice = input_data.isel(t=t_idx).sel(c=input_channel_names)

    # Check if all values are zeros or NaN
    if _check_nan_n_zeros(czyx_slice.values):
        click.echo(f"All values at t={t_idx} are zero or Nan, skipping reconstruction.")
        return

    # Apply transformation (returns xr.DataArray CZYX)
    output_czyx = func(czyx_slice, **kwargs)

    # Add t dimension from input coords
    t_coord = input_data.coords["t"].values[t_idx : t_idx + 1]
    t_attrs = input_data.coords["t"].attrs

    output_xa = output_czyx.expand_dims(dim={"t": t_coord}, axis=0)
    output_xa["t"].attrs = t_attrs

    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_position:
        output_position.write_xarray(output_xa)

    if verbose:
        click.echo(f"Finished writing t={t_idx}")


def estimate_resources(shape, settings, num_processes):
    T, C, Z, Y, X = shape

    gb_ram_per_cpu = 0
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    voxel_resource_multiplier = 4
    fourier_resource_multiplier = 32
    input_memory = Z * Y * X * gb_per_element

    if settings.birefringence is not None:
        gb_ram_per_cpu += input_memory * voxel_resource_multiplier
    if settings.phase is not None:
        gb_ram_per_cpu += input_memory * fourier_resource_multiplier
    if settings.fluorescence is not None:
        gb_ram_per_cpu += input_memory * fourier_resource_multiplier
    ram_multiplier = 1
    gb_ram_per_cpu = np.ceil(np.max([1, ram_multiplier * gb_ram_per_cpu])).astype(int)
    num_cpus = np.min([32, num_processes])

    return num_cpus, gb_ram_per_cpu


def _check_nan_n_zeros(input_array):
    """
    Checks if data are all zeros or nan
    """
    return np.all(np.isnan(input_array)) or np.all(input_array == 0)


def convert_data(tif_path, latest_out_path, prefix=""):
    """
    Converts Micro-Manager ome-tif to .zarr
    """
    converter = TIFFConverter(
        os.path.join(tif_path, prefix),
        latest_out_path,
    )
    converter()


def run_convert(ome_tif_path):
    """
    Converts Micro-Manager ome-tif to .zarr and returns output path
    """
    ome_tif_folder_path = Path(ome_tif_path).absolute()
    out_path = os.path.join(
        ome_tif_folder_path.parent.absolute(),
        ome_tif_folder_path.name + "_converted" + ".zarr",
    )
    if not Path(out_path).exists():
        convert_data(ome_tif_folder_path, out_path)
    else:
        print("Not converting ome-tif to zarr: Destination folder exists.")
    return out_path


def validate_and_process_paths(value: str) -> list[Path]:
    """
    Validates and Processes Converted Micro-Manager ome-tif zarr paths
    """
    # Sort and validate the input paths, expanding plates into lists of positions
    input_paths = [Path(value)]
    # Filter out non-directories (e.g., zarr.json files from glob expansion)
    input_paths = [path for path in input_paths if path.is_dir()]
    for path in input_paths:
        with open_ome_zarr(path, mode="r") as dataset:
            if isinstance(dataset, Plate):
                plate_path = input_paths.pop()
                for position in dataset.positions():
                    input_paths.append(plate_path / position[0])

    return input_paths


def check_folder_for_ometiff(input_data_folder: Path) -> bool:
    """
    Checks for Micro-Manager ome-tif folder
    """
    try:
        data_type, extra_info = _infer_format(input_data_folder)
    except (ValueError, RuntimeError):
        return False
    if data_type == "ometiff":
        return True
    return False


def get_dataset_info(path: str):
    """Retrieve summary information for a dataset.

    Parameters
    ----------
    path : StrOrBytesPath
        Path to the dataset
    """
    path = Path(path).resolve()
    try:
        fmt, extra_info = _infer_format(path)
        if fmt == "omezarr" and extra_info in ("0.4", "0.5"):
            reader = open_ome_zarr(path, mode="r", version=extra_info)
        else:
            reader = read_images(path, data_type=fmt)
    except (ValueError, RuntimeError):
        print("Error: No compatible dataset is found.")
        return None

    fmt_msg = f"Format:\t\t\t {fmt}"
    if extra_info:
        if extra_info.startswith("0."):
            fmt_msg += " v" + extra_info
    sum_msg = "=== Summary ==="
    ch_msg = f"Channel names:\t\t {reader.channel_names}"
    msgs = []
    if isinstance(reader, BaseFOVMapping):
        _, first_fov = next(iter(reader))
        shape_msg = ", ".join([f"{a}={s}" for s, a in zip(first_fov.shape, ("T", "C", "Z", "Y", "X"))])
        msgs.extend(
            [
                sum_msg,
                fmt_msg,
                f"FOVs:\t\t\t {len(reader)}",
                f"FOV shape:\t\t {shape_msg}",
                ch_msg,
                f"(Z, Y, X) scale (um):\t {first_fov.zyx_scale}",
            ]
        )
        if reader.micromanager_summary:
            result_string = "\n".join(f"{key}:\t\t {value}" for key, value in reader.micromanager_summary.items())
            msgs.append("============")
            msgs.append(result_string)
    elif isinstance(reader, NGFFNode):
        msgs.extend(
            [
                sum_msg,
                fmt_msg,
                "".join(["Axes:\t\t\t "] + [f"{a.name} ({a.type}); " for a in reader.axes]),
                ch_msg,
            ]
        )
        if isinstance(reader, Plate):
            meta = reader.metadata
            msgs.extend(
                [
                    f"Row names:\t\t {[r.name for r in meta.rows]}",
                    f"Column names:\t\t {[c.name for c in meta.columns]}",
                    f"Wells:\t\t\t {len(meta.wells)}",
                ]
            )
            positions = list(reader.positions())
            total_bytes_uncompressed = sum(p["0"].nbytes for _, p in positions)
            msgs.append(f"Positions:\t\t {len(positions)}")
            msgs.append(f"Chunk size:\t\t {positions[0][1][0].chunks}")
            msgs.append(
                f"No. bytes decompressed:\t\t {total_bytes_uncompressed} [{sizeof_fmt(total_bytes_uncompressed)}]"
            )
        else:
            total_bytes_uncompressed = reader["0"].nbytes
            msgs.append(f"(Z, Y, X) scale (um):\t {tuple(reader.scale[2:])}")
            msgs.append(f"Chunk size:\t\t {reader['0'].chunks}")
            msgs.append(
                f"No. bytes decompressed:\t {total_bytes_uncompressed} [{sizeof_fmt(total_bytes_uncompressed)}]"
            )
        reader.close()

    if len(msgs) == 0:
        return None
    return str.join("\n", msgs)

from pathlib import Path

import click
import numpy as np
import xarray as xr
from iohub.ngff import open_ome_zarr


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

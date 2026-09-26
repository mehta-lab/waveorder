import hashlib
import json
from pathlib import Path

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


def apply_inverse_czyx(
    czyx: np.ndarray,
    model_function,
    czyx_coords: dict,
    **kwargs,
) -> np.ndarray:
    """CZYX numpy in/out wrapper around a ``waveorder.api`` apply-inverse function.

    iohub's ``process_single_position`` reads each CZYX volume once, hands it
    to ``func`` as a numpy array, and writes back the array ``func`` returns.
    The ``waveorder.api`` functions take and return a labeled
    ``xr.DataArray``, so this labels the volume, calls ``model_function`` and
    unwraps the result. Module-level so it pickles into iohub's spawn pool.

    Parameters
    ----------
    czyx : np.ndarray
        CZYX input volume, channels in ``input_channel_names`` order.
    model_function : callable
        One of the ``waveorder.api.*.apply_inverse_transfer_function``
        functions: xr.DataArray CZYX in, xr.DataArray CZYX out.
    czyx_coords : dict
        Coordinates for the input volume, as ``{dim: (dim, values, attrs)}``,
        taken once from the input store's ``Position.to_xarray()`` so they
        come from the same store as the data.
    **kwargs
        Passed to ``model_function``.
    """
    czyx_data = xr.DataArray(czyx, dims=("c", "z", "y", "x"), coords=czyx_coords)
    return model_function(czyx_data, **kwargs).values


def reconstruction_fingerprint(settings, transfer_function_settings: dict, transfer_function: dict) -> str:
    """Identify what determines a reconstruction's output, as an iohub resume token.

    iohub mixes the token into each finished unit's record, so resuming after
    any of these changed recomputes instead of reusing stale output: the
    reconstruction settings, the settings the transfer function was computed
    with, and the transfer function's array shapes. ``time_indices`` is left
    out because it selects which timepoints run, not what each one produces.
    """
    payload = json.dumps(
        {
            "settings": settings.model_dump(mode="json", exclude={"time_indices"}),
            "transfer_function_settings": transfer_function_settings,
            "transfer_function_shapes": {key: list(tensor.shape) for key, tensor in transfer_function.items()},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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

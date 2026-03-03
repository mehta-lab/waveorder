"""Shared helpers for the API layer."""

from typing import Literal

import numpy as np
import torch
import xarray as xr


def _position_list_from_shape_scale_offset(shape: int, scale: float, offset: float) -> list:
    """
    Generates a list of positions based on the given array shape,
    pixel size (scale), and offset.

    Examples
    --------
    >>> _position_list_from_shape_scale_offset(5, 1.0, 0.0)
    [2.0, 1.0, 0.0, -1.0, -2.0]
    >>> _position_list_from_shape_scale_offset(4, 0.5, 1.0)
    [1.5, 1.0, 0.5, 0.0]
    """
    return list((-np.arange(shape) + (shape // 2) + offset) * scale)


def _named_dataarray(array, name):
    """Create a DataArray with dimension names prefixed by variable name.

    This prevents dimension-name collisions when variables with different
    shapes are placed in the same xr.Dataset.
    """
    dims = tuple(f"{name}_d{i}" for i in range(array.ndim))
    return xr.DataArray(array, dims=dims)


def _to_tensor(ds: xr.Dataset, key: str) -> torch.Tensor:
    """Extract a variable from an xr.Dataset as a torch.Tensor."""
    return torch.from_numpy(ds[key].values.copy())


def _to_singular_system(
    ds: xr.Dataset, prefix: str = "singular_system"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (U, S, Vh) singular system tuple from an xr.Dataset."""
    return (
        _to_tensor(ds, f"{prefix}_U"),
        _to_tensor(ds, f"{prefix}_S"),
        _to_tensor(ds, f"{prefix}_Vh"),
    )


def _biref_inverse_kwargs(settings) -> dict:
    """Extract model kwargs from birefringence apply-inverse settings."""
    return settings.apply_inverse.model_dump(exclude={"background_path", "wavelength_illumination"})


def _output_channel_names(
    recon_biref: bool = False,
    recon_phase: bool = False,
    recon_fluo: bool = False,
    recon_dim: Literal[2, 3] = 3,
    fluor_channel_name: str = "",
) -> list[str]:
    """Return the output channel names for a given reconstruction config.

    This is the single source of truth for output channel naming.
    """
    names = []
    if recon_biref:
        names += [
            "Retardance",
            "Orientation",
            "Transmittance",
            "Depolarization",
        ]
    if recon_phase:
        names.append("Phase2D" if recon_dim == 2 else "Phase3D")
    if recon_biref and recon_phase and recon_dim == 3:
        names += [
            "Retardance_Joint_Decon",
            "Orientation_Joint_Decon",
            "Phase_Joint_Decon",
        ]
    if recon_fluo:
        names.append(f"{fluor_channel_name}_Density{'2D' if recon_dim == 2 else '3D'}")
    return names


def radians_to_nanometers(retardance_rad: torch.Tensor, wavelength_illumination_um: float) -> torch.Tensor:
    """
    waveorder returns retardance in radians, while waveorder displays and saves
    retardance in nanometers. This function converts from radians to nanometers
    using the illumination wavelength (which is internally handled in um
    in waveorder).
    """
    return retardance_rad * wavelength_illumination_um * 1e3 / (2 * np.pi)


def _build_output_xarray(
    data: np.ndarray,
    channel_names: list[str],
    input_data: xr.DataArray,
    singleton_z: bool = False,
) -> xr.DataArray:
    """Build a CZYX xr.DataArray inheriting coords from input.

    Parameters
    ----------
    data : np.ndarray
        CZYX output array.
    channel_names : list[str]
        Channel names for the C dimension.
    input_data : xr.DataArray
        CZYX input DataArray (for inheriting Z/Y/X coords).
    singleton_z : bool
        If True, use [0.0] for Z coords; otherwise copy from input.
    """
    if singleton_z:
        z_values = np.array([0.0])
    else:
        z_values = input_data.coords["z"].values

    coords = {
        "c": ("c", channel_names),
        "z": ("z", z_values, input_data.coords["z"].attrs),
        "y": (
            "y",
            input_data.coords["y"].values,
            input_data.coords["y"].attrs,
        ),
        "x": (
            "x",
            input_data.coords["x"].values,
            input_data.coords["x"].attrs,
        ),
    }

    return xr.DataArray(
        data,
        dims=("c", "z", "y", "x"),
        coords=coords,
    )

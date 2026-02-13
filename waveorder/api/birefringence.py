"""Birefringence reconstruction: settings, transfer functions, and inverse."""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import xarray as xr
from pydantic import Field, field_validator

from waveorder.api._settings import MyBaseModel, WavelengthIllumination
from waveorder.api._utils import (
    _biref_inverse_kwargs,
    _build_output_xarray,
    _named_dataarray,
    _output_channel_names,
    _to_tensor,
    radians_to_nanometers,
)
from waveorder.models import inplane_oriented_thick_pol3d

# --- Settings ---


class TransferFunctionSettings(MyBaseModel):
    swing: float = Field(default=0.1, description="swing of the liquid crystal (0 to 1)")

    @field_validator("swing")
    @classmethod
    def swing_range(cls, v):
        if v <= 0 or v >= 1.0:
            raise ValueError(f"swing = {v} should be between 0 and 1.")
        return v


class ApplyInverseSettings(WavelengthIllumination):
    background_path: Union[str, Path] = Field(
        default="",
        description="path to background zarr (empty = no background)",
    )

    @field_validator("background_path")
    @classmethod
    def check_background_path(cls, v):
        if v == "":
            return v

        raw_dir = r"{}".format(v)
        if not os.path.isdir(raw_dir):
            raise ValueError(f"{v} is not an existing directory")
        return raw_dir

    remove_estimated_background: bool = Field(default=False, description="estimate and remove background")
    flip_orientation: bool = Field(default=False, description="flip the orientation angle")
    rotate_orientation: bool = Field(default=False, description="rotate orientation by 90 degrees")


class Settings(MyBaseModel):
    transfer_function: TransferFunctionSettings = TransferFunctionSettings()
    apply_inverse: ApplyInverseSettings = ApplyInverseSettings()


# --- Functions ---


def simulate(
    settings: Settings = None,
    yx_shape: tuple[int, int] = (256, 256),
    z_depth: int = 5,
    yx_pixel_size: float = 0.1,
    z_pixel_size: float = 0.25,
    scheme: str = "4-State",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Simulate polarization data from a star-target phantom.

    Returns (phantom, data) as CZYX xr.DataArrays.
    Phantom channels: Retardance, Orientation, Transmittance, Depolarization.
    Data channels: State0, State1, ... (one per polarization state).
    """
    if settings is None:
        settings = Settings()

    retardance, orientation, transmittance, depolarization = inplane_oriented_thick_pol3d.generate_test_phantom(
        yx_shape
    )
    intensity_to_stokes = inplane_oriented_thick_pol3d.calculate_transfer_function(
        swing=settings.transfer_function.swing, scheme=scheme
    )
    cyx_data = inplane_oriented_thick_pol3d.apply_transfer_function(
        retardance,
        orientation,
        transmittance,
        depolarization,
        intensity_to_stokes,
    )
    cyx_data = cyx_data.squeeze()  # (C, 1, Y, X) -> (C, Y, X)

    # Tile along Z
    num_states = cyx_data.shape[0]
    czyx_data = np.repeat(cyx_data.numpy()[:, None, :, :], z_depth, axis=1)
    czyx_phantom = np.stack(
        [
            np.repeat(retardance.numpy()[None, :, :], z_depth, axis=0),
            np.repeat(orientation.numpy()[None, :, :], z_depth, axis=0),
            np.repeat(transmittance.numpy()[None, :, :], z_depth, axis=0),
            np.repeat(depolarization.numpy()[None, :, :], z_depth, axis=0),
        ]
    )

    zyx_coords = {
        "z": np.arange(z_depth) * z_pixel_size,
        "y": np.arange(yx_shape[0]) * yx_pixel_size,
        "x": np.arange(yx_shape[1]) * yx_pixel_size,
    }
    phantom = xr.DataArray(
        czyx_phantom,
        dims=("c", "z", "y", "x"),
        coords={
            "c": [
                "Retardance",
                "Orientation",
                "Transmittance",
                "Depolarization",
            ],
            **zyx_coords,
        },
    )
    data = xr.DataArray(
        czyx_data,
        dims=("c", "z", "y", "x"),
        coords={
            "c": [f"State{i}" for i in range(num_states)],
            **zyx_coords,
        },
    )
    return phantom, data


def compute_transfer_function(
    czyx_data: xr.DataArray,
    settings: Settings = None,
    input_channel_names: list[str] = None,
) -> xr.Dataset:
    """Compute birefringence transfer function.

    Returns xr.Dataset with:
    - "intensity_to_stokes_matrix": 2D array mapping intensities to Stokes
    """
    if settings is None:
        settings = Settings()

    if input_channel_names is None:
        input_channel_names = list(czyx_data.coords["c"].values)

    intensity_to_stokes_matrix = inplane_oriented_thick_pol3d.calculate_transfer_function(
        scheme=str(len(input_channel_names)) + "-State",
        **settings.transfer_function.model_dump(),
    )

    return xr.Dataset(
        {
            "intensity_to_stokes_matrix": _named_dataarray(
                intensity_to_stokes_matrix.cpu().numpy(),
                "intensity_to_stokes_matrix",
            ),
        }
    )


def apply_inverse_transfer_function(
    czyx_data: xr.DataArray,
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
    cyx_no_sample_data: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Reconstruct birefringence from polarization data.

    Returns CZYX xr.DataArray with channels
    [Retardance (nm), Orientation, Transmittance, Depolarization].
    """
    if settings is None:
        settings = Settings()

    wavelength = settings.apply_inverse.wavelength_illumination
    biref_kwargs = _biref_inverse_kwargs(settings)

    czyx_tensor = torch.tensor(czyx_data.values, dtype=torch.float32)
    bg_tensor = torch.tensor(cyx_no_sample_data, dtype=torch.float32) if cyx_no_sample_data is not None else None

    reconstructed_parameters = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
        czyx_tensor,
        _to_tensor(transfer_function, "intensity_to_stokes_matrix"),
        cyx_no_sample_data=bg_tensor,
        project_stokes_to_2d=(recon_dim == 2),
        **biref_kwargs,
    )

    retardance = radians_to_nanometers(reconstructed_parameters[0], wavelength)

    output = torch.stack((retardance,) + reconstructed_parameters[1:]).numpy()

    return _build_output_xarray(
        output,
        _output_channel_names(recon_biref=True, recon_dim=recon_dim),
        czyx_data,
        singleton_z=(recon_dim == 2),
    )


def reconstruct(
    czyx_data: xr.DataArray,
    settings: Settings = None,
    input_channel_names: list[str] = None,
    recon_dim: Literal[2, 3] = 3,
    cyx_no_sample_data: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Reconstruct birefringence from polarization data (one-liner).

    Chains compute_transfer_function + apply_inverse_transfer_function.
    """
    if settings is None:
        settings = Settings()

    tf = compute_transfer_function(czyx_data, settings, input_channel_names)
    return apply_inverse_transfer_function(czyx_data, tf, recon_dim, settings, cyx_no_sample_data)

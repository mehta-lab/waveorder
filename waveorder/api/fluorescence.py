"""Fluorescence reconstruction: settings, transfer functions, inverse, and simulation."""

import warnings
from typing import Literal, Optional

import numpy as np
import torch
import xarray as xr
from pydantic import Field, PositiveFloat, model_validator

from waveorder.api._settings import (
    FourierApplyInverseSettings,
    FourierTransferFunctionSettings,
    MyBaseModel,
)
from waveorder.api._utils import (
    _build_output_xarray,
    _named_dataarray,
    _output_channel_names,
    _position_list_from_shape_scale_offset,
    _to_singular_system,
    _to_tensor,
)
from waveorder.models import (
    isotropic_fluorescent_thick_3d,
    isotropic_fluorescent_thin_3d,
)

# --- Settings ---


class TransferFunctionSettings(FourierTransferFunctionSettings):
    wavelength_emission: PositiveFloat = Field(default=0.532, description="emission wavelength in micrometers")
    confocal_pinhole_diameter: Optional[PositiveFloat] = Field(
        default=None,
        description="confocal pinhole diameter (null = widefield)",
    )

    @model_validator(mode="after")
    def warn_unit_consistency(self):
        ratio = self.yx_pixel_size / self.wavelength_emission
        if ratio < 1.0 / 20 or ratio > 20:
            warnings.warn(
                f"yx_pixel_size ({self.yx_pixel_size}) / wavelength_illumination ({self.wavelength_emission}) = {ratio}. Did you use consistent units?",
                UserWarning,
            )
        return self


ApplyInverseSettings = FourierApplyInverseSettings


class Settings(MyBaseModel):
    transfer_function: TransferFunctionSettings = TransferFunctionSettings()
    apply_inverse: ApplyInverseSettings = ApplyInverseSettings()


# --- Functions ---


def simulate(
    settings: Settings = None,
    recon_dim: Literal[2, 3] = 3,
    zyx_shape: tuple[int, int, int] = (100, 256, 256),
    sphere_radius: float = 5,
    channel_name: str = "GFP",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Simulate fluorescence data from a spherical phantom.

    For recon_dim=3: thick 3D phantom imaged to 3D data.
    For recon_dim=2: thin 2D phantom imaged through defocus to 3D data.

    Returns (phantom, data) as CZYX xr.DataArrays.
    """
    if settings is None:
        settings = Settings()

    s = settings.transfer_function
    Z, Y, X = zyx_shape
    zyx_coords = {
        "z": np.arange(Z) * s.z_pixel_size,
        "y": np.arange(Y) * s.yx_pixel_size,
        "x": np.arange(X) * s.yx_pixel_size,
    }

    if recon_dim == 3:
        zyx_fluorescence = isotropic_fluorescent_thick_3d.generate_test_phantom(
            zyx_shape,
            s.yx_pixel_size,
            s.z_pixel_size,
            sphere_radius=sphere_radius,
        )
        otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
            zyx_shape,
            s.yx_pixel_size,
            s.z_pixel_size,
            wavelength_emission=s.wavelength_emission,
            z_padding=0,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_detection=s.numerical_aperture_detection,
        )
        zyx_data = isotropic_fluorescent_thick_3d.apply_transfer_function(zyx_fluorescence, otf, z_padding=0)
        phantom_zyx = zyx_fluorescence.numpy()

    elif recon_dim == 2:
        yx_shape = (Y, X)
        yx_fluorescence = isotropic_fluorescent_thin_3d.generate_test_phantom(
            yx_shape=yx_shape,
            yx_pixel_size=s.yx_pixel_size,
            sphere_radius=sphere_radius,
        )
        z_position_list = _position_list_from_shape_scale_offset(
            shape=Z,
            scale=s.z_pixel_size,
            offset=0,
        )
        fluorescent_tf = isotropic_fluorescent_thin_3d.calculate_transfer_function(
            yx_shape=yx_shape,
            yx_pixel_size=s.yx_pixel_size,
            wavelength_emission=s.wavelength_emission,
            z_position_list=z_position_list,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_detection=s.numerical_aperture_detection,
        )
        zyx_data = isotropic_fluorescent_thin_3d.apply_transfer_function(yx_fluorescence, fluorescent_tf)
        # Tile 2D phantom along Z so it appears at every defocus plane
        phantom_zyx = np.broadcast_to(yx_fluorescence.numpy()[None, :, :], (Z, Y, X)).copy()

    phantom = xr.DataArray(
        phantom_zyx[None, ...],
        dims=("c", "z", "y", "x"),
        coords={"c": ["Phantom"], **zyx_coords},
    )
    data = xr.DataArray(
        zyx_data.numpy()[None, ...],
        dims=("c", "z", "y", "x"),
        coords={"c": [channel_name], **zyx_coords},
    )
    return phantom, data


def compute_transfer_function(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
) -> xr.Dataset:
    """Compute fluorescence transfer function.

    For 2D: returns xr.Dataset with singular_system_U, _S, _Vh.
    For 3D: returns xr.Dataset with optical_transfer_function.
    """
    if settings is None:
        settings = Settings()

    zyx_shape = czyx_data.shape[1:]  # CZYX -> ZYX
    settings_dict = settings.transfer_function.model_dump()

    if recon_dim == 2:
        settings_dict["yx_shape"] = [zyx_shape[1], zyx_shape[2]]
        settings_dict["z_position_list"] = _position_list_from_shape_scale_offset(
            shape=zyx_shape[0],
            scale=settings_dict["z_pixel_size"],
            offset=settings_dict["z_focus_offset"],
        )
        settings_dict.pop("z_pixel_size")
        settings_dict.pop("z_padding")
        settings_dict.pop("z_focus_offset")

        fluorescent_2d_to_3d_tf = isotropic_fluorescent_thin_3d.calculate_transfer_function(
            **settings_dict,
        )
        U, S, Vh = isotropic_fluorescent_thin_3d.calculate_singular_system(fluorescent_2d_to_3d_tf)

        return xr.Dataset(
            {
                "singular_system_U": _named_dataarray(U.cpu().numpy(), "singular_system_U"),
                "singular_system_S": _named_dataarray(S.cpu().numpy(), "singular_system_S"),
                "singular_system_Vh": _named_dataarray(Vh.cpu().numpy(), "singular_system_Vh"),
            }
        )

    elif recon_dim == 3:
        settings_dict.pop("z_focus_offset")

        optical_tf = isotropic_fluorescent_thick_3d.calculate_transfer_function(zyx_shape=zyx_shape, **settings_dict)

        return xr.Dataset(
            {
                "optical_transfer_function": _named_dataarray(optical_tf.cpu().numpy(), "optical_transfer_function"),
            }
        )


def apply_inverse_transfer_function(
    czyx_data: xr.DataArray,
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
    fluor_channel_name: str = "",
) -> xr.DataArray:
    """Reconstruct fluorescence density.

    Returns CZYX xr.DataArray with a single fluorescence density channel.
    Uses thin (2D) or thick (3D) model depending on recon_dim.
    """
    if settings is None:
        settings = Settings()

    czyx_tensor = torch.tensor(czyx_data.values, dtype=torch.float32)

    # [fluo, 2]
    if recon_dim == 2:
        output = isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
            czyx_tensor[0],
            _to_singular_system(transfer_function),
            **settings.apply_inverse.model_dump(),
        )
    # [fluo, 3]
    elif recon_dim == 3:
        output = isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
            czyx_tensor[0],
            _to_tensor(transfer_function, "optical_transfer_function"),
            settings.transfer_function.z_padding,
            **settings.apply_inverse.model_dump(),
        )
    # Pad to CZYX
    while output.ndim != 4:
        output = torch.unsqueeze(output, 0)

    return _build_output_xarray(
        output.numpy(),
        _output_channel_names(
            recon_fluo=True,
            recon_dim=recon_dim,
            fluor_channel_name=fluor_channel_name,
        ),
        czyx_data,
        singleton_z=(recon_dim == 2),
    )


def reconstruct(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
    fluor_channel_name: str = "",
) -> xr.DataArray:
    """Reconstruct fluorescence density (one-liner).

    Chains compute_transfer_function + apply_inverse_transfer_function.
    """
    if settings is None:
        settings = Settings()

    tf = compute_transfer_function(czyx_data, recon_dim, settings)
    return apply_inverse_transfer_function(czyx_data, tf, recon_dim, settings, fluor_channel_name)

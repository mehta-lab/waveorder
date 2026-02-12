"""Phase reconstruction: settings, transfer functions, inverse, and simulation."""

from typing import Literal

import numpy as np
import torch
import xarray as xr
from pydantic import Field, NonNegativeFloat, model_validator

from waveorder.api._settings import (
    FourierApplyInverseSettings,
    FourierTransferFunctionSettings,
    MyBaseModel,
    WavelengthIllumination,
)
from waveorder.api._utils import (
    _build_output_xarray,
    _named_dataarray,
    _output_channel_names,
    _position_list_from_shape_scale_offset,
    _to_singular_system,
    _to_tensor,
)
from waveorder.models import isotropic_thin_3d, phase_thick_3d

# --- Settings ---


class TransferFunctionSettings(
    FourierTransferFunctionSettings,
    WavelengthIllumination,
):
    numerical_aperture_illumination: NonNegativeFloat = Field(
        default=0.9, description="condenser numerical aperture"
    )
    invert_phase_contrast: bool = Field(
        default=False,
        description="invert contrast for positive/negative phase",
    )

    @model_validator(mode="after")
    def validate_numerical_aperture_illumination(self):
        if (
            self.numerical_aperture_illumination
            > self.index_of_refraction_media
        ):
            raise ValueError(
                f"numerical_aperture_illumination = {self.numerical_aperture_illumination} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
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
    index_of_refraction_sample: float = 1.33,
    sphere_radius: float = 5,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Simulate brightfield phase data from a spherical phantom.

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
        zyx_phase = phase_thick_3d.generate_test_phantom(
            zyx_shape,
            s.yx_pixel_size,
            s.z_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            index_of_refraction_media=s.index_of_refraction_media,
            index_of_refraction_sample=index_of_refraction_sample,
            sphere_radius=sphere_radius,
        )
        real_tf, _ = phase_thick_3d.calculate_transfer_function(
            zyx_shape,
            s.yx_pixel_size,
            s.z_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            z_padding=0,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_illumination=s.numerical_aperture_illumination,
            numerical_aperture_detection=s.numerical_aperture_detection,
        )
        zyx_data = phase_thick_3d.apply_transfer_function(
            zyx_phase, real_tf, z_padding=0, brightness=1e3
        )
        phantom_zyx = zyx_phase.numpy()

    elif recon_dim == 2:
        yx_shape = (Y, X)
        _, yx_phase = isotropic_thin_3d.generate_test_phantom(
            yx_shape=yx_shape,
            yx_pixel_size=s.yx_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            index_of_refraction_media=s.index_of_refraction_media,
            index_of_refraction_sample=index_of_refraction_sample,
            sphere_radius=sphere_radius,
        )
        # Half-circle phantom (matching model-level example)
        yx_phase[Y // 2 :] = 0

        # Use same z_position_list as compute_transfer_function
        z_position_list = _position_list_from_shape_scale_offset(
            shape=Z,
            scale=s.z_pixel_size,
            offset=0,
        )
        absorption_tf, phase_tf = (
            isotropic_thin_3d.calculate_transfer_function(
                yx_shape=yx_shape,
                yx_pixel_size=s.yx_pixel_size,
                wavelength_illumination=s.wavelength_illumination,
                z_position_list=z_position_list,
                index_of_refraction_media=s.index_of_refraction_media,
                numerical_aperture_illumination=s.numerical_aperture_illumination,
                numerical_aperture_detection=s.numerical_aperture_detection,
                invert_phase_contrast=s.invert_phase_contrast,
            )
        )
        # Zero absorption, phase-only object
        yx_absorption = torch.zeros_like(yx_phase)
        zyx_data = isotropic_thin_3d.apply_transfer_function(
            yx_absorption, yx_phase, absorption_tf, phase_tf
        )
        # Tile 2D phantom along Z so it appears at every defocus plane
        phantom_zyx = np.broadcast_to(
            yx_phase.numpy()[None, :, :], (Z, Y, X)
        ).copy()

    phantom = xr.DataArray(
        phantom_zyx[None, ...],
        dims=("c", "z", "y", "x"),
        coords={"c": ["Phantom"], **zyx_coords},
    )
    data = xr.DataArray(
        zyx_data.numpy()[None, ...],
        dims=("c", "z", "y", "x"),
        coords={"c": ["Brightfield"], **zyx_coords},
    )
    return phantom, data


def compute_transfer_function(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
) -> xr.Dataset:
    """Compute phase transfer function.

    For 2D: returns xr.Dataset with singular_system_U, _S, _Vh.
    For 3D: returns xr.Dataset with real/imaginary_potential_transfer_function.
    """
    if settings is None:
        settings = Settings()

    zyx_shape = czyx_data.shape[1:]  # CZYX -> ZYX
    settings_dict = settings.transfer_function.model_dump()

    if recon_dim == 2:
        settings_dict["yx_shape"] = [zyx_shape[1], zyx_shape[2]]
        settings_dict["z_position_list"] = (
            _position_list_from_shape_scale_offset(
                shape=zyx_shape[0],
                scale=settings_dict["z_pixel_size"],
                offset=settings_dict["z_focus_offset"],
            )
        )
        settings_dict.pop("z_pixel_size")
        settings_dict.pop("z_padding")
        settings_dict.pop("z_focus_offset")

        absorption_tf, phase_tf = (
            isotropic_thin_3d.calculate_transfer_function(**settings_dict)
        )
        U, S, Vh = isotropic_thin_3d.calculate_singular_system(
            absorption_tf, phase_tf
        )

        return xr.Dataset(
            {
                "singular_system_U": _named_dataarray(
                    U.cpu().numpy(), "singular_system_U"
                ),
                "singular_system_S": _named_dataarray(
                    S.cpu().numpy(), "singular_system_S"
                ),
                "singular_system_Vh": _named_dataarray(
                    Vh.cpu().numpy(), "singular_system_Vh"
                ),
            }
        )

    elif recon_dim == 3:
        settings_dict.pop("z_focus_offset")

        real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape, **settings_dict
        )

        return xr.Dataset(
            {
                "real_potential_transfer_function": _named_dataarray(
                    real_tf.cpu().numpy(),
                    "real_potential_transfer_function",
                ),
                "imaginary_potential_transfer_function": _named_dataarray(
                    imag_tf.cpu().numpy(),
                    "imaginary_potential_transfer_function",
                ),
            }
        )


def apply_inverse_transfer_function(
    czyx_data: xr.DataArray,
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
) -> xr.DataArray:
    """Reconstruct phase from brightfield data.

    Returns CZYX xr.DataArray with a single Phase channel.
    Uses thin (2D) or thick (3D) model depending on recon_dim.
    """
    if settings is None:
        settings = Settings()

    czyx_tensor = torch.tensor(czyx_data.values, dtype=torch.float32)

    # [phase only, 2]
    if recon_dim == 2:
        (
            absorption_yx,
            phase_yx,
        ) = isotropic_thin_3d.apply_inverse_transfer_function(
            czyx_tensor[0],
            _to_singular_system(transfer_function),
            **settings.apply_inverse.model_dump(),
        )
        output = phase_yx[None, None]

    # [phase only, 3]
    elif recon_dim == 3:
        output = phase_thick_3d.apply_inverse_transfer_function(
            czyx_tensor[0],
            _to_tensor(transfer_function, "real_potential_transfer_function"),
            _to_tensor(
                transfer_function, "imaginary_potential_transfer_function"
            ),
            z_padding=settings.transfer_function.z_padding,
            **settings.apply_inverse.model_dump(),
        )

    # Pad to CZYX
    while output.ndim != 4:
        output = torch.unsqueeze(output, 0)

    return _build_output_xarray(
        output.numpy(),
        _output_channel_names(recon_phase=True, recon_dim=recon_dim),
        czyx_data,
        singleton_z=(recon_dim == 2),
    )


def reconstruct(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
) -> xr.DataArray:
    """Reconstruct phase from brightfield data (one-liner).

    Chains compute_transfer_function + apply_inverse_transfer_function.
    """
    if settings is None:
        settings = Settings()

    tf = compute_transfer_function(czyx_data, recon_dim, settings)
    return apply_inverse_transfer_function(czyx_data, tf, recon_dim, settings)

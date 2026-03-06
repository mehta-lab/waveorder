"""Fluorescence reconstruction: settings, transfer functions, inverse, and simulation."""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
import torch
import xarray as xr
from pydantic import Field, PositiveFloat, model_validator

from waveorder.api._settings import (
    FourierApplyInverseSettings,
    MyBaseModel,
    OptimizableFourierTransferFunctionSettings,
)
from waveorder.api._utils import (
    _build_output_xarray,
    _named_dataarray,
    _output_channel_names,
    _position_list_from_shape_scale_offset,
    _to_singular_system,
    _to_tensor,
)
from waveorder.focus import compute_midband_power
from waveorder.models import (
    isotropic_fluorescent_thick_3d,
    isotropic_fluorescent_thin_3d,
)
from waveorder.optim import (
    PrintLogger,
    TensorBoardLogger,
    extract_optimizable_params,
    optimize_reconstruction,
)

# --- Settings ---


class TransferFunctionSettings(OptimizableFourierTransferFunctionSettings):
    wavelength_emission: PositiveFloat = Field(default=0.532, description="emission wavelength in micrometers")
    confocal_pinhole_diameter: Optional[PositiveFloat] = Field(
        default=None,
        description="confocal pinhole diameter (null = widefield)",
    )

    @model_validator(mode="after")
    def warn_wavelength_consistency(self):
        ratio = self.yx_pixel_size / self.wavelength_emission
        if ratio < 1.0 / 20 or ratio > 20:
            warnings.warn(
                f"yx_pixel_size ({self.yx_pixel_size}) / wavelength_emission ({self.wavelength_emission}) = {ratio}. Did you use consistent units?",
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

    For ``recon_dim=3``: thick 3D phantom imaged to 3D data.
    For ``recon_dim=2``: thin 2D phantom imaged through defocus to 3D data.

    Parameters
    ----------
    settings : Settings, optional
        Fluorescence reconstruction settings. Uses defaults if None.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    zyx_shape : tuple of int
        (Z, Y, X) shape of the output arrays.
    sphere_radius : float
        Radius of the phantom sphere in micrometers.
    channel_name : str
        Name for the data channel coordinate.

    Returns
    -------
    phantom : xr.DataArray
        CZYX array with a single Phantom channel.
    data : xr.DataArray
        CZYX array with a single fluorescence channel.
    """
    if settings is None:
        settings = Settings()

    s = settings.transfer_function.resolve_floats()
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
            offset=s.z_focus_offset,
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

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input CZYX data array (shape is used to determine ZYX dimensions).
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Fluorescence reconstruction settings. Uses defaults if None.

    Returns
    -------
    xr.Dataset
        For 2D: contains ``singular_system_U``, ``singular_system_S``,
        ``singular_system_Vh``.
        For 3D: contains ``optical_transfer_function``.
    """
    if settings is None:
        settings = Settings()

    zyx_shape = czyx_data.shape[1:]  # CZYX -> ZYX
    s = settings.transfer_function.resolve_floats()

    if recon_dim == 2:
        z_position_list = _position_list_from_shape_scale_offset(
            shape=zyx_shape[0],
            scale=s.z_pixel_size,
            offset=s.z_focus_offset,
        )

        fluorescent_tf = isotropic_fluorescent_thin_3d.calculate_transfer_function(
            yx_shape=[zyx_shape[1], zyx_shape[2]],
            yx_pixel_size=s.yx_pixel_size,
            z_position_list=z_position_list,
            wavelength_emission=s.wavelength_emission,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_detection=s.numerical_aperture_detection,
        )
        U, S, Vh = isotropic_fluorescent_thin_3d.calculate_singular_system(fluorescent_tf)

        return xr.Dataset(
            {
                "singular_system_U": _named_dataarray(U.cpu().numpy(), "singular_system_U"),
                "singular_system_S": _named_dataarray(S.cpu().numpy(), "singular_system_S"),
                "singular_system_Vh": _named_dataarray(Vh.cpu().numpy(), "singular_system_Vh"),
            }
        )

    elif recon_dim == 3:
        optical_tf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape,
            yx_pixel_size=s.yx_pixel_size,
            z_pixel_size=s.z_pixel_size,
            wavelength_emission=s.wavelength_emission,
            z_padding=s.z_padding,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_detection=s.numerical_aperture_detection,
        )

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

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input CZYX fluorescence data.
    transfer_function : xr.Dataset
        Transfer function from ``compute_transfer_function``.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Fluorescence reconstruction settings. Uses defaults if None.
    fluor_channel_name : str
        Name for the output fluorescence channel.

    Returns
    -------
    xr.DataArray
        CZYX array with a single fluorescence density channel.
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
    """Reconstruct fluorescence density.

    Convenience function that chains ``compute_transfer_function`` and
    ``apply_inverse_transfer_function``.

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input CZYX fluorescence data.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Fluorescence reconstruction settings. Uses defaults if None.
    fluor_channel_name : str
        Name for the output fluorescence channel.

    Returns
    -------
    xr.DataArray
        CZYX array with a single fluorescence density channel.
    """
    if settings is None:
        settings = Settings()

    tf = compute_transfer_function(czyx_data, recon_dim, settings)
    return apply_inverse_transfer_function(czyx_data, tf, recon_dim, settings, fluor_channel_name)


def optimize(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3] = 2,
    settings: Settings = None,
    max_iterations: int = 10,
    method: str = "adam",
    convergence_tol: float | None = None,
    convergence_patience: int | None = 5,
    use_gradients: bool | None = None,
    grid_points: int = 7,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
    log_dir: str | None = None,
    log_images: bool = False,
) -> tuple[Settings, xr.DataArray]:
    """Optimize fluorescence reconstruction parameters.

    Parameters
    ----------
    czyx_data : xr.DataArray
        CZYX fluorescence data (single channel).
    recon_dim : {2, 3}
        Reconstruction dimensionality (2 or 3).
    settings : Settings
        Fluorescence settings with OptimizableFloat fields.
    max_iterations : int
        Maximum optimizer steps (ignored by grid_search).
    method : str
        Optimizer method: "adam", "lbfgs", "nelder_mead", "grid_search".
    convergence_tol : float, optional
        Stop early if loss does not improve by at least this amount.
    convergence_patience : int, optional
        Iterations without improvement before early stopping.
    use_gradients : bool, optional
        Whether to compute gradients. Auto-detected from method if None.
    grid_points : int
        Number of grid points per parameter (grid_search only).
    midband_fractions : tuple[float, float]
        Inner and outer fractions of cutoff frequency for the loss annulus.
    log_dir : str, optional
        TensorBoard log directory. None = print-only logging.

    Returns
    -------
    tuple[Settings, xr.DataArray]
        Updated settings with optimized values, and the final reconstruction.
    """
    if settings is None:
        settings = Settings()

    opt_params, _ = extract_optimizable_params(settings.transfer_function)

    if not opt_params:
        print("No optimizable parameters found. Running standard reconstruction.")
        return settings, reconstruct(czyx_data, recon_dim=recon_dim, settings=settings)

    logger = TensorBoardLogger(log_dir) if log_dir else PrintLogger()

    s = settings.transfer_function.resolve_floats()
    zyx_data = torch.tensor(czyx_data.values[0], dtype=torch.float32)
    Z = zyx_data.shape[0]

    def reconstruct_fn(data, **tensor_params):
        na_det = tensor_params.get("numerical_aperture_detection", s.numerical_aperture_detection)
        z_offset = tensor_params.get("z_focus_offset", s.z_focus_offset)

        if recon_dim == 2:
            z_position_list = (-torch.arange(Z) + (Z // 2) + z_offset) * s.z_pixel_size
            return isotropic_fluorescent_thin_3d.reconstruct(
                data,
                yx_pixel_size=s.yx_pixel_size,
                z_position_list=z_position_list,
                wavelength_emission=s.wavelength_emission,
                index_of_refraction_media=s.index_of_refraction_media,
                numerical_aperture_detection=na_det,
                regularization_strength=settings.apply_inverse.regularization_strength,
            )
        else:
            return isotropic_fluorescent_thick_3d.reconstruct(
                data,
                yx_pixel_size=s.yx_pixel_size,
                z_pixel_size=s.z_pixel_size,
                wavelength_emission=s.wavelength_emission,
                z_padding=s.z_padding,
                index_of_refraction_media=s.index_of_refraction_media,
                numerical_aperture_detection=na_det,
                regularization_strength=settings.apply_inverse.regularization_strength,
            )

    def loss_fn(recon):
        loss = -compute_midband_power(
            recon,
            NA_det=s.numerical_aperture_detection,
            lambda_ill=s.wavelength_emission,
            pixel_size=s.yx_pixel_size,
            midband_fractions=midband_fractions,
        )
        if loss.ndim > 0:
            loss = loss.mean()
        return loss

    optim_kwargs = dict(
        max_iterations=max_iterations,
        method=method,
        use_gradients=use_gradients,
        grid_points=grid_points,
        log_images=log_images,
    )
    if convergence_tol is not None:
        optim_kwargs["convergence_tol"] = convergence_tol
    if convergence_patience is not None:
        optim_kwargs["convergence_patience"] = convergence_patience

    result = optimize_reconstruction(
        data=zyx_data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params=opt_params,
        logger=logger,
        **optim_kwargs,
    )

    new_tf_dict = settings.transfer_function.model_dump()
    for dotted_name, value in result.optimized_values.items():
        field_name = dotted_name.split(".")[-1]
        new_tf_dict[field_name] = value

    new_settings = Settings(
        transfer_function=TransferFunctionSettings(**new_tf_dict),
        apply_inverse=settings.apply_inverse,
    )

    final_recon = reconstruct(czyx_data, recon_dim=recon_dim, settings=new_settings)

    return new_settings, final_recon

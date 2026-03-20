"""Phase reconstruction: settings, transfer functions, inverse, and simulation."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch
import xarray as xr
from pydantic import Field, NonNegativeFloat, model_validator

from waveorder import optics, util
from waveorder.api._settings import (
    FourierApplyInverseSettings,
    MyBaseModel,
    OptimizableFourierTransferFunctionSettings,
    WavelengthIllumination,
    _float_val,
)
from waveorder.api._utils import (
    _named_dataarray,
    _output_channel_names,
    _position_list_from_shape_scale_offset,
    _to_singular_system,
    _to_tensor,
    _wrap_output_tensor,
)
from waveorder.device import resolve_device
from waveorder.models import isotropic_thin_3d, phase_thick_3d
from waveorder.optim import (
    NullLogger,
    TensorBoardLogger,
    extract_optimizable_params,
    optimize_reconstruction,
)
from waveorder.optim._types import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings, build_loss_fn

# --- Settings ---


class TransferFunctionSettings(
    OptimizableFourierTransferFunctionSettings,
    WavelengthIllumination,
):
    numerical_aperture_illumination: Union[NonNegativeFloat, OptimizableFloat] = Field(
        default=0.9, description="(optimizable) condenser numerical aperture"
    )
    invert_phase_contrast: bool = Field(
        default=False,
        description="invert contrast for positive/negative phase",
    )

    @model_validator(mode="after")
    def validate_numerical_aperture_illumination(self):
        na_ill = _float_val(self.numerical_aperture_illumination)
        if na_ill > self.index_of_refraction_media:
            raise ValueError(
                f"numerical_aperture_illumination = {na_ill} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
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

    For ``recon_dim=3``: thick 3D phantom imaged to 3D data.
    For ``recon_dim=2``: thin 2D phantom imaged through defocus to 3D data.

    Parameters
    ----------
    settings : Settings, optional
        Phase reconstruction settings. Uses defaults if None.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    zyx_shape : tuple of int
        (Z, Y, X) shape of the output arrays.
    index_of_refraction_sample : float
        Refractive index of the sample sphere.
    sphere_radius : float
        Radius of the phantom sphere in micrometers.

    Returns
    -------
    phantom : xr.DataArray
        CZYX array with a single Phantom channel.
    data : xr.DataArray
        CZYX array with a single Brightfield channel.
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
            tilt_angle_zenith=s.tilt_angle_zenith,
            tilt_angle_azimuth=s.tilt_angle_azimuth,
        )
        zyx_data = phase_thick_3d.apply_transfer_function(zyx_phase, real_tf, z_padding=0, brightness=1e3)
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
        z_position_list = _position_list_from_shape_scale_offset(
            shape=Z,
            scale=s.z_pixel_size,
            offset=s.z_focus_offset,
        )
        absorption_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
            yx_shape=yx_shape,
            yx_pixel_size=s.yx_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            z_position_list=z_position_list,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_illumination=s.numerical_aperture_illumination,
            numerical_aperture_detection=s.numerical_aperture_detection,
            invert_phase_contrast=s.invert_phase_contrast,
            tilt_angle_zenith=s.tilt_angle_zenith,
            tilt_angle_azimuth=s.tilt_angle_azimuth,
        )
        # Zero absorption, phase-only object
        yx_absorption = torch.zeros_like(yx_phase)
        zyx_data = isotropic_thin_3d.apply_transfer_function(yx_absorption, yx_phase, absorption_tf, phase_tf)
        # Tile 2D phantom along Z so it appears at every defocus plane
        phantom_zyx = np.broadcast_to(yx_phase.numpy()[None, :, :], (Z, Y, X)).copy()

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
    device: str | torch.device | None = None,
) -> xr.Dataset:
    """Compute phase transfer function.

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input CZYX data array (shape is used to determine ZYX dimensions).
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Phase reconstruction settings. Uses defaults if None.
    device : str, torch.device, or None
        Compute device. None = CPU, "auto" = best available.

    Returns
    -------
    xr.Dataset
        For 2D: contains ``singular_system_U``, ``singular_system_S``,
        ``singular_system_Vh``.
        For 3D: contains ``real_potential_transfer_function``,
        ``imaginary_potential_transfer_function``.
    """
    if settings is None:
        settings = Settings()
    device = resolve_device(device)

    zyx_shape = czyx_data.shape[1:]  # CZYX -> ZYX
    s = settings.transfer_function.resolve_floats()

    if recon_dim == 2:
        z_position_list = torch.tensor(
            _position_list_from_shape_scale_offset(
                shape=zyx_shape[0],
                scale=s.z_pixel_size,
                offset=s.z_focus_offset,
            ),
            dtype=torch.float32,
            device=device,
        )

        absorption_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
            yx_shape=[zyx_shape[1], zyx_shape[2]],
            yx_pixel_size=s.yx_pixel_size,
            z_position_list=z_position_list,
            wavelength_illumination=s.wavelength_illumination,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_illumination=s.numerical_aperture_illumination,
            numerical_aperture_detection=s.numerical_aperture_detection,
            invert_phase_contrast=s.invert_phase_contrast,
            tilt_angle_zenith=s.tilt_angle_zenith,
            tilt_angle_azimuth=s.tilt_angle_azimuth,
        )
        U, S, Vh = isotropic_thin_3d.calculate_singular_system(absorption_tf, phase_tf)

        return xr.Dataset(
            {
                "singular_system_U": _named_dataarray(U.cpu().numpy(), "singular_system_U"),
                "singular_system_S": _named_dataarray(S.cpu().numpy(), "singular_system_S"),
                "singular_system_Vh": _named_dataarray(Vh.cpu().numpy(), "singular_system_Vh"),
            }
        )

    elif recon_dim == 3:
        zernike_coeffs = s.get_zernike_coefficients()
        has_zernike = any(c != 0 for c in zernike_coeffs)
        real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape,
            yx_pixel_size=s.yx_pixel_size,
            z_pixel_size=s.z_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            z_padding=s.z_padding,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_illumination=s.numerical_aperture_illumination,
            numerical_aperture_detection=s.numerical_aperture_detection,
            invert_phase_contrast=s.invert_phase_contrast,
            tilt_angle_zenith=s.tilt_angle_zenith,
            tilt_angle_azimuth=s.tilt_angle_azimuth,
            zernike_coefficients=zernike_coeffs if has_zernike else None,
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
    czyx_data: xr.DataArray | list[xr.DataArray],
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
    device: str | torch.device | None = None,
) -> xr.DataArray | list[xr.DataArray]:
    """Reconstruct phase from brightfield data.

    Parameters
    ----------
    czyx_data : xr.DataArray or list[xr.DataArray]
        Input CZYX brightfield data. When a list is provided, tiles are
        stacked into a batch for efficient processing and the result
        is returned as a list of xr.DataArrays.
    transfer_function : xr.Dataset
        Transfer function from ``compute_transfer_function``.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Phase reconstruction settings. Uses defaults if None.
    device : str, torch.device, or None
        Compute device. None = CPU, "auto" = best available.

    Returns
    -------
    xr.DataArray or list[xr.DataArray]
        CZYX array(s) with a single Phase channel.
    """
    if settings is None:
        settings = Settings()
    device = resolve_device(device)

    # Normalize input: list -> (B,Z,Y,X), single -> (Z,Y,X)
    is_list = isinstance(czyx_data, list)
    if is_list:
        data_list = czyx_data
        zyx_tensor = torch.stack(
            [torch.tensor(d.values[0], dtype=torch.float32, device=device) for d in data_list]
        )  # (B, Z, Y, X)
    else:
        zyx_tensor = torch.tensor(czyx_data.values[0], dtype=torch.float32, device=device)  # (Z, Y, X)

    # Single model call — handles both (Z,Y,X) and (B,Z,Y,X)
    # [phase only, 2]
    if recon_dim == 2:
        U, S, Vh = _to_singular_system(transfer_function)
        _, output = isotropic_thin_3d.apply_inverse_transfer_function(
            zyx_tensor,
            (U.to(device), S.to(device), Vh.to(device)),
            **settings.apply_inverse.model_dump(),
        )
    # [phase only, 3]
    elif recon_dim == 3:
        output = phase_thick_3d.apply_inverse_transfer_function(
            zyx_tensor,
            _to_tensor(transfer_function, "real_potential_transfer_function").to(device),
            _to_tensor(transfer_function, "imaginary_potential_transfer_function").to(device),
            z_padding=settings.transfer_function.z_padding,
            **settings.apply_inverse.model_dump(),
        )

    # Wrap output tensor(s) back into xr.DataArray(s)
    ch = _output_channel_names(recon_phase=True, recon_dim=recon_dim)
    sz = recon_dim == 2
    if is_list:
        return [_wrap_output_tensor(output[i], ch, data_list[i], sz) for i in range(len(data_list))]
    return _wrap_output_tensor(output, ch, czyx_data, sz)


def reconstruct(
    czyx_data: xr.DataArray,
    recon_dim: Literal[2, 3],
    settings: Settings = None,
    device: str | torch.device | None = None,
) -> xr.DataArray:
    """Reconstruct phase from brightfield data.

    Convenience function that chains ``compute_transfer_function`` and
    ``apply_inverse_transfer_function``.

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input CZYX brightfield data.
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings, optional
        Phase reconstruction settings. Uses defaults if None.
    device : str, torch.device, or None
        Compute device. None = CPU, "auto" = best available.

    Returns
    -------
    xr.DataArray
        CZYX array with a single Phase channel.
    """
    if settings is None:
        settings = Settings()

    tf = compute_transfer_function(czyx_data, recon_dim, settings, device=device)
    return apply_inverse_transfer_function(czyx_data, tf, recon_dim, settings, device=device)


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
    loss_settings=None,
    log_dir: str | None = None,
    log_images: bool = False,
    device: str | torch.device | None = None,
) -> tuple[Settings, xr.DataArray]:
    """Optimize reconstruction parameters.

    Parameters
    ----------
    czyx_data : xr.DataArray
        CZYX brightfield data (single channel).
    recon_dim : {2, 3}
        Reconstruction dimensionality.
    settings : Settings
        Phase settings. Fields with ``lr > 0`` will be optimized.
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
    loss_settings : LossSettings, optional
        Loss function configuration. Defaults to MidbandPowerLossSettings.
    log_dir : str, optional
        TensorBoard log directory. None = no logging.
    log_images : bool
        If True, log reconstruction images to TensorBoard each iteration.
    device : str, torch.device, or None
        Compute device. None = CPU, "auto" = best available.

    Returns
    -------
    tuple[Settings, xr.DataArray]
        Updated settings with optimized values, and the final reconstruction.
    """
    if settings is None:
        settings = Settings()
    device = resolve_device(device)

    opt_params, _ = extract_optimizable_params(settings.transfer_function)

    if not opt_params:
        print("No optimizable parameters found. Running standard reconstruction.")
        return settings, reconstruct(czyx_data, recon_dim=recon_dim, settings=settings, device=device)

    logger = TensorBoardLogger(log_dir) if log_dir else NullLogger()

    s = settings.transfer_function.resolve_floats()
    zyx_data = torch.tensor(czyx_data.values[0], dtype=torch.float32, device=device)
    Z = zyx_data.shape[0]
    yx_shape = (zyx_data.shape[1], zyx_data.shape[2])

    # Soft pupil cutoff for smooth gradients during optimization.
    # The final reconstruction (after optimize returns) uses the default 1e4.
    optim_steepness = 100.0

    # Pre-allocate z index tensor on device (avoids GPU sync per iteration)
    z_indices = -torch.arange(Z, device=device) + (Z // 2)

    # Zernike field names for resolving tensor params
    zernike_field_names = s.zernike_field_names
    zernike_defaults = s.get_zernike_coefficients()

    def reconstruct_fn(data, **tensor_params):
        na_ill = tensor_params.get("numerical_aperture_illumination", s.numerical_aperture_illumination)
        na_det = tensor_params.get("numerical_aperture_detection", s.numerical_aperture_detection)
        z_offset = tensor_params.get("z_focus_offset", s.z_focus_offset)
        tilt_zenith = tensor_params.get("tilt_angle_zenith", s.tilt_angle_zenith)
        tilt_azimuth = tensor_params.get("tilt_angle_azimuth", s.tilt_angle_azimuth)

        # Collect Zernike coefficients from tensor_params or defaults
        zernike_coeffs = [
            tensor_params.get(name, default) for name, default in zip(zernike_field_names, zernike_defaults)
        ]
        # Only pass zernike if any are non-zero or optimizable
        has_zernike = any(not (isinstance(c, (int, float)) and c == 0) for c in zernike_coeffs)
        zernike_arg = zernike_coeffs if has_zernike else None

        if recon_dim == 2:
            z_positions = (z_indices + z_offset) * s.z_pixel_size
            return isotropic_thin_3d.reconstruct(
                data,
                yx_pixel_size=s.yx_pixel_size,
                z_position_list=z_positions,
                wavelength_illumination=s.wavelength_illumination,
                index_of_refraction_media=s.index_of_refraction_media,
                numerical_aperture_illumination=na_ill,
                numerical_aperture_detection=na_det,
                invert_phase_contrast=s.invert_phase_contrast,
                regularization_strength=settings.apply_inverse.regularization_strength,
                tilt_angle_zenith=tilt_zenith,
                tilt_angle_azimuth=tilt_azimuth,
                pupil_steepness=optim_steepness,
                zernike_coefficients=zernike_arg,
            )[1]  # [1] = phase
        else:
            return phase_thick_3d.reconstruct(
                data,
                yx_pixel_size=s.yx_pixel_size,
                z_pixel_size=s.z_pixel_size,
                wavelength_illumination=s.wavelength_illumination,
                z_padding=s.z_padding,
                index_of_refraction_media=s.index_of_refraction_media,
                numerical_aperture_illumination=na_ill,
                numerical_aperture_detection=na_det,
                invert_phase_contrast=s.invert_phase_contrast,
                regularization_strength=settings.apply_inverse.regularization_strength,
                tilt_angle_zenith=tilt_zenith,
                tilt_angle_azimuth=tilt_azimuth,
                pupil_steepness=optim_steepness,
                zernike_coefficients=zernike_arg,
            )

    if loss_settings is None:
        loss_settings = MidbandPowerLossSettings()

    loss_fn = build_loss_fn(
        loss_settings,
        NA_det=s.numerical_aperture_detection,
        wavelength=s.wavelength_illumination,
        pixel_size=s.yx_pixel_size,
    )

    def log_extras(step, lgr, param_tensors):
        if not log_images:
            return
        tilt_z = param_tensors.get("tilt_angle_zenith", s.tilt_angle_zenith)
        tilt_a = param_tensors.get("tilt_angle_azimuth", s.tilt_angle_azimuth)
        na_ill_t = param_tensors.get("numerical_aperture_illumination", s.numerical_aperture_illumination)
        fyy, fxx = util.generate_frequencies(yx_shape, s.yx_pixel_size)
        pupil = optics.generate_tilted_pupil(
            fxx,
            fyy,
            na_ill_t,
            s.wavelength_illumination,
            s.index_of_refraction_media,
            tilt_z,
            tilt_a,
        )
        lgr.log_image("illumination_pupil", torch.fft.fftshift(pupil).detach(), step)

    optim_kwargs = dict(
        max_iterations=max_iterations,
        method=method,
        use_gradients=use_gradients,
        grid_points=grid_points,
        log_images=log_images,
        log_extras_fn=log_extras,
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

    # Build updated settings
    new_tf_dict = settings.transfer_function.model_dump()
    for dotted_name, value in result.optimized_values.items():
        field_name = dotted_name.split(".")[-1]
        new_tf_dict[field_name] = value

    new_settings = Settings(
        transfer_function=TransferFunctionSettings(**new_tf_dict),
        apply_inverse=settings.apply_inverse,
    )

    final_recon = reconstruct(czyx_data, recon_dim=recon_dim, settings=new_settings, device=device)
    return new_settings, final_recon

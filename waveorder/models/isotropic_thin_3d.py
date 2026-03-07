from __future__ import annotations

import warnings
from typing import Literal, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.filter import apply_filter_bank


def generate_test_phantom(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    wavelength_illumination: float,
    index_of_refraction_media: float,
    index_of_refraction_sample: float,
    sphere_radius: float,
) -> Tuple[Tensor, Tensor]:
    sphere, _, _ = util.generate_sphere_target(
        (3,) + yx_shape,
        yx_pixel_size,
        z_pixel_size=1.0,
        radius=sphere_radius,
        blur_size=2 * yx_pixel_size,
    )
    yx_phase = (
        sphere[1] * (index_of_refraction_sample - index_of_refraction_media) * 0.1 / wavelength_illumination
    )  # phase in radians

    yx_absorption = torch.clone(yx_phase)

    return yx_absorption, yx_phase


def calculate_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    """Calculate the transfer function for 2D phase imaging.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        Shape of YX dimensions
    yx_pixel_size : float
        Pixel size in YX plane
    z_position_list : list or Tensor
        Defocus distances in micrometers
    wavelength_illumination : float
        Wavelength of illumination light
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture
    invert_phase_contrast : bool, optional
        Invert phase contrast, by default False
    tilt_angle_zenith : float or Tensor, optional
        Illumination tilt zenith angle in radians, by default 0.0
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0

    Returns
    -------
    Tuple[Tensor, Tensor]
        absorption_2d_to_3d_transfer_function, phase_2d_to_3d_transfer_function
    """
    # Extract float values for Nyquist computation (not in gradient chain)
    na_ill_val = float(torch.as_tensor(numerical_aperture_illumination).detach())
    na_det_val = float(torch.as_tensor(numerical_aperture_detection).detach())

    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        na_ill_val,
        na_det_val,
    )
    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))

    (
        absorption_2d_to_3d_transfer_function,
        phase_2d_to_3d_transfer_function,
    ) = _calculate_wrap_unsafe_transfer_function(
        (
            yx_shape[0] * yx_factor,
            yx_shape[1] * yx_factor,
        ),
        yx_pixel_size / yx_factor,
        z_position_list,
        wavelength_illumination,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )

    return (
        sampling.nd_fourier_central_cuboid(absorption_2d_to_3d_transfer_function, yx_shape),
        sampling.nd_fourier_central_cuboid(phase_2d_to_3d_transfer_function, yx_shape),
    )


def _calculate_wrap_unsafe_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    na_ill = torch.as_tensor(numerical_aperture_illumination, dtype=torch.float32)
    na_det = torch.as_tensor(numerical_aperture_detection, dtype=torch.float32)

    # Clamp illumination NA if >= detection NA (differentiable)
    clamped_ill = torch.where(
        na_ill >= na_det,
        0.9 * na_det,
        na_ill,
    )
    if (na_ill >= na_det).any():
        warnings.warn(
            "numerical_aperture_illumination is >= "
            "numerical_aperture_detection. Setting "
            "numerical_aperture_illumination to 0.9 * "
            "numerical_aperture_detection to avoid singularities."
        )

    z_positions = torch.as_tensor(z_position_list, dtype=torch.float32)
    if invert_phase_contrast:
        z_positions = -z_positions

    with torch.no_grad():
        fyy, fxx = util.generate_frequencies(yx_shape, yx_pixel_size)
        fyy, fxx = fyy.to(z_positions.device), fxx.to(z_positions.device)
        radial_frequencies = torch.sqrt(fyy**2 + fxx**2)

    illumination_pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        clamped_ill,
        wavelength_illumination,
        index_of_refraction_media,
        tilt_angle_zenith,
        tilt_angle_azimuth,
    )

    detection_pupil = optics.generate_pupil(
        radial_frequencies,
        na_det,
        wavelength_illumination,
        steepness=pupil_steepness,
    )
    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        detection_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_positions,
    )

    # Batched WOTF: (Y,X) * (Z,Y,X) broadcasts to (Z,Y,X)
    return optics.compute_weak_object_transfer_function_2d(
        illumination_pupil, detection_pupil.unsqueeze(0) * propagation_kernel
    )


def calculate_singular_system(
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the singular system of the absorption and phase transfer
    functions.

    Together, the transfer functions form a (2, Z, Vy, Vx) tensor, where
    (2,) is the object-space dimension (abs, phase), (Z,) is the data-space
    dimension, and (Vy, Vx) are the spatial frequency dimensions.

    Uses a norm-based decomposition that is faster than full SVD and
    supports gradient flow through complex tensors.

    Parameters
    ----------
    absorption_2d_to_3d_transfer_function : Tensor
        ZYX transfer function for absorption
    phase_2d_to_3d_transfer_function : Tensor
        ZYX transfer function for phase

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        U (2, 2, Vy, Vx), S (2, Vy, Vx), Vh (2, Z, Vy, Vx)
    """
    # sfYX shape: (s=2, Z, Vy, Vx)
    sfYX = torch.stack(
        (
            absorption_2d_to_3d_transfer_function,
            phase_2d_to_3d_transfer_function,
        ),
        dim=0,
    )
    s, Z, Vy, Vx = sfYX.shape

    # Per-channel norms: S[k] = norm(H[k, :]) for k in {absorption, phase}
    S = torch.sqrt(
        torch.clamp(
            torch.sum(torch.abs(sfYX) ** 2, dim=1),
            min=1e-12,
        )
    )  # (s=2, Vy, Vx)

    # Normalized rows: Vh[k, z] = H[k, z] / S[k]
    Vh = sfYX / (S[:, None] + 1e-12)  # (s=2, Z, Vy, Vx)

    # U = identity (each channel reconstructs independently)
    U = torch.zeros(s, s, Vy, Vx, dtype=sfYX.dtype, device=sfYX.device)
    for i in range(s):
        U[i, i] = 1.0

    return U, S, Vh


def visualize_transfer_function(
    viewer,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> None:
    """Note: unlike other `visualize_transfer_function` calls, this transfer
    function is a mixed 3D-to-2D transfer function, so it cannot reuse
    util.add_transfer_function_to_viewer. If more 3D-to-2D transfer functions
    are added, consider refactoring.
    """
    arrays = [
        (torch.imag(absorption_2d_to_3d_transfer_function), "Im(absorb TF)"),
        (torch.real(absorption_2d_to_3d_transfer_function), "Re(absorb TF)"),
        (torch.imag(phase_2d_to_3d_transfer_function), "Im(phase TF)"),
        (torch.real(phase_2d_to_3d_transfer_function), "Re(phase TF)"),
    ]

    for array in arrays:
        lim = (0.5 * torch.max(torch.abs(array[0]))).item()
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (2, 0, 1)


def visualize_point_spread_function(
    viewer,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> None:
    arrays = [
        (torch.fft.ifftn(absorption_2d_to_3d_transfer_function), "absorb PSF"),
        (torch.fft.ifftn(phase_2d_to_3d_transfer_function), "phase PSF"),
    ]

    for array in arrays:
        lim = (0.5 * torch.max(torch.abs(array[0]))).item()
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (0, 1, 2)


def apply_transfer_function(
    yx_absorption: Tensor,
    yx_phase: Tensor,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> Tensor:
    # Very simple simulation, consider adding noise and bkg knobs

    # simulate absorbing object
    yx_absorption_hat = torch.fft.fftn(yx_absorption)
    zyx_absorption_data_hat = yx_absorption_hat[None, ...] * absorption_2d_to_3d_transfer_function
    zyx_absorption_data = torch.real(torch.fft.ifftn(zyx_absorption_data_hat, dim=(1, 2)))

    # simulate phase object
    yx_phase_hat = torch.fft.fftn(yx_phase)
    zyx_phase_data_hat = yx_phase_hat[None, ...] * phase_2d_to_3d_transfer_function
    zyx_phase_data = torch.real(torch.fft.ifftn(zyx_phase_data_hat, dim=(1, 2)))

    # sum and add background
    data = zyx_absorption_data + zyx_phase_data
    data = data + 10  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    singular_system: Tuple[Tensor, Tensor, Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    reg_p: float = 1e-6,  # TODO: use this parameter
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    bg_filter: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Reconstructs absorption and phase from zyx_data and a pair of
    3D-to-2D transfer functions named absorption_2d_to_3d_transfer_function and
    phase_2d_to_3d_transfer_function, providing options for reconstruction
    algorithms.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, label-free defocus stack
    singular_system : Tuple[Tensor, Tensor, Tensor]
        singular system of the transfer function bank
    reconstruction_algorithm : Literal["Tikhonov";, "TV";], optional
        "Tikhonov" or "TV", by default "Tikhonov"
        "TV" is not implemented.
    regularization_strength : float, optional
        regularization parameter, by default 1e-3
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
        "TV" is not implemented.
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
        "TV" is not implemented.
    bg_filter : bool, optional
        option for slow-varying 2D background normalization with uniform filter
        by default False

    Returns
    -------
    Tuple[Tensor]
        yx_absorption (unitless)
        yx_phase (radians)

    Raises
    ------
    NotImplementedError
        TV is not implemented
    """
    # Normalize
    zyx = util.inten_normalization(zyx_data, bg_filter=bg_filter)

    # TODO Consider refactoring with vectorial transfer function SVD
    if reconstruction_algorithm == "Tikhonov":
        U, S, Vh = singular_system
        S_reg = S / (S**2 + regularization_strength)
        sfyx_inverse_filter = torch.einsum("sj...,j...,jf...->fs...", U, S_reg, Vh)

        absorption_yx, phase_yx = apply_filter_bank(sfyx_inverse_filter, zyx)

    # ADMM deconvolution with anisotropic TV regularization
    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    return absorption_yx, phase_yx


def reconstruct(
    zyx_data: Tensor,
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor] = 0.9,
    numerical_aperture_detection: Union[float, Tensor] = 1.2,
    invert_phase_contrast: bool = False,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    reg_p: float = 1e-6,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    bg_filter: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct 2D absorption and phase from a brightfield defocus stack.

    Chains calculate_transfer_function, calculate_singular_system,
    and apply_inverse_transfer_function.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, label-free defocus stack
    yx_pixel_size : float
        Pixel size in the transverse (Y, X) dimensions
    z_position_list : list or Tensor
        Defocus distances in micrometers
    wavelength_illumination : float
        Wavelength of illumination light
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture
    invert_phase_contrast : bool, optional
        Invert phase contrast, by default False
    reconstruction_algorithm : str, optional
        "Tikhonov" or "TV", by default "Tikhonov"
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    bg_filter : bool, optional
        Slow-varying 2D background normalization, by default False
    tilt_angle_zenith : float or Tensor, optional
        Illumination tilt zenith angle in radians, by default 0.0
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0

    Returns
    -------
    Tuple[Tensor, Tensor]
        yx_absorption, yx_phase
    """
    absorption_tf, phase_tf = calculate_transfer_function(
        zyx_data.shape[-2:],
        yx_pixel_size,
        z_position_list,
        wavelength_illumination,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )
    singular_system = calculate_singular_system(absorption_tf, phase_tf)
    return apply_inverse_transfer_function(
        zyx_data,
        singular_system,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        reg_p=reg_p,
        TV_rho_strength=TV_rho_strength,
        TV_iterations=TV_iterations,
        bg_filter=bg_filter,
    )

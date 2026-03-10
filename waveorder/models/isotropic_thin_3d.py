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
        Scalar for shared tilt; ``(B,)`` tensor for per-tile tilt
        (produces batched output).
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
        Scalar for shared tilt; ``(B,)`` tensor for per-tile tilt.
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(absorption_tf, phase_tf)`` with shape ``(Z, Y, X)`` or
        ``(B, Z, Y, X)`` when batched tilt angles are provided.
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

    # nd_fourier_central_cuboid preserves leading batch dims
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
        fyy, fxx = util.generate_frequencies(yx_shape, yx_pixel_size, device=z_positions.device)
        radial_frequencies = torch.sqrt(fyy**2 + fxx**2)

    # Detect batched tilt angles
    tilt_zenith_t = torch.as_tensor(tilt_angle_zenith, dtype=torch.float32)
    tilt_azimuth_t = torch.as_tensor(tilt_angle_azimuth, dtype=torch.float32)
    batched = tilt_zenith_t.ndim >= 1 and tilt_zenith_t.shape[0] > 1

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

    # det_prop: (Z, Yos, Xos)
    det_prop = detection_pupil.unsqueeze(0) * propagation_kernel

    # Generate tilted illumination pupil
    # For batched (B,) tilt angles, reshape to (B, 1, 1) so
    # generate_tilted_pupil broadcasts against (Yos, Xos) grids
    if batched:
        tilt_angle_zenith = tilt_zenith_t[:, None, None]
        tilt_angle_azimuth = tilt_azimuth_t[:, None, None]

    illumination_pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        clamped_ill,
        wavelength_illumination,
        index_of_refraction_media,
        tilt_angle_zenith,
        tilt_angle_azimuth,
    )  # (Yos, Xos) or (B, Yos, Xos)

    if not batched:
        # Unbatched WOTF: ill (Yos, Xos) broadcasts against det_prop (Z, Yos, Xos)
        return optics.compute_weak_object_transfer_function_2d(illumination_pupil, det_prop)

    # Batched WOTF: ill (B, 1, Yos, Xos) broadcasts against
    # det_prop (1, Z, Yos, Xos) -> (B, Z, Yos, Xos)
    return optics.compute_weak_object_transfer_function_2d(illumination_pupil[:, None], det_prop[None])


def calculate_singular_system(
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the singular system of the absorption and phase transfer
    functions.

    Parameters
    ----------
    absorption_2d_to_3d_transfer_function : Tensor
        Transfer function for absorption, shape ``(Z, Vy, Vx)`` or
        ``(B, Z, Vy, Vx)``
    phase_2d_to_3d_transfer_function : Tensor
        Transfer function for phase, same shape as absorption TF

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        - U : ``(2, 2, Vy, Vx)`` or ``(B, 2, 2, Vy, Vx)``
        - S : ``(2, Vy, Vx)`` or ``(B, 2, Vy, Vx)``
        - Vh : ``(2, Z, Vy, Vx)`` or ``(B, 2, Z, Vy, Vx)``
    """
    batched = absorption_2d_to_3d_transfer_function.ndim == 4
    if not batched:
        absorption_2d_to_3d_transfer_function = absorption_2d_to_3d_transfer_function.unsqueeze(0)
        phase_2d_to_3d_transfer_function = phase_2d_to_3d_transfer_function.unsqueeze(0)

    # sfYX shape: (B, s=2, Z, Vy, Vx)
    sfYX = torch.stack(
        (
            absorption_2d_to_3d_transfer_function,
            phase_2d_to_3d_transfer_function,
        ),
        dim=1,
    )
    B, s, Z, Vy, Vx = sfYX.shape

    # Per-channel norms: S[b, k] = norm(H[b, k, :])
    S = torch.sqrt(
        torch.clamp(
            torch.sum(torch.abs(sfYX) ** 2, dim=2),
            min=1e-12,
        )
    )  # (B, s=2, Vy, Vx)

    # Normalized rows: Vh[b, k, z] = H[b, k, z] / S[b, k]
    Vh = sfYX / (S[:, :, None] + 1e-12)  # (B, s=2, Z, Vy, Vx)

    # U = identity (each channel reconstructs independently)
    U = torch.zeros(B, s, s, Vy, Vx, dtype=sfYX.dtype, device=sfYX.device)
    for i in range(s):
        U[:, i, i] = 1.0

    if not batched:
        U = U.squeeze(0)
        S = S.squeeze(0)
        Vh = Vh.squeeze(0)

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
    """Reconstructs absorption and phase from zyx_data.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    singular_system : Tuple[Tensor, Tensor, Tensor]
        Singular system ``(U, S, Vh)``. Unbatched shapes:
        ``(2, 2, Vy, Vx)``, ``(2, Vy, Vx)``, ``(2, Z, Vy, Vx)``.
        Batched shapes have a leading ``B`` dimension.
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov". "TV" is not implemented.
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
    TV_rho_strength : float, optional
        TV-specific rho strength, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    bg_filter : bool, optional
        Slow-varying 2D background normalization, by default False

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(yx_absorption, yx_phase)`` with shape ``(Y, X)`` or
        ``(B, Y, X)``.
    """
    batched = zyx_data.ndim == 4
    if not batched:
        zyx_data = zyx_data.unsqueeze(0)

    # Normalize: (B, Z, Y, X)
    zyx = util.inten_normalization(zyx_data, bg_filter=bg_filter)

    # TODO Consider refactoring with vectorial transfer function SVD
    if reconstruction_algorithm == "Tikhonov":
        U, S, Vh = singular_system
        batched_ss = S.ndim == 4  # (B, 2, Vy, Vx)

        if not batched_ss:
            # Shared singular system: use apply_filter_bank per tile
            S_reg = S / (S**2 + regularization_strength)
            sfyx_inverse_filter = torch.einsum("sj...,j...,jf...->fs...", U, S_reg, Vh)
            results = []
            for b in range(zyx.shape[0]):
                results.append(apply_filter_bank(sfyx_inverse_filter, zyx[b]))
            output = torch.stack(results, dim=0)  # (B, 2, Y, X)
        else:
            # Per-tile singular system: batched direct FFT multiply
            S_reg = S / (S**2 + regularization_strength)  # (B, 2, Y, X)
            # inverse_filter: S_reg * Vh -> (B, 2, Z, Y, X)
            inverse_filter = S_reg[:, :, None, :, :] * Vh  # (B, s=2, Z=f, Y, X)
            # Transpose to (B, Z=f, s=2, Y, X) for matrix multiply
            inverse_filter = inverse_filter.permute(0, 2, 1, 3, 4)

            data_fft = torch.fft.fft2(zyx)  # (B, Z, Y, X)
            # filter (B, Z, 2, Y, X) * data (B, Z, 1, Y, X) -> sum over Z -> (B, 2, Y, X)
            output_fft = (inverse_filter * data_fft[:, :, None, :, :]).sum(dim=1)
            output = torch.fft.ifft2(output_fft).real  # (B, 2, Y, X)

    # ADMM deconvolution with anisotropic TV regularization
    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    absorption_yx = output[:, 0]  # (B, Y, X)
    phase_yx = output[:, 1]  # (B, Y, X)

    if not batched:
        absorption_yx = absorption_yx.squeeze(0)
        phase_yx = phase_yx.squeeze(0)

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

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
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
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov".
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
        Scalar for shared tilt, ``(B,)`` tensor for per-tile tilt.
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
        Scalar for shared tilt, ``(B,)`` tensor for per-tile tilt.
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(yx_absorption, yx_phase)`` with shape ``(Y, X)`` or ``(B, Y, X)``.
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

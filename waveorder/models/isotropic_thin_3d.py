from typing import Literal, Tuple

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
        sphere[1]
        * (index_of_refraction_sample - index_of_refraction_media)
        * 0.1
        / wavelength_illumination
    )  # phase in radians

    yx_absorption = torch.clone(yx_phase)

    return yx_absorption, yx_phase


def calculate_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: list,
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    invert_phase_contrast: bool = False,
    illumination_sector_angles: list[tuple[float, float]] = None,
) -> Tuple[Tensor, Tensor]:
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        numerical_aperture_illumination,
        numerical_aperture_detection,
    )
    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))

    # Handle sector illumination case (or single channel with full aperture)
    if illumination_sector_angles is None:
        # Single channel with full aperture - wrap as [(0, 360)]
        illumination_sector_angles = [(0, 360)]

    absorption_tfs = []
    phase_tfs = []

    for start_angle, end_angle in illumination_sector_angles:
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
            sector_angle_start=start_angle,
            sector_angle_end=end_angle,
        )

        absorption_2d_to_3d_transfer_function_out = torch.zeros(
            (len(z_position_list),) + tuple(yx_shape), dtype=torch.complex64
        )
        phase_2d_to_3d_transfer_function_out = torch.zeros(
            (len(z_position_list),) + tuple(yx_shape), dtype=torch.complex64
        )

        for z in range(len(z_position_list)):
            absorption_2d_to_3d_transfer_function_out[z] = (
                sampling.nd_fourier_central_cuboid(
                    absorption_2d_to_3d_transfer_function[z], yx_shape
                )
            )
            phase_2d_to_3d_transfer_function_out[z] = (
                sampling.nd_fourier_central_cuboid(
                    phase_2d_to_3d_transfer_function[z], yx_shape
                )
            )

        absorption_tfs.append(absorption_2d_to_3d_transfer_function_out)
        phase_tfs.append(phase_2d_to_3d_transfer_function_out)

    # Always return (C, Z, Y, X) arrays, even for single channel
    return (
        torch.stack(absorption_tfs, dim=0),
        torch.stack(phase_tfs, dim=0),
    )


def _calculate_wrap_unsafe_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: list,
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    invert_phase_contrast: bool = False,
    sector_angle_start: float = None,
    sector_angle_end: float = None,
) -> Tuple[Tensor, Tensor]:
    if numerical_aperture_illumination >= numerical_aperture_detection:
        print(
            "Warning: numerical_aperture_illumination is >= "
            "numerical_aperture_detection. Setting "
            "numerical_aperture_illumination to 0.9 * "
            "numerical_aperture_detection to avoid singularities."
        )
        numerical_aperture_illumination = 0.9 * numerical_aperture_detection

    if invert_phase_contrast:
        z_position_list = [-1 * x for x in z_position_list]
    radial_frequencies = util.generate_radial_frequencies(
        yx_shape, yx_pixel_size
    )

    # Generate illumination pupil (sector or full aperture)
    if sector_angle_start is not None and sector_angle_end is not None:
        fyy, fxx = util.generate_frequencies(yx_shape, yx_pixel_size)
        illumination_pupil = optics.generate_sector_pupil(
            radial_frequencies,
            fxx,
            fyy,
            numerical_aperture_illumination,
            wavelength_illumination,
            sector_angle_start,
            sector_angle_end,
        )
    else:
        illumination_pupil = optics.generate_pupil(
            radial_frequencies,
            numerical_aperture_illumination,
            wavelength_illumination,
        )

    detection_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_illumination,
    )
    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        detection_pupil,
        wavelength_illumination / index_of_refraction_media,
        torch.tensor(z_position_list),
    )

    zyx_shape = (len(z_position_list),) + tuple(yx_shape)
    absorption_2d_to_3d_transfer_function = torch.zeros(
        zyx_shape, dtype=torch.complex64
    )
    phase_2d_to_3d_transfer_function = torch.zeros(
        zyx_shape, dtype=torch.complex64
    )
    for z in range(len(z_position_list)):
        (
            absorption_2d_to_3d_transfer_function[z],
            phase_2d_to_3d_transfer_function[z],
        ) = optics.compute_weak_object_transfer_function_2d(
            illumination_pupil, detection_pupil * propagation_kernel[z]
        )

    return (
        absorption_2d_to_3d_transfer_function,
        phase_2d_to_3d_transfer_function,
    )


def calculate_singular_system(
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the singular system of the absoprtion and phase transfer
    functions for multi-channel data.

    Together, the transfer functions form a (2, C, Z, Vy, Vx) tensor, where
    (2,) is the object-space dimension (abs, phase), (C,) is the channel dimension,
    (Z,) is the data-space dimension, and (Vy, Vx) are the spatial frequency dimensions.

    The SVD is computed over the (2, C*Z) dimensions, flattening channels and z together.

    Parameters
    ----------
    absorption_2d_to_3d_transfer_function : Tensor
        CZYX transfer function for absorption
    phase_2d_to_3d_transfer_function : Tensor
        CZYX transfer function for phase

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        U, S, Vh for the singular system
    """
    # absorption_2d_to_3d_transfer_function shape: (C, Z, Y, X)
    # phase_2d_to_3d_transfer_function shape: (C, Z, Y, X)

    # Stack absorption and phase: (2, C, Z, Y, X)
    sfYX_transfer_function = torch.stack(
        (
            absorption_2d_to_3d_transfer_function,
            phase_2d_to_3d_transfer_function,
        ),
        dim=0,
    )

    # Flatten C and Z dimensions: (2, C*Z, Y, X)
    num_channels = sfYX_transfer_function.shape[1]
    num_z = sfYX_transfer_function.shape[2]
    sfYX_transfer_function = sfYX_transfer_function.reshape(
        2,
        num_channels * num_z,
        sfYX_transfer_function.shape[3],
        sfYX_transfer_function.shape[4],
    )

    # Permute for SVD: (Y, X, 2, C*Z)
    YXsf_transfer_function = sfYX_transfer_function.permute(2, 3, 0, 1)
    Up, Sp, Vhp = torch.linalg.svd(YXsf_transfer_function, full_matrices=False)
    U = Up.permute(2, 3, 0, 1)
    S = Sp.permute(2, 0, 1)
    Vh = Vhp.permute(2, 3, 0, 1)
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
    zyx_absorption_data_hat = yx_absorption_hat[None, ...] * torch.real(
        absorption_2d_to_3d_transfer_function
    )
    zyx_absorption_data = torch.real(
        torch.fft.ifftn(zyx_absorption_data_hat, dim=(1, 2))
    )

    # simulate phase object
    yx_phase_hat = torch.fft.fftn(yx_phase)
    zyx_phase_data_hat = yx_phase_hat[None, ...] * torch.real(
        phase_2d_to_3d_transfer_function
    )
    zyx_phase_data = torch.real(
        torch.fft.ifftn(zyx_phase_data_hat, dim=(1, 2))
    )

    # sum and add background
    data = zyx_absorption_data + zyx_phase_data
    data += 10  # Add a direct background
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
    """Reconstructs absorption and phase from multi-channel zyx_data and the
    singular system, combining all illumination channels into single absorption
    and phase estimates.

    Parameters
    ----------
    zyx_data : Tensor
        Multi-channel 3D raw data with shape (C, Z, Y, X).
        For single channel (full aperture), C=1.
    singular_system : Tuple[Tensor, Tensor, Tensor]
        singular system of the multi-channel transfer function bank
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
        yx_absorption (unitless) with shape (Y, X)
        yx_phase (radians) with shape (Y, X)

    Raises
    ------
    NotImplementedError
        TV is not implemented
    """
    # zyx_data shape: (C, Z, Y, X)
    num_channels = zyx_data.shape[0]
    num_z = zyx_data.shape[1]

    # Flatten C and Z dimensions: (C*Z, Y, X)
    czyx_data = zyx_data.reshape(
        num_channels * num_z, zyx_data.shape[2], zyx_data.shape[3]
    )

    # Normalize
    czyx = util.inten_normalization(czyx_data, bg_filter=bg_filter)

    # TODO Consider refactoring with vectorial transfer function SVD
    if reconstruction_algorithm == "Tikhonov":
        print("Computing inverse filter")
        U, S, Vh = singular_system
        S_reg = S / (S**2 + regularization_strength)
        sfyx_inverse_filter = torch.einsum(
            "sj...,j...,jf...->fs...", U, S_reg, Vh
        )

        absorption_yx, phase_yx = apply_filter_bank(sfyx_inverse_filter, czyx)

    # ADMM deconvolution with anisotropic TV regularization
    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    return absorption_yx, phase_yx

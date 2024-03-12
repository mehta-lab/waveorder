from typing import Literal, Tuple

import torch
from torch import Tensor

from waveorder import optics, util


def generate_test_phantom(
    yx_shape,
    yx_pixel_size,
    wavelength_illumination,
    index_of_refraction_media,
    index_of_refraction_sample,
    sphere_radius,
):
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

    yx_absorption = 0.02 * sphere[1]

    return yx_absorption, yx_phase


def calculate_transfer_function(
    yx_shape,
    yx_pixel_size,
    z_position_list,
    wavelength_illumination,
    index_of_refraction_media,
    numerical_aperture_illumination,
    numerical_aperture_detection,
    invert_phase_contrast=False,
):
    if invert_phase_contrast:
        z_position_list = torch.flip(torch.tensor(z_position_list), dims=(0,))

    radial_frequencies = util.generate_radial_frequencies(
        yx_shape, yx_pixel_size
    )

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


def visualize_transfer_function(
    viewer,
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
):
    # TODO: consider generalizing w/ phase_thick_3d.visualize_transfer_function
    arrays = [
        (torch.imag(absorption_2d_to_3d_transfer_function), "Im(absorb TF)"),
        (torch.real(absorption_2d_to_3d_transfer_function), "Re(absorb TF)"),
        (torch.imag(phase_2d_to_3d_transfer_function), "Im(phase TF)"),
        (torch.real(phase_2d_to_3d_transfer_function), "Re(phase TF)"),
    ]

    for array in arrays:
        lim = 0.5 * torch.max(torch.abs(array[0]))
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (0, 1, 2)


def visualize_point_spread_function(
    viewer,
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
):
    arrays = [
        (torch.fft.ifftn(absorption_2d_to_3d_transfer_function), "absorb PSF"),
        (torch.fft.ifftn(phase_2d_to_3d_transfer_function), "phase PSF"),
    ]

    for array in arrays:
        lim = 0.5 * torch.max(torch.abs(array[0]))
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (0, 1, 2)


def apply_transfer_function(
    yx_absorption,
    yx_phase,
    phase_2d_to_3d_transfer_function,
    absorption_2d_to_3d_transfer_function,
):
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
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-6,
    reg_p: float = 1e-6,  # TODO: use this parameter
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    bg_filter: bool = True,
) -> Tuple[Tensor]:
    """Reconstructs absorption and phase from zyx_data and a pair of
    3D-to-2D transfer functions named absorption_2d_to_3d_transfer_function and
    phase_2d_to_3d_transfer_function, providing options for reconstruction
    algorithms.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, label-free defocus stack
    absorption_2d_to_3d_transfer_function : Tensor
        3D-to-2D absorption transfer function, see calculate_transfer_function above
    phase_2d_to_3d_transfer_function : Tensor
        3D-to-2D phase transfer function, see calculate_transfer_function above
    reconstruction_algorithm : Literal[&quot;Tikhonov&quot;, &quot;TV&quot;], optional
        "Tikhonov" or "TV", by default "Tikhonov"
        "TV" is not implemented.
    regularization_strength : float, optional
        regularization parameter, by default 1e-6
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
        "TV" is not implemented.
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
        "TV" is not implemented.
    bg_filter : bool, optional
        option for slow-varying 2D background normalization with uniform filter
        by default True

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
    zyx_data_normalized = util.inten_normalization(
        zyx_data, bg_filter=bg_filter
    )

    zyx_data_hat = torch.fft.fft2(zyx_data_normalized, dim=(1, 2))

    # TODO AHA and b_vec calculations should be moved into tikhonov/tv calculations
    AHA = [
        torch.sum(torch.abs(absorption_2d_to_3d_transfer_function) ** 2, dim=0)
        + regularization_strength,
        torch.sum(
            torch.conj(absorption_2d_to_3d_transfer_function)
            * phase_2d_to_3d_transfer_function,
            dim=0,
        ),
        torch.sum(
            torch.conj(
                phase_2d_to_3d_transfer_function,
            )
            * absorption_2d_to_3d_transfer_function,
            dim=0,
        ),
        torch.sum(
            torch.abs(
                phase_2d_to_3d_transfer_function,
            )
            ** 2,
            dim=0,
        )
        + reg_p,
    ]

    b_vec = [
        torch.sum(
            torch.conj(absorption_2d_to_3d_transfer_function) * zyx_data_hat,
            dim=0,
        ),
        torch.sum(
            torch.conj(
                phase_2d_to_3d_transfer_function,
            )
            * zyx_data_hat,
            dim=0,
        ),
    ]

    # Deconvolution with Tikhonov regularization
    if reconstruction_algorithm == "Tikhonov":
        absorption, phase = util.dual_variable_tikhonov_deconvolution_2d(
            AHA, b_vec
        )

    # ADMM deconvolution with anisotropic TV regularization
    elif reconstruction_algorithm == "TV":
        raise NotImplementedError
        absorption, phase = util.dual_variable_admm_tv_deconv_2d(
            AHA, b_vec, rho=TV_rho_strength, itr=TV_iterations
        )

    phase -= torch.mean(phase)

    return absorption, phase

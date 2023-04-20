import torch
import numpy as np
from waveorder import optics, util


def calculate_transfer_function(
    yx_shape,
    yx_pixel_size,
    z_position_list,
    wavelength_illumination,
    index_of_refraction_media,
    numerical_aperture_illumination,
    numerical_aperture_detection,
):
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

    zyx_shape = (len(z_position_list),) + yx_shape
    absorption_2D_to_3D_transfer_function = torch.zeros(
        zyx_shape, dtype=torch.complex64
    )
    phase_2D_to_3D_transfer_function = torch.zeros(
        zyx_shape, dtype=torch.complex64
    )
    for z in range(len(z_position_list)):
        (
            absorption_2D_to_3D_transfer_function[z],
            phase_2D_to_3D_transfer_function[z],
        ) = optics.compute_weak_object_transfer_function_2D(
            illumination_pupil, detection_pupil * propagation_kernel[z]
        )

    return (
        absorption_2D_to_3D_transfer_function,
        phase_2D_to_3D_transfer_function,
    )


def visualize_transfer_function(
    viewer,
    absorption_2D_to_3D_transfer_function,
    phase_2D_to_3D_transfer_function,
):
    # TODO: consider generalizing w/ phase3Dto3D.visualize_TF
    arrays = [
        (torch.imag(absorption_2D_to_3D_transfer_function), "Im(absorb TF)"),
        (torch.real(absorption_2D_to_3D_transfer_function), "Re(absorb TF)"),
        (torch.imag(phase_2D_to_3D_transfer_function), "Im(phase TF)"),
        (torch.real(phase_2D_to_3D_transfer_function), "Re(phase TF)"),
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
    viewer.dims.order = (2, 0, 1)


def apply_transfer_function(zyx_object, phase_2D_to_3D_transfer_function):
    # Very simple simulation, consider adding noise and bkg knobs
    # TODO: extend to absorption, or restrict to just phase
    zyx_object_hat = torch.fft.fftn(zyx_object, dim=(1, 2))
    zyx_data = zyx_object_hat * torch.real(phase_2D_to_3D_transfer_function)
    data = torch.real(torch.fft.ifftn(zyx_data, dim=(1, 2)))

    data = torch.tensor(data + 10)  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data,
    absorption_2D_to_3D_transfer_function,
    phase_2D_to_3D_transfer_function,
    method="Tikhonov",
    reg_u=1e-6,
    reg_p=1e-6,
    bg_filter=True,
    **kwargs
):
    zyx_data_normalized = util.inten_normalization(
        zyx_data, bg_filter=bg_filter
    )

    zyx_data_hat = torch.fft.fft2(zyx_data_normalized, dim=(1, 2))

    # TODO AHA and b_vec calculations should be moved into tikhonov calculations
    AHA = [
        torch.sum(torch.abs(absorption_2D_to_3D_transfer_function) ** 2, dim=0)
        + reg_u,
        torch.sum(
            torch.conj(absorption_2D_to_3D_transfer_function)
            * phase_2D_to_3D_transfer_function,
            dim=0,
        ),
        torch.sum(
            torch.conj(
                phase_2D_to_3D_transfer_function,
            )
            * absorption_2D_to_3D_transfer_function,
            dim=0,
        ),
        torch.sum(
            torch.abs(
                phase_2D_to_3D_transfer_function,
            )
            ** 2,
            dim=0,
        )
        + reg_p,
    ]

    b_vec = [
        torch.sum(
            torch.conj(absorption_2D_to_3D_transfer_function) * zyx_data_hat,
            dim=0,
        ),
        torch.sum(
            torch.conj(
                phase_2D_to_3D_transfer_function,
            )
            * zyx_data_hat,
            dim=0,
        ),
    ]

    # Deconvolution with Tikhonov regularization
    if method == "Tikhonov":
        mu_sample, phi_sample = util.dual_variable_tikhonov_deconvolution_2D(
            AHA, b_vec
        )

    # ADMM deconvolution with anisotropic TV regularization
    elif method == "TV":
        mu_sample, phi_sample = util.dual_variable_admm_tv_deconv_2D(
            AHA, b_vec, **kwargs
        )

    phi_sample -= torch.mean(phi_sample)

    return phi_sample

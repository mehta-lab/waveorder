import numpy as np
import torch
from waveorder import optics, util
from waveorder.models import isotropic_fluorescent_thick_3d


def generate_test_phantom(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_illumination,
    index_of_refraction_media,
    index_of_refraction_sample,
    sphere_radius,
):
    sphere, _, _ = util.generate_sphere_target(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        radius=sphere_radius,
        blur_size=2 * yx_pixel_size,
    )
    zyx_phase = (
        sphere
        * (index_of_refraction_sample - index_of_refraction_media)
        * z_pixel_size
        / wavelength_illumination
    )  # phase in radians

    return zyx_phase


def calculate_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_illumination,
    z_padding,
    index_of_refraction_media,
    numerical_aperture_illumination,
    numerical_aperture_detection,
    axial_flip=False,
):
    radial_frequencies = util.generate_radial_frequencies(
        zyx_shape[1:], yx_pixel_size
    )
    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    if axial_flip:
        z_position_list = torch.flip(z_position_list, dims=(0,))

    ill_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_illumination,
        wavelength_illumination,
    )
    det_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_illumination,
    )
    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
    )
    greens_function_z = optics.generate_greens_function_z(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
    )

    (
        real_potential_transfer_function,
        imag_potential_transfer_function,
    ) = optics.compute_weak_object_transfer_function_3D(
        ill_pupil,
        ill_pupil,
        det_pupil,
        propagation_kernel,
        greens_function_z,
        z_pixel_size,
    )

    return real_potential_transfer_function, imag_potential_transfer_function


def visualize_transfer_function(
    viewer,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    zyx_scale,
):
    # TODO: consider generalizing w/ phase2Dto3D.visualize_TF
    arrays = [
        (torch.real(imag_potential_transfer_function), "Re(imag pot. TF)"),
        (torch.imag(imag_potential_transfer_function), "Im(imag pot. TF)"),
        (torch.real(real_potential_transfer_function), "Re(real pot. TF)"),
        (torch.imag(real_potential_transfer_function), "Im(real pot. TF)"),
    ]

    for array in arrays:
        lim = 0.5 * torch.max(torch.abs(array[0]))
        viewer.add_image(
            torch.fft.ifftshift(array[0]).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / zyx_scale,
        )
    viewer.dims.order = (0, 1, 2)


def apply_transfer_function(
    zyx_object, real_potential_transfer_function, z_padding
):
    # This simplified forward model only handles phase, so it resuses the fluorescence forward model
    # TODO: extend to absorption
    return isotropic_fluorescent_thick_3d.apply_transfer_function(
        zyx_object, real_potential_transfer_function, z_padding
    )


def apply_inverse_transfer_function(
    zyx_data,
    real_potential_transfer_function,
    imaginary_potential_transfer_function,
    z_padding,
    z_pixel_size,  # TODO: MOVE THIS PARAM TO OTF? (leaky param)
    wavelength_illumination,  # TOOD: MOVE THIS PARAM TO OTF? (leaky param)
    absorption_ratio=0.0,
    reconstruction_algorithm="Tikhonov",
    regularization_strength=1e-3,
    TV_rho_strength=1e-3,
    TV_iterations=10,
):
    # Handle padding
    zyx_padded = util.pad_zyx_along_z(zyx_data, z_padding)

    # Normalize
    zyx = util.inten_normalization_3D(zyx_padded)

    # Prepare TF
    effective_transfer_function = (
        real_potential_transfer_function
        + absorption_ratio * imaginary_potential_transfer_function
    )

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        f_real = util.single_variable_tikhonov_deconvolution_3D(
            zyx, effective_transfer_function, reg_re=regularization_strength
        )

    elif reconstruction_algorithm == "TV":
        raise NotImplementedError
        f_real = util.single_variable_admm_tv_deconvolution_3D(
            zyx,
            effective_transfer_function,
            reg_re=regularization_strength,
            rho=TV_rho_strength,
            itr=TV_iterations,
        )

    # Unpad
    if z_padding != 0:
        f_real = f_real[z_padding:-z_padding]

    return f_real * z_pixel_size / 4 / np.pi * wavelength_illumination

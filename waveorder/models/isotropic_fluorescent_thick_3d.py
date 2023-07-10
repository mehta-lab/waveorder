import torch
from waveorder import optics, util


def generate_test_phantom(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    sphere_radius,
):
    sphere, _, _ = util.generate_sphere_target(
        zyx_shape, yx_pixel_size, z_pixel_size, sphere_radius
    )

    return sphere


def calculate_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_emission,
    z_padding,
    index_of_refraction_media,
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

    det_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_emission,
    )

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_emission / index_of_refraction_media,
        z_position_list,
    )

    point_spread_function = (
        torch.abs(torch.fft.ifft2(propagation_kernel, dim=(1, 2))) ** 2
    )
    optical_transfer_function = torch.fft.fftn(
        point_spread_function, dim=(0, 1, 2)
    )
    optical_transfer_function /= torch.max(
        torch.abs(optical_transfer_function)
    )  # normalize

    return optical_transfer_function


def visualize_transfer_function(viewer, optical_transfer_function, zyx_scale):
    arrays = [
        (torch.imag(optical_transfer_function), "Im(OTF)"),
        (torch.real(optical_transfer_function), "Re(OTF)"),
    ]

    for array in arrays:
        lim = 0.1 * torch.max(torch.abs(array[0]))
        viewer.add_image(
            torch.fft.ifftshift(array[0]).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / zyx_scale,
        )
    viewer.dims.order = (0, 1, 2)


def apply_transfer_function(zyx_object, optical_transfer_function, z_padding):
    if (
        zyx_object.shape[0] + 2 * z_padding
        != optical_transfer_function.shape[0]
    ):
        raise ValueError(
            "Please check padding: ZYX_obj.shape[0] + 2 * Z_pad != H_re.shape[0]"
        )
    if z_padding > 0:
        optical_transfer_function = optical_transfer_function[
            z_padding:-z_padding
        ]

    # Very simple simulation, consider adding noise and bkg knobs
    zyx_obj_hat = torch.fft.fftn(zyx_object)
    zyx_data = zyx_obj_hat * optical_transfer_function
    data = torch.real(torch.fft.ifftn(zyx_data))

    data += 10  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data,
    optical_transfer_function,
    z_padding,
    reconstruction_algorithm="Tikhonov",
    regularization_strength=1e-3,
    TV_rho_strength=1e-3,
    TV_iterations=10,
):
    # Handle padding
    zyx_padded = util.pad_zyx_along_z(zyx_data, z_padding)

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        f_real = util.single_variable_tikhonov_deconvolution_3D(
            zyx_padded,
            optical_transfer_function,
            reg_re=regularization_strength,
        )

    elif reconstruction_algorithm == "TV":
        raise NotImplementedError
        f_real = util.single_variable_admm_tv_deconvolution_3D(
            zyx_padded,
            optical_transfer_function,
            reg_re=regularization_strength,
            rho=TV_rho_strength,
            itr=TV_iterations,
        )

    # Unpad
    if z_padding != 0:
        f_real = f_real[z_padding:-z_padding]

    return f_real

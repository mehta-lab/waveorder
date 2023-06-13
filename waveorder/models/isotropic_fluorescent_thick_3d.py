import torch
from waveorder import optics, util


def generate_test_phantom(yx_shape):
    star, _, _ = util.generate_star_target(yx_shape, blur_px=0.1)
    return star


def calculate_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_illumination,
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
        wavelength_illumination,
    )

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
    )

    point_spread_function = (
        torch.abs(torch.fft.ifft2(propagation_kernel, dim=(0, 1))) ** 2
    )
    optical_transfer_function = torch.fft.fftn(
        point_spread_function, dim=(0, 1, 2)
    )
    optical_transfer_function /= torch.max(
        torch.abs(optical_transfer_function)
    )  # normalize

    return optical_transfer_function


def visualize_transfer_function():
    return NotImplementedError


def apply_transfer_function():
    return NotImplementedError


def apply_inverse_transfer_function():
    return NotImplementedError

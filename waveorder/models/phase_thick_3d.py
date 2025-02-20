from typing import Literal

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.filter import apply_filter_bank
from waveorder.models import isotropic_fluorescent_thick_3d
from waveorder.reconstruct import tikhonov_regularized_inverse_filter
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer


def generate_test_phantom(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    index_of_refraction_media: float,
    index_of_refraction_sample: float,
    sphere_radius: float,
) -> np.ndarray:
    sphere, _, _ = util.generate_sphere_target(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        radius=sphere_radius,
        blur_size=2 * yx_pixel_size,
    )
    zyx_phase = sphere * (
        index_of_refraction_sample - index_of_refraction_media
    )  # refractive index increment

    return zyx_phase


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    invert_phase_contrast: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        numerical_aperture_illumination,
        numerical_aperture_detection,
    )
    axial_nyquist = sampling.axial_nyquist(
        wavelength_illumination,
        numerical_aperture_detection,
        index_of_refraction_media,
    )

    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))
    z_factor = int(np.ceil(z_pixel_size / axial_nyquist))

    (
        real_potential_transfer_function,
        imag_potential_transfer_function,
    ) = _calculate_wrap_unsafe_transfer_function(
        (
            zyx_shape[0] * z_factor,
            zyx_shape[1] * yx_factor,
            zyx_shape[2] * yx_factor,
        ),
        yx_pixel_size / yx_factor,
        z_pixel_size / z_factor,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
    )

    zyx_out_shape = (zyx_shape[0] + 2 * z_padding,) + zyx_shape[1:]
    return (
        sampling.nd_fourier_central_cuboid(
            real_potential_transfer_function, zyx_out_shape
        ),
        sampling.nd_fourier_central_cuboid(
            imag_potential_transfer_function, zyx_out_shape
        ),
    )


def _calculate_wrap_unsafe_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    invert_phase_contrast: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    radial_frequencies = util.generate_radial_frequencies(
        zyx_shape[1:], yx_pixel_size
    )
    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    if invert_phase_contrast:
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
        axially_even=False,
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
    real_potential_transfer_function: np.ndarray,
    imag_potential_transfer_function: np.ndarray,
    zyx_scale: tuple[float, float, float],
) -> None:
    add_transfer_function_to_viewer(
        viewer,
        imag_potential_transfer_function,
        zyx_scale,
        layer_name="Imag pot. TF",
    )

    add_transfer_function_to_viewer(
        viewer,
        real_potential_transfer_function,
        zyx_scale,
        layer_name="Real pot. TF",
    )


def apply_transfer_function(
    zyx_object: np.ndarray,
    real_potential_transfer_function: np.ndarray,
    z_padding: int,
    brightness: float,
) -> np.ndarray:
    # This simplified forward model only handles phase, so it resuses the fluorescence forward model
    # TODO: extend to absorption
    return (
        isotropic_fluorescent_thick_3d.apply_transfer_function(
            zyx_object,
            real_potential_transfer_function,
            z_padding,
            background=0,
        )
        * brightness
        + brightness
    )


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    real_potential_transfer_function: Tensor,
    imaginary_potential_transfer_function: Tensor,
    z_padding: int,
    absorption_ratio: float = 0.0,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
) -> Tensor:
    """Reconstructs 3D phase from labelfree defocus zyx_data and a pair of
    complex 3D transfer functions real_potential_transfer_function and
    imag_potential_transfer_function, providing options for reconstruction
    algorithms.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, label-free defocus stack
    real_potential_transfer_function : Tensor
        Real potential transfer function, see calculate_transfer_function abov
    imaginary_potential_transfer_function : Tensor
        Imaginary potential transfer function, see calculate_transfer_function abov
    z_padding : int
        Padding for axial dimension. Use zero for defocus stacks that
        extend ~3 PSF widths beyond the sample. Pad by ~3 PSF widths otherwise.
    absorption_ratio : float, optional,
        Absorption-to-phase ratio in the sample.
        Use default 0 for purely phase objects.
    reconstruction_algorithm : str, optional
        "Tikhonov" or "TV", by default "Tikhonov"
        "TV" is not implemented.
    regularization_strength : float, optional
        regularization parameter, by default 1e-3
    TV_rho_strength : _type_, optional
        TV-specific regularization parameter, by default 1e-3
        "TV" is not implemented.
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
        "TV" is not implemented.

    Returns
    -------
    Tensor
        zyx_phase (radians)

    Raises
    ------
    NotImplementedError
        TV is not implemented
    """
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
        inverse_filter = tikhonov_regularized_inverse_filter(
            effective_transfer_function, regularization_strength
        )

        # [None]s and [0] are for applying a 1x1 "bank" of filters.
        # For further uniformity, consider returning (1, Z, Y, X)
        f_real = apply_filter_bank(inverse_filter[None, None], zyx[None])[0]

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

    return f_real

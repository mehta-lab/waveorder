from typing import Literal, Tuple

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.filter import apply_filter_bank


def generate_test_phantom(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    sphere_radius: float,
) -> Tensor:
    """Generate a test phantom for fluorescent thin object.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        Shape of YX dimensions
    yx_pixel_size : float
        Pixel size in YX plane
    wavelength_emission : float
        Emission wavelength
    sphere_radius : float
        Radius of spherical phantom

    Returns
    -------
    Tensor
        YX fluorescence density map
    """
    sphere, _, _ = util.generate_sphere_target(
        (3,) + yx_shape,
        yx_pixel_size,
        z_pixel_size=1.0,
        radius=sphere_radius,
        blur_size=2 * yx_pixel_size,
    )

    # Use middle slice as thin fluorescent object
    yx_fluorescence_density = sphere[1]

    return yx_fluorescence_density


def calculate_transfer_function(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    z_position_list: list,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
) -> Tensor:
    """Calculate transfer function for fluorescent thin object imaging.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        Shape of YX dimensions
    yx_pixel_size : float
        Pixel size in YX plane
    z_position_list : list
        List of Z positions for defocus stack
    wavelength_emission : float
        Emission wavelength
    index_of_refraction_media : float
        Refractive index of imaging medium
    numerical_aperture_detection : float
        Numerical aperture of detection objective
    confocal_pinhole_diameter : float | None, optional
        Diameter of confocal pinhole. Not implemented for 2D fluorescence.

    Returns
    -------
    Tensor
        Fluorescent 2D-to-3D transfer function

    Raises
    ------
    NotImplementedError
        If confocal_pinhole_diameter is not None
    """
    if confocal_pinhole_diameter is not None:
        raise NotImplementedError(
            "Confocal reconstruction is not implemented for 2D fluorescence"
        )

    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_emission,
        numerical_aperture_detection,  # ill = det for fluorescence
        numerical_aperture_detection,
    )
    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))

    fluorescent_2d_to_3d_transfer_function = (
        _calculate_wrap_unsafe_transfer_function(
            (
                yx_shape[0] * yx_factor,
                yx_shape[1] * yx_factor,
            ),
            yx_pixel_size / yx_factor,
            z_position_list,
            wavelength_emission,
            index_of_refraction_media,
            numerical_aperture_detection,
        )
    )

    fluorescent_2d_to_3d_transfer_function_out = torch.zeros(
        (len(z_position_list),) + tuple(yx_shape), dtype=torch.complex64
    )

    for z in range(len(z_position_list)):
        fluorescent_2d_to_3d_transfer_function_out[z] = (
            sampling.nd_fourier_central_cuboid(
                fluorescent_2d_to_3d_transfer_function[z], yx_shape
            )
        )

    return fluorescent_2d_to_3d_transfer_function_out


def calculate_singular_system(
    fluorescent_2d_to_3d_transfer_function: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the singular system of the fluorescent transfer function.

    The transfer function has shape (Z, Vy, Vx), where (Z,) is the data-space
    dimension, and (Vy, Vx) are the spatial frequency dimensions.

    The SVD is computed over the (Z,) dimension.

    Parameters
    ----------
    fluorescent_2d_to_3d_transfer_function : Tensor
        ZYX transfer function for fluorescence

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        U, S, Vh components of the SVD
    """
    # For fluorescence, we have only one object property (fluorescence density)
    # Input shape: (Z, Vy, Vx)

    # We need to create the format: (1, Z, Vy, Vx) where 1 represents single object type
    sfYX_transfer_function = fluorescent_2d_to_3d_transfer_function[None]

    # Permute to: (Vy, Vx, 1, Z) for SVD
    YXsf_transfer_function = sfYX_transfer_function.permute(2, 3, 0, 1)
    Up, Sp, Vhp = torch.linalg.svd(YXsf_transfer_function, full_matrices=False)
    # SVD gives us: Up: (Vy, Vx, 1, min(1,Z)), Sp: (Vy, Vx, min(1,Z)), Vhp: (Vy, Vx, min(1,Z), Z)

    # Permute back to match expected format:
    U = Up.permute(2, 3, 0, 1)  # (1, min(1,Z), Vy, Vx) -> (1, Z, Vy, Vx)
    S = Sp.permute(2, 0, 1)  # (min(1,Z), Vy, Vx) -> (1, Vy, Vx)
    Vh = Vhp.permute(2, 3, 0, 1)  # (min(1,Z), Z, Vy, Vx) -> (1, 1, Vy, Vx)
    return U, S, Vh


def _calculate_wrap_unsafe_transfer_function(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    z_position_list: list,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
) -> Tensor:
    """Calculate wrap-unsafe transfer function for fluorescent imaging."""
    radial_frequencies = util.generate_radial_frequencies(
        yx_shape, yx_pixel_size
    )

    det_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_emission,
    )

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_emission / index_of_refraction_media,
        torch.tensor(z_position_list),
    )

    zyx_shape = (len(z_position_list),) + tuple(yx_shape)
    fluorescent_2d_to_3d_transfer_function = torch.zeros(
        zyx_shape, dtype=torch.complex64
    )

    for z in range(len(z_position_list)):
        # For fluorescent imaging, the transfer function is the squared magnitude
        # of the coherent transfer function (incoherent imaging)
        point_spread_function = (
            torch.abs(torch.fft.ifft2(propagation_kernel[z], dim=(0, 1))) ** 2
        )
        fluorescent_2d_to_3d_transfer_function[z] = torch.fft.fft2(
            point_spread_function
        )

    # Normalize
    max_val = torch.max(torch.abs(fluorescent_2d_to_3d_transfer_function))
    if max_val > 0:
        fluorescent_2d_to_3d_transfer_function /= max_val

    return fluorescent_2d_to_3d_transfer_function


def visualize_transfer_function(
    viewer,
    fluorescent_2d_to_3d_transfer_function: Tensor,
    zyx_scale: tuple[float, float, float],
) -> None:
    """Visualize the fluorescent transfer function in napari."""
    arrays = [
        (
            torch.imag(fluorescent_2d_to_3d_transfer_function),
            "Im(fluorescent TF)",
        ),
        (
            torch.real(fluorescent_2d_to_3d_transfer_function),
            "Re(fluorescent TF)",
        ),
    ]

    for array in arrays:
        lim = (0.5 * torch.max(torch.abs(array[0]))).item()
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=zyx_scale,
        )
    viewer.dims.order = (2, 0, 1)


def apply_transfer_function(
    yx_fluorescence_density: Tensor,
    fluorescent_2d_to_3d_transfer_function: Tensor,
    background: int = 10,
) -> Tensor:
    """Simulate fluorescent imaging by applying the transfer function.

    Parameters
    ----------
    yx_fluorescence_density : Tensor
        2D fluorescence density map
    fluorescent_2d_to_3d_transfer_function : Tensor
        3D transfer function
    background : int, optional
        Background counts, by default 10

    Returns
    -------
    Tensor
        Simulated 3D fluorescent data stack
    """
    # Simulate fluorescent object imaging
    yx_fluorescence_hat = torch.fft.fftn(yx_fluorescence_density)
    zyx_fluorescence_data_hat = yx_fluorescence_hat[None] * torch.real(
        fluorescent_2d_to_3d_transfer_function
    )
    zyx_fluorescence_data = torch.real(
        torch.fft.ifftn(zyx_fluorescence_data_hat, dim=(1, 2))
    )

    # Add background
    data = zyx_fluorescence_data + background
    return data


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    singular_system: Tuple[Tensor, Tensor, Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
) -> Tensor:
    """Reconstruct fluorescence density from zyx_data and singular system.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, fluorescence defocus stack
    singular_system : Tuple[Tensor, Tensor, Tensor]
        Singular system of the fluorescent transfer function
    reconstruction_algorithm : Literal["Tikhonov", "TV"], optional
        Reconstruction algorithm, by default "Tikhonov"
        "TV" is not implemented
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
        "TV" is not implemented
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
        "TV" is not implemented

    Returns
    -------
    Tensor
        YX fluorescence density reconstruction

    Raises
    ------
    NotImplementedError
        TV is not implemented
    """
    if reconstruction_algorithm == "Tikhonov":
        print("Computing inverse filter")
        U, S, Vh = singular_system
        S_reg = S / (S**2 + regularization_strength)
        sfyx_inverse_filter = torch.einsum(
            "sj...,j...,jf...->fs...", U, S_reg, Vh
        )

        # Apply filter bank - returns tuple but we only have one object type
        yx_fluorescence_density = apply_filter_bank(
            sfyx_inverse_filter, zyx_data
        )[0]

    elif reconstruction_algorithm == "TV":
        raise NotImplementedError("TV reconstruction is not implemented")

    return yx_fluorescence_density

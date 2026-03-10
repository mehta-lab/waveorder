from typing import Literal

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.reconstruct import tikhonov_regularized_inverse_filter
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer


def generate_test_phantom(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    sphere_radius: float,
) -> Tensor:
    sphere, _, _ = util.generate_sphere_target(zyx_shape, yx_pixel_size, z_pixel_size, sphere_radius)

    return sphere


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
) -> Tensor:
    """Calculate the optical transfer function for fluorescence imaging.

    Supports both widefield and confocal microscopy modes. When
    confocal_pinhole_diameter is None, computes widefield OTF. When specified,
    computes confocal OTF by multiplying excitation and detection PSFs, where
    the detection PSF is downweighted by the pinhole aperture function.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Shape of the 3D volume
    yx_pixel_size : float
        Pixel size in YX plane
    z_pixel_size : float
        Pixel size in Z dimension
    wavelength_emission : float
        Emission wavelength
    z_padding : int
        Padding for axial dimension
    index_of_refraction_media : float
        Refractive index of imaging medium
    numerical_aperture_detection : float
        Numerical aperture of detection objective
    confocal_pinhole_diameter : float | None, optional
        Diameter of confocal pinhole in image space (demagnified). If None,
        computes widefield OTF. If specified, computes confocal OTF.

    Returns
    -------
    Tensor
        3D optical transfer function
    """
    na_det_val = float(torch.as_tensor(numerical_aperture_detection).detach())
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_emission,
        na_det_val,  # ill = det for fluorescence
        na_det_val,
    )
    axial_nyquist = sampling.axial_nyquist(
        wavelength_emission,
        na_det_val,
        index_of_refraction_media,
    )

    # For confocal, double the Nyquist range (half the sampling requirement)
    if confocal_pinhole_diameter is not None:
        transverse_nyquist = transverse_nyquist / 2
        axial_nyquist = axial_nyquist / 2

    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))
    z_factor = int(np.ceil(z_pixel_size / axial_nyquist))

    optical_transfer_function = _calculate_wrap_unsafe_transfer_function(
        (
            zyx_shape[0] * z_factor,
            zyx_shape[1] * yx_factor,
            zyx_shape[2] * yx_factor,
        ),
        yx_pixel_size / yx_factor,
        z_pixel_size / z_factor,
        wavelength_emission,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_detection,
        confocal_pinhole_diameter,
    )
    zyx_out_shape = (zyx_shape[0] + 2 * z_padding,) + zyx_shape[1:]
    return sampling.nd_fourier_central_cuboid(optical_transfer_function, zyx_out_shape)


def _calculate_pinhole_aperture_otf(
    radial_frequencies: Tensor,
    pinhole_diameter: float,
) -> Tensor:
    """Calculate the pinhole aperture OTF for confocal microscopy.

    The pinhole acts as a spatial filter in the image plane. A smaller pinhole
    (approaching a point) gives a broader OTF (approaching flat/ones).
    A larger pinhole gives a narrower OTF (approaching a delta function).

    Parameters
    ----------
    radial_frequencies : Tensor
        Radial spatial frequencies (units of 1/length)
    pinhole_diameter : float
        Diameter (not radius) of the confocal pinhole (units of length, matching
        radial_frequencies)

    Returns
    -------
    Tensor
        Pinhole aperture OTF (jinc^2 function)
    """
    argument = pinhole_diameter * radial_frequencies
    j1_values = torch.special.bessel_j1(np.pi * argument)
    jinc = torch.where(argument > 1e-10, j1_values / (2 * argument), 0.5)
    return jinc**2


def _calculate_wrap_unsafe_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
) -> Tensor:
    radial_frequencies = util.generate_radial_frequencies(zyx_shape[1:], yx_pixel_size)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift((torch.arange(z_total) - z_total // 2) * z_pixel_size)

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

    point_spread_function = torch.abs(torch.fft.ifft2(propagation_kernel, dim=(1, 2))) ** 2
    optical_transfer_function = torch.fft.fftn(point_spread_function, dim=(0, 1, 2))

    # Confocal: multiply excitation PSF with detection PSF (downweighted by pinhole)
    if confocal_pinhole_diameter is not None:
        pinhole_otf_2d = _calculate_pinhole_aperture_otf(radial_frequencies, confocal_pinhole_diameter)
        # Detection OTF is downweighted by pinhole
        otf_detection = optical_transfer_function * pinhole_otf_2d[None, :, :]

        # Convert to PSFs
        psf_excitation = torch.abs(torch.fft.ifftn(optical_transfer_function, dim=(0, 1, 2)))
        psf_detection = torch.abs(torch.fft.ifftn(otf_detection, dim=(0, 1, 2)))

        # Confocal PSF = excitation PSF * detection PSF (in real space)
        psf_confocal = psf_excitation * psf_detection

        # Convert back to OTF
        optical_transfer_function = torch.fft.fftn(psf_confocal, dim=(0, 1, 2))

    optical_transfer_function = optical_transfer_function / torch.clamp(
        torch.max(torch.abs(optical_transfer_function)), min=1e-12
    )

    return optical_transfer_function


def visualize_transfer_function(
    viewer,
    optical_transfer_function: Tensor,
    zyx_scale: tuple[float, float, float],
) -> None:
    add_transfer_function_to_viewer(
        viewer,
        torch.real(optical_transfer_function),
        zyx_scale,
        clim_factor=0.05,
    )


def apply_transfer_function(
    zyx_object: Tensor,
    optical_transfer_function: Tensor,
    z_padding: int,
    background: int = 10,
) -> Tensor:
    """Simulate imaging by applying a transfer function

    Parameters
    ----------
    zyx_object : torch.Tensor
    optical_transfer_function : torch.Tensor
    z_padding : int
    background : int, optional
        constant background counts added to each voxel, by default 10

    Returns
    -------
    Simulated data : torch.Tensor

    """
    if zyx_object.shape[0] + 2 * z_padding != optical_transfer_function.shape[0]:
        raise ValueError("Please check padding: ZYX_obj.shape[0] + 2 * Z_pad != H_re.shape[0]")
    if z_padding > 0:
        optical_transfer_function = optical_transfer_function[z_padding:-z_padding]

    # Very simple simulation, consider adding noise and bkg knobs
    zyx_obj_hat = torch.fft.fftn(zyx_object)
    zyx_data = zyx_obj_hat * optical_transfer_function
    data = torch.real(torch.fft.ifftn(zyx_data))

    data += background  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    optical_transfer_function: Tensor,
    z_padding: int,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
) -> Tensor:
    """Reconstructs fluorescence density from defocus data.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    optical_transfer_function : Tensor
        3D optical transfer function (shared, not batched)
    z_padding : int
        Padding for axial dimension. Use zero for defocus stacks that
        extend ~3 PSF widths beyond the sample. Pad by ~3 PSF widths otherwise.
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov". "TV" is not implemented.
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10

    Returns
    -------
    Tensor
        Fluorescence density, shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    """
    batched = zyx_data.ndim == 4
    if not batched:
        zyx_data = zyx_data.unsqueeze(0)

    # Handle padding: (B, Z, Y, X) -> (B, Z+2*pad, Y, X)
    zyx_padded = util.pad_zyx_along_z(zyx_data, z_padding)

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        inverse_filter = tikhonov_regularized_inverse_filter(optical_transfer_function, regularization_strength)

        # Batched FFT multiply: inverse_filter (Z,Y,X) broadcasts over B
        zyx_fft = torch.fft.fftn(zyx_padded, dim=(-3, -2, -1))
        f_real = torch.real(torch.fft.ifftn(zyx_fft * inverse_filter, dim=(-3, -2, -1)))

    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    # Unpad
    if z_padding != 0:
        f_real = f_real[:, z_padding:-z_padding]

    if not batched:
        f_real = f_real.squeeze(0)

    return f_real


def reconstruct(
    zyx_data: Tensor,
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
) -> Tensor:
    """Reconstruct 3D fluorescence density from a defocus stack.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    yx_pixel_size : float
        Pixel size in the transverse (Y, X) dimensions
    z_pixel_size : float
        Pixel size in the axial (Z) dimension
    wavelength_emission : float
        Emission wavelength
    z_padding : int
        Padding for axial dimension
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_detection : float
        Detection numerical aperture
    confocal_pinhole_diameter : float | None, optional
        Confocal pinhole diameter, by default None (widefield)
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov".
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10

    Returns
    -------
    Tensor
        Fluorescence density, shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    """
    # Use last 3 dims as zyx_shape for TF computation
    zyx_shape = zyx_data.shape[-3:]
    optical_transfer_function = calculate_transfer_function(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_emission,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_detection,
        confocal_pinhole_diameter=confocal_pinhole_diameter,
    )
    return apply_inverse_transfer_function(
        zyx_data,
        optical_transfer_function,
        z_padding,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        TV_rho_strength=TV_rho_strength,
        TV_iterations=TV_iterations,
    )

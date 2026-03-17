from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.filter import apply_filter_bank
from waveorder.models import isotropic_fluorescent_thick_3d
from waveorder.reconstruct import tikhonov_regularized_inverse_filter
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer

"""
Phase Thick 3D Model - Units and Conventions
=============================================

This module implements phase-from-defocus optical diffraction tomography (ODT)
for thick phase objects using the weak object transfer function (first Born
approximation).

Units Convention
----------------
This model uses "cycles" as the fundamental unit for phase:
    - 1 cycle = 2π radians = 1 wavelength of optical path difference

Phantom (input):
    Phase in cycles per voxel = (Δn × z_pixel_size) / λ_medium
    where:
        - Δn = n_sample - n_media (refractive index difference)
        - z_pixel_size = voxel thickness
        - λ_medium = λ_vacuum / n_media (wavelength in medium)

Reconstruction (output):
    Phase in cycles per voxel (same units as phantom)

Converting Between Units
------------------------
From cycles to radians:
    phase_radians = 2 * np.pi * phase_cycles

From cycles to refractive index difference:
    wavelength_medium = wavelength_vacuum / n_media
    delta_n = phase_cycles * wavelength_medium / z_pixel_size

From cycles to optical path length:
    optical_path_length = phase_cycles * wavelength_medium

Physics Background
------------------
The weak object approximation (first Born approximation) assumes:
1. Small refractive index variations: |Δn| << n_media
2. Weak scattering: no multiple scattering
3. Linear relationship between object and measured intensity

Reference
---------
J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative 3D
tomographic imaging for partially coherent light microscopy,"
Opt. Express 25, 15699-15712 (2017)
"""


def generate_test_phantom(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    index_of_refraction_media: float,
    index_of_refraction_sample: float,
    sphere_radius: float,
) -> np.ndarray:
    """
    Generate a spherical phantom with phase in cycles per voxel.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Shape of the 3D volume (Z, Y, X)
    yx_pixel_size : float
        Pixel size in transverse (Y, X) dimensions (length)
    z_pixel_size : float
        Pixel size in axial (Z) dimension (length)
    wavelength_illumination : float
        Wavelength of illumination light (length, same units as pixel sizes)
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    index_of_refraction_sample : float
        Refractive index of the sphere
    sphere_radius : float
        Radius of the sphere (length, same units as pixel sizes)

    Returns
    -------
    np.ndarray
        3D array of phase in cycles per voxel.
        Units: (n_sample - n_media) × z_pixel_size / λ_medium [cycles/voxel]

        Each voxel value represents the phase shift (in cycles) that light
        acquires when passing through that voxel. This matches the units
        returned by apply_inverse_transfer_function().
    """
    sphere, _, _ = util.generate_sphere_target(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        radius=sphere_radius,
        blur_size=2 * yx_pixel_size,
    )

    # Compute refractive index difference
    delta_n = sphere * (index_of_refraction_sample - index_of_refraction_media)

    # Convert to phase in cycles per voxel
    wavelength_medium = wavelength_illumination / index_of_refraction_media
    zyx_phase = delta_n * z_pixel_size / wavelength_medium

    return zyx_phase


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 1e4,
) -> tuple[Tensor, Tensor]:
    na_ill_val = float(torch.as_tensor(numerical_aperture_illumination).detach())
    na_det_val = float(torch.as_tensor(numerical_aperture_detection).detach())

    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        na_ill_val,
        na_det_val,
    )
    axial_nyquist = sampling.axial_nyquist(
        wavelength_illumination,
        na_det_val,
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
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )

    zyx_out_shape = (zyx_shape[0] + 2 * z_padding,) + zyx_shape[1:]
    return (
        sampling.nd_fourier_central_cuboid(real_potential_transfer_function, zyx_out_shape),
        sampling.nd_fourier_central_cuboid(imag_potential_transfer_function, zyx_out_shape),
    )


def _calculate_wrap_unsafe_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 1e4,
) -> tuple[Tensor, Tensor]:
    fyy, fxx = util.generate_frequencies(zyx_shape[1:], yx_pixel_size)
    radial_frequencies = torch.sqrt(fyy**2 + fxx**2)
    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift((torch.arange(z_total) - z_total // 2) * z_pixel_size)
    if invert_phase_contrast:
        z_position_list = torch.flip(z_position_list, dims=(0,))

    ill_pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        numerical_aperture_illumination,
        wavelength_illumination,
        index_of_refraction_media,
        tilt_angle_zenith,
        tilt_angle_azimuth,
    )
    det_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_illumination,
        steepness=pupil_steepness,
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
        zyx_phase : Phase in cycles per voxel
            Units: (Δn × z_pixel_size) / λ_medium [cycles/voxel]

            Each voxel represents the phase shift (in cycles) that light acquires
            when passing through that voxel. This matches the units of the input
            phantom from generate_test_phantom().

            To convert to phase in radians:
                phase_radians = 2 * np.pi * zyx_phase

            To convert to refractive index difference:
                wavelength_medium = wavelength_illumination / index_of_refraction_media
                delta_n = zyx_phase * wavelength_medium / z_pixel_size

            Note: One cycle corresponds to 2π radians of phase shift, or one
            wavelength of optical path length difference.

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
        real_potential_transfer_function + absorption_ratio * imaginary_potential_transfer_function
    )

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        inverse_filter = tikhonov_regularized_inverse_filter(effective_transfer_function, regularization_strength)

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


def reconstruct(
    zyx_data: Tensor,
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor] = 0.9,
    numerical_aperture_detection: Union[float, Tensor] = 1.2,
    invert_phase_contrast: bool = False,
    absorption_ratio: float = 0.0,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 1e4,
) -> Tensor:
    """Reconstruct 3D phase from a brightfield defocus stack.

    Chains calculate_transfer_function and apply_inverse_transfer_function.

    Parameters
    ----------
    zyx_data : Tensor
        3D raw data, label-free defocus stack
    yx_pixel_size : float
        Pixel size in the transverse (Y, X) dimensions
    z_pixel_size : float
        Pixel size in the axial (Z) dimension
    wavelength_illumination : float
        Wavelength of illumination light
    z_padding : int
        Padding for axial dimension
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture
    invert_phase_contrast : bool, optional
        Invert phase contrast, by default False
    absorption_ratio : float, optional
        Absorption-to-phase ratio, by default 0.0
    reconstruction_algorithm : str, optional
        "Tikhonov" or "TV", by default "Tikhonov"
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    tilt_angle_zenith : float or Tensor, optional
        Illumination tilt zenith angle in radians, by default 0.0
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 1e4

    Returns
    -------
    Tensor
        zyx_phase in cycles per voxel
    """
    real_tf, imag_tf = calculate_transfer_function(
        zyx_data.shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )
    return apply_inverse_transfer_function(
        zyx_data,
        real_tf,
        imag_tf,
        z_padding,
        absorption_ratio=absorption_ratio,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        TV_rho_strength=TV_rho_strength,
        TV_iterations=TV_iterations,
    )


def calculate_skewed_transfer_function(
    raw_data_shape: tuple[int, int, int],
    yx_pixel_size: float,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 1e4,
) -> tuple[Tensor, Tensor]:
    """Compute phase transfer function in skewed (raw oblique) coordinates.

    Uses the Fourier affine theorem to resample the standard deskewed
    transfer function onto the skewed data's Fourier grid. The deskew
    transform is an affine mapping A (skewed → deskewed), so in Fourier
    space: H_skewed(k_s) = H_deskewed(A^{-T} k_s) / |det A|.

    The frequency mapping (derived from the biahub deskew affine) is:

        kZ_d = -(cos(θ) · kz_scan + ky_tilt) / sin(θ)
        kY_d = -kx_cover
        kX_d = kz_scan

    where all frequencies are in physical units (cycles/μm).

    Parameters
    ----------
    raw_data_shape : tuple[int, int, int]
        Shape (Z_scan, Y_tilt, X_cover) of the raw oblique data.
    yx_pixel_size : float
        Camera pixel size in micrometers.
    ls_angle_deg : float
        Light sheet angle in degrees (angle from optical axis).
    px_to_scan_ratio : float
        pixel_size / scan_step.
    wavelength_illumination : float
        Illumination wavelength in micrometers.
    z_padding : int
        Number of z-slices to pad for boundary effects.
    index_of_refraction_media : float
        Refractive index of the imaging medium.
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture.
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture.
    invert_phase_contrast : bool
        Invert phase contrast sign.
    tilt_angle_zenith : float or Tensor
        Illumination tilt zenith angle in radians.
    tilt_angle_azimuth : float or Tensor
        Illumination tilt azimuth angle in radians.
    pupil_steepness : float
        Sigmoid steepness for smooth pupil cutoff.

    Returns
    -------
    real_tf_skewed : Tensor
        Real potential transfer function in skewed coordinates.
        Shape: (Z_scan + 2*z_padding, Y_tilt, X_cover).
    imag_tf_skewed : Tensor
        Imaginary potential transfer function in skewed coordinates.
        Shape: (Z_scan + 2*z_padding, Y_tilt, X_cover).
    """
    from scipy.ndimage import map_coordinates

    theta = np.radians(ls_angle_deg)
    st = np.sin(theta)
    ct = np.cos(theta)
    scan_step = yx_pixel_size / px_to_scan_ratio

    Z_scan, Y_tilt, X_cover = raw_data_shape

    # Deskewed geometry (no averaging)
    dz_d = st * yx_pixel_size
    Nz_d = Y_tilt
    Ny_d = X_cover
    Nx_d = int(np.ceil(Z_scan / px_to_scan_ratio))

    # Compute standard TF in deskewed space (includes z_padding)
    real_tf_d, imag_tf_d = calculate_transfer_function(
        (Nz_d, Ny_d, Nx_d),
        yx_pixel_size,
        dz_d,
        wavelength_illumination,
        z_padding=z_padding,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_illumination=numerical_aperture_illumination,
        numerical_aperture_detection=numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )

    # --- Fourier resampling ---
    # The deskewed TF is in FFT order. We fftshift to make frequencies
    # monotonic for interpolation, resample, then ifftshift back.

    Nz_d_padded = Nz_d + 2 * z_padding
    Z_s_padded = Z_scan + 2 * z_padding

    # Deskewed frequency axes (physical, fftshifted = monotonic)
    kz_d = np.fft.fftshift(np.fft.fftfreq(Nz_d_padded, dz_d))
    ky_d = np.fft.fftshift(np.fft.fftfreq(Ny_d, yx_pixel_size))
    kx_d = np.fft.fftshift(np.fft.fftfreq(Nx_d, yx_pixel_size))

    # Skewed frequency axes (physical, fftshifted)
    kz_s = np.fft.fftshift(np.fft.fftfreq(Z_s_padded, scan_step))
    ky_s = np.fft.fftshift(np.fft.fftfreq(Y_tilt, yx_pixel_size))
    kx_s = np.fft.fftshift(np.fft.fftfreq(X_cover, yx_pixel_size))

    # Build 3D meshgrid of target (skewed) frequencies
    KZ_s, KY_s, KX_s = np.meshgrid(kz_s, ky_s, kx_s, indexing="ij")

    # Map to deskewed frequencies using the affine Fourier theorem:
    #   H_skewed(k_s) = H_deskewed(A^{-T} k_s) / |det A|
    # where A is the deskew matrix (skewed pixel → deskewed pixel).
    KZ_d_target = -(ct * KZ_s + KY_s) / st
    KY_d_target = -KX_s
    KX_d_target = KZ_s

    # Convert physical frequencies to fractional indices in the
    # fftshifted deskewed grid (for map_coordinates)
    def _freq_to_index(k_target, k_axis):
        """Map physical frequency to continuous index in a sorted array."""
        dk = k_axis[1] - k_axis[0] if len(k_axis) > 1 else 1.0
        return (k_target - k_axis[0]) / dk

    iz = _freq_to_index(KZ_d_target, kz_d)
    iy = _freq_to_index(KY_d_target, ky_d)
    ix = _freq_to_index(KX_d_target, kx_d)

    coords = np.array([iz.ravel(), iy.ravel(), ix.ravel()])

    # Determinant of the deskew matrix (skewed pixel → deskewed pixel)
    # A = [[0,-1,0],[0,0,-1],[1/R,-ct,0]], det(A) = 1/R
    det_A = 1.0 / px_to_scan_ratio

    def _resample(tf_tensor):
        tf_shifted = np.fft.fftshift(tf_tensor.numpy())
        # Interpolate real and imaginary parts
        out_real = map_coordinates(tf_shifted.real, coords, order=1, mode="constant", cval=0.0).reshape(KZ_s.shape)
        out_imag = map_coordinates(tf_shifted.imag, coords, order=1, mode="constant", cval=0.0).reshape(KZ_s.shape)
        # Apply normalization and ifftshift back to FFT order
        result = (out_real + 1j * out_imag) / abs(det_A)
        return torch.from_numpy(np.fft.ifftshift(result)).to(tf_tensor.dtype)

    real_tf_skewed = _resample(real_tf_d)
    imag_tf_skewed = _resample(imag_tf_d)

    return real_tf_skewed, imag_tf_skewed

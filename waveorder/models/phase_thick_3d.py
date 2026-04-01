from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder.models import isotropic_fluorescent_thick_3d
from waveorder.phantoms import single_bead
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
    delta_n = index_of_refraction_sample - index_of_refraction_media
    phantom = single_bead(
        shape=zyx_shape,
        pixel_sizes=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        bead_radius_um=sphere_radius,
        refractive_index_diff=delta_n,
        blur_size_um=2 * yx_pixel_size,
    )

    # Convert dn to phase in cycles per voxel
    wavelength_medium = wavelength_illumination / index_of_refraction_media
    zyx_phase = phantom.phase * z_pixel_size / wavelength_medium

    return zyx_phase


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor] = 0.9,
    numerical_aperture_detection: Union[float, Tensor] = 1.2,
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 1e4,
) -> tuple[Tensor, Tensor]:
    """Compute the 3D phase transfer function.

    When any parameter is a 1-D tensor of length B, all varying
    parameters are broadcast to length B and the output is
    ``(B, Z, Y, X)``.  Scalar parameters are shared across the batch.
    Shared optical components (detection pupil, propagation kernel,
    Green's function) are computed once regardless of batch size.

    Returns ``(Z, Y, X)`` when all parameters are scalar.
    """
    # Detect batch size (1 = scalar)
    batchable = {
        "numerical_aperture_illumination": numerical_aperture_illumination,
        "numerical_aperture_detection": numerical_aperture_detection,
        "tilt_angle_zenith": tilt_angle_zenith,
        "tilt_angle_azimuth": tilt_angle_azimuth,
    }
    batch_size = 1
    for name, val in batchable.items():
        t = torch.as_tensor(val)
        if t.ndim >= 1 and t.shape[0] > 1:
            if batch_size == 1:
                batch_size = t.shape[0]
            elif t.shape[0] != batch_size:
                raise ValueError(
                    f"Batched parameters must have the same length, got {batch_size} and {t.shape[0]} for '{name}'"
                )
    unbatched = batch_size == 1

    # Broadcast all batchable params to (B,)
    def _to_batch(val):
        t = torch.as_tensor(val, dtype=torch.float32)
        return t.expand(batch_size) if t.ndim == 0 else t

    na_ill = _to_batch(numerical_aperture_illumination)
    na_det = _to_batch(numerical_aperture_detection)
    zen = _to_batch(tilt_angle_zenith)
    azi = _to_batch(tilt_angle_azimuth)

    # Nyquist upsampling
    na_ill_0 = float(na_ill[0])
    na_det_0 = float(na_det[0])
    yx_factor = int(np.ceil(yx_pixel_size / sampling.transverse_nyquist(wavelength_illumination, na_ill_0, na_det_0)))
    z_factor = int(
        np.ceil(z_pixel_size / sampling.axial_nyquist(wavelength_illumination, na_det_0, index_of_refraction_media))
    )

    up_shape = (
        zyx_shape[0] * z_factor,
        zyx_shape[1] * yx_factor,
        zyx_shape[2] * yx_factor,
    )
    up_yx = yx_pixel_size / yx_factor
    up_z = z_pixel_size / z_factor
    zyx_out_shape = (zyx_shape[0] + 2 * z_padding,) + zyx_shape[1:]

    # Shared optics (computed once, moved to input device)
    # Pass original tensors (not floats) to preserve gradient graph
    fyy, fxx, det_pupil, propagation_kernel, greens_function_z = _compute_shared_optics(
        up_shape,
        up_yx,
        up_z,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        na_det[0],
        invert_phase_contrast,
        pupil_steepness,
    )

    device = zen.device
    fyy = fyy.to(device)
    fxx = fxx.to(device)
    det_pupil = det_pupil.to(device)
    propagation_kernel = propagation_kernel.to(device)
    greens_function_z = greens_function_z.to(device)

    # Batched illumination pupils: (B, Y, X) or (1, Y, X)
    ill_pupils = optics.generate_tilted_pupil(
        fxx, fyy, na_ill[0], wavelength_illumination, index_of_refraction_media, zen[:, None, None], azi[:, None, None]
    )

    # Batched WOTF: (B, Z, Y, X) or (1, Z, Y, X)
    real_tf, imag_tf = optics.compute_weak_object_transfer_function_3D(
        ill_pupils, ill_pupils, det_pupil, propagation_kernel, greens_function_z, up_z
    )

    # Downsample each element
    real_tfs = torch.stack([sampling.nd_fourier_central_cuboid(real_tf[i], zyx_out_shape) for i in range(batch_size)])
    imag_tfs = torch.stack([sampling.nd_fourier_central_cuboid(imag_tf[i], zyx_out_shape) for i in range(batch_size)])

    if unbatched:
        return real_tfs.squeeze(0), imag_tfs.squeeze(0)
    return real_tfs, imag_tfs


def _compute_shared_optics(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_illumination,
    z_padding,
    index_of_refraction_media,
    numerical_aperture_detection,
    invert_phase_contrast=False,
    pupil_steepness=1e4,
):
    """Compute optical components independent of illumination tilt."""
    fyy, fxx = util.generate_frequencies(zyx_shape[1:], yx_pixel_size)
    radial_frequencies = torch.sqrt(fyy**2 + fxx**2)
    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift((torch.arange(z_total) - z_total // 2) * z_pixel_size)
    if invert_phase_contrast:
        z_position_list = torch.flip(z_position_list, dims=(0,))

    det_pupil = optics.generate_pupil(
        radial_frequencies, numerical_aperture_detection, wavelength_illumination, steepness=pupil_steepness
    )
    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies, det_pupil, wavelength_illumination / index_of_refraction_media, z_position_list
    )
    greens_function_z = optics.generate_greens_function_z(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
        axially_even=False,
    )

    return fyy, fxx, det_pupil, propagation_kernel, greens_function_z


def _calculate_wrap_unsafe_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_illumination,
    z_padding,
    index_of_refraction_media,
    numerical_aperture_illumination,
    numerical_aperture_detection,
    invert_phase_contrast=False,
    tilt_angle_zenith=0.0,
    tilt_angle_azimuth=0.0,
    pupil_steepness=1e4,
):
    fyy, fxx, det_pupil, propagation_kernel, greens_function_z = _compute_shared_optics(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_detection,
        invert_phase_contrast,
        pupil_steepness,
    )

    ill_pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        numerical_aperture_illumination,
        wavelength_illumination,
        index_of_refraction_media,
        tilt_angle_zenith,
        tilt_angle_azimuth,
    )

    return optics.compute_weak_object_transfer_function_3D(
        ill_pupil, ill_pupil, det_pupil, propagation_kernel, greens_function_z, z_pixel_size
    )


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
    """Reconstructs 3D phase from labelfree defocus data.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    real_potential_transfer_function : Tensor
        Real potential transfer function, ``(Z, Y, X)`` shared across
        batch or ``(B, Z, Y, X)`` applied 1-to-1 with batched data.
    imaginary_potential_transfer_function : Tensor
        Imaginary potential transfer function, same shape options.
    z_padding : int
        Padding for axial dimension. Use zero for defocus stacks that
        extend ~3 PSF widths beyond the sample. Pad by ~3 PSF widths otherwise.
    absorption_ratio : float, optional
        Absorption-to-phase ratio, by default 0.0
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
        zyx_phase : Phase in cycles per voxel, shape ``(Z, Y, X)`` or
        ``(B, Z, Y, X)``
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
    """
    batched = zyx_data.ndim == 4
    if not batched:
        zyx_data = zyx_data.unsqueeze(0)

    # Handle padding: (B, Z, Y, X) -> (B, Z+2*pad, Y, X)
    zyx_padded = util.pad_zyx_along_z(zyx_data, z_padding)

    # Normalize
    zyx = util.inten_normalization_3D(zyx_padded)

    # Prepare TF (shared)
    effective_transfer_function = (
        real_potential_transfer_function + absorption_ratio * imaginary_potential_transfer_function
    )

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        inverse_filter = tikhonov_regularized_inverse_filter(effective_transfer_function, regularization_strength)

        # Batched FFT multiply: inverse_filter (Z,Y,X) broadcasts over B
        zyx_fft = torch.fft.fftn(zyx, dim=(-3, -2, -1))
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

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
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
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov".
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
        Phase in cycles per voxel, shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    """
    # Use last 3 dims as zyx_shape for TF computation
    zyx_shape = zyx_data.shape[-3:]
    real_tf, imag_tf = calculate_transfer_function(
        zyx_shape,
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

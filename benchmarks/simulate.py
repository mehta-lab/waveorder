"""Forward model simulation: phantom + config → simulated measurement."""

from __future__ import annotations

from torch import Tensor

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d
from waveorder.phantoms import Phantom


def _pixel_sizes(phantom: Phantom) -> tuple[float, float]:
    """Extract (z_pixel_size, yx_pixel_size) from a phantom."""
    return phantom.pixel_sizes[0], phantom.pixel_sizes[1]


def simulate_phase_3d(
    phantom: Phantom,
    wavelength_illumination: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_illumination: float = 0.9,
    numerical_aperture_detection: float = 1.2,
    brightness: float = 1e3,
) -> Tensor:
    """Simulate 3D phase measurement from a phantom.

    Applies the weak object transfer function to the phantom's phase
    channel to produce a simulated brightfield measurement.

    Parameters
    ----------
    phantom : Phantom
        Ground truth phantom. Uses ``phantom.phase`` (units: dn).
    wavelength_illumination : float
        Illumination wavelength in um.
    index_of_refraction_media : float
        Refractive index of the surrounding medium.
    numerical_aperture_illumination : float
        Condenser NA.
    numerical_aperture_detection : float
        Objective NA.
    brightness : float
        Brightness scaling for the forward model.

    Returns
    -------
    Tensor
        Simulated 3D brightfield data, shape (Z, Y, X).
    """
    z_pixel_size, yx_pixel_size = _pixel_sizes(phantom)
    zyx_shape = tuple(phantom.phase.shape)

    # Convert dn to phase in cycles per voxel
    wavelength_medium = wavelength_illumination / index_of_refraction_media
    zyx_phase = phantom.phase * z_pixel_size / wavelength_medium

    real_tf, _ = phase_thick_3d.calculate_transfer_function(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_illumination=wavelength_illumination,
        z_padding=0,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_illumination=numerical_aperture_illumination,
        numerical_aperture_detection=numerical_aperture_detection,
    )

    return phase_thick_3d.apply_transfer_function(zyx_phase, real_tf, z_padding=0, brightness=brightness)


def simulate_fluorescence_3d(
    phantom: Phantom,
    wavelength_emission: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_detection: float = 1.2,
    background: int = 10,
) -> Tensor:
    """Simulate 3D fluorescence measurement from a phantom.

    Applies the optical transfer function to the phantom's fluorescence
    channel to produce a simulated widefield measurement.

    Parameters
    ----------
    phantom : Phantom
        Ground truth phantom. Uses ``phantom.fluorescence``.
    wavelength_emission : float
        Emission wavelength in um.
    index_of_refraction_media : float
        Refractive index of the surrounding medium.
    numerical_aperture_detection : float
        Objective NA.
    background : int
        Background offset added to the simulated data.

    Returns
    -------
    Tensor
        Simulated 3D fluorescence data, shape (Z, Y, X).
    """
    z_pixel_size, yx_pixel_size = _pixel_sizes(phantom)
    zyx_shape = tuple(phantom.fluorescence.shape)

    otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_emission=wavelength_emission,
        z_padding=0,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_detection=numerical_aperture_detection,
    )

    return isotropic_fluorescent_thick_3d.apply_transfer_function(
        phantom.fluorescence, otf, z_padding=0, background=background
    )

"""Forward model simulation: phantom + config → simulated measurement."""

from __future__ import annotations

from torch import Tensor

from waveorder.models import (
    isotropic_fluorescent_thick_3d,
    phase_thick_3d,
    shift_variant_fluorescent_3d,
)
from waveorder.phantoms import Phantom
from waveorder.zernike import SpatialPolynomialPupil


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
    z_pixel_size, yx_pixel_size, _ = phantom.pixel_sizes
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
    z_pixel_size, yx_pixel_size, _ = phantom.pixel_sizes
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


def simulate_shift_variant_fluorescence_3d(
    phantom: Phantom,
    spatial_pupil_coefficients: dict[tuple[int, int, int], float] | dict[str, float],
    n_tiles_yx: tuple[int, int] = (8, 8),
    wavelength_emission: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_detection: float = 1.2,
    background: float = 0.0,
) -> Tensor:
    """Simulate a smoothly shift-variant 3D fluorescence measurement.

    The complex pupil at field position :math:`r_o = (x_o, y_o)` is

    .. math::

        P(\\nu; r_o) = P_0(\\nu)\\,\\exp\\!\\left(i\\,2\\pi
        \\sum_{j, m, n} c_{j,m,n}\\, Z_j(\\nu)\\, x_o^m\\, y_o^n\\right),

    expanded with ``spatial_pupil_coefficients``. The simulator partitions
    the FOV into ``n_tiles_yx`` tiles, evaluates the local PSF at each
    tile center, and convolves the partition-of-unity-windowed phantom
    with that PSF.

    Parameters
    ----------
    phantom : Phantom
        Ground-truth phantom (uses ``phantom.fluorescence``).
    spatial_pupil_coefficients : dict
        ``{(j, m, n): c}`` or ``{"j_m_n": c}`` map of coefficients in
        waves (RMS). String keys are accepted to round-trip cleanly
        through YAML.
    n_tiles_yx : tuple[int, int]
        Number of partition tiles along ``(Y, X)``.
    wavelength_emission : float
        Emission wavelength in microns.
    index_of_refraction_media : float
        Refractive index of the imaging medium.
    numerical_aperture_detection : float
        Detection NA.
    background : float
        Constant background offset added to the simulated measurement.

    Returns
    -------
    Tensor
        Simulated 3D measurement, shape ``(Z, Y, X)``.
    """
    z_pixel_size, yx_pixel_size, _ = phantom.pixel_sizes
    Z, Y, X = phantom.fluorescence.shape
    field_extent_um = (Y * yx_pixel_size / 2, X * yx_pixel_size / 2)

    coefs = _normalize_pupil_coefficients(spatial_pupil_coefficients)
    spatial_pupil = SpatialPolynomialPupil(coefs, field_extent_um=field_extent_um)

    return shift_variant_fluorescent_3d.apply_shift_variant_forward(
        phantom.fluorescence,
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=z_pixel_size,
        wavelength_emission=wavelength_emission,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_detection=numerical_aperture_detection,
        spatial_pupil=spatial_pupil,
        n_tiles_yx=tuple(n_tiles_yx),
        background=background,
    )


def _normalize_pupil_coefficients(
    raw: dict,
) -> dict[tuple[int, int, int], float]:
    """Normalize ``{"j_m_n": c}`` or ``{(j, m, n): c}`` into tuple keys.

    YAML serialises tuple keys as strings, so configs use the
    underscore-separated form ``"4_2_0": 0.2``.
    """
    out: dict[tuple[int, int, int], float] = {}
    for key, value in raw.items():
        if isinstance(key, tuple):
            j, m, n = key
        else:
            parts = str(key).split("_")
            if len(parts) != 3:
                raise ValueError(f"Pupil coefficient key {key!r} must be 'j_m_n'")
            j, m, n = (int(p) for p in parts)
        out[(int(j), int(m), int(n))] = float(value)
    return out

"""Differentiable fluorescence PSF/OTF synthesis from imaging-system parameters.

Builds the widefield fluorescence point spread function from a pupil with
named Zernike aberrations::

    P(nu) = P0(nu) * exp(i 2 pi sum_j c_j Z_j(nu)),
    h(z, y, x) = | IFFT2[ P(nu) exp(i 2 pi z n_z(nu)) ] |^2,

with the Zernike coefficients ``c_j`` in **waves** (orthonormal Noll
normalization; see :mod:`waveorder.calibrate.zernike_names`). If your
coefficients are in radians of pupil phase (the cross-track exchange
currency of the 2026-10-30 sprint), divide by ``2 * pi`` before calling
in here. This mirrors
:func:`waveorder.models.shift_variant_fluorescent_3d.calculate_psf_at_field_position`
but takes the coefficients directly and preserves torch autograd through the
coefficients, the detection NA (via the sigmoid pupil edge) and the focus
offset, so the PSF can sit inside a gradient-based calibration loop.

Notes
-----
The transfer function of ``wo compute-tf``
(:func:`waveorder.models.isotropic_fluorescent_thick_3d.calculate_transfer_function`)
supersamples to Nyquist before cropping back; this module evaluates the
pupil on the native grid (like the shift-variant model), so the two agree
to a small discretization difference when the data is undersampled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from waveorder import optics
from waveorder.calibrate.zernike_names import name_to_noll
from waveorder.zernike import make_pupil_grid, zernike_noll


def _as_noll_coefficients(zernike_coefficients: Mapping[str | int, float | Tensor] | None) -> dict[int, float | Tensor]:
    """Normalize a name- or Noll-keyed coefficient mapping to Noll keys."""
    if zernike_coefficients is None:
        return {}
    out: dict[int, float | Tensor] = {}
    for key, value in zernike_coefficients.items():
        j = key if isinstance(key, int) else name_to_noll(key)
        if j in out:
            raise ValueError(f"Zernike term {key!r} (Noll {j}) given more than once")
        out[j] = value
    return out


def aberrated_pupil(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    wavelength_emission: float,
    numerical_aperture_detection: float | Tensor,
    zernike_coefficients: Mapping[str | int, float | Tensor] | None = None,
) -> Tensor:
    """Complex detection pupil with named Zernike aberrations.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        ``(Y, X)`` grid shape.
    yx_pixel_size : float
        Transverse pixel size in um.
    wavelength_emission : float
        Emission wavelength in um (free space).
    numerical_aperture_detection : float or Tensor
        Detection NA. A tensor keeps autograd through the (sigmoid) pupil
        edge.
    zernike_coefficients : mapping, optional
        ``{name_or_noll_index: coefficient}`` with coefficients in waves;
        float or 0-dim tensor values (tensors keep autograd).

    Returns
    -------
    Tensor
        Complex pupil, shape ``(Y, X)``, FFT frequency order (DC at
        ``[0, 0]``).
    """
    _, _, frr, theta = make_pupil_grid(yx_shape, yx_pixel_size, fft_order=True)
    pupil_amplitude = optics.generate_pupil(frr, numerical_aperture_detection, wavelength_emission)

    coefficients = _as_noll_coefficients(zernike_coefficients)
    if not coefficients:
        return pupil_amplitude.to(torch.complex64)

    na = (
        numerical_aperture_detection
        if isinstance(numerical_aperture_detection, Tensor)
        else torch.tensor(float(numerical_aperture_detection))
    )
    rho = (frr * wavelength_emission / na).clamp(max=1.0)
    phase_waves = torch.zeros_like(frr)
    for j, c in coefficients.items():
        phase_waves = phase_waves + c * zernike_noll(j, rho, theta)
    return pupil_amplitude * torch.exp(2j * math.pi * phase_waves)


def fluorescence_psf(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float | Tensor,
    zernike_coefficients: Mapping[str | int, float | Tensor] | None = None,
    z_focus_offset: float | Tensor = 0.0,
) -> Tensor:
    """Widefield fluorescence intensity PSF from imaging-system parameters.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        ``(Z, Y, X)`` output shape.
    yx_pixel_size : float
        Transverse pixel size in um.
    z_pixel_size : float
        Axial pixel size in um.
    wavelength_emission : float
        Emission wavelength in um (free space).
    index_of_refraction_media : float
        Refractive index of the imaging medium.
    numerical_aperture_detection : float or Tensor
        Detection NA.
    zernike_coefficients : mapping, optional
        ``{name_or_noll_index: coefficient}``, coefficients in waves
        (see :mod:`waveorder.calibrate.zernike_names`).
    z_focus_offset : float or Tensor
        Focus offset in slices; shifts the defocus axis.

    Returns
    -------
    Tensor
        Real PSF, shape ``(Z, Y, X)``, sum-normalized to 1, in FFT order on
        every axis (matching the shift-variant model): the in-focus slice
        is at index 0 along z and the peak sits at ``[0, 0, 0]``, so
        ``fftn(psf)`` is a phase-ramp-free convolution kernel. Apply
        ``torch.fft.fftshift`` for a center-origin view.
    """
    z_total, _, _ = zyx_shape
    yx_shape = zyx_shape[1:]

    pupil = aberrated_pupil(
        yx_shape,
        yx_pixel_size,
        wavelength_emission,
        numerical_aperture_detection,
        zernike_coefficients,
    )
    _, _, frr, _ = make_pupil_grid(yx_shape, yx_pixel_size, fft_order=True)
    pupil_support = optics.generate_pupil(frr, numerical_aperture_detection, wavelength_emission)

    z_steps = torch.arange(z_total) - z_total // 2
    if isinstance(z_focus_offset, Tensor) or z_focus_offset != 0.0:
        z_steps = z_steps - z_focus_offset
    z_position_list = torch.fft.ifftshift(z_steps * z_pixel_size)

    propagation_kernel = optics.generate_propagation_kernel(
        frr,
        pupil_support,
        wavelength_emission / index_of_refraction_media,
        z_position_list,
    )
    coherent_psf = torch.fft.ifft2(propagation_kernel * pupil[None, :, :], dim=(1, 2))
    psf = torch.abs(coherent_psf) ** 2

    total = psf.sum()
    return psf / total.clamp(min=1e-30)


def fluorescence_otf(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float | Tensor,
    zernike_coefficients: Mapping[str | int, float | Tensor] | None = None,
    z_focus_offset: float | Tensor = 0.0,
    normalize: str = "peak",
) -> Tensor:
    """Widefield fluorescence OTF from imaging-system parameters.

    Same parameters as :func:`fluorescence_psf`, plus:

    Parameters
    ----------
    normalize : {"peak", "none"}
        ``"peak"`` scales to ``max |OTF| = 1`` (the WaveOrder transfer
        function convention, matching
        ``isotropic_fluorescent_thick_3d.calculate_transfer_function``);
        ``"none"`` returns the FFT of the sum-normalized PSF, whose DC
        value is exactly 1 up to the pupil discretization.

    Returns
    -------
    Tensor
        Complex OTF, shape ``(Z, Y, X)``, FFT frequency order on every
        axis — directly usable in ``g_fft = f_fft * otf`` forward models
        and in :func:`waveorder.calibrate.scores.whittle_evidence`.
    """
    psf = fluorescence_psf(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_emission,
        index_of_refraction_media,
        numerical_aperture_detection,
        zernike_coefficients,
        z_focus_offset,
    )
    otf = torch.fft.fftn(psf, dim=(0, 1, 2))
    if normalize == "peak":
        otf = otf / torch.clamp(torch.max(torch.abs(otf)), min=1e-12)
    elif normalize != "none":
        raise ValueError(f"normalize must be 'peak' or 'none', got {normalize!r}")
    return otf

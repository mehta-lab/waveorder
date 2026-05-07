"""Shift-variant fluorescence forward model with a spatial-polynomial pupil.

Builds on :mod:`waveorder.models.isotropic_fluorescent_thick_3d` but lets
the complex pupil :math:`P(\\nu; r_o)` vary smoothly with field position
:math:`r_o`, expanded as

.. math::

    P(\\nu; r_o) = P_0(\\nu)\\,\\exp\\!\\left(i\\,2\\pi
    \\sum_{j, m, n} c_{j, m, n}\\, Z_j(\\nu)\\, x_o^m\\, y_o^n\\right).

For a point source at ``r_o``, the 3D PRF is

.. math::

    h(r_d; r_o) = \\big| \\mathcal{F}^{-1}_\\nu
    [P(\\nu; r_o) e^{i 2\\pi z\\, n_z(\\nu)}] \\big|^2.

Dense forward simulation uses a partition of unity over field tiles:
each tile is windowed, convolved with its tile-center 3D PSF, and summed.

This module is compute-only — see ``benchmarks/simulate.py`` for the
benchmark wrapper.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from waveorder import optics
from waveorder.zernike import SpatialPolynomialPupil, make_pupil_grid


def calculate_psf_at_field_position(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    spatial_pupil: SpatialPolynomialPupil,
    x_o_um: float,
    y_o_um: float,
    z_padding: int = 0,
) -> Tensor:
    """Compute the 3D PSF at one field position from a spatial-polynomial pupil.

    The pupil is evaluated at ``(x_o_um, y_o_um)``; the propagation
    kernel ``exp(i 2 pi z n_z)`` is shared across positions and applied
    after the pupil multiplication. Returns the intensity PSF
    :math:`|\\mathrm{IFFT}[P]|^2` summed to unit volume.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        ``(Z, Y, X)`` shape of the output PSF.
    yx_pixel_size : float
        Transverse pixel size in microns.
    z_pixel_size : float
        Axial pixel size in microns.
    wavelength_emission : float
        Emission wavelength in microns.
    index_of_refraction_media : float
        Refractive index of the surrounding medium.
    numerical_aperture_detection : float
        Detection NA.
    spatial_pupil : SpatialPolynomialPupil
        Field-position dependent pupil.
    x_o_um, y_o_um : float
        Field position relative to the FOV center, in microns.
    z_padding : int
        Extra Z padding above/below ``zyx_shape[0]``. Mainly here to
        match ``isotropic_fluorescent_thick_3d`` if you want to reuse
        the OTF directly.

    Returns
    -------
    Tensor
        3D PSF, shape ``(Z + 2 z_padding, Y, X)``, real-valued, sum
        normalised to ``1``.
    """
    z_total = zyx_shape[0] + 2 * z_padding
    yx_shape = zyx_shape[1:]

    _, _, frr, theta = make_pupil_grid(yx_shape, yx_pixel_size, fft_order=True)
    pupil_amp = optics.generate_pupil(frr, numerical_aperture_detection, wavelength_emission)
    rho = (frr * wavelength_emission / numerical_aperture_detection).clamp(max=1.0)
    phi = spatial_pupil.aberration_waves(rho, theta, x_o_um, y_o_um)
    pupil = pupil_amp.to(torch.complex64) * torch.exp(2j * math.pi * phi.to(torch.complex64))

    z_position_list = torch.fft.ifftshift((torch.arange(z_total) - z_total // 2) * z_pixel_size)
    propagation_kernel = optics.generate_propagation_kernel(
        frr,
        pupil_amp,
        wavelength_emission / index_of_refraction_media,
        z_position_list,
    )
    pupil_3d = propagation_kernel * pupil[None, :, :]
    coherent_psf = torch.fft.ifft2(pupil_3d, dim=(1, 2))
    psf = torch.abs(coherent_psf) ** 2

    total = psf.sum()
    if total > 0:
        psf = psf / total
    return psf


def _hann_window_1d(n: int) -> Tensor:
    """Periodic Hann window of length ``n``, summed to one when overlapped at half-period."""
    if n <= 1:
        return torch.ones(n)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(n) / n)


def make_partition_of_unity_weights(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    n_tiles_yx: tuple[int, int],
    overlap_fraction: float = 0.5,
) -> tuple[Tensor, list[tuple[int, int]], list[tuple[float, float]]]:
    """Build per-tile partition-of-unity weights for shift-variant tiling.

    Tile centers are placed on a uniform ``n_tiles_yx`` grid spanning the
    full FOV. Each tile is given a raised-cosine window with FWHM equal
    to ``2 / overlap_fraction`` times the tile spacing. Weights are
    normalised so they sum to one at every pixel.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        ``(Ny, Nx)`` real-space image shape.
    yx_pixel_size : float
        Transverse pixel size (used only to report tile centers in microns).
    n_tiles_yx : tuple[int, int]
        Number of tiles along ``(Y, X)``.
    overlap_fraction : float
        Width of each window relative to tile spacing. ``0.5`` means
        each window has full width = 2 tile spacings (50% overlap).

    Returns
    -------
    weights : Tensor
        Shape ``(n_tiles, Ny, Nx)`` partition-of-unity weights, summing
        to ``1`` at every pixel.
    tile_indices : list[tuple[int, int]]
        ``(iy, ix)`` tile lattice indices, one per output channel.
    tile_centers_um : list[tuple[float, float]]
        ``(y_um, x_um)`` tile centers relative to the FOV center.
    """
    ny_tiles, nx_tiles = n_tiles_yx
    Ny, Nx = yx_shape

    y_extent_um = Ny * yx_pixel_size
    x_extent_um = Nx * yx_pixel_size
    y_centers_um = torch.linspace(-y_extent_um / 2, y_extent_um / 2, ny_tiles + 2)[1:-1]
    x_centers_um = torch.linspace(-x_extent_um / 2, x_extent_um / 2, nx_tiles + 2)[1:-1]

    spacing_y_um = y_extent_um / (ny_tiles + 1)
    spacing_x_um = x_extent_um / (nx_tiles + 1)
    width_y_um = spacing_y_um / max(overlap_fraction, 1e-3)
    width_x_um = spacing_x_um / max(overlap_fraction, 1e-3)

    y_um = (torch.arange(Ny) - Ny // 2 + 0.5) * yx_pixel_size
    x_um = (torch.arange(Nx) - Nx // 2 + 0.5) * yx_pixel_size

    weights = []
    tile_indices = []
    tile_centers_um = []
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            yc = y_centers_um[iy]
            xc = x_centers_um[ix]
            wy = torch.cos(math.pi * (y_um - yc) / width_y_um).clamp(min=0) ** 2
            wx = torch.cos(math.pi * (x_um - xc) / width_x_um).clamp(min=0) ** 2
            weights.append(wy[:, None] * wx[None, :])
            tile_indices.append((iy, ix))
            tile_centers_um.append((float(yc), float(xc)))

    w = torch.stack(weights, dim=0)
    w = w / w.sum(dim=0, keepdim=True).clamp(min=1e-12)
    return w, tile_indices, tile_centers_um


def apply_shift_variant_forward(
    zyx_object: Tensor,
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    spatial_pupil: SpatialPolynomialPupil,
    n_tiles_yx: tuple[int, int] = (8, 8),
    background: float = 0.0,
    return_tile_psfs: bool = False,
) -> Tensor | tuple[Tensor, dict]:
    """Apply a smoothly shift-variant fluorescence forward model.

    Decomposes the FOV into a partition of unity over ``n_tiles_yx``
    tiles. Each tile contributes::

        g_t(r_d) = h_t(r_d) ⊛ [w_t(r_d_xy) * f(r_d)],

    where ``h_t`` is the 3D PSF evaluated at the tile-center field
    position, ``w_t`` is the tile's partition-of-unity weight, and the
    convolution is computed in the Fourier domain. The total simulated
    measurement is :math:`g(r_d) = \\sum_t g_t(r_d) + \\text{background}`.

    Parameters
    ----------
    zyx_object : Tensor
        3D fluorescence object, shape ``(Z, Y, X)``.
    yx_pixel_size : float
        Transverse pixel size in microns.
    z_pixel_size : float
        Axial pixel size in microns.
    wavelength_emission : float
        Emission wavelength in microns.
    index_of_refraction_media : float
        Refractive index of the imaging medium.
    numerical_aperture_detection : float
        Detection NA.
    spatial_pupil : SpatialPolynomialPupil
        Field-position dependent pupil expansion.
    n_tiles_yx : tuple[int, int]
        Number of partition tiles in ``(Y, X)``.
    background : float
        Additive background.
    return_tile_psfs : bool
        If True, also return ``{'tile_psfs': (T, Z, Y, X), 'tile_centers_um': [(y, x), ...]}``.

    Returns
    -------
    Tensor or tuple[Tensor, dict]
        Simulated 3D measurement of shape ``(Z, Y, X)``. When
        ``return_tile_psfs`` is True, a dict with the tile PSFs and
        tile centers is returned alongside.
    """
    Z, Y, X = zyx_object.shape

    weights, tile_indices, tile_centers_um = make_partition_of_unity_weights((Y, X), yx_pixel_size, n_tiles_yx)

    object_fft_cache: dict[int, Tensor] = {}
    accumulator = torch.zeros_like(zyx_object)
    psfs: list[Tensor] = []

    for t, (yc_um, xc_um) in enumerate(tile_centers_um):
        psf = calculate_psf_at_field_position(
            (Z, Y, X),
            yx_pixel_size,
            z_pixel_size,
            wavelength_emission,
            index_of_refraction_media,
            numerical_aperture_detection,
            spatial_pupil,
            x_o_um=xc_um,
            y_o_um=yc_um,
        )
        if return_tile_psfs:
            psfs.append(psf)

        windowed = zyx_object * weights[t][None, :, :]
        otf = torch.fft.fftn(psf)
        windowed_fft = torch.fft.fftn(windowed)
        contrib = torch.real(torch.fft.ifftn(windowed_fft * otf))
        accumulator = accumulator + contrib

    accumulator = accumulator + background
    if return_tile_psfs:
        return accumulator, {"tile_psfs": torch.stack(psfs, dim=0), "tile_centers_um": tile_centers_um}
    return accumulator


def calculate_otf_at_field_position(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    spatial_pupil: SpatialPolynomialPupil,
    x_o_um: float,
    y_o_um: float,
) -> Tensor:
    """Convenience wrapper: 3D OTF for a single field position.

    The OTF is normalised so that ``max(|OTF|) == 1``, matching the
    shift-invariant ``isotropic_fluorescent_thick_3d.calculate_transfer_function``.
    """
    psf = calculate_psf_at_field_position(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_emission,
        index_of_refraction_media,
        numerical_aperture_detection,
        spatial_pupil,
        x_o_um=x_o_um,
        y_o_um=y_o_um,
    )
    otf = torch.fft.fftn(psf)
    otf = otf / torch.clamp(torch.max(torch.abs(otf)), min=1e-12)
    return otf

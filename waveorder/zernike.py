"""Zernike polynomials and spatial-polynomial pupils for shift-variant imaging.

Provides:

* :func:`zernike_noll` and :func:`zernike_modes` to evaluate one or many
  Noll-indexed Zernike polynomials on a normalized pupil grid.
* :class:`SpatialPolynomialPupil`, an expansion

  .. math::

      P(\\nu; r_o) = P_0(\\nu) \\exp\\!\\left(i\\,2\\pi
      \\sum_{j, m, n} c_{j, m, n}\\, Z_j(\\nu)\\, x_o^m\\, y_o^n\\right),

  where :math:`P_0(\\nu)` is the diffraction-limited circular pupil and
  :math:`c_{j, m, n}` are the user-specified coefficients (units: waves).

Both Noll indexing (``j = 1`` is piston) and the spatial-polynomial sums
follow the convention used in :mod:`waveorder.models.shift_variant_fluorescent_3d`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from waveorder import optics


def _noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll index ``j`` (1-based) to ``(n, m)`` order/azimuthal pair.

    Parameters
    ----------
    j : int
        Noll index, ``j >= 1``. ``j = 1`` is piston.

    Returns
    -------
    tuple[int, int]
        ``(n, m)`` with ``n >= 0`` and ``-n <= m <= n``,
        ``(n - |m|) % 2 == 0``.
    """
    if j < 1:
        raise ValueError(f"Noll index j must be >= 1, got {j}")
    n = 0
    cumulative = 0
    while cumulative + (n + 1) < j:
        n += 1
        cumulative += n
    k = j - cumulative - 1
    if n % 2 == 0:
        ms = [0] + [v for pair in zip(range(2, n + 1, 2), range(-2, -n - 1, -2)) for v in pair]
    else:
        ms = [v for pair in zip(range(1, n + 1, 2), range(-1, -n - 1, -2)) for v in pair]
    m = ms[k]
    if j % 2 == 0 and m < 0:
        m = -m
    if j % 2 == 1 and m > 0:
        m = -m
    if j == 1:
        m = 0
    return n, m


def _radial_polynomial(n: int, m: int, rho: Tensor) -> Tensor:
    """Evaluate the Zernike radial polynomial :math:`R_n^{|m|}(\\rho)`.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency (sign ignored).
    rho : Tensor
        Normalized radius in ``[0, 1]``.

    Returns
    -------
    Tensor
        Same shape as ``rho``.
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return torch.zeros_like(rho)
    out = torch.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        coef = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k) * math.factorial((n + m_abs) // 2 - k) * math.factorial((n - m_abs) // 2 - k)
        )
        out = out + coef * rho ** (n - 2 * k)
    return out


def zernike_noll(j: int, rho: Tensor, theta: Tensor) -> Tensor:
    """Evaluate a single Noll-indexed Zernike polynomial.

    Uses the orthonormal convention: ``int |Z|^2 dA / (pi R^2) = 1`` over
    the unit disk, so the prefactor is ``sqrt(2(n+1))`` for ``m != 0`` and
    ``sqrt(n+1)`` for ``m == 0``.

    Parameters
    ----------
    j : int
        Noll index (1-based). ``j = 1`` is piston.
    rho : Tensor
        Normalized radius in ``[0, 1]``.
    theta : Tensor
        Azimuthal angle in radians.

    Returns
    -------
    Tensor
        Zernike polynomial evaluated on ``(rho, theta)``. Same shape as
        the broadcast of inputs.
    """
    n, m = _noll_to_nm(j)
    r = _radial_polynomial(n, abs(m), rho)
    if m > 0:
        out = r * torch.cos(m * theta)
        norm = math.sqrt(2 * (n + 1))
    elif m < 0:
        out = r * torch.sin(-m * theta)
        norm = math.sqrt(2 * (n + 1))
    else:
        out = r
        norm = math.sqrt(n + 1)
    return norm * out


def zernike_modes(noll_indices: list[int], rho: Tensor, theta: Tensor) -> Tensor:
    """Evaluate several Noll-indexed Zernike modes stacked along axis 0.

    Parameters
    ----------
    noll_indices : list[int]
        Noll indices to compute.
    rho : Tensor
        Normalized pupil radius in ``[0, 1]``.
    theta : Tensor
        Azimuthal pupil angle in radians.

    Returns
    -------
    Tensor
        Shape ``(len(noll_indices), *rho.shape)``.
    """
    return torch.stack([zernike_noll(j, rho, theta) for j in noll_indices], dim=0)


class SpatialPolynomialPupil:
    """Field-position dependent complex pupil from spatial-polynomial coefficients.

    The pupil is

    .. math::

        P(\\nu; r_o) = P_0(\\nu)
        \\exp\\!\\left(i\\,2\\pi \\Phi(\\nu; r_o)\\right),

        \\Phi(\\nu; r_o) = \\sum_{j, m, n} c_{j, m, n}\\,
        Z_j(\\nu)\\, x_o^m\\, y_o^n,

    where :math:`Z_j` is the Noll-indexed Zernike polynomial,
    :math:`(x_o, y_o)` are normalized field-of-view positions in
    ``[-1, 1]``, and the coefficients :math:`c_{j,m,n}` are in waves.

    Parameters
    ----------
    coefficients : dict
        Mapping ``(j, m, n) -> c`` of spatial-polynomial coefficients in
        waves. Keys are tuples of ints. Coefficients with ``c == 0`` are
        omitted.
    field_extent_um : tuple[float, float]
        Half-extent of the field of view in microns, ``(half_y, half_x)``.
        Used to normalize field positions: ``x_o_norm = x_um / half_x``.

    Examples
    --------
    >>> coefs = {(4, 0, 0): 0.0, (4, 2, 0): 0.5, (4, 0, 2): 0.5}  # quadratic defocus
    >>> spp = SpatialPolynomialPupil(coefs, field_extent_um=(12.8, 12.8))
    """

    def __init__(self, coefficients: dict[tuple[int, int, int], float], field_extent_um: tuple[float, float]):
        self.coefficients = {k: float(v) for k, v in coefficients.items() if v != 0}
        self.field_extent_um = tuple(float(v) for v in field_extent_um)

        seen_jmn = set(self.coefficients.keys())
        self.noll_indices = sorted({j for (j, _, _) in seen_jmn})

    @property
    def max_spatial_order(self) -> int:
        """Highest combined spatial order ``m + n`` present in the expansion."""
        if not self.coefficients:
            return 0
        return max(m + n for (_, m, n) in self.coefficients.keys())

    def aberration_waves(self, rho: Tensor, theta: Tensor, x_o_um: float, y_o_um: float) -> Tensor:
        """Evaluate :math:`\\Phi(\\nu; r_o)` (in waves) on the pupil grid.

        Parameters
        ----------
        rho : Tensor
            Normalized pupil radius, shape ``(Ny, Nx)``.
        theta : Tensor
            Azimuthal angle, shape ``(Ny, Nx)``.
        x_o_um, y_o_um : float
            Field position in microns.

        Returns
        -------
        Tensor
            Wavefront aberration in waves, shape ``(Ny, Nx)``.
        """
        if not self.coefficients:
            return torch.zeros_like(rho)
        x_norm = x_o_um / self.field_extent_um[1]
        y_norm = y_o_um / self.field_extent_um[0]
        zernike_cache: dict[int, Tensor] = {}
        out = torch.zeros_like(rho)
        for (j, m, n), c in self.coefficients.items():
            if j not in zernike_cache:
                zernike_cache[j] = zernike_noll(j, rho, theta)
            out = out + c * zernike_cache[j] * (x_norm**m) * (y_norm**n)
        return out

    def complex_pupil(
        self,
        radial_frequencies: Tensor,
        azimuthal_angles: Tensor,
        numerical_aperture: float,
        wavelength: float,
        x_o_um: float,
        y_o_um: float,
    ) -> Tensor:
        """Evaluate the complex pupil :math:`P(\\nu; r_o)`.

        Parameters
        ----------
        radial_frequencies : Tensor
            Pupil radial frequencies, shape ``(Ny, Nx)``, in 1/length.
        azimuthal_angles : Tensor
            Pupil azimuthal angles, shape ``(Ny, Nx)``, in radians.
        numerical_aperture : float
            Detection NA. Sets the pupil cutoff.
        wavelength : float
            Wavelength in length units matching ``radial_frequencies``.
        x_o_um, y_o_um : float
            Field position in microns.

        Returns
        -------
        Tensor
            Complex pupil, shape ``(Ny, Nx)``, dtype ``complex64``.
        """
        amplitude = optics.generate_pupil(radial_frequencies, numerical_aperture, wavelength)
        rho = (radial_frequencies * wavelength / numerical_aperture).clamp(max=1.0)
        phi = self.aberration_waves(rho, azimuthal_angles, x_o_um, y_o_um)
        return amplitude.to(torch.complex64) * torch.exp(2j * math.pi * phi.to(torch.complex64))


class SeidelPupil:
    """Rotationally symmetric field-dependent pupil from primary Seidel coefficients.

    The wavefront aberration is

    .. math::

        \\Phi(\\rho, \\theta; r_o) =\\;
            & W_d\\,\\rho^2
            + W_{040}\\,\\rho^4 \\\\
            & + W_{131}\\,\\eta\\,\\rho^3\\,\\cos(\\theta - \\phi)
            + W_{222}\\,\\eta^2\\,\\rho^2\\,\\cos^2(\\theta - \\phi) \\\\
            & + W_{220}\\,\\eta^2\\,\\rho^2
            + W_{311}\\,\\eta^3\\,\\rho\\,\\cos(\\theta - \\phi),

    where :math:`(\\rho, \\theta)` are normalized pupil polar coordinates,
    :math:`\\eta = \\sqrt{x_o^2 + y_o^2}` is the normalized field radius,
    and :math:`\\phi = \\mathrm{atan2}(y_o, x_o)` is the field azimuth.
    All coefficients are in waves of optical path difference.

    Because the wavefront depends on :math:`\\theta - \\phi`, the system is
    rotationally symmetric: PSFs at the same :math:`\\eta` are rotations of
    one another. This is the "linear revolution invariance" (LRI)
    property exploited by Ring Deconvolution Microscopy.

    Parameters
    ----------
    coefficients : dict
        Mapping from coefficient name to value in waves. Accepted names:
        ``"sphere"`` (:math:`W_{040}`), ``"coma"`` (:math:`W_{131}`),
        ``"astigmatism"`` (:math:`W_{222}`), ``"field_curvature"``
        (:math:`W_{220}`), ``"distortion"`` (:math:`W_{311}`),
        ``"defocus"`` (:math:`W_d`). Unspecified entries are zero.
    field_extent_um : tuple[float, float]
        Half-extent of the field in microns, ``(half_y, half_x)``. Field
        positions are normalised so the FOV corner is at
        :math:`\\eta = \\sqrt{2}`.

    Examples
    --------
    >>> pupil = SeidelPupil({"sphere": 0.2, "coma": 0.5}, field_extent_um=(12.8, 12.8))
    >>> import torch
    >>> rho = torch.linspace(0, 1, 4)[:, None].expand(4, 4)
    >>> theta = torch.zeros(4, 4)
    >>> w_on_axis = pupil.aberration_waves(rho, theta, 0.0, 0.0)
    >>> torch.allclose(w_on_axis, 0.2 * rho ** 4)
    True
    """

    COEFFICIENT_NAMES: tuple[str, ...] = (
        "sphere",
        "coma",
        "astigmatism",
        "field_curvature",
        "distortion",
        "defocus",
    )

    def __init__(
        self,
        coefficients: dict[str, float],
        field_extent_um: tuple[float, float],
    ):
        unknown = set(coefficients) - set(self.COEFFICIENT_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown Seidel coefficient name(s): {sorted(unknown)}. Accepted: {self.COEFFICIENT_NAMES}"
            )
        self.coefficients = {name: float(coefficients.get(name, 0.0)) for name in self.COEFFICIENT_NAMES}
        self.field_extent_um = tuple(float(v) for v in field_extent_um)

    def aberration_waves(
        self,
        rho: Tensor,
        theta: Tensor,
        x_o_um: float,
        y_o_um: float,
    ) -> Tensor:
        """Evaluate the Seidel wavefront aberration (in waves) on a pupil grid.

        Parameters
        ----------
        rho : Tensor
            Normalized pupil radius, shape ``(Ny, Nx)``.
        theta : Tensor
            Azimuthal pupil angle in radians, shape ``(Ny, Nx)``.
        x_o_um, y_o_um : float
            Field position relative to the FOV center, in microns.

        Returns
        -------
        Tensor
            Wavefront aberration in waves, same shape as ``rho``.
        """
        x_norm = x_o_um / self.field_extent_um[1]
        y_norm = y_o_um / self.field_extent_um[0]
        eta = math.sqrt(x_norm * x_norm + y_norm * y_norm)
        phi_field = math.atan2(y_norm, x_norm) if eta > 0 else 0.0

        c = self.coefficients
        out = torch.zeros_like(rho)
        if c["defocus"]:
            out = out + c["defocus"] * rho**2
        if c["sphere"]:
            out = out + c["sphere"] * rho**4
        if eta > 0:
            cos_dtheta = torch.cos(theta - phi_field)
            if c["coma"]:
                out = out + c["coma"] * eta * rho**3 * cos_dtheta
            if c["astigmatism"]:
                out = out + c["astigmatism"] * (eta**2) * (rho**2) * (cos_dtheta**2)
            if c["field_curvature"]:
                out = out + c["field_curvature"] * (eta**2) * (rho**2)
            if c["distortion"]:
                out = out + c["distortion"] * (eta**3) * rho * cos_dtheta
        return out


def make_pupil_grid(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    fft_order: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate Cartesian and polar pupil-frequency grids.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        ``(Ny, Nx)`` pupil grid shape.
    yx_pixel_size : float
        Real-space pixel size in length units (e.g. um).
    fft_order : bool
        If True (default), DC at index ``(0, 0)`` matching
        ``torch.fft.fftfreq``. If False, DC at the array center
        (visualisation-friendly).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(fxx, fyy, frr, theta)`` each of shape ``(Ny, Nx)``.
        ``fxx, fyy`` are Cartesian frequencies (cycles per length unit),
        ``frr = sqrt(fxx**2 + fyy**2)``, and ``theta = arctan2(fyy, fxx)``.
    """
    ny, nx = yx_shape
    fy = torch.fft.fftfreq(ny, d=yx_pixel_size)
    fx = torch.fft.fftfreq(nx, d=yx_pixel_size)
    if not fft_order:
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    frr = torch.sqrt(fxx**2 + fyy**2)
    theta = torch.atan2(fyy, fxx)
    return fxx, fyy, frr, theta

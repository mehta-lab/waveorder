"""Unmatched back projectors that accelerate Richardson-Lucy deconvolution.

Richardson-Lucy usually back-projects with the transpose of the forward
projector, ``conj(OTF)``. It does not have to. What sets the convergence rate
is the **spectral product** ``|DFT(f) * DFT(b)|``, read per spatial frequency:

- product near 1 -> that frequency is recovered in one iteration
- product of 0.01 -> it needs roughly 100

The transpose gives ``|OTF|**2``, which spans orders of magnitude between DC and
the resolution limit, so the iteration count is set by the slowest, highest
frequency. Choosing ``b`` to flatten that product instead is a preconditioner,
and it is what turns "ten or more iterations" into one.

Every kind except ``"gaussian"`` is an inversion term times an apodization term:

===================== ==================== ==========================
inversion             apodization: none    apodization: Butterworth
===================== ==================== ==========================
``conj(OTF)``         ``"matched"``        --
Wiener                ``"wiener"``         ``"wiener_butterworth"``
``1`` (Dirac delta)   (noise, unusable)    ``"butterworth"``
===================== ==================== ==========================

The inversion term (``alpha``) decides how flat the product is; the apodization
term (``beta``, ``order``) decides how hard everything past the resolution limit
is suppressed. Only the Wiener term amplifies, so only the kinds containing it
flatten the product appreciably -- which is why ``"wiener_butterworth"``, with
both, is the one Guo et al. recommend.

``"gaussian"`` stands apart: designed in real space to match the PSF FWHM, no
free parameters, and it only ever attenuates.

Caveat: an unmatched ``b`` is not an adjoint, so Richardson-Lucy's convergence
guarantee no longer holds and over-iterating introduces artifacts. Guo et al.
suggest one iteration as a rule of thumb (up to five at low ``order``).

Reference: Guo et al. 2020, Supplementary Note 2
(`doi.org/10.1038/s41587-020-0560-x <https://doi.org/10.1038/s41587-020-0560-x>`_),
following ``BackProjector.m`` in
`eguomin/regDeconProject <https://github.com/eguomin/regDeconProject>`_.
"""


import math
from typing import Literal, Optional

import torch
from torch import Tensor

_EPS = 1e-12

BackProjectorType = Literal[
    "matched",
    "gaussian",
    "butterworth",
    "wiener",
    "wiener_butterworth",
]
ResolutionMode = Literal["fwhm", "fwhm_over_sqrt2", "manual"]
BetaConvention = Literal["reference", "paper"]

_BACK_PROJECTOR_TYPES = (
    "matched",
    "gaussian",
    "butterworth",
    "wiener",
    "wiener_butterworth",
)
_RESOLUTION_MODES = ("fwhm", "fwhm_over_sqrt2", "manual")
_BETA_CONVENTIONS = ("reference", "paper")


def calculate_back_projector(
    optical_transfer_function: Tensor,
    back_projector: BackProjectorType = "matched",
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    order: int = 8,
    resolution_mode: ResolutionMode = "fwhm",
    resolution_zyx_px: Optional[tuple[float, float, float]] = None,
    beta_convention: BetaConvention = "reference",
) -> Tensor:
    """Build a back projector in Fourier space from a forward-projector OTF.

    Drop-in replacement for ``conj(OTF)`` in the Richardson-Lucy back-projection
    step: same FFT convention (DC at index zero), same shape, device and dtype,
    so the adjoint stays ``ifftn(fftn(y) * back_projector)``.

    Which knobs each kind reads, and what turning them does:

    ==================== ===== ==== ===== ==========================================
    kind                 alpha beta order effect
    ==================== ===== ==== ===== ==========================================
    ``matched``          --    --   --    plain Richardson-Lucy, no free parameters
    ``gaussian``         --    --   --    attenuates only; the mildest option
    ``butterworth``      --    x    x     suppresses past the cutoff, no inversion
    ``wiener``           x     --   --    inverts, but amplifies noise unchecked
    ``wiener_butterworth`` x   x    x     inverts and suppresses; the recommended one
    ==================== ===== ==== ===== ==========================================

    Lower ``alpha`` -> flatter spectral product -> fewer iterations, more noise.
    Lower ``beta`` / higher ``order`` -> harder suppression past the resolution
    limit, at the cost of real-space ringing.

    Parameters
    ----------
    optical_transfer_function : Tensor
        Forward-projector OTF, complex, shape ``(Z, Y, X)``, DC at index zero.
        Every kind except ``"matched"`` assumes a unit-peak OTF and normalizes
        internally if needed.
    back_projector : {"matched", "gaussian", "butterworth", "wiener", \
"wiener_butterworth"}, optional
        Which one to build, by default ``"matched"``.
    alpha : float, optional
        Wiener regularization: the term added to ``|OTF|**2`` that keeps the
        inversion from dividing by a vanishing OTF. It sets **how hard the
        deconvolution inverts**, since the spectral product is
        ``|OTF|**2 / (|OTF|**2 + alpha)`` -- flat (so one iteration) only when
        ``alpha`` is near ``|OTF(cutoff)|**2``, and increasingly damped above
        that. Guo et al. use 0.001-0.05.

        ``None`` (default) substitutes the mean cutoff gain (Eq. 28) SQUARED --
        that gain is ``beta``'s scale, and ``alpha`` is added to a squared
        magnitude, so using it unsquared leaves ``alpha`` orders of magnitude
        too large.
    beta : float, optional
        Cutoff gain: the spectral amplitude still passed **at** the resolution
        limit, so smaller means sharper suppression beyond it. Enters as
        ``eps**2``; see Notes. Guo et al. use 0.001-0.05 (Table S2.1). ``None``
        (default) substitutes the mean cutoff gain (Eq. 28).
    order : int, optional
        Butterworth order, the exponent that sets **how steep the transition
        is** at the cutoff, by default 8. Higher is flatter in the passband but
        closer to a brick wall, which rings in real space; lower is gentler but
        gives up amplitude near the cutoff, so more iterations are needed. Guo
        et al. pair 8-10 with a single iteration for single- and dual-view
        microscopes, dropping to 5 (and 2-5 iterations) for quad-view and
        reflective geometries, which ring more readily.
    resolution_mode : {"fwhm", "fwhm_over_sqrt2", "manual"}, optional
        Where the cutoff frequency sits. ``"fwhm"`` (default) uses the measured
        PSF FWHM; ``"fwhm_over_sqrt2"`` puts the cutoff a factor sqrt(2) higher,
        for resolution-doubling instruments such as iSIM; ``"manual"`` takes
        ``resolution_zyx_px``. Ignored by ``"matched"`` and ``"gaussian"``.
    resolution_zyx_px : tuple of float, optional
        Resolution limit per axis **in pixels**, required by and only valid with
        ``resolution_mode="manual"``. Divide a physical resolution by the pixel
        size first.
    beta_convention : {"reference", "paper"}, optional
        How ``beta`` calibrates the Wiener-Butterworth transition, by default
        ``"reference"``. See Notes. Ignored by every other kind.

    Returns
    -------
    Tensor
        Complex back projector, same shape, device and dtype as the input OTF.

    Notes
    -----
    Guo et al.'s text and their reference code disagree on how ``beta`` maps to
    the Butterworth transition width. Both belong to one family,
    ``eps**2 = beta_w**p / beta**2 - 1``, with ``beta_w`` the Wiener term's own
    gain at the lateral cutoff: Eq. 27 has ``p = 2``, the reference code
    ``p = 1``. They agree only at ``beta_w == 1``, which never occurs -- the
    Wiener term amplifies near the cutoff, putting ``beta_w`` around ten.

    So the filter's actual gain at the cutoff is ``beta`` under ``"paper"`` and
    ``beta * sqrt(beta_w)`` under ``"reference"``. This module implements the
    paper's formula, rescaling ``beta`` by ``sqrt(beta_w)`` for ``"reference"``,
    which is the default so published Table S2.1 values reproduce published
    results.

    References
    ----------
    Guo, M. et al. Rapid image deconvolution and multiview fusion for optical
    microscopy. *Nat. Biotechnol.* **38**, 1337-1346 (2020), Supplementary
    Note 2.
    """
    if back_projector not in _BACK_PROJECTOR_TYPES:
        raise ValueError(f"back_projector must be one of {_BACK_PROJECTOR_TYPES}, got {back_projector!r}")
    if resolution_mode not in _RESOLUTION_MODES:
        raise ValueError(f"resolution_mode must be one of {_RESOLUTION_MODES}, got {resolution_mode!r}")
    if beta_convention not in _BETA_CONVENTIONS:
        raise ValueError(f"beta_convention must be one of {_BETA_CONVENTIONS}, got {beta_convention!r}")
    if optical_transfer_function.ndim != 3:
        raise ValueError(
            f"optical_transfer_function must be 3D (Z, Y, X), got shape {tuple(optical_transfer_function.shape)}"
        )
    if not optical_transfer_function.is_complex():
        raise ValueError(f"optical_transfer_function must be complex, got dtype {optical_transfer_function.dtype}")

    # The transpose is the true adjoint whatever the OTF normalization, so it
    # short-circuits before any of the filter-design machinery below.
    if back_projector == "matched":
        return torch.conj_physical(optical_transfer_function)

    if resolution_mode == "manual":
        if resolution_zyx_px is None:
            raise ValueError("resolution_mode='manual' requires resolution_zyx_px")
        if len(resolution_zyx_px) != 3:
            raise ValueError(f"resolution_zyx_px must have 3 entries (Z, Y, X), got {len(resolution_zyx_px)}")
        if any(r <= 0 for r in resolution_zyx_px):
            raise ValueError(f"resolution_zyx_px entries must be positive, got {resolution_zyx_px}")
    elif resolution_zyx_px is not None:
        raise ValueError(f"resolution_zyx_px is only valid with resolution_mode='manual', got {resolution_mode!r}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    shape = tuple(optical_transfer_function.shape)
    device = optical_transfer_function.device
    real_dtype = optical_transfer_function.real.dtype

    # The Gaussian back projector has no free parameters and, per the reference
    # implementation, matches the PSF FWHM whatever resolution_mode says.
    if back_projector == "gaussian":
        fwhm_zyx_px = _psf_fwhm_zyx_px(optical_transfer_function)
        gaussian = _gaussian_back_projector(shape, fwhm_zyx_px, device, real_dtype)
        return gaussian.to(optical_transfer_function.dtype)

    if resolution_mode == "manual":
        resolution = tuple(float(r) for r in resolution_zyx_px)
    else:
        fwhm_zyx_px = _psf_fwhm_zyx_px(optical_transfer_function)
        divisor = 1.0 if resolution_mode == "fwhm" else math.sqrt(2.0)
        resolution = tuple(f / divisor for f in fwhm_zyx_px)

    # Cutoff as a signed frequency index, matching the reference: the
    # Fourier-domain pixel size is 1/S, so the cutoff sits at S / resolution.
    cutoff_indices = tuple(size / res for size, res in zip(shape, resolution))

    alpha_value = None if alpha is None else float(alpha)
    beta_value = None if beta is None else float(beta)
    if back_projector != "butterworth" or alpha_value is None or beta_value is None:
        normalized_otf = optical_transfer_function / torch.clamp(
            torch.max(torch.abs(optical_transfer_function)), min=_EPS
        )
        normalized_magnitude = torch.abs(normalized_otf)
        if alpha_value is None or beta_value is None:
            matched_cutoff_gain = _matched_cutoff_gain(normalized_magnitude, cutoff_indices)
            # Eq. 28 returns a gain, which is beta's scale. alpha is added to
            # |OTF|**2, so it needs that gain squared.
            alpha_value = matched_cutoff_gain**2 if alpha_value is None else alpha_value
            beta_value = matched_cutoff_gain if beta_value is None else beta_value

    if alpha_value <= 0:
        raise ValueError(f"alpha must be positive, got {alpha_value}")
    if beta_value <= 0:
        raise ValueError(f"beta must be positive, got {beta_value}")

    if back_projector == "butterworth":
        # beta = 1 / sqrt(1 + eps**2), so eps**2 = 1 / beta**2 - 1.
        if beta_value > 1.0:
            raise ValueError(f"butterworth requires beta <= 1 (it is a gain at the cutoff), got {beta_value}")
        epsilon_squared = 1.0 / beta_value**2 - 1.0
        mask = _butterworth_mask(shape, cutoff_indices, epsilon_squared, order, device, real_dtype)
        return mask.to(optical_transfer_function.dtype)

    wiener = torch.conj(normalized_otf) / (normalized_magnitude.square() + alpha_value)
    if back_projector == "wiener":
        return wiener.to(optical_transfer_function.dtype)

    # Wiener-Butterworth. See Notes on the two beta conventions.
    wiener_cutoff_gain = _wiener_cutoff_gain(wiener, cutoff_indices)
    if beta_convention == "reference":
        effective_beta = beta_value * math.sqrt(wiener_cutoff_gain)
    else:
        effective_beta = beta_value
    if effective_beta > wiener_cutoff_gain:
        raise ValueError(
            f"wiener_butterworth requires an effective beta <= the Wiener cutoff gain "
            f"({wiener_cutoff_gain:.4g}), got {effective_beta:.4g}. Lower beta or raise alpha."
        )
    epsilon_squared = (wiener_cutoff_gain / effective_beta) ** 2 - 1.0
    mask = _butterworth_mask(shape, cutoff_indices, epsilon_squared, order, device, real_dtype)
    return (wiener * mask).to(optical_transfer_function.dtype)


def _signed_frequency_indices(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Signed frequency index per FFT bin, ``[0, 1, ..., -2, -1]``, unshifted."""
    return torch.fft.fftfreq(size, device=device, dtype=dtype) * size


def _broadcast_along(values: Tensor, axis: int) -> Tensor:
    """Reshape a 1D tensor so it broadcasts along ``axis`` of a 3D volume."""
    return values.reshape([-1 if a == axis else 1 for a in range(3)])


def _ellipsoidal_radius_squared(
    shape: tuple[int, int, int],
    scales: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Squared radius on an unshifted ellipsoidal frequency grid."""
    radius_squared = torch.zeros(shape, device=device, dtype=dtype)
    for axis, (size, scale) in enumerate(zip(shape, scales)):
        indices = _signed_frequency_indices(size, device, dtype) / scale
        radius_squared.add_(_broadcast_along(indices.square(), axis))
    return radius_squared


def _butterworth_mask(
    shape: tuple[int, int, int],
    cutoff_indices: tuple[float, float, float],
    epsilon_squared: float,
    order: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Butterworth low-pass over an ellipsoidal cutoff surface, unshifted.

    Evaluated in float64 where supported because ``radius**(2 * order)`` can
    overflow float32. MPS uses an equivalent log-space float32 calculation.
    """
    work_dtype = dtype if device.type == "mps" else torch.float64
    radius_squared = _ellipsoidal_radius_squared(shape, cutoff_indices, device, work_dtype)
    if work_dtype == torch.float64:
        mask = radius_squared.pow_(order).mul_(epsilon_squared).add_(1.0).sqrt_().reciprocal_()
    elif epsilon_squared == 0:
        mask = torch.ones_like(radius_squared)
    else:
        log_attenuation = radius_squared.log_().mul_(order).add_(math.log(epsilon_squared))
        mask = torch.nn.functional.softplus(log_attenuation).mul_(-0.5).exp_()
    return mask.to(dtype)


def _gaussian_back_projector(
    shape: tuple[int, int, int],
    fwhm_zyx_px: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """OTF of a unit-sum Gaussian kernel whose FWHM matches the PSF."""
    sigmas = tuple(fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0))) for fwhm in fwhm_zyx_px)
    work_dtype = dtype if device.type == "mps" else torch.float64
    exponent = _ellipsoidal_radius_squared(shape, sigmas, device, work_dtype)
    kernel = torch.exp(-0.5 * exponent)
    kernel.div_(torch.clamp(kernel.sum(), min=_EPS))
    return torch.fft.fftn(kernel.to(dtype), dim=(-3, -2, -1))


def _psf_fwhm_zyx_px(optical_transfer_function: Tensor) -> tuple[float, float, float]:
    """Measure the PSF full width at half maximum per axis, in pixels."""
    psf = torch.fft.fftshift(torch.real(torch.fft.ifftn(optical_transfer_function, dim=(-3, -2, -1))))
    peak_z, peak_y, peak_x = (int(index) for index in torch.unravel_index(torch.argmax(psf), psf.shape))
    return (
        _fwhm_1d(psf[:, peak_y, peak_x], "z"),
        _fwhm_1d(psf[peak_z, :, peak_x], "y"),
        _fwhm_1d(psf[peak_z, peak_y, :], "x"),
    )


def _fwhm_1d(profile: Tensor, axis_name: str) -> float:
    """Full width at half maximum of a peaked 1D profile, in pixels.

    Both half-maximum crossings are located by linear interpolation between
    the bracketing samples. A profile that never falls below half maximum on
    one side is undersampled relative to the PSF, which the caller cannot
    recover from, so this raises rather than returning a sentinel.
    """
    peak_index = int(torch.argmax(profile))
    peak_value = float(profile[peak_index])
    if not peak_value > 0:
        raise ValueError(f"cannot measure PSF FWHM along {axis_name}: the profile peak is not positive")
    normalized = profile.cpu().to(torch.float64) / peak_value

    below_left = torch.nonzero(normalized[: peak_index + 1] < 0.5).flatten()
    below_right = torch.nonzero(normalized[peak_index:] < 0.5).flatten()
    if below_left.numel() == 0 or below_right.numel() == 0:
        raise ValueError(
            f"cannot measure PSF FWHM along {axis_name}: the profile never falls below half maximum, so "
            f"the PSF is undersampled along this axis. Pass resolution_mode='manual' with an explicit "
            f"resolution_zyx_px, or reconstruct on a finer grid."
        )

    def _interpolate(low_index: int, high_index: int) -> float:
        low_value = float(normalized[low_index])
        high_value = float(normalized[high_index])
        if high_value == low_value:
            raise ValueError(f"cannot measure PSF FWHM along {axis_name}: the profile is flat at half maximum")
        return low_index + (0.5 - low_value) / (high_value - low_value) * (high_index - low_index)

    left_index = int(below_left[-1])
    right_index = peak_index + int(below_right[0])
    return _interpolate(right_index - 1, right_index) - _interpolate(left_index, left_index + 1)


def _mean_gain_at_cutoff(profile: Tensor, cutoff: float) -> float:
    """Average a shifted 1D profile at the two cutoff frequencies."""
    size = profile.shape[0]
    center = size // 2
    low = max(int(round(center - cutoff)), 0)
    high = min(int(round(center + cutoff)), size - 1)
    return float((profile[low] + profile[high]) / 2)


def _matched_cutoff_gain(normalized_magnitude: Tensor, cutoff_indices: tuple[float, float, float]) -> float:
    """Mean cutoff gain of the matched back projector (Eq. 28).

    Per axis, the OTF magnitude is maximum-projected onto that axis and
    sampled at both cutoff frequencies; the three per-axis gains are averaged.
    This is the value substituted when ``alpha`` or ``beta`` is left unset.
    """
    magnitude = torch.fft.fftshift(normalized_magnitude)
    gains = []
    for axis, cutoff in enumerate(cutoff_indices):
        other_axes = tuple(a for a in range(3) if a != axis)
        gains.append(_mean_gain_at_cutoff(torch.amax(magnitude, dim=other_axes), cutoff))
    return sum(gains) / 3.0


def _wiener_cutoff_gain(wiener: Tensor, cutoff_indices: tuple[float, float, float]) -> float:
    """Gain of the Wiener term at the lateral (X) cutoff.

    Unlike :func:`_matched_cutoff_gain` this reads the central Z slice
    rather than a maximum projection, matching the reference implementation.
    """
    magnitude = torch.fft.fftshift(torch.abs(wiener))
    central_slice = magnitude[magnitude.shape[0] // 2]  # (Y, X)
    return _mean_gain_at_cutoff(torch.amax(central_slice, dim=0), cutoff_indices[2])

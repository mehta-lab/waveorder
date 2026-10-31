"""Calibration scores: Whittle/Gaussian evidence and the calibration penalty.

The evidence score is the exact negative log marginal likelihood of the
imaging system H for the Gaussian model (design document section 6, steps
3-5)::

    g = H f + n,   f ~ N(0, S),   n ~ N(0, sigma^2 I),

which is diagonal per spatial frequency k with variance
``v_k = |H_k|^2 S_k + sigma^2``::

    -log p(g | H) = sum_k [ |G_k|^2 / (2 v_k) + (1/2) log(2 pi v_k) ],

the Whittle likelihood, with ``G = FFT(g) / sqrt(N)`` (unitary
normalization). The Tikhonov prior with weight ``lambda`` is
``S_k = sigma^2 / lambda`` (the prior implied by the Wiener filter
``H* / (|H|^2 + lambda)``). The first term rewards systems that explain the
observed power; the log-det "volume" term stops the score from always
preferring the largest transfer function.

All terms are differentiable by torch autograd through ``otf`` (and
through anything ``otf`` was built from, e.g. Zernike coefficients via
:mod:`waveorder.calibrate.synthesis`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def whittle_evidence(
    data: Tensor | None = None,
    otf: Tensor | None = None,
    noise_sigma: float | Tensor = 1.0,
    regularization_strength: float | Tensor = 1e-3,
    object_spectrum: Tensor | None = None,
    data_fft: Tensor | None = None,
    exclude_dc: bool = False,
) -> Tensor:
    """Negative log evidence ``-log p(g | H)`` under the Gaussian model.

    Lower is better. Compare values only across calls with identical data
    and identical ``exclude_dc``.

    Parameters
    ----------
    data : Tensor, optional
        Real data ``g``, shape ``(Z, Y, X)`` (any number of dims works).
        Either ``data`` or ``data_fft`` must be given.
    otf : Tensor
        Transfer function ``H_k``, same shape as the data, FFT frequency
        order (e.g. from :func:`waveorder.calibrate.synthesis.fluorescence_otf`).
    noise_sigma : float or Tensor
        Noise standard deviation ``sigma`` in data units.
    regularization_strength : float or Tensor
        Tikhonov weight ``lambda``; the object prior spectrum is
        ``S_k = sigma^2 / lambda``. Ignored when ``object_spectrum`` is given.
    object_spectrum : Tensor, optional
        Explicit object power spectrum ``S_k`` (same shape as ``otf``),
        for ``tikhonov_spectral`` priors.
    data_fft : Tensor, optional
        Precomputed ``torch.fft.fftn(data, norm="ortho")``; pass this when
        scoring many candidate systems against the same data.
    exclude_dc : bool
        Drop the ``k = 0`` term. Useful on real data with a large constant
        background that the object prior does not model.

    Returns
    -------
    Tensor
        Scalar negative log evidence (sums over all frequencies).
    """
    if data_fft is None:
        if data is None:
            raise ValueError("provide either 'data' or 'data_fft'")
        data_fft = torch.fft.fftn(data, norm="ortho")
    if otf is None:
        raise ValueError("'otf' is required")

    sigma2 = torch.as_tensor(noise_sigma, dtype=torch.float64) ** 2
    power = torch.abs(data_fft) ** 2
    h2 = torch.abs(otf) ** 2
    if object_spectrum is not None:
        s_k = object_spectrum
    else:
        s_k = sigma2 / torch.as_tensor(regularization_strength, dtype=torch.float64)
    v_k = h2 * s_k + sigma2

    terms = power / (2.0 * v_k) + 0.5 * torch.log(2.0 * math.pi * v_k)
    if exclude_dc:
        flat = terms.reshape(-1)
        return flat[1:].sum()
    return terms.sum()


def calibration_penalty(
    values: dict[str, float | Tensor],
    nominals: dict[str, float],
    tolerances: dict[str, float | str],
) -> Tensor:
    """The calibration term ``C = sum_i (h_i - h_i0)^2 / (2 tau_i^2)``.

    Always added to the chosen score; the tolerances are its weights
    (design document section 4.4, "Score"). Parameters with tolerance
    ``"free"`` contribute nothing.

    Parameters
    ----------
    values : dict[str, float or Tensor]
        Current parameter values by name.
    nominals : dict[str, float]
        Prior centers ``h_i0`` by name.
    tolerances : dict[str, float or "free"]
        Gaussian prior standard deviations ``tau_i`` by name, or ``"free"``.

    Returns
    -------
    Tensor
        Scalar penalty.
    """
    total = torch.zeros((), dtype=torch.float64)
    for name, value in values.items():
        tau = tolerances.get(name, "free")
        if tau == "free":
            continue
        nominal = nominals[name]
        residual = torch.as_tensor(value, dtype=torch.float64) - nominal
        total = total + residual**2 / (2.0 * float(tau) ** 2)
    return total

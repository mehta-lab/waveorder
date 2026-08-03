"""Richardson-Lucy (RL) and Gradient-Consensus (RLGC) inference core.

This module provides an operator-agnostic, PyTorch implementation of
Richardson-Lucy deconvolution and Andrew G. York's "Gradient Consensus"
variant. It works with any linear forward operator ``H`` and its adjoint
``H_T``, so it is not specific to deconvolution: any Poisson-noisy linear
measurement (blurring, projection, binning, ...) can be inverted with it.

Richardson-Lucy iteratively maximizes the Poisson log-likelihood of the
measurement. It sharpens well, but over many iterations it overfits the
noise, producing the characteristic "starry night" of spurious bright
speckles. Gradient Consensus resists this: at each iteration it splits the
photons into two random halves and only updates voxels where both halves
agree on the update direction, freezing voxels where the data disagrees
with itself.

The scaled-gradient step, Poisson noise model, and log-likelihood are
adapted from Andrew G. York's Gradient Consensus demo,
`doi.org/10.5281/zenodo.10278918 <https://doi.org/10.5281/zenodo.10278918>`_.
"""

from typing import Callable, Literal, Optional

import torch
from torch import Tensor

_EPS = 1e-12


def clip(x: Tensor, eps: float = _EPS) -> Tensor:
    """Clamp nonpositive entries up to a small positive number.

    FFT-based operators return small negative values from numerical error;
    the Poisson model requires strictly positive rates, so we floor them.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float, optional
        Floor value, by default ``1e-12``.

    Returns
    -------
    torch.Tensor
        ``x`` with entries below ``eps`` replaced by ``eps``.
    """
    return torch.clamp(x, min=eps)


def poisson_log_likelihood(expected_counts: Tensor, measured_counts: Tensor) -> Tensor:
    """Poisson log-likelihood of a measurement given expected rates.

    Parameters
    ----------
    expected_counts : torch.Tensor
        Expected (mean) counts per pixel, i.e. ``H(estimate)``.
    measured_counts : torch.Tensor
        Measured counts per pixel.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood ``sum(m * log(e) - e - lgamma(1 + m))``.
    """
    e = clip(expected_counts)
    m = measured_counts
    return torch.sum(m * torch.log(e) - e - torch.lgamma(1 + m))


def _coinflip(counts: Tensor, probability: float, generator: Optional[torch.Generator]) -> Tensor:
    """Binomially thin ``counts``, keeping each event with ``probability``.

    Equivalent to placing a beam splitter in the detection path: each photon
    independently lands in the "heads" arm with the given probability.
    """
    counts = torch.round(clip(counts, 0.0))
    probs = torch.full_like(counts, probability)
    return torch.binomial(counts, probs, generator=generator)


def scaled_gradient_step(
    estimate: Tensor,
    measured: Tensor,
    forward: Callable[[Tensor], Tensor],
    transpose: Callable[[Tensor], Tensor],
    transpose_ones: Tensor,
    *,
    method: Literal["RL", "RLGC"] = "RL",
    background: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """One multiplicative RL (or RLGC) update of ``estimate``.

    The update is ``estimate + gradient * step_size`` where ``gradient`` is
    the gradient of the Poisson log-likelihood and ``step_size`` is the
    Richardson-Lucy step that guarantees the estimate stays nonnegative.

    Parameters
    ----------
    estimate : torch.Tensor
        Current object estimate.
    measured : torch.Tensor
        Measured photon counts.
    forward : callable
        Forward operator ``H`` (object space -> measurement space).
    transpose : callable
        Adjoint operator ``H_T`` (measurement space -> object space).
    transpose_ones : torch.Tensor
        Precomputed ``H_T(1)``, the Richardson-Lucy step-size normalizer.
    method : {"RL", "RLGC"}, optional
        ``"RL"`` maximizes the Poisson likelihood. ``"RLGC"`` additionally
        freezes voxels where two random halves of the photons disagree on
        the update direction over the ``H_T(H(.))`` crosstalk neighborhood.
        By default ``"RL"``.
    background : float, optional
        Constant additive background (dark counts / offset) folded into the
        expected rates, by default ``0.0``.
    generator : torch.Generator, optional
        RNG for the RLGC coin flip, for reproducibility. Unused for RL.

    Returns
    -------
    updated_estimate : torch.Tensor
        The updated estimate.
    step_size : torch.Tensor
        The per-voxel step size actually used (zeros mark frozen voxels for
        RLGC); useful as a convergence diagnostic.
    """
    rates = clip(forward(estimate) + background)
    gradient = transpose(measured / rates - 1.0)
    step_size = estimate / transpose_ones
    if method == "RLGC":
        heads = _coinflip(measured, 0.5, generator)
        heads_gradient = transpose(heads / rates - 0.5)
        tails_gradient = gradient - heads_gradient  # faster than a second H_T call
        # Crosstalk neighborhood H_T(H(.)) defines which voxels touch the
        # same detector pixels; a nonpositive local dot product means the
        # two photon halves (locally) disagree, so we freeze those voxels.
        local_dot_product = transpose(forward(heads_gradient * tails_gradient))
        step_size = torch.where(local_dot_product <= 0, torch.zeros_like(step_size), step_size)
    return estimate + gradient * step_size, step_size


def richardson_lucy(
    measured: Tensor,
    forward: Callable[[Tensor], Tensor],
    transpose: Callable[[Tensor], Tensor],
    *,
    num_iterations: int,
    method: Literal["RL", "RLGC"] = "RL",
    background: float = 0.0,
    stopping_tolerance: Optional[float] = None,
    guess: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Reconstruct an object by iterating :func:`scaled_gradient_step`.

    Parameters
    ----------
    measured : torch.Tensor
        Measured photon counts.
    forward : callable
        Forward operator ``H``.
    transpose : callable
        Adjoint operator ``H_T``.
    num_iterations : int
        Maximum number of update steps.
    method : {"RL", "RLGC"}, optional
        Update rule, by default ``"RL"``. See :func:`scaled_gradient_step`.
    background : float, optional
        Constant additive background folded into the expected rates, by
        default ``0.0``.
    stopping_tolerance : float, optional
        If set, stop early once the relative change of the estimate,
        ``||new - old|| / ||old||``, falls below this value. RLGC also stops
        automatically once every voxel is frozen. By default ``None`` (run
        all iterations).
    guess : torch.Tensor, optional
        Initial estimate. Defaults to an array of ones, which is smoother
        than the noisy measurement and avoids baking noise into the result.
    generator : torch.Generator, optional
        RNG for the RLGC coin flip, for reproducibility.

    Returns
    -------
    torch.Tensor
        The final object estimate, same shape as ``transpose(measured)``.
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    if method not in ("RL", "RLGC"):
        raise ValueError(f"method must be 'RL' or 'RLGC', got {method!r}")

    transpose_ones = clip(transpose(torch.ones_like(measured)))
    if guess is None:
        estimate = torch.ones_like(transpose_ones)
    else:
        estimate = clip(guess.clone())

    for _ in range(num_iterations):
        updated, step_size = scaled_gradient_step(
            estimate,
            measured,
            forward,
            transpose,
            transpose_ones,
            method=method,
            background=background,
            generator=generator,
        )
        updated = clip(updated)
        if method == "RLGC" and torch.all(step_size == 0):
            estimate = updated
            break
        if stopping_tolerance is not None:
            change = torch.linalg.vector_norm(updated - estimate)
            scale = torch.linalg.vector_norm(estimate)
            estimate = updated
            if scale > 0 and (change / scale) < stopping_tolerance:
                break
        else:
            estimate = updated

    return estimate

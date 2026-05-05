"""Blends — weight kernels + reduction for tile-overlap accumulation.

Modeled after flox ``Aggregation``: a ``Blend`` carries four callables
describing the reduction's full lifecycle — ``init`` turns a contributing
tile into intermediate state, ``combine`` is the associative pairwise
merge used by the tree-reducer, and ``finalize`` turns accumulated state
into the blended output. ``weight_kernel`` stays separate so any spatial
weighting can be paired with any reduction.

Built-in blends:
    * ``uniform_mean`` — arithmetic mean (all contributors weighted 1).
    * ``gaussian_mean(sigma_fraction)`` — separable Gaussian weighting.
    * ``max_blend`` / ``min_blend`` — pixelwise extrema.
    * ``clipped_mean(lo, hi, base=…)`` — wrap any mean-style blend with clipping.

Custom blends: instantiate ``Blend`` directly. Associativity contract:
``combine(combine(a, b), c) == combine(a, combine(b, c))`` must hold,
else tree-reduction changes the result. Non-associative reductions
(weighted median, mode) need to carry full state through ``combine``
and reduce at ``finalize`` — documented as O(N) memory.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache, reduce

import numpy as np

Intermediates = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Blend:
    """Flox-style blend: weight kernel + associative tree reduction.

    ``init`` receives values + weights already cropped to the cell region
    and returns a tuple of ND arrays (the intermediate state). ``combine``
    pairwise-merges two intermediates (must be associative). ``finalize``
    reduces accumulated state to the blended output.
    """

    name: str
    weight_kernel: Callable[[tuple[int, ...]], np.ndarray]
    init: Callable[[np.ndarray, np.ndarray], Intermediates]
    combine: Callable[[Intermediates, Intermediates], Intermediates]
    finalize: Callable[[Intermediates], np.ndarray]
    fill_value: float = field(default=float("nan"))


def _broadcast_weights(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Promote spatial weights to full leading+spatial shape."""
    extra = values.ndim - weights.ndim
    if extra < 0:
        raise ValueError(f"weights have more dims ({weights.ndim}) than values ({values.ndim})")
    if extra:
        weights = weights.reshape((1,) * extra + weights.shape)
    return np.broadcast_to(weights, values.shape)


def _weighted_mean_init(values: np.ndarray, weights: np.ndarray) -> Intermediates:
    v = values.astype(np.float64, copy=False)
    w = _broadcast_weights(v, weights.astype(np.float64, copy=False))
    # Mask values at zero-weight locations (the engine pads contributions
    # outside the intersection with fill_value=NaN; NaN * 0 = NaN in IEEE 754,
    # so we explicitly zero those before multiplying to keep the accumulator clean).
    masked_v = np.where(w > 0, v, 0.0)
    return (masked_v * w, np.asarray(w, dtype=np.float64).copy())


def _sum_combine(a: Intermediates, b: Intermediates) -> Intermediates:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _weighted_mean_finalize(t: Intermediates) -> np.ndarray:
    s, w = t
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, s / w, np.nan)


@lru_cache(maxsize=16)
def _ones_kernel(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=np.float64)


@lru_cache(maxsize=16)
def _gaussian_kernel(shape: tuple[int, ...], sigma_fraction: float) -> np.ndarray:
    kernels: list[np.ndarray] = []
    for size in shape:
        if size <= 1:
            kernels.append(np.ones(size, dtype=np.float64))
            continue
        sigma = max(1.0, size * sigma_fraction)
        x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
        k = np.exp(-(x**2) / (2.0 * sigma**2))
        kernels.append(k / k.max())
    return reduce(np.multiply.outer, kernels)


def uniform_mean() -> Blend:
    """Arithmetic mean across contributors."""
    return Blend(
        name="uniform_mean",
        weight_kernel=_ones_kernel,
        init=_weighted_mean_init,
        combine=_sum_combine,
        finalize=_weighted_mean_finalize,
    )


def gaussian_mean(sigma_fraction: float = 1.0 / 8.0) -> Blend:
    """Weighted mean with separable Gaussian kernel (peak at tile center).

    ``sigma_fraction`` is σ as a fraction of each axis length; 1/8 matches
    patchly's default and suppresses edge contributions without killing them.
    """

    def _kernel(shape: tuple[int, ...]) -> np.ndarray:
        return _gaussian_kernel(shape, sigma_fraction)

    return Blend(
        name=f"gaussian_mean(sigma_fraction={sigma_fraction:.4g})",
        weight_kernel=_kernel,
        init=_weighted_mean_init,
        combine=_sum_combine,
        finalize=_weighted_mean_finalize,
    )


def _extremum_init(values: np.ndarray, weights: np.ndarray) -> Intermediates:  # noqa: ARG001
    return (values,)


def _max_combine(a: Intermediates, b: Intermediates) -> Intermediates:
    return (np.maximum(a[0], b[0]),)


def _min_combine(a: Intermediates, b: Intermediates) -> Intermediates:
    return (np.minimum(a[0], b[0]),)


def _identity_finalize(t: Intermediates) -> np.ndarray:
    return t[0]


def max_blend() -> Blend:
    """Pixelwise max across contributors. Weights ignored."""
    return Blend(
        name="max",
        weight_kernel=_ones_kernel,
        init=_extremum_init,
        combine=_max_combine,
        finalize=_identity_finalize,
        fill_value=-np.inf,
    )


def min_blend() -> Blend:
    """Pixelwise min across contributors. Weights ignored."""
    return Blend(
        name="min",
        weight_kernel=_ones_kernel,
        init=_extremum_init,
        combine=_min_combine,
        finalize=_identity_finalize,
        fill_value=np.inf,
    )


def clipped_mean(lo: float, hi: float, *, base: Blend | None = None) -> Blend:
    """Wrap a mean-style blend with finalize-time clipping."""
    b = base or uniform_mean()

    def _finalize(t: Intermediates) -> np.ndarray:
        return np.clip(b.finalize(t), lo, hi)

    return Blend(
        name=f"clipped_mean({lo},{hi},base={b.name})",
        weight_kernel=b.weight_kernel,
        init=b.init,
        combine=b.combine,
        finalize=_finalize,
        fill_value=b.fill_value,
    )

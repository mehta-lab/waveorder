"""Pure-python tests for the Blend dataclass + built-in factories.

Verifies the associativity contract (combine(combine(a,b),c) ==
combine(a,combine(b,c))) and that each built-in produces the documented
reduction over a small fixed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from waveorder.tile_stitch.blend import (
    Blend,
    clipped_mean,
    gaussian_mean,
    max_blend,
    min_blend,
    uniform_mean,
)


def _reduce(blend: Blend, contributors: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Drive a blend through its full lifecycle on a list of contributors."""
    intermediates = [blend.init(v, w) for v, w in zip(contributors, weights, strict=True)]
    acc = intermediates[0]
    for nxt in intermediates[1:]:
        acc = blend.combine(acc, nxt)
    return blend.finalize(acc)


# --- uniform_mean ---


def test_uniform_mean_arithmetic_mean_three_contributors():
    blend = uniform_mean()
    a, b, c = np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])
    w = np.array([1.0, 1.0])
    out = _reduce(blend, [a, b, c], [w, w, w])
    np.testing.assert_allclose(out, [3.0, 4.0])


def test_uniform_mean_associativity():
    blend = uniform_mean()
    a, b, c = np.array([1.5, 2.5]), np.array([3.5, 4.5]), np.array([5.5, 6.5])
    w = np.array([1.0, 1.0])
    ia, ib, ic = blend.init(a, w), blend.init(b, w), blend.init(c, w)
    left = blend.combine(blend.combine(ia, ib), ic)
    right = blend.combine(ia, blend.combine(ib, ic))
    np.testing.assert_allclose(blend.finalize(left), blend.finalize(right))


def test_uniform_mean_zero_weight_yields_nan():
    blend = uniform_mean()
    out = _reduce(blend, [np.array([1.0])], [np.array([0.0])])
    assert np.isnan(out[0])


# --- gaussian_mean ---


def test_gaussian_mean_kernel_peaks_at_center():
    """Gaussian kernel is normalized so peak == 1.0; off-center values are smaller."""
    blend = gaussian_mean(sigma_fraction=0.25)
    kernel = blend.weight_kernel((9, 9))
    assert kernel.shape == (9, 9)
    center = kernel[4, 4]
    edge = kernel[0, 0]
    assert center == pytest.approx(1.0)
    assert edge < center


def test_gaussian_mean_single_axis_kernel_is_separable():
    """Separable kernel: 1D outer product == 2D evaluation on the same shape."""
    blend = gaussian_mean(sigma_fraction=0.25)
    k_1d = blend.weight_kernel((9,))
    k_2d = blend.weight_kernel((9, 9))
    np.testing.assert_allclose(k_2d, np.multiply.outer(k_1d, k_1d))


def test_gaussian_mean_name_carries_sigma():
    """sigma_fraction is part of the blend identity (used by tokenizers)."""
    b1 = gaussian_mean(sigma_fraction=0.1)
    b2 = gaussian_mean(sigma_fraction=0.2)
    assert b1.name != b2.name
    assert "sigma_fraction=0.1" in b1.name


# --- max_blend / min_blend ---


def test_max_blend_pixelwise_max():
    blend = max_blend()
    a, b, c = np.array([1.0, 5.0, 3.0]), np.array([2.0, 4.0, 7.0]), np.array([0.0, 6.0, 1.0])
    w = np.array([1.0, 1.0, 1.0])
    out = _reduce(blend, [a, b, c], [w, w, w])
    np.testing.assert_array_equal(out, [2.0, 6.0, 7.0])


def test_min_blend_pixelwise_min():
    blend = min_blend()
    a, b, c = np.array([1.0, 5.0, 3.0]), np.array([2.0, 4.0, 7.0]), np.array([0.0, 6.0, 1.0])
    w = np.array([1.0, 1.0, 1.0])
    out = _reduce(blend, [a, b, c], [w, w, w])
    np.testing.assert_array_equal(out, [0.0, 4.0, 1.0])


def test_max_blend_associativity():
    blend = max_blend()
    a, b, c = np.array([1.0, 5.0]), np.array([3.0, 2.0]), np.array([4.0, 4.0])
    w = np.array([1.0, 1.0])
    ia, ib, ic = blend.init(a, w), blend.init(b, w), blend.init(c, w)
    left = blend.finalize(blend.combine(blend.combine(ia, ib), ic))
    right = blend.finalize(blend.combine(ia, blend.combine(ib, ic)))
    np.testing.assert_array_equal(left, right)


def test_max_blend_fill_value_is_neg_inf():
    assert max_blend().fill_value == -np.inf


def test_min_blend_fill_value_is_pos_inf():
    assert min_blend().fill_value == np.inf


# --- clipped_mean ---


def test_clipped_mean_clips_at_finalize():
    blend = clipped_mean(lo=0.0, hi=10.0)
    out = _reduce(blend, [np.array([15.0, -5.0])], [np.array([1.0, 1.0])])
    np.testing.assert_array_equal(out, [10.0, 0.0])


def test_clipped_mean_wraps_gaussian_base():
    base = gaussian_mean(sigma_fraction=0.2)
    blend = clipped_mean(lo=-1.0, hi=1.0, base=base)
    assert "gaussian_mean" in blend.name
    assert blend.weight_kernel is base.weight_kernel


# --- Custom Blend ---


def test_custom_blend_dataclass_is_frozen():
    blend = uniform_mean()
    with pytest.raises(Exception):
        blend.name = "renamed"  # type: ignore[misc]

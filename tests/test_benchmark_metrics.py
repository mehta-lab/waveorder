"""Tests for benchmarks.metrics."""

import torch

from benchmarks.metrics import (
    comparison_metrics,
    compute_metrics,
    distribution_stats,
)
from benchmarks.utils import render_histogram

# Shared optical parameters for tests
_OPTICS = {"NA_det": 1.2, "wavelength": 0.532, "pixel_size": 0.1}


def _random_volume(shape=(16, 32, 32), seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen)


class TestDistributionStats:
    def test_keys(self):
        stats = distribution_stats(_random_volume())
        assert {"max", "min", "mean", "histogram"} <= set(stats.keys())

    def test_histogram_bins(self):
        stats = distribution_stats(_random_volume(), n_bins=20)
        assert len(stats["histogram"]["counts"]) == 20
        assert len(stats["histogram"]["bin_edges"]) == 21

    def test_values_correct(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = distribution_stats(v)
        assert stats["max"] == 5.0
        assert stats["min"] == 1.0
        assert stats["mean"] == 3.0


class TestComparisonMetrics:
    def test_identical_volumes(self):
        v = _random_volume()
        m = comparison_metrics(v, v)
        assert m["mse"] == 0.0
        assert abs(m["ssim"] - 1.0) < 1e-5
        assert abs(m["correlation"] - 1.0) < 1e-5

    def test_different_volumes(self):
        a = _random_volume(seed=0)
        b = _random_volume(seed=1)
        m = comparison_metrics(a, b)
        assert m["mse"] > 0
        assert -1 <= m["ssim"] <= 1
        assert -1 <= m["correlation"] <= 1


class TestComputeMetrics:
    def test_image_quality_always_present(self):
        result = compute_metrics(_random_volume(), **_OPTICS)
        assert "image_quality" in result
        assert "max" in result["image_quality"]
        assert "midband_power" in result["image_quality"]
        assert "spectral_flatness" in result["image_quality"]
        assert "total_variation" in result["image_quality"]

    def test_midband_power_positive(self):
        result = compute_metrics(_random_volume(), **_OPTICS)
        assert result["image_quality"]["midband_power"] > 0

    def test_with_reference(self):
        v = _random_volume()
        result = compute_metrics(v, **_OPTICS, reference=v)
        assert "with_reference" in result
        assert "ssim" in result["with_reference"]

    def test_with_reference_absent_without_reference(self):
        result = compute_metrics(_random_volume(), **_OPTICS)
        assert "with_reference" not in result

    def test_with_phantom(self):
        v = _random_volume()
        result = compute_metrics(v, **_OPTICS, phantom=v)
        assert "with_phantom" in result
        assert "mse" in result["with_phantom"]

    def test_with_phantom_absent_without_phantom(self):
        result = compute_metrics(_random_volume(), **_OPTICS)
        assert "with_phantom" not in result


class TestRenderHistogram:
    def test_renders_single_line(self):
        stats = distribution_stats(_random_volume())
        output = render_histogram(stats["histogram"])
        assert isinstance(output, str)
        assert "\n" not in output

    def test_empty_histogram(self):
        hist = {"bin_edges": [0, 1], "counts": [0]}
        assert render_histogram(hist) == "(empty)"

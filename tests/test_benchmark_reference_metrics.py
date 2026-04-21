"""Tests for reference_metrics schema and runtime bound check."""

import pytest

from benchmarks.config import CaseConfig, ReferenceMetric
from benchmarks.runner import check_reference_metrics


class TestReferenceMetricSchema:
    def test_min_only(self):
        rm = ReferenceMetric(min=0.7)
        assert rm.min == 0.7
        assert rm.max is None

    def test_max_only(self):
        rm = ReferenceMetric(max=1.0)
        assert rm.max == 1.0
        assert rm.min is None

    def test_both(self):
        rm = ReferenceMetric(min=0.0, max=1.0)
        assert rm.min == 0.0 and rm.max == 1.0

    def test_neither_rejected(self):
        with pytest.raises(Exception, match="at least one"):
            ReferenceMetric()

    def test_min_gt_max_rejected(self):
        with pytest.raises(Exception, match="must be <="):
            ReferenceMetric(min=1.0, max=0.5)

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            ReferenceMetric(min=0.0, foo=1)


class TestCaseConfigWithReferenceMetrics:
    def test_parses(self):
        case = CaseConfig(
            type="hpc",
            input="/a.zarr",
            position="A/1/000000",
            reference_metrics={
                "with_phantom.ssim": {"min": 0.7},
                "image_quality.midband_power": {"min": 150.0},
            },
        )
        assert case.reference_metrics["with_phantom.ssim"].min == 0.7
        assert case.reference_metrics["image_quality.midband_power"].min == 150.0

    def test_defaults_to_none(self):
        case = CaseConfig(type="synthetic")
        assert case.reference_metrics is None


# Sample metrics dict shaped like what compute_metrics returns
_METRICS = {
    "image_quality": {"midband_power": 200.0, "spectral_flatness": 0.3},
    "with_phantom": {"ssim": 0.85, "mse": 0.001, "correlation": 0.99},
}


class TestCheckReferenceMetrics:
    def test_returns_none_when_no_refs(self):
        assert check_reference_metrics(_METRICS, None) is None
        assert check_reference_metrics(_METRICS, {}) is None

    def test_min_pass(self):
        refs = {"with_phantom.ssim": ReferenceMetric(min=0.7)}
        result = check_reference_metrics(_METRICS, refs)
        assert result["all_pass"] is True
        entry = result["per_metric"]["with_phantom.ssim"]
        assert entry["value"] == 0.85
        assert entry["min"] == 0.7
        assert entry["pass"] is True

    def test_min_fail(self):
        refs = {"with_phantom.ssim": ReferenceMetric(min=0.95)}
        result = check_reference_metrics(_METRICS, refs)
        assert result["all_pass"] is False
        assert result["per_metric"]["with_phantom.ssim"]["pass"] is False

    def test_max_pass_and_fail(self):
        refs_pass = {"with_phantom.mse": ReferenceMetric(max=0.01)}
        refs_fail = {"with_phantom.mse": ReferenceMetric(max=0.0001)}
        assert check_reference_metrics(_METRICS, refs_pass)["all_pass"] is True
        assert check_reference_metrics(_METRICS, refs_fail)["all_pass"] is False

    def test_range_both_bounds(self):
        refs = {"with_phantom.ssim": ReferenceMetric(min=0.8, max=0.9)}
        assert check_reference_metrics(_METRICS, refs)["all_pass"] is True

    def test_multi_metrics_one_fails(self):
        refs = {
            "with_phantom.ssim": ReferenceMetric(min=0.7),
            "image_quality.midband_power": ReferenceMetric(min=500.0),
        }
        result = check_reference_metrics(_METRICS, refs)
        assert result["all_pass"] is False
        assert result["per_metric"]["with_phantom.ssim"]["pass"] is True
        assert result["per_metric"]["image_quality.midband_power"]["pass"] is False

    def test_edge_equal_min_passes(self):
        refs = {"with_phantom.ssim": ReferenceMetric(min=0.85)}
        assert check_reference_metrics(_METRICS, refs)["all_pass"] is True

    def test_missing_path_raises(self):
        refs = {"with_phantom.not_a_field": ReferenceMetric(min=0.0)}
        with pytest.raises(ValueError, match="not_a_field"):
            check_reference_metrics(_METRICS, refs)

    def test_missing_top_level_group_raises(self):
        refs = {"with_reference.ssim": ReferenceMetric(min=0.0)}
        with pytest.raises(ValueError):
            check_reference_metrics(_METRICS, refs)

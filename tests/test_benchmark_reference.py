"""Tests for the unified ``reference`` min/max regression check.

Covers both metric paths (``with_phantom.ssim``, ``image_quality.midband_power``)
and parameter paths (``parameter.z_focus_offset``) through the same schema.
"""

from pathlib import Path

import pytest
import yaml

from benchmarks.config import CaseConfig, ReferenceBound
from benchmarks.runner import check_reference


class TestReferenceBoundSchema:
    def test_min_only(self):
        rb = ReferenceBound(min=0.7)
        assert rb.min == 0.7 and rb.max is None

    def test_max_only(self):
        rb = ReferenceBound(max=1.0)
        assert rb.max == 1.0 and rb.min is None

    def test_both(self):
        rb = ReferenceBound(min=-0.25, max=0.25)
        assert rb.min == -0.25 and rb.max == 0.25

    def test_neither_rejected(self):
        with pytest.raises(Exception, match="at least one"):
            ReferenceBound()

    def test_min_gt_max_rejected(self):
        with pytest.raises(Exception, match="must be <="):
            ReferenceBound(min=1.0, max=0.5)

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            ReferenceBound(min=0.0, foo=1)


class TestCaseConfigReference:
    def test_parses_mixed_keys(self):
        case = CaseConfig(
            type="hpc",
            input="/a.zarr",
            position="A/1/000000",
            reference={
                "with_phantom.ssim": {"min": 0.7},
                "parameter.z_focus_offset": {"min": -0.25, "max": 0.25},
            },
        )
        assert case.reference["with_phantom.ssim"].min == 0.7
        assert case.reference["parameter.z_focus_offset"].max == 0.25

    def test_defaults_to_none(self):
        assert CaseConfig(type="synthetic").reference is None


_METRICS = {
    "image_quality": {"midband_power": 200.0, "spectral_flatness": 0.3},
    "with_phantom": {"ssim": 0.85, "mse": 0.001},
}


def _write_optimized_config(case_dir: Path, tf_values: dict):
    cfg = {
        "reconstruction_dimension": 2,
        "input_channel_names": ["BF"],
        "phase": {
            "transfer_function": {
                "yx_pixel_size": 0.325,
                "z_pixel_size": 2.0,
                "wavelength_illumination": 0.45,
                "index_of_refraction_media": 1.0,
                "numerical_aperture_illumination": 0.4,
                "numerical_aperture_detection": 0.55,
                **tf_values,
            },
            "apply_inverse": {"regularization_strength": 0.001},
        },
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "config_optimized.yml").write_text(yaml.dump(cfg))


class TestMetricChecks:
    def test_returns_none(self, tmp_path):
        assert check_reference(tmp_path, _METRICS, None) is None
        assert check_reference(tmp_path, _METRICS, {}) is None

    def test_min_pass(self, tmp_path):
        refs = {"with_phantom.ssim": ReferenceBound(min=0.7)}
        result = check_reference(tmp_path, _METRICS, refs)
        assert result["all_pass"] is True
        assert result["per_ref"]["with_phantom.ssim"]["value"] == 0.85

    def test_min_fail(self, tmp_path):
        refs = {"with_phantom.ssim": ReferenceBound(min=0.95)}
        result = check_reference(tmp_path, _METRICS, refs)
        assert result["all_pass"] is False

    def test_max_bounds(self, tmp_path):
        good = {"with_phantom.mse": ReferenceBound(max=0.01)}
        bad = {"with_phantom.mse": ReferenceBound(max=0.0001)}
        assert check_reference(tmp_path, _METRICS, good)["all_pass"] is True
        assert check_reference(tmp_path, _METRICS, bad)["all_pass"] is False

    def test_two_sided_bounds(self, tmp_path):
        refs = {"with_phantom.ssim": ReferenceBound(min=0.8, max=0.9)}
        assert check_reference(tmp_path, _METRICS, refs)["all_pass"] is True

    def test_missing_path_raises(self, tmp_path):
        refs = {"with_phantom.not_a_field": ReferenceBound(min=0.0)}
        with pytest.raises(ValueError, match="not_a_field"):
            check_reference(tmp_path, _METRICS, refs)


class TestParameterChecks:
    def test_pass(self, tmp_path):
        _write_optimized_config(
            tmp_path,
            {"z_focus_offset": -0.05, "tilt_angle_zenith": 0.02, "tilt_angle_azimuth": 0.3},
        )
        refs = {
            "parameter.z_focus_offset": ReferenceBound(min=-0.25, max=0.25),
            "parameter.tilt_angle_zenith": ReferenceBound(min=-0.1, max=0.1),
            "parameter.tilt_angle_azimuth": ReferenceBound(min=-3.14, max=3.14),
        }
        result = check_reference(tmp_path, {}, refs)
        assert result["all_pass"] is True
        assert result["per_ref"]["parameter.z_focus_offset"]["value"] == pytest.approx(-0.05)

    def test_one_fails(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": -0.5})
        refs = {"parameter.z_focus_offset": ReferenceBound(min=-0.25, max=0.25)}
        result = check_reference(tmp_path, {}, refs)
        assert result["all_pass"] is False
        assert result["per_ref"]["parameter.z_focus_offset"]["pass"] is False

    def test_raises_when_no_optimized_config(self, tmp_path):
        refs = {"parameter.z_focus_offset": ReferenceBound(min=-0.25, max=0.25)}
        with pytest.raises(ValueError, match="not written"):
            check_reference(tmp_path, {}, refs)

    def test_unknown_parameter_raises(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": 0.0})
        refs = {"parameter.not_a_real_field": ReferenceBound(min=0.0, max=0.1)}
        with pytest.raises(ValueError, match="not_a_real_field"):
            check_reference(tmp_path, {}, refs)


class TestMixedChecks:
    def test_both_types_together(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": 0.0})
        refs = {
            "with_phantom.ssim": ReferenceBound(min=0.7),
            "parameter.z_focus_offset": ReferenceBound(min=-0.25, max=0.25),
        }
        result = check_reference(tmp_path, _METRICS, refs)
        assert result["all_pass"] is True
        assert set(result["per_ref"]) == {"with_phantom.ssim", "parameter.z_focus_offset"}

    def test_metric_fails_param_passes(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": 0.0})
        refs = {
            "with_phantom.ssim": ReferenceBound(min=0.95),
            "parameter.z_focus_offset": ReferenceBound(min=-0.25, max=0.25),
        }
        result = check_reference(tmp_path, _METRICS, refs)
        assert result["all_pass"] is False
        assert result["per_ref"]["with_phantom.ssim"]["pass"] is False
        assert result["per_ref"]["parameter.z_focus_offset"]["pass"] is True

"""Tests for reference_parameters schema and runtime drift check."""

from pathlib import Path

import pytest
import yaml

from benchmarks.config import CaseConfig, ReferenceParameter
from benchmarks.runner import check_reference_parameters


class TestReferenceParameterSchema:
    def test_valid(self):
        rp = ReferenceParameter(value=0.0, tolerance=0.25)
        assert rp.value == 0.0
        assert rp.tolerance == 0.25

    def test_negative_value_allowed(self):
        rp = ReferenceParameter(value=-0.5, tolerance=0.1)
        assert rp.value == -0.5

    def test_zero_tolerance_rejected(self):
        with pytest.raises(Exception):
            ReferenceParameter(value=0.0, tolerance=0.0)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(Exception):
            ReferenceParameter(value=0.0, tolerance=-0.1)

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            ReferenceParameter(value=0.0, tolerance=0.25, foo="bar")


class TestCaseConfigWithReferenceParameters:
    def test_parses(self):
        case = CaseConfig(
            type="hpc",
            input="/a/b.zarr",
            position="A/1/000000",
            config="cfg.yml",
            reference_parameters={
                "z_focus_offset": {"value": 0.0, "tolerance": 0.25},
                "tilt_angle_zenith": {"value": 0.0, "tolerance": 0.1},
            },
        )
        assert case.reference_parameters["z_focus_offset"].value == 0.0
        assert case.reference_parameters["tilt_angle_zenith"].tolerance == 0.1

    def test_defaults_to_none(self):
        case = CaseConfig(type="synthetic")
        assert case.reference_parameters is None


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


class TestCheckReferenceParameters:
    def test_returns_none_when_no_references(self, tmp_path):
        assert check_reference_parameters(tmp_path, None) is None
        assert check_reference_parameters(tmp_path, {}) is None

    def test_raises_when_no_optimized_config(self, tmp_path):
        refs = {"z_focus_offset": ReferenceParameter(value=0.0, tolerance=0.25)}
        with pytest.raises(ValueError, match="no optimization ran"):
            check_reference_parameters(tmp_path, refs)

    def test_all_pass(self, tmp_path):
        _write_optimized_config(
            tmp_path,
            {"z_focus_offset": -0.05, "tilt_angle_zenith": 0.02, "tilt_angle_azimuth": 0.3},
        )
        refs = {
            "z_focus_offset": ReferenceParameter(value=0.0, tolerance=0.25),
            "tilt_angle_zenith": ReferenceParameter(value=0.0, tolerance=0.1),
            "tilt_angle_azimuth": ReferenceParameter(value=0.0, tolerance=3.14),
        }
        result = check_reference_parameters(tmp_path, refs)
        assert result["all_pass"] is True
        assert result["per_param"]["z_focus_offset"]["drift"] == pytest.approx(0.05)
        assert all(p["pass"] for p in result["per_param"].values())

    def test_one_fails(self, tmp_path):
        _write_optimized_config(
            tmp_path,
            {"z_focus_offset": -0.5, "tilt_angle_zenith": 0.02, "tilt_angle_azimuth": 0.3},
        )
        refs = {
            "z_focus_offset": ReferenceParameter(value=0.0, tolerance=0.25),
            "tilt_angle_zenith": ReferenceParameter(value=0.0, tolerance=0.1),
        }
        result = check_reference_parameters(tmp_path, refs)
        assert result["all_pass"] is False
        assert result["per_param"]["z_focus_offset"]["pass"] is False
        assert result["per_param"]["tilt_angle_zenith"]["pass"] is True

    def test_exact_tolerance_is_pass(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": 0.25})
        refs = {"z_focus_offset": ReferenceParameter(value=0.0, tolerance=0.25)}
        result = check_reference_parameters(tmp_path, refs)
        assert result["per_param"]["z_focus_offset"]["pass"] is True

    def test_unknown_param_raises(self, tmp_path):
        _write_optimized_config(tmp_path, {"z_focus_offset": 0.0})
        refs = {"not_a_real_field": ReferenceParameter(value=0.0, tolerance=0.1)}
        with pytest.raises(ValueError, match="not_a_real_field"):
            check_reference_parameters(tmp_path, refs)

    def test_fluorescence_path(self, tmp_path):
        cfg = {
            "reconstruction_dimension": 2,
            "input_channel_names": ["GFP"],
            "fluorescence": {
                "transfer_function": {
                    "yx_pixel_size": 0.1,
                    "z_pixel_size": 0.25,
                    "wavelength_emission": 0.532,
                    "index_of_refraction_media": 1.3,
                    "numerical_aperture_detection": 1.2,
                    "z_focus_offset": 0.05,
                },
                "apply_inverse": {"regularization_strength": 0.001},
            },
        }
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config_optimized.yml").write_text(yaml.dump(cfg))
        refs = {"z_focus_offset": ReferenceParameter(value=0.0, tolerance=0.1)}
        result = check_reference_parameters(tmp_path, refs)
        assert result["all_pass"] is True

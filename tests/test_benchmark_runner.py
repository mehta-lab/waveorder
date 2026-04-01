"""Tests for benchmarks.runner and benchmarks.config."""

import shutil

import pytest
import yaml

from benchmarks.config import ExperimentConfig, PhantomConfig, load_experiment
from benchmarks.runner import _build_phantom, run_synthetic_case


class TestPhantomConfig:
    def test_valid(self):
        pc = PhantomConfig(function="single_bead", shape=(16, 32, 32))
        assert pc.function == "single_bead"
        assert pc.shape == (16, 32, 32)

    def test_extra_kwargs_allowed(self):
        pc = PhantomConfig(
            function="single_bead",
            shape=(16, 32, 32),
            bead_radius_um=2.0,
            refractive_index_diff=0.05,
        )
        assert pc.model_dump()["bead_radius_um"] == 2.0

    def test_invalid_function_rejected(self):
        with pytest.raises(Exception):
            PhantomConfig(function="invalid_phantom")


class TestExperimentConfig:
    def test_valid(self):
        exp = ExperimentConfig(
            name="test",
            cases={
                "case_a": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
                }
            },
        )
        assert exp.name == "test"
        assert exp.cases["case_a"].phantom.function == "single_bead"

    def test_base_phantom_inherited(self):
        exp = ExperimentConfig(
            name="test",
            base_phantom={"function": "single_bead", "shape": [16, 32, 32]},
            cases={"case_a": {"type": "synthetic"}, "case_b": {"type": "synthetic"}},
        )
        assert exp.cases["case_a"].phantom.function == "single_bead"
        assert exp.cases["case_b"].phantom.function == "single_bead"

    def test_case_phantom_not_overwritten_by_base(self):
        exp = ExperimentConfig(
            name="test",
            base_phantom={"function": "single_bead", "shape": [16, 32, 32]},
            cases={
                "custom": {
                    "type": "synthetic",
                    "phantom": {"function": "random_beads", "shape": [8, 16, 16], "n_beads": 3, "bead_radius_um": 0.3},
                }
            },
        )
        assert exp.cases["custom"].phantom.function == "random_beads"

    def test_base_config_inherited(self):
        exp = ExperimentConfig(
            name="test",
            base_config="configs/phase.yml",
            cases={"case_a": {"type": "synthetic", "phantom": {"function": "single_bead"}}},
        )
        assert exp.cases["case_a"].config == "configs/phase.yml"


class TestLoadExperiment:
    def test_from_yaml(self, tmp_path):
        exp_dict = {
            "name": "test",
            "cases": {
                "case_a": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
                }
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp_dict))
        exp = load_experiment(path)
        assert isinstance(exp, ExperimentConfig)
        assert exp.cases["case_a"].phantom.shape == (16, 32, 32)

    def test_overrides(self, tmp_path):
        base_config = {
            "reconstruction_dimension": 3,
            "phase": {"apply_inverse": {"regularization_strength": 0.01}},
        }
        config_path = tmp_path / "base.yml"
        config_path.write_text(yaml.dump(base_config))

        exp_dict = {
            "name": "test",
            "base_config": str(config_path),
            "cases": {
                "case_a": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead"},
                    "overrides": {"phase.apply_inverse.regularization_strength": 0.001},
                }
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp_dict))
        exp = load_experiment(path)

        from benchmarks.config import resolve_recon_config

        resolved = resolve_recon_config(exp.cases["case_a"], tmp_path)
        assert resolved["phase"]["apply_inverse"]["regularization_strength"] == 0.001


class TestBuildPhantom:
    def test_from_config(self):
        pc = PhantomConfig(function="single_bead", shape=(16, 32, 32), bead_radius_um=1.0)
        phantom = _build_phantom(pc)
        assert phantom.phase.shape == (16, 32, 32)

    def test_from_dict(self):
        phantom = _build_phantom({"function": "single_bead", "shape": [16, 32, 32]})
        assert phantom.phase.shape == (16, 32, 32)


class TestLoadRegressionExperiment:
    def test_regression_yml_valid(self):
        """Validate the committed regression.yml experiment."""
        from pathlib import Path

        regression_path = Path(__file__).parent.parent / "benchmarks" / "experiments" / "regression.yml"
        if not regression_path.exists():
            pytest.skip("regression.yml not found")
        exp = load_experiment(regression_path)
        assert exp.name == "regression"
        assert "phase_3d_beads" in exp.cases
        assert "fluor_3d_beads" in exp.cases


@pytest.mark.skipif(not shutil.which("wo"), reason="wo CLI not installed")
class TestRunSyntheticCase:
    """End-to-end tests that shell out to ``wo rec``."""

    def test_phase_end_to_end(self, tmp_path):
        phantom_config = PhantomConfig(
            function="single_bead",
            shape=(32, 64, 64),
            pixel_sizes=(0.25, 0.1, 0.1),
            bead_radius_um=2.0,
            refractive_index_diff=0.05,
        )
        recon_config = {
            "reconstruction_dimension": 3,
            "input_channel_names": ["Brightfield"],
            "phase": {
                "transfer_function": {
                    "yx_pixel_size": 0.1,
                    "z_pixel_size": 0.25,
                    "wavelength_illumination": 0.532,
                    "index_of_refraction_media": 1.3,
                    "numerical_aperture_illumination": 0.9,
                    "numerical_aperture_detection": 1.2,
                },
            },
        }
        case_dir = tmp_path / "phase_case"
        metrics = run_synthetic_case(phantom_config, recon_config, case_dir, modality="phase")

        assert "image_quality" in metrics
        assert "with_phantom" in metrics
        assert (case_dir / "timing.json").exists()
        assert (case_dir / "metrics.json").exists()
        assert (case_dir / "phantom_config.json").exists()
        assert (case_dir / "cli_command.sh").exists()
        assert (case_dir / "simulated.zarr").exists()
        assert (case_dir / "reconstruction.zarr").exists()

    def test_fluorescence_end_to_end(self, tmp_path):
        phantom_config = PhantomConfig(
            function="single_bead",
            shape=(32, 64, 64),
            pixel_sizes=(0.25, 0.1, 0.1),
            bead_radius_um=2.0,
            fluorescence_intensity=1.0,
        )
        recon_config = {
            "reconstruction_dimension": 3,
            "input_channel_names": ["GFP"],
            "fluorescence": {
                "transfer_function": {
                    "yx_pixel_size": 0.1,
                    "z_pixel_size": 0.25,
                    "wavelength_emission": 0.532,
                    "index_of_refraction_media": 1.3,
                    "numerical_aperture_detection": 1.2,
                },
            },
        }
        case_dir = tmp_path / "fluor_case"
        metrics = run_synthetic_case(phantom_config, recon_config, case_dir, modality="fluorescence")

        assert "image_quality" in metrics
        assert "with_phantom" in metrics
        assert (case_dir / "cli_command.sh").exists()

"""Tests for benchmarks.runner and benchmarks.config."""

import shutil

import pytest
import yaml

from benchmarks.config import load_experiment
from benchmarks.runner import _build_phantom, run_synthetic_case


class TestConfig:
    def test_load_experiment_basic(self, tmp_path):
        exp = {
            "name": "test",
            "cases": {
                "case_a": {
                    "type": "synthetic",
                    "phantom": {
                        "function": "single_bead",
                        "shape": [16, 32, 32],
                    },
                }
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp))
        loaded = load_experiment(path)
        assert loaded["name"] == "test"
        assert "case_a" in loaded["cases"]

    def test_base_phantom_inherited(self, tmp_path):
        exp = {
            "name": "test",
            "base_phantom": {
                "function": "single_bead",
                "shape": [16, 32, 32],
            },
            "cases": {
                "case_a": {"type": "synthetic"},
                "case_b": {"type": "synthetic"},
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp))
        loaded = load_experiment(path)
        assert loaded["cases"]["case_a"]["phantom"]["function"] == "single_bead"
        assert loaded["cases"]["case_b"]["phantom"]["function"] == "single_bead"

    def test_overrides_applied(self, tmp_path):
        base_config = {
            "reconstruction_dimension": 3,
            "phase": {
                "apply_inverse": {"regularization_strength": 0.01},
            },
        }
        config_path = tmp_path / "base.yml"
        config_path.write_text(yaml.dump(base_config))

        exp = {
            "name": "test",
            "base_config": str(config_path),
            "cases": {
                "case_a": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
                    "overrides": {
                        "phase.apply_inverse.regularization_strength": 0.001,
                    },
                }
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp))
        loaded = load_experiment(path)
        resolved = loaded["cases"]["case_a"]["_resolved_config"]
        assert resolved["phase"]["apply_inverse"]["regularization_strength"] == 0.001


class TestBuildPhantom:
    def test_single_bead(self):
        phantom = _build_phantom(
            {
                "function": "single_bead",
                "shape": [16, 32, 32],
                "bead_radius_um": 1.0,
            }
        )
        assert phantom.phase.shape == (16, 32, 32)

    def test_random_beads(self):
        phantom = _build_phantom(
            {
                "function": "random_beads",
                "shape": [16, 32, 32],
                "n_beads": 3,
                "bead_radius_um": 0.3,
                "seed": 42,
            }
        )
        assert phantom.phase.shape == (16, 32, 32)


@pytest.mark.skipif(not shutil.which("wo"), reason="wo CLI not installed")
class TestRunSyntheticCase:
    """End-to-end tests that shell out to ``wo rec``."""

    def test_phase_end_to_end(self, tmp_path):
        phantom_config = {
            "function": "single_bead",
            "shape": [32, 64, 64],
            "pixel_sizes": [0.25, 0.1, 0.1],
            "bead_radius_um": 2.0,
            "refractive_index_diff": 0.05,
        }
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
        phantom_config = {
            "function": "single_bead",
            "shape": [32, 64, 64],
            "pixel_sizes": [0.25, 0.1, 0.1],
            "bead_radius_um": 2.0,
            "fluorescence_intensity": 1.0,
        }
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

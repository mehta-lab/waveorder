"""3.x -> 4.0 migration: deterministic, lossless, and round-trippable."""

from pathlib import Path

import pytest
import yaml

from waveorder.cli.settings import ReconstructionSettings
from waveorder.config import (
    MigrationError,
    detect_version,
    migrate_3x_dict,
    v4_to_legacy,
    v4_to_yaml_text,
)
from waveorder.config.schema import V4Config

REPO_ROOT = Path(__file__).parent.parent.parent

# Every 3.x phase / fluorescence configuration in the repository.
MIGRATABLE_CONFIG_PATHS = sorted(
    [
        *(REPO_ROOT / "docs/examples/cli/configs").glob("*.yml"),
        *(REPO_ROOT / "benchmarks/configs").glob("*.yml"),
        *(REPO_ROOT / "docs/examples/optimization").glob("*.yml"),
    ]
)
MIGRATABLE_CONFIG_PATHS = [
    p for p in MIGRATABLE_CONFIG_PATHS if "birefringence" not in p.name and "tile-stitch" not in p.name
]

BIREFRINGENCE_CONFIG_PATHS = sorted((REPO_ROOT / "docs/examples/cli/configs").glob("birefringence*.yml"))


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_version_detection():
    assert detect_version({}) == 3
    assert detect_version({"input_channel_names": ["BF"]}) == 3
    assert detect_version({"version": 4}) == 4


@pytest.mark.parametrize("config_path", MIGRATABLE_CONFIG_PATHS, ids=lambda p: p.name)
def test_migration_round_trip(config_path):
    """migrate(3.x) -> legacy reproduces the directly-parsed settings field for field."""
    raw = _load(config_path)
    direct = ReconstructionSettings(**raw)
    migrated = migrate_3x_dict(raw)
    round_tripped = v4_to_legacy(migrated)
    assert round_tripped.model_dump() == direct.model_dump()


@pytest.mark.parametrize("config_path", MIGRATABLE_CONFIG_PATHS, ids=lambda p: p.name)
def test_migration_is_deterministic(config_path):
    raw = _load(config_path)
    assert migrate_3x_dict(raw).model_dump() == migrate_3x_dict(raw).model_dump()


@pytest.mark.parametrize("config_path", MIGRATABLE_CONFIG_PATHS, ids=lambda p: p.name)
def test_migrated_yaml_reparses(config_path):
    """The written 4.0 YAML parses back to the same configuration."""
    migrated = migrate_3x_dict(_load(config_path))
    reparsed = V4Config(**yaml.safe_load(v4_to_yaml_text(migrated)))
    assert reparsed.model_dump() == migrated.model_dump()


@pytest.mark.parametrize("config_path", BIREFRINGENCE_CONFIG_PATHS, ids=lambda p: p.name)
def test_birefringence_not_migratable(config_path):
    with pytest.raises(MigrationError):
        migrate_3x_dict(_load(config_path))


def test_block_regrouping():
    """Spot-check the field map of design document section 2."""
    raw = {
        "input_channel_names": ["BF"],
        "time_indices": [0, 5],
        "reconstruction_dimension": 2,
        "phase": {
            "transfer_function": {
                "yx_pixel_size": 0.325,
                "z_pixel_size": 2.0,
                "z_padding": 3,
                "wavelength_illumination": 0.45,
                "index_of_refraction_media": 1.0,
                "numerical_aperture_illumination": 0.4,
                "numerical_aperture_detection": 0.55,
                "invert_phase_contrast": True,
            },
            "apply_inverse": {
                "reconstruction_algorithm": "Tikhonov",
                "regularization_strength": 0.001,
            },
        },
    }
    cfg = migrate_3x_dict(raw)
    assert cfg.version == 4
    assert cfg.data.input_channel_names == ["BF"]
    assert cfg.data.time_indices == [0, 5]
    assert cfg.data.pixel_size.yx.y == 0.325
    assert cfg.data.pixel_size.z == 2.0
    assert cfg.object.dimension == 2
    assert cfg.solver.padding.z == 3
    assert cfg.imaging_system.contrast == "phase"
    assert cfg.imaging_system.index_of_refraction_media == 1.0
    assert cfg.imaging_system.detection.numerical_aperture == 0.55
    assert cfg.imaging_system.illumination.wavelength == 0.45
    assert cfg.imaging_system.illumination.numerical_aperture == 0.4
    assert cfg.imaging_system.illumination.invert_phase_contrast is True
    assert cfg.object.prior == "tikhonov"
    assert cfg.object.weight == 0.001
    assert cfg.solver.reconstruction == "closed_form"
    assert cfg.noise.kind == "gaussian"
    assert cfg.noise.gaussian.sigma == 1.0


def test_tv_maps_to_admm():
    raw = {
        "input_channel_names": ["BF"],
        "phase": {
            "transfer_function": {},
            "apply_inverse": {
                "reconstruction_algorithm": "TV",
                "regularization_strength": 0.01,
                "TV_rho_strength": 0.005,
                "TV_iterations": 7,
            },
        },
    }
    cfg = migrate_3x_dict(raw)
    assert cfg.object.prior == "tv"
    assert cfg.object.weight == 0.01
    assert cfg.solver.reconstruction.admm.rho == 0.005
    assert cfg.solver.reconstruction.admm.iterations == 7
    round_tripped = v4_to_legacy(cfg)
    assert round_tripped.model_dump() == ReconstructionSettings(**raw).model_dump()


def test_rl_maps_to_richardson_lucy_with_poisson():
    raw = {
        "input_channel_names": ["GFP"],
        "fluorescence": {
            "transfer_function": {},
            "apply_inverse": {
                "reconstruction_algorithm": "RLGC",
                "rl": {"iterations": 13, "background": 2.0},
            },
        },
    }
    cfg = migrate_3x_dict(raw)
    assert cfg.object.prior == "none"
    assert cfg.noise.kind == "poisson"
    rl = cfg.solver.reconstruction.richardson_lucy
    assert rl.iterations == 13
    assert rl.background == 2.0
    assert rl.gradient_consensus is True
    round_tripped = v4_to_legacy(cfg)
    assert round_tripped.model_dump() == ReconstructionSettings(**raw).model_dump()


def test_confocal_pinhole_maps_to_detection():
    raw = {
        "input_channel_names": ["GFP"],
        "fluorescence": {
            "transfer_function": {"confocal_pinhole_diameter": 0.5, "wavelength_emission": 0.52},
            "apply_inverse": {},
        },
    }
    cfg = migrate_3x_dict(raw)
    assert cfg.imaging_system.detection.wavelength == 0.52
    assert cfg.imaging_system.detection.pinhole.diameter_um == 0.5
    round_tripped = v4_to_legacy(cfg)
    assert round_tripped.model_dump() == ReconstructionSettings(**raw).model_dump()


def test_optimizable_floats_map_to_free_parameters():
    raw = {
        "input_channel_names": ["BF"],
        "reconstruction_dimension": 2,
        "phase": {
            "transfer_function": {
                "z_focus_offset": {"init": 1.5, "lr": 0.05},
                "tilt_angle_zenith": {"init": 0.0, "lr": 0.005},
            },
            "apply_inverse": {"regularization_strength": 0.01},
        },
        "optimization": {"max_iterations": 50, "method": "adam"},
    }
    cfg = migrate_3x_dict(raw)
    z_offset = cfg.imaging_system.detection.z_focus_offset
    assert z_offset.nominal == 1.5
    assert z_offset.tolerance == "free"
    calibration = cfg.solver.calibration
    assert calibration.score == "sharpness"
    assert calibration.search[0].gradient.max_iter == 50
    assert calibration.advanced.optimizer == "adam"
    assert calibration.advanced.learning_rates == {
        "detection.z_focus_offset": 0.05,
        "illumination.tilt.zenith": 0.005,
    }
    round_tripped = v4_to_legacy(cfg)
    assert round_tripped.model_dump() == ReconstructionSettings(**raw).model_dump()


def test_auto_regularization_maps_to_weight_auto():
    raw = {
        "input_channel_names": ["GFP"],
        "fluorescence": {
            "transfer_function": {},
            "apply_inverse": {"auto_regularization": {"rule": "l_curve", "num_samples": 11}},
        },
    }
    cfg = migrate_3x_dict(raw)
    assert cfg.object.weight.auto.rule == "l_curve"
    assert cfg.object.weight.auto.num_samples == 11
    round_tripped = v4_to_legacy(cfg)
    assert round_tripped.model_dump() == ReconstructionSettings(**raw).model_dump()


def test_detect_version_rejects_empty_file():
    with pytest.raises(ValueError, match="mapping"):
        detect_version(None)


def test_free_parameters_without_calibration_fail_loudly():
    """A 4.0 config with tolerances but no calibration block must not silently
    run the 3.x sharpness optimizer (the design default score is evidence)."""
    from waveorder.config import NotRunnableInLegacyError

    raw = {
        "version": 4,
        "data": {"input_channel_names": ["BF"], "pixel_size": {"yx": 0.1, "z": 0.25}},
        "imaging_system": {
            "contrast": "phase",
            "detection": {"numerical_aperture": {"nominal": 1.2, "tolerance": "free"}},
            "illumination": {"wavelength": 0.532, "numerical_aperture": 0.9},
        },
    }
    with pytest.raises(NotRunnableInLegacyError, match="solver.calibration"):
        v4_to_legacy(V4Config(**raw))
    # An explicit sharpness block selects the 3.x optimizer path.
    raw["solver"] = {"calibration": {"score": "sharpness", "search": [{"gradient": {"max_iter": 10}}]}}
    legacy = v4_to_legacy(V4Config(**raw))
    assert legacy.phase.transfer_function.numerical_aperture_detection.is_optimizable

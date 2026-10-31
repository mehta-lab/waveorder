"""4.0 configs drive the existing Zernike recovery; tf.zarr round-trips."""

import numpy as np
import pytest
import torch
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from typer.testing import CliRunner

from waveorder.calibrate.run import build_recovery_settings, zernike_calibration_requested
from waveorder.calibrate.tfzarr import read_tf_zarr, write_tf_zarr
from waveorder.cli.main import app
from waveorder.config.schema import V4Config
from waveorder.models.shift_variant_fluorescent_3d import apply_shift_variant_forward
from waveorder.zernike import SpatialPolynomialPupil


def _base_config(**imaging_system_updates) -> dict:
    cfg = {
        "version": 4,
        "data": {
            "input_channel_names": ["GFP"],
            "pixel_size": {"yx": 0.2, "z": 0.4},
        },
        "imaging_system": {
            "contrast": "fluorescence",
            "index_of_refraction_media": 1.3,
            "detection": {
                "wavelength": 0.532,
                "numerical_aperture": 1.2,
                "pupil_aberrations": {"zernike_up_to_order": {"order": 4, "tolerance": "free"}},
            },
        },
        "solver": {
            "calibration": {
                "score": "sharpness",
                "search": [{"gradient": {"max_iter": 30}}],
            }
        },
    }
    cfg["imaging_system"].update(imaging_system_updates)
    return cfg


def test_zernike_calibration_requested():
    cfg = V4Config(**_base_config())
    assert zernike_calibration_requested(cfg)
    plain = _base_config()
    plain["imaging_system"]["detection"].pop("pupil_aberrations")
    plain.pop("solver")
    assert not zernike_calibration_requested(V4Config(**plain))


def test_case_5_shift_invariant_settings():
    """Design case 5 (AMBSF): order-4 free Zernikes, one full-FOV tile."""
    cfg = V4Config(**_base_config())
    settings, noll_indices = build_recovery_settings(cfg, (128, 96))
    assert noll_indices == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert settings.tile_size_yx == {"y": 128, "x": 96}
    assert settings.loss == "midband"
    assert settings.n_iter == 30


def test_case_17_tiled_settings():
    """Design case 17 (patchwise recovery): per-tile independent estimates."""
    cfg = V4Config(**_base_config(field_dependence={"tiled": {"size_yx": [32, 32], "overlap_yx": [8, 8]}}))
    settings, _ = build_recovery_settings(cfg, (128, 128))
    assert settings.tile_size_yx == {"y": 32, "x": 32}
    assert settings.tile_overlap_yx == {"y": 8.0, "x": 8.0}


def test_named_term_selects_single_mode():
    raw = _base_config()
    raw["imaging_system"]["detection"]["pupil_aberrations"] = {"astigmatism_0": {"nominal": 0.0, "tolerance": "free"}}
    _, noll_indices = build_recovery_settings(V4Config(**raw), (64, 64))
    assert noll_indices == [6]


def test_unsupported_paths_raise():
    evidence = _base_config()
    evidence["solver"]["calibration"]["score"] = "evidence"
    with pytest.raises(NotImplementedError, match="score"):
        build_recovery_settings(V4Config(**evidence), (64, 64))

    numeric = _base_config()
    numeric["imaging_system"]["detection"]["pupil_aberrations"] = {
        "zernike_up_to_order": {"order": 4, "tolerance": 0.05}
    }
    with pytest.raises(NotImplementedError, match="tolerance"):
        build_recovery_settings(V4Config(**numeric), (64, 64))


def test_tfzarr_round_trip(tmp_path):
    parameters = np.arange(6, dtype=np.float64).reshape(2, 3)
    tile_grid = {"size_yx": [32, 32], "overlap_yx": [0, 0], "centers_yx": [[-8.0, 0.0], [8.0, 0.0]]}
    tf = (np.random.default_rng(0).random((2, 4, 8, 8)) + 0j).astype(np.complex64)
    write_tf_zarr(
        tmp_path / "tf.zarr",
        parameters=parameters,
        parameter_names=["defocus", "astigmatism_45", "astigmatism_0"],
        tile_grid=tile_grid,
        provenance={"kind": "refined", "sources": ["x.zarr/0/0/0"]},
        transfer_function=tf,
    )
    store = read_tf_zarr(tmp_path / "tf.zarr")
    np.testing.assert_array_equal(store.parameters, parameters)
    np.testing.assert_array_equal(store.parameters_covariance, np.zeros((2, 3, 3)))
    np.testing.assert_array_equal(store.transfer_function, tf)
    assert store.parameter_names == ["defocus", "astigmatism_45", "astigmatism_0"]
    assert store.tile_grid == tile_grid
    assert store.provenance["kind"] == "refined"


def test_tfzarr_shape_validation(tmp_path):
    with pytest.raises(ValueError, match="N_tiles"):
        write_tf_zarr(tmp_path / "bad.zarr", np.zeros(3), ["a", "b", "c"], {})
    with pytest.raises(ValueError, match="parameter_names"):
        write_tf_zarr(tmp_path / "bad.zarr", np.zeros((1, 3)), ["a"], {})


@pytest.fixture
def aberrated_beads_zarr(tmp_path):
    """A small bead grid imaged through a constant astigmatic pupil."""
    torch.manual_seed(0)
    Z, Y, X = 24, 64, 64
    phantom = torch.zeros(Z, Y, X)
    for y in (16, 32, 48):
        for x in (16, 32, 48):
            phantom[Z // 2, y, x] = 1.0

    truth = 0.15
    pupil = SpatialPolynomialPupil({(6, 0, 0): truth}, field_extent_um=(6.4, 6.4))
    data = apply_shift_variant_forward(
        phantom,
        yx_pixel_size=0.2,
        z_pixel_size=0.4,
        wavelength_emission=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_detection=1.2,
        spatial_pupil=pupil,
        n_tiles_yx=(2, 2),
    )

    input_path = tmp_path / "beads.zarr"
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=["GFP"])
    position = dataset.create_position("0", "0", "0")
    position.create_image(
        "0",
        data.numpy()[None, None].astype(np.float32),
        transform=[TransformationMeta(type="scale", scale=[1, 1, 0.4, 0.2, 0.2])],
    )
    dataset.close()
    return input_path, truth


def test_calibrate_tf_cli_end_to_end(tmp_path, aberrated_beads_zarr):
    input_path, truth = aberrated_beads_zarr
    raw = _base_config()
    raw["imaging_system"]["detection"]["pupil_aberrations"] = {
        "astigmatism_0": {"nominal": 0.0, "tolerance": "free"},
        "astigmatism_45": {"nominal": 0.0, "tolerance": "free"},
    }
    raw["solver"]["calibration"]["search"] = [{"gradient": {"max_iter": 120}}]
    config_path = tmp_path / "calibrate.yml"
    config_path.write_text(yaml.safe_dump(raw))

    output_path = tmp_path / "tf_hat.zarr"
    result = CliRunner().invoke(
        app,
        ["calibrate-tf", "-i", str(input_path / "0" / "0" / "0"), "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    store = read_tf_zarr(output_path)
    assert store.parameter_names == ["astigmatism_45", "astigmatism_0"]
    assert store.parameters.shape == (1, 2)
    assert store.transfer_function is not None  # shift-invariant stores carry H
    assert store.provenance["kind"] == "refined"
    assert store.tile_grid["size_yx"] == [64, 64]

    recovered = dict(zip(store.parameter_names, store.parameters[0]))
    # The sharpness score sees |H| only, so the sign of a lone astigmatism
    # pair is ambiguous; check the recovered magnitude.
    magnitude = np.hypot(recovered["astigmatism_0"], recovered["astigmatism_45"])
    assert abs(magnitude - truth) < 0.08, recovered

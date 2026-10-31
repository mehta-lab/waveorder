"""Bead calibration (prior: beads) recovers signed aberrations."""

import math

import numpy as np
import pytest
import torch
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from typer.testing import CliRunner

from waveorder.calibrate.beads import detect_beads, run_bead_calibration, sphere_form_factor
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.calibrate.tfzarr import read_tf_zarr
from waveorder.cli.main import app
from waveorder.config.schema import V4Config

ZYX_SHAPE = (16, 48, 48)
OPTICS = dict(
    yx_pixel_size=0.2,
    z_pixel_size=0.4,
    wavelength_emission=0.532,
    index_of_refraction_media=1.3,
    numerical_aperture_detection=1.2,
)
DIAMETER = 0.3
TRUTH = {"astigmatism_0": 0.15}
POSITIONS = [(8, 12, 12), (8, 12, 36), (8, 36, 24)]


def _bead_config(max_iter: int = 300) -> V4Config:
    return V4Config(
        **{
            "version": 4,
            "data": {"input_channel_names": ["GFP"], "pixel_size": {"yx": 0.2, "z": 0.4}},
            "imaging_system": {
                "contrast": "fluorescence",
                "index_of_refraction_media": 1.3,
                "detection": {
                    "wavelength": 0.532,
                    "numerical_aperture": 1.2,
                    "pupil_aberrations": {"astigmatism_0": {"nominal": 0.0, "tolerance": "free"}},
                },
            },
            "object": {"prior": {"beads": {"diameter": DIAMETER}}},
            "noise": {"gaussian": {"sigma": 1.0}},
            "solver": {"calibration": {"score": "evidence", "search": [{"gradient": {"max_iter": max_iter}}]}},
        }
    )


@pytest.fixture(scope="module")
def bead_volume():
    """Spheres imaged through an astigmatic PSF with Gaussian noise."""
    otf = fluorescence_otf(ZYX_SHAPE, zernike_coefficients=TRUTH, **OPTICS)
    form = sphere_form_factor(ZYX_SHAPE, OPTICS["yx_pixel_size"], OPTICS["z_pixel_size"], DIAMETER)

    freqs = [torch.fft.fftfreq(n) for n in ZYX_SHAPE]
    grids = torch.meshgrid(*freqs, indexing="ij")
    object_fft = torch.zeros(ZYX_SHAPE, dtype=torch.complex64)
    for z, y, x in POSITIONS:
        phase = grids[0] * z + grids[1] * y + grids[2] * x
        object_fft = object_fft + 3000.0 * torch.exp(-2j * math.pi * phase)
    object_fft = object_fft * form

    clean = torch.real(torch.fft.ifftn(object_fft * otf)) + 10.0
    generator = torch.Generator().manual_seed(3)
    return (clean + torch.randn(ZYX_SHAPE, generator=generator)).numpy()


def test_sphere_form_factor_dc_and_decay():
    form = sphere_form_factor(ZYX_SHAPE, 0.2, 0.4, DIAMETER)
    assert float(form[0, 0, 0]) == 1.0
    assert float(form.abs().max()) <= 1.0


def test_detect_beads_finds_the_peaks(bead_volume):
    positions = detect_beads(torch.as_tensor(bead_volume))
    found = {tuple(p.tolist()) for p in positions}
    for truth_position in POSITIONS:
        assert any(max(abs(a - b) for a, b in zip(p, truth_position)) <= 2 for p in found), (truth_position, found)


def test_bead_calibration_recovers_signed_aberration(bead_volume):
    result = run_bead_calibration(_bead_config(), bead_volume)
    recovered = dict(zip(result.parameter_names, result.parameters[0]))
    # The known object breaks the global sign-flip twin: the sign is
    # identifiable here, unlike the blind evidence path.
    assert abs(recovered["astigmatism_0"] - TRUTH["astigmatism_0"]) < 0.04, recovered
    assert result.positions_zyx.shape[0] >= len(POSITIONS)
    variances = np.diag(result.parameters_covariance[0])
    assert np.all(np.isfinite(variances)) and np.all(variances > 0)


def test_bead_calibration_cli(tmp_path, bead_volume):
    input_path = tmp_path / "beads.zarr"
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=["GFP"])
    position = dataset.create_position("0", "0", "0")
    position.create_image(
        "0",
        np.asarray(bead_volume)[None, None].astype(np.float32),
        transform=[TransformationMeta(type="scale", scale=[1, 1, 0.4, 0.2, 0.2])],
    )
    dataset.close()

    config_path = tmp_path / "beads.yml"
    config_path.write_text(yaml.safe_dump(_bead_config(max_iter=150).model_dump()))

    output_path = tmp_path / "tf_beads.zarr"
    result = CliRunner().invoke(
        app,
        ["calibrate-tf", "-i", str(input_path / "0" / "0" / "0"), "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    store = read_tf_zarr(output_path)
    assert store.provenance["kind"] == "beads"
    assert store.parameter_names == ["astigmatism_0"]
    assert abs(store.parameters[0, 0] - TRUTH["astigmatism_0"]) < 0.06
    assert store.parameters_covariance[0, 0, 0] > 0


def test_detect_beads_adaptive_relaxation(bead_volume):
    # A deliberately strict threshold finds nothing at first; the adaptive
    # fallback relaxes it (with a warning) instead of crashing.
    scaled = torch.as_tensor(bead_volume)
    strict = 1e6
    with pytest.raises(ValueError, match="snr_threshold"):
        detect_beads(scaled, snr_threshold=strict, adaptive=False)
    # Moderate SNR: pick a threshold ~1.5x the actual peak SNR so only the
    # relaxed pass finds the beads.
    median = scaled.median()
    mad = (scaled - median).abs().median()
    peak_snr = float((scaled.max() - median) / (1.4826 * mad))
    with pytest.warns(UserWarning, match="relaxed"):
        positions = detect_beads(scaled, snr_threshold=1.5 * peak_snr)
    assert positions.shape[0] >= 1


def test_bead_detection_knobs_from_config(bead_volume):
    config = _bead_config(max_iter=1)
    config.object.prior.beads.snr_threshold = 3.0
    config.object.prior.beads.max_beads = 2
    result = run_bead_calibration(config, bead_volume)
    assert result.positions_zyx.shape[0] == 2

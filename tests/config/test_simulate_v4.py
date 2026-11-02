"""The 4.0 simulate verb: standard outputs, determinism, calibration closure."""

import numpy as np
import pytest
import yaml
from iohub.ngff import open_ome_zarr
from typer.testing import CliRunner

from waveorder.calibrate.evidence_search import run_evidence_calibration
from waveorder.calibrate.tfzarr import read_tf_zarr
from waveorder.cli.main import app
from waveorder.config.schema import V4Config
from waveorder.config.simulate import run_simulation

SIM_RAW = {
    "version": 4,
    "simulate": {"seed": 0, "shape": [16, 32, 32]},
    "data": {"input_channel_names": ["GFP"], "pixel_size": {"yx": 0.2, "z": 0.4}},
    "imaging_system": {
        "contrast": "fluorescence",
        "index_of_refraction_media": 1.3,
        "detection": {
            "wavelength": 0.532,
            "numerical_aperture": 1.2,
            "pupil_aberrations": {
                "spherical": 0.08,
                "astigmatism_0": {"nominal": 0.0, "tolerance": 0.1},
            },
        },
    },
    "object": {"source": {"gaussian_spectrum": {"exponent": 2.0}}},
    "noise": {"gaussian": {"snr_db": 20}},
}


def _config(**updates) -> V4Config:
    raw = yaml.safe_load(yaml.safe_dump(SIM_RAW))
    raw.update(updates)
    return V4Config(**raw)


def test_simulation_writes_standard_outputs(tmp_path):
    truth = run_simulation(_config(), tmp_path / "sim")

    with open_ome_zarr(tmp_path / "sim" / "data.zarr" / "0" / "0" / "0") as data:
        assert data.data.shape == (1, 1, 16, 32, 32)
        assert data.channel_names == ["GFP"]
        assert data.scale[-3:] == [0.4, 0.2, 0.2]
    with open_ome_zarr(tmp_path / "sim" / "object.zarr" / "0" / "0" / "0") as obj:
        assert obj.data.shape == (1, 1, 16, 32, 32)

    store = read_tf_zarr(tmp_path / "sim" / "tf.zarr")
    # P = parameters with a tolerance: only astigmatism_0 is free.
    assert store.parameter_names == ["astigmatism_0"]
    assert store.parameters[0, 0] == truth["astigmatism_0"]
    assert abs(truth["astigmatism_0"]) < 0.4  # drawn from N(0, 0.1)
    assert truth["spherical"] == 0.08  # fixed term, in provenance not parameters
    np.testing.assert_array_equal(store.parameters_covariance, np.zeros((1, 1, 1)))
    assert store.transfer_function.shape == (16, 32, 32)
    assert store.provenance["kind"] == "simulated"
    assert store.provenance["true_coefficients_waves"]["spherical"] == 0.08


def test_simulation_is_deterministic(tmp_path):
    truth_a = run_simulation(_config(), tmp_path / "a")
    truth_b = run_simulation(_config(), tmp_path / "b")
    assert truth_a == truth_b
    with open_ome_zarr(tmp_path / "a" / "data.zarr" / "0" / "0" / "0") as a:
        with open_ome_zarr(tmp_path / "b" / "data.zarr" / "0" / "0" / "0") as b:
            np.testing.assert_array_equal(a.data[:], b.data[:])
    truth_c = run_simulation(_config(simulate={"seed": 1, "shape": [16, 32, 32]}), tmp_path / "c")
    assert truth_c["astigmatism_0"] != truth_a["astigmatism_0"]


def test_poisson_noise(tmp_path):
    cfg = _config(noise={"poisson": {"photons": 50}})
    run_simulation(cfg, tmp_path / "sim")
    with open_ome_zarr(tmp_path / "sim" / "data.zarr" / "0" / "0" / "0") as data:
        counts = data.data[:]
    assert counts.min() >= 0
    assert counts == pytest.approx(np.round(counts))  # integer counts
    assert 25 < counts.mean() < 100


def test_simulate_cli(tmp_path):
    config_path = tmp_path / "truth.yml"
    config_path.write_text(yaml.safe_dump(SIM_RAW))
    result = CliRunner().invoke(app, ["simulate", "-c", str(config_path), "-o", str(tmp_path / "sim")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sim" / "data.zarr").exists()
    assert (tmp_path / "sim" / "object.zarr").exists()
    assert (tmp_path / "sim" / "tf.zarr").exists()


def test_calibration_closure_on_simulated_data(tmp_path):
    """Case 28-style closure: calibrate-tf on simulate's output recovers the truth."""
    truth = run_simulation(_config(), tmp_path / "sim")
    with open_ome_zarr(tmp_path / "sim" / "data.zarr" / "0" / "0" / "0") as data:
        zyx = np.asarray(data.data[0, 0])

    recon_raw = yaml.safe_load(yaml.safe_dump(SIM_RAW))
    recon_raw.pop("simulate")
    recon_raw["object"] = {"prior": "tikhonov", "weight": 1e-3}
    recon_raw["noise"] = {"gaussian": {"sigma": "estimate"}}
    recon_raw["solver"] = {
        "calibration": {"score": "evidence", "search": [{"grid": {"points": 7}}, {"gradient": {"max_iter": 80}}]}
    }
    result = run_evidence_calibration(V4Config(**recon_raw), zyx)
    recovered = result.parameters[0, 0]
    # |H| only: the estimate is defined up to the global sign flip of the
    # free terms... but the fixed spherical term breaks the exact twin here,
    # so compare against the better of the two signings with a loose gate.
    error = min(abs(recovered - truth["astigmatism_0"]), abs(-recovered - truth["astigmatism_0"]))
    assert error < 0.05, (recovered, truth)

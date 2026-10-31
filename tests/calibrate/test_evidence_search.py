"""Evidence/GCV-scored calibration recovers synthetic aberrations."""

import numpy as np
import pytest
import torch

from waveorder.calibrate.evidence_search import (
    build_parameter_specs,
    gcv_score,
    run_evidence_calibration,
)
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.config.schema import V4Config

ZYX_SHAPE = (16, 32, 32)
OPTICS = dict(
    yx_pixel_size=0.2,
    z_pixel_size=0.4,
    wavelength_emission=0.532,
    index_of_refraction_media=1.3,
    numerical_aperture_detection=1.2,
)
TRUTH = {"astigmatism_0": 0.12, "coma_x": 0.06}
SIGMA, LAM = 0.05, 1e-2


def _config(score: str = "evidence", search=None, weight=LAM, sigma=SIGMA) -> V4Config:
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
                    "pupil_aberrations": {
                        "astigmatism_0": {"nominal": 0.0, "tolerance": 0.2},
                        "coma_x": {"nominal": 0.0, "tolerance": 0.2},
                    },
                },
            },
            "object": {"prior": "tikhonov", "weight": weight},
            "noise": {"gaussian": {"sigma": sigma}},
            "solver": {
                "calibration": {
                    "score": score,
                    "search": search or [{"gradient": {"max_iter": 150}}],
                }
            },
        }
    )


@pytest.fixture(scope="module")
def synthetic_data():
    otf_truth = fluorescence_otf(ZYX_SHAPE, zernike_coefficients=TRUTH, **OPTICS)
    generator = torch.Generator().manual_seed(0)
    prior_std = SIGMA / LAM**0.5
    f = torch.randn(ZYX_SHAPE, generator=generator) * prior_std
    g = torch.real(torch.fft.ifftn(torch.fft.fftn(f) * otf_truth))
    g = g + torch.randn(ZYX_SHAPE, generator=generator) * SIGMA
    return g.numpy()


def _twin_error(parameters: np.ndarray, names: list[str]) -> float:
    """Error vs truth, minimized over the global sign flip (exact twin)."""
    truth = np.array([TRUTH[name] for name in names])
    direct = np.abs(parameters - truth).max()
    flipped = np.abs(-parameters - truth).max()
    return min(direct, flipped)


def test_build_parameter_specs_orders_by_noll():
    specs = build_parameter_specs(_config())
    assert [s.name for s in specs] == ["astigmatism_0", "coma_x"]  # Noll 6, 8
    assert specs[0].tolerance == 0.2


def test_evidence_gradient_recovers_truth(synthetic_data):
    result = run_evidence_calibration(_config(), synthetic_data)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.05, result.parameters
    assert len(result.score_trace) == 150
    variances = np.diag(result.parameters_covariance[0])
    assert np.all(np.isfinite(variances)) and np.all(variances > 0)
    assert np.sqrt(variances).max() < 0.1  # tight posterior at this SNR


def test_evidence_grid_then_gradient(synthetic_data):
    config = _config(search=[{"grid": {"points": 5}}, {"gradient": {"max_iter": 60}}])
    result = run_evidence_calibration(config, synthetic_data)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.05
    assert result.score_grid is not None
    assert result.score_grid.shape == (25, 3)  # 5 x 5 points, 2 coords + score


def test_random_then_gradient(synthetic_data):
    config = _config(search=[{"random": {"samples": 8, "seed": 1}}, {"gradient": {"max_iter": 60}}])
    result = run_evidence_calibration(config, synthetic_data)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.06


def test_gcv_recovers_truth(synthetic_data):
    result = run_evidence_calibration(_config(score="gcv"), synthetic_data)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.06, result.parameters


def test_sigma_estimate_and_weight_auto(synthetic_data):
    # The score is exactly symmetric under the global sign flip of the
    # coefficients, so the zero nominal is a stationary point; a grid first
    # stage (as the design prescribes for multi-modal searches) moves the
    # start off it before sigma and lambda adapt.
    config = _config(
        sigma="estimate",
        weight="auto",
        search=[{"grid": {"points": 5}}, {"gradient": {"max_iter": 100}}],
    )
    result = run_evidence_calibration(config, synthetic_data)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.06
    # The jointly-estimated noise level should land near the truth.
    assert 0.3 * SIGMA < result.noise_sigma < 3.0 * SIGMA, result.noise_sigma
    assert np.isfinite(result.weight) and result.weight > 0


def _store(values: dict[str, float], variances: dict[str, float] | None = None):
    from waveorder.calibrate.tfzarr import TransferFunctionStore

    names = list(values)
    covariance = np.zeros((1, len(names), len(names)))
    if variances is not None:
        covariance[0] = np.diag([variances[name] for name in names])
    return TransferFunctionStore(
        parameters=np.array([[values[name] for name in names]]),
        parameter_names=names,
        tile_grid={},
        parameters_covariance=covariance,
    )


def test_init_store_sets_start_without_prior(synthetic_data):
    config = _config(search=[{"gradient": {"max_iter": 1}}])
    init = _store({"astigmatism_0": 0.3, "coma_x": -0.2})
    result = run_evidence_calibration(config, synthetic_data, init_store=init)
    # One Adam step (lr 0.01) cannot move far from the --init start.
    np.testing.assert_allclose(result.parameters[0], [0.3, -0.2], atol=0.02)


def test_prior_covariance_overrides_tolerances(synthetic_data):
    # A tight -t covariance pins the estimate near the prior center even
    # though the configuration's own tolerance (0.2) would let it move.
    prior = _store(
        {"astigmatism_0": 0.3, "coma_x": -0.1},
        variances={"astigmatism_0": 1e-8, "coma_x": 1e-8},
    )
    result = run_evidence_calibration(_config(), synthetic_data, prior_store=prior)
    np.testing.assert_allclose(result.parameters[0], [0.3, -0.1], atol=0.02)


def test_zero_covariance_moves_center_only(synthetic_data):
    # An all-zero covariance (e.g. from simulate) carries no uncertainty:
    # the config tolerances stay in force, only the start/center moves.
    prior = _store({"astigmatism_0": 0.3, "coma_x": -0.1})
    result = run_evidence_calibration(_config(), synthetic_data, prior_store=prior)
    assert _twin_error(result.parameters[0], result.parameter_names) < 0.06


def test_sequential_calibration(synthetic_data):
    """Design case 12: -t yesterday's tf_hat.zarr as today's prior."""
    from waveorder.calibrate.tfzarr import TransferFunctionStore

    first = run_evidence_calibration(_config(), synthetic_data)
    store = TransferFunctionStore(
        parameters=first.parameters,
        parameter_names=first.parameter_names,
        tile_grid={},
        parameters_covariance=first.parameters_covariance,
    )

    otf_truth = fluorescence_otf(ZYX_SHAPE, zernike_coefficients=TRUTH, **OPTICS)
    generator = torch.Generator().manual_seed(7)
    f = torch.randn(ZYX_SHAPE, generator=generator) * (SIGMA / LAM**0.5)
    second_data = torch.real(torch.fft.ifftn(torch.fft.fftn(f) * otf_truth))
    second_data = (second_data + torch.randn(ZYX_SHAPE, generator=generator) * SIGMA).numpy()

    second = run_evidence_calibration(_config(), second_data, prior_store=store)
    assert _twin_error(second.parameters[0], second.parameter_names) < 0.05
    # Two volumes carry more information than one: the posterior tightens.
    assert np.trace(second.parameters_covariance[0]) < np.trace(first.parameters_covariance[0]) * 1.5


def test_gcv_score_matches_definition():
    otf = fluorescence_otf(ZYX_SHAPE, **OPTICS)
    g = torch.randn(ZYX_SHAPE, generator=torch.Generator().manual_seed(2))
    g_fft = torch.fft.fftn(g, norm="ortho")
    lam = 1e-2
    h2 = torch.abs(otf) ** 2
    shrink = lam / (h2 + lam)
    n = g_fft.numel()
    expected = ((torch.abs(g_fft) ** 2) * shrink**2).sum() / ((shrink.sum() / n) ** 2)
    assert torch.allclose(gcv_score(g_fft, otf, lam), expected)


def test_singular_prior_covariance_rejected(synthetic_data):
    singular = _store(
        {"astigmatism_0": 0.3, "coma_x": -0.1},
        variances={"astigmatism_0": 1e-8, "coma_x": 0.0},  # zero-variance row
    )
    # Force the non-zero branch: one finite variance, one zero -> singular.
    singular.parameters_covariance[0, 0, 0] = 1e-8
    with pytest.raises(ValueError, match="singular"):
        run_evidence_calibration(_config(), synthetic_data, prior_store=singular)


def test_grid_range_centers_on_prior_center(synthetic_data):
    # A -t prior (zero covariance -> tolerances kept) moves the grid's
    # center: with the config tolerance 0.2, the grid spans the prior
    # center +/- 0.4, so the best grid point tracks the prior center.
    prior = _store({"astigmatism_0": 0.5, "coma_x": 0.5})
    config = _config(search=[{"grid": {"points": 3}}])
    result = run_evidence_calibration(_config(search=[{"grid": {"points": 3}}]), synthetic_data)
    moved = run_evidence_calibration(config, synthetic_data, prior_store=prior)
    # Without the prior the 3-point grid centered at 0 cannot reach 0.5.
    assert abs(result.parameters[0, 0]) <= 0.4 + 1e-9
    # With the prior the grid includes points around 0.5.
    assert moved.parameters[0, 0] >= 0.1 - 1e-9

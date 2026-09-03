"""Tests for automatic regularization-strength selection.

The reference numbers in this file were produced by the CZ Biohub weight-search
scripts that :mod:`waveorder.optim.autoreg` ports (``compMicro/utils.py`` and
``compMicro/weight_search.py``), so a change in behaviour shows up as a failure
here rather than as a quietly different pick.
"""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from waveorder.models import isotropic_fluorescent_thick_3d, phase_thick_3d
from waveorder.optim import autoreg
from waveorder.optim.autoreg import AutoRegularizationSettings

PHASE_TF_KWARGS = dict(
    yx_pixel_size=0.15,
    z_pixel_size=1.0,
    wavelength_illumination=0.441,
    index_of_refraction_media=1.3,
    numerical_aperture_illumination=1.15,
    numerical_aperture_detection=0.52,
)
FLUOR_TF_KWARGS = dict(
    yx_pixel_size=0.325,
    z_pixel_size=1.0,
    wavelength_emission=0.525,
    index_of_refraction_media=1.0,
    numerical_aperture_detection=0.55,
)


def _measurement(zyx_shape=(12, 64, 64), seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.random(zyx_shape) * 100 + 500).astype(np.float32))


def _phase_tfs(zyx_shape, z_padding):
    return phase_thick_3d.calculate_transfer_function(zyx_shape=zyx_shape, z_padding=z_padding, **PHASE_TF_KWARGS)


def _fluor_tf(zyx_shape, z_padding):
    return isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape=zyx_shape, z_padding=z_padding, **FLUOR_TF_KWARGS
    )


def _sweep_one(data, tf, contrast, z_padding, strength, apodization_rolloff=0.0):
    """Reproduce a single point of the sweep, the way select_regularization does."""
    measurement = autoreg.preprocess_measurement(data, contrast, z_padding)
    spectrum = torch.fft.fftn(measurement, dim=(-3, -2, -1))
    recon = torch.real(
        torch.fft.ifftn(
            spectrum * autoreg._inverse_filter(tf, strength, apodization_rolloff),
            dim=(-3, -2, -1),
        )
    )
    return recon[z_padding : recon.shape[0] - z_padding] if z_padding else recon


# --- the load-bearing test: the sweep scores the production reconstruction ---


@pytest.mark.parametrize("z_padding", [0, 3])
@pytest.mark.parametrize("strength", [1e-8, 1e-3])
def test_sweep_reconstruction_matches_phase_model(z_padding, strength):
    """A swept reconstruction is what phase_thick_3d would have produced."""
    zyx_shape = (12, 64, 64)
    data = _measurement(zyx_shape)
    real_tf, imag_tf = _phase_tfs(zyx_shape, z_padding)

    expected = phase_thick_3d.apply_inverse_transfer_function(
        data,
        real_tf,
        imag_tf,
        z_padding=z_padding,
        regularization_strength=strength,
    )
    actual = _sweep_one(data, real_tf, "phase", z_padding, strength)

    assert torch.allclose(expected, actual, atol=1e-6)


@pytest.mark.parametrize("z_padding", [0, 3])
@pytest.mark.parametrize("strength", [1e-8, 1e-3])
def test_sweep_reconstruction_matches_fluorescence_model(z_padding, strength):
    """A swept reconstruction is what isotropic_fluorescent_thick_3d would have produced."""
    zyx_shape = (12, 64, 64)
    data = _measurement(zyx_shape)
    otf = _fluor_tf(zyx_shape, z_padding)

    expected = isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
        data,
        otf,
        z_padding=z_padding,
        regularization_strength=strength,
    )
    actual = _sweep_one(data, otf, "fluorescence", z_padding, strength)

    assert torch.allclose(expected, actual, atol=1e-6)


def test_sweep_applies_apodization_like_the_model():
    """apodization_rolloff reshapes the inverse filter, so the sweep must apply it too.

    Uses a pixel size that undersamples the optical band, which is the case the
    knob exists for. At a well-sampled pixel size the window is ~1 wherever the
    filter carries energy and this test would pass with or without the fix.
    """
    zyx_shape, rolloff = (12, 64, 64), 0.25
    data = _measurement(zyx_shape)
    undersampled = {**PHASE_TF_KWARGS, "yx_pixel_size": 0.5}
    real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(zyx_shape=zyx_shape, z_padding=0, **undersampled)

    expected = phase_thick_3d.apply_inverse_transfer_function(
        data, real_tf, imag_tf, z_padding=0, regularization_strength=1e-3, apodization_rolloff=rolloff
    )
    unapodized = _sweep_one(data, real_tf, "phase", 0, 1e-3)
    apodized = _sweep_one(data, real_tf, "phase", 0, 1e-3, rolloff)

    # The window has to make a real difference here, or the assertion below is empty.
    assert (expected - unapodized).abs().max() > 0.05 * (expected.max() - expected.min())
    assert torch.allclose(expected, apodized, atol=1e-6)


def test_preprocess_measurement_normalizes_phase_only():
    """Phase intensity-normalizes before its inverse filter; fluorescence does not."""
    data = _measurement((6, 16, 16))

    phase = autoreg.preprocess_measurement(data, "phase", z_padding=0)
    fluorescence = autoreg.preprocess_measurement(data, "fluorescence", z_padding=0)

    assert torch.allclose(phase, data / data.mean() - 1)
    assert torch.allclose(fluorescence, data)

    padded = autoreg.preprocess_measurement(data, "fluorescence", z_padding=2)
    assert padded.shape == (10, 16, 16)

    with pytest.raises(ValueError, match="contrast"):
        autoreg.preprocess_measurement(data, "birefringence", z_padding=0)


# --- rules, against reference values from the scripts they port ---

#: ``compMicro/utils.py::batched_otsu_cnr`` on the fixture built below.
OTSU_REFERENCE = [
    3.579247,
    3.343898,
    3.276220,
    3.673673,
    3.484225,
    4.079302,
    4.594685,
    6.974761,
    7.500283,
]


def _otsu_fixture():
    rng = np.random.default_rng(20240101)
    arr = rng.standard_normal((9, 8, 48, 48)).astype(np.float32)
    # A block whose contrast grows along the sweep, so the scores are ordered.
    arr[:, :, 10:30, 10:30] += np.linspace(0.5, 4.0, 9, dtype=np.float32).reshape(9, 1, 1, 1)
    return torch.from_numpy(arr)


def test_otsu_cnr_scores_match_reference():
    scores = autoreg.otsu_cnr_scores(_otsu_fixture()).cpu().numpy()
    np.testing.assert_allclose(scores, OTSU_REFERENCE, rtol=1e-5)


def test_otsu_cnr_scores_use_one_shared_slice():
    """Every reconstruction is scored on the same z, so scores stay comparable."""
    recons = _otsu_fixture()
    # Zeroing a slice that is not the most structured must not change the scores.
    perturbed = recons.clone()
    flat_z = int(torch.argmin(autoreg._local_variance(recons).var(dim=(2, 3)).mean(dim=0)))
    perturbed[0, flat_z] = 0.0

    assert autoreg.otsu_cnr_scores(recons)[1:].allclose(autoreg.otsu_cnr_scores(perturbed)[1:])


def _noisy_l_curve():
    """A gently bent, noisy trace, of the kind real sweeps produce."""
    rng = np.random.default_rng(7)
    residual = np.logspace(0, 1.2, 25) * (1 + 0.02 * rng.standard_normal(25))
    solution = np.logspace(2.6, 1.1, 25) * (1 + 0.02 * rng.standard_normal(25))
    return residual, solution


_NOISY_L_CURVE = _noisy_l_curve()


@pytest.mark.parametrize(
    "residual, solution, expected",
    [
        # A real L: two turning points, corner between them.
        (
            np.concatenate([np.linspace(1, 1.05, 12), np.linspace(1.05, 40, 13)]),
            np.concatenate([np.linspace(500, 60, 12), np.linspace(60, 55, 13)]),
            11,
        ),
        # Nearly a straight line in log-log: no corner, so the search runs to an end.
        (np.logspace(0, 1.5, 25), np.logspace(2.4, 1.0, 25), 24),
        # Noisy and only gently bent, which is what real data tends to look like.
        (*_NOISY_L_CURVE, 13),
    ],
)
def test_find_l_curve_corner_matches_reference(residual, solution, expected):
    assert autoreg.find_l_curve_corner(residual, solution) == expected


def test_find_l_curve_corner_degenerate_inputs():
    assert autoreg.find_l_curve_corner(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0
    # Constant norms carry no curvature at all.
    flat = np.ones(9)
    assert 0 <= autoreg.find_l_curve_corner(flat, flat) < 9


def test_compute_l_curve_norms():
    rng = np.random.default_rng(3)
    tf = torch.from_numpy(rng.standard_normal((6, 16, 16)).astype(np.float32)).to(torch.complex64)
    data = torch.from_numpy(rng.standard_normal((6, 16, 16)).astype(np.float32))
    recons = torch.from_numpy(rng.standard_normal((4, 6, 16, 16)).astype(np.float32))

    residual_norms, solution_norms = autoreg.compute_l_curve_norms(data, tf, recons)

    assert residual_norms.shape == solution_norms.shape == (4,)
    np.testing.assert_allclose(
        solution_norms,
        torch.linalg.vector_norm(recons, dim=(-3, -2, -1)).numpy(),
        rtol=1e-6,
    )


# --- crop selection ---


def test_select_crop_finds_the_structured_region():
    data = torch.zeros(4, 512, 512)
    data[:, 300:400, 100:200] = torch.from_numpy(
        np.random.default_rng(0).standard_normal((4, 100, 100)).astype(np.float32)
    )

    y0, x0, size = autoreg.select_crop(data, crop_size=128)

    assert size == 128
    # The crop must overlap the structure it was supposed to find.
    assert y0 < 400 and y0 + size > 300
    assert x0 < 200 and x0 + size > 100


def test_select_crop_returns_full_frame_when_small():
    assert autoreg.select_crop(torch.zeros(3, 64, 64), crop_size=256) == (0, 0, 64)


def test_select_crop_reaches_the_far_edge():
    """The candidate stride must not leave a strip of the frame unreachable."""
    data = torch.zeros(2, 300, 300)
    data[:, 270:300, 270:300] = 100.0

    y0, x0, size = autoreg.select_crop(data, crop_size=128)

    assert y0 + size == 300
    assert x0 + size == 300


@pytest.mark.parametrize("shape, crop_size", [((4, 512, 512), 128), ((6, 301, 457), 128), ((4, 400, 400), 256)])
def test_select_crop_matches_an_exhaustive_search(shape, crop_size):
    """The integral-image shortcut picks what re-reducing every window would."""
    rng = np.random.default_rng(1)
    array = rng.normal(1000, 5, size=shape).astype(np.float32)
    array[:, 40:90, 60:110] += 400
    data = torch.from_numpy(array)

    size = min(crop_size, shape[1], shape[2])
    step = max(size // 2, 1)

    def starts(extent):
        last = extent - size
        return sorted({*range(0, last + 1, step), last})

    expected = max(
        ((y0, x0) for y0 in starts(shape[1]) for x0 in starts(shape[2])),
        key=lambda c: float(data[:, c[0] : c[0] + size, c[1] : c[1] + size].double().var(unbiased=False)),
    )

    assert autoreg.select_crop(data, crop_size)[:2] == expected


# --- settings ---


def test_settings_defaults_and_validation():
    settings = AutoRegularizationSettings()
    assert settings.rule == "otsu_cnr"
    assert settings.search_scale == "relative"
    assert settings.report_path is None

    with pytest.raises(ValidationError, match="search_min must be less than search_max"):
        AutoRegularizationSettings(search_min=2.0, search_max=-6.0)
    with pytest.raises(ValidationError, match="num_samples must be at least 3"):
        AutoRegularizationSettings(num_samples=2)
    with pytest.raises(ValidationError):
        AutoRegularizationSettings(rule="not_a_rule")
    with pytest.raises(ValidationError):
        AutoRegularizationSettings(typo=1)


# --- select_regularization ---


def _select(rule="otsu_cnr", zyx_shape=(12, 64, 64), z_padding=0, **kwargs):
    data = _measurement(zyx_shape)
    real_tf, _ = _phase_tfs(zyx_shape, z_padding)
    settings = AutoRegularizationSettings(rule=rule, num_samples=7, **kwargs)
    return data, real_tf, autoreg.select_regularization(data, real_tf, settings, contrast="phase", z_padding=z_padding)


@pytest.mark.parametrize("rule", ["otsu_cnr", "l_curve"])
def test_select_regularization_returns_a_swept_value(rule):
    _, _, result = _select(rule)

    assert result.rule == rule
    assert len(result.regularization_strengths) == 7
    assert result.regularization_strength == pytest.approx(result.regularization_strengths[result.index])
    assert result.regularization_strength > 0
    # Ascending sweep, so the pick is inside the swept span.
    assert result.regularization_strengths[0] <= result.regularization_strength
    assert result.regularization_strength <= result.regularization_strengths[-1]

    if rule == "otsu_cnr":
        assert len(result.scores) == 7 and len(result.residual_norms) == 0
    else:
        assert len(result.residual_norms) == 7 and len(result.scores) == 0


def test_relative_scale_anchors_to_the_transfer_function_peak():
    """`relative` reads the search bounds as log10(lambda / |H|^2max)."""
    zyx_shape = (12, 64, 64)
    data = _measurement(zyx_shape)
    real_tf, _ = _phase_tfs(zyx_shape, 0)
    h2max = float((real_tf.abs() ** 2).max())

    relative = autoreg.select_regularization(
        data,
        real_tf,
        AutoRegularizationSettings(num_samples=5, search_min=-2.0, search_max=2.0, search_scale="relative"),
        contrast="phase",
    )
    absolute = autoreg.select_regularization(
        data,
        real_tf,
        AutoRegularizationSettings(num_samples=5, search_min=-2.0, search_max=2.0, search_scale="absolute"),
        contrast="phase",
    )

    np.testing.assert_allclose(relative.regularization_strengths, absolute.regularization_strengths * h2max, rtol=1e-6)
    assert relative.transfer_function_peak_squared == pytest.approx(h2max)
    assert relative.lambda_over_h2max == pytest.approx(relative.regularization_strength / h2max)


def test_pick_on_the_sweep_edge_warns():
    """A pick at either end means the optimum may be outside the search bounds."""
    zyx_shape = (12, 64, 64)
    data = _measurement(zyx_shape)
    real_tf, _ = _phase_tfs(zyx_shape, 0)

    # The measurement is structureless noise and the whole range is heavily
    # over-regularized, so Otsu CNR climbs monotonically with smoothing and the
    # argmax has nowhere to sit but the top of the sweep.
    result = autoreg.select_regularization(
        data,
        real_tf,
        AutoRegularizationSettings(rule="otsu_cnr", num_samples=5, search_min=3.0, search_max=4.0),
        contrast="phase",
    )

    assert result.index == len(result.regularization_strengths) - 1
    assert any("end of the sweep" in message for message in result.warnings)

    with pytest.warns(UserWarning, match="end of the sweep"):
        autoreg.warn_all(result)


def test_shape_mismatch_is_rejected():
    data = _measurement((12, 64, 64))
    real_tf, _ = _phase_tfs((12, 32, 32), 0)

    with pytest.raises(ValueError, match="does not match"):
        autoreg.select_regularization(data, real_tf, AutoRegularizationSettings(), contrast="phase")


def test_result_serializes_for_the_report(tmp_path):
    _, _, result = _select("l_curve")

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    loaded = json.loads(report_path.read_text())

    assert loaded["rule"] == "l_curve"
    assert loaded["regularization_strength"] == pytest.approx(result.regularization_strength)
    assert loaded["crop"] == {"y0": 0, "x0": 0, "size": 0}
    assert len(loaded["residual_norms"]) == 7

"""Correctness tests for the unmatched back projectors.

These check :mod:`waveorder.backprojector` and its wiring into 3D fluorescence
Richardson-Lucy:

* the matched back projector still reproduces ``conj(OTF)`` exactly,
* the resolution limit inferred from the PSF FWHM lands on the real OTF band
  edge, which is the assumption every unmatched choice rests on,
* Wiener-Butterworth flattens the spectral product across the passband, which
  is the mechanism behind the speed-up,
* both beta conventions hit the cutoff gain they promise,
* Wiener-Butterworth reaches in one iteration what the matched back
  projector needs tens of iterations for,
* unmatched back projectors are refused for RLGC and warn when over-iterated.
"""

import warnings

import pytest
import torch

from waveorder import backprojector
from waveorder.api import fluorescence
from waveorder.backprojector import calculate_back_projector
from waveorder.models import isotropic_fluorescent_thick_3d as thick

_OTF_KWARGS = dict(
    yx_pixel_size=0.1,
    z_pixel_size=0.3,
    wavelength_emission=0.515,
    z_padding=0,
    index_of_refraction_media=1.4,
    numerical_aperture_detection=1.2,
)

# Guo et al. Table S2.1 lists 0.001-0.05 for both; these sit at the sharp end.
_FILTER_KWARGS = dict(alpha=0.001, beta=0.001, order=8)

_UNMATCHED = ["gaussian", "butterworth", "wiener", "wiener_butterworth"]
_ALL_KINDS = ["matched"] + _UNMATCHED


@pytest.fixture(scope="module")
def otf():
    return thick.calculate_transfer_function((24, 64, 64), **_OTF_KWARGS)


def _cutoff_indices(otf):
    fwhm = backprojector._psf_fwhm_zyx_px(otf)
    return tuple(size / res for size, res in zip(otf.shape, fwhm))


def _kx_profile(volume):
    """Shifted |.| profile along kx through the DC row, for a (Z, Y, X) volume."""
    magnitude = torch.fft.fftshift(torch.abs(volume))
    return magnitude[magnitude.shape[0] // 2, magnitude.shape[1] // 2, :]


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_shape_dtype_and_device_are_preserved(otf, kind):
    back_projector = calculate_back_projector(otf, kind, **_FILTER_KWARGS)
    assert back_projector.shape == otf.shape
    assert back_projector.dtype == otf.dtype
    assert back_projector.device == otf.device


def test_matched_is_exactly_the_conjugate_otf(otf):
    """The matched choice must be an independent, materialized conjugate."""
    back_projector = calculate_back_projector(otf, "matched")
    assert torch.equal(back_projector, torch.conj(otf))
    assert not back_projector.is_conj()
    assert back_projector.untyped_storage().data_ptr() != otf.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_projectors_support_mps(otf):
    mps_otf = otf.to("mps")
    for kind in _ALL_KINDS:
        expected = calculate_back_projector(otf, kind, **_FILTER_KWARGS)
        actual = calculate_back_projector(mps_otf, kind, **_FILTER_KWARGS).cpu()
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind", ["matched", "gaussian", "butterworth"])
def test_unit_dc_gain(otf, kind):
    """These kernels sum to one, so they neither brighten nor dim the estimate."""
    back_projector = calculate_back_projector(otf, kind, **_FILTER_KWARGS)
    assert float(torch.abs(back_projector[0, 0, 0])) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["wiener", "wiener_butterworth"])
def test_wiener_dc_gain_is_set_by_alpha(otf, kind):
    """The Wiener term scales DC by 1/(1+alpha).

    Richardson-Lucy divides this constant back out through ``H_T(1)``, so it is
    harmless, but it should stay predictable rather than drift.
    """
    alpha = _FILTER_KWARGS["alpha"]
    back_projector = calculate_back_projector(otf, kind, **_FILTER_KWARGS)
    assert float(torch.abs(back_projector[0, 0, 0])) == pytest.approx(1.0 / (1.0 + alpha), abs=1e-5)


def test_fwhm_cutoff_lands_on_the_otf_band_edge(otf):
    """The FWHM heuristic must actually locate the resolution limit.

    Every unmatched back projector places its Butterworth transition at this
    inferred cutoff, so if the heuristic were far from the true band edge the
    filters would either apodize inside the passband or amplify pure noise.
    """
    profile = _kx_profile(otf)
    center = profile.shape[0] // 2
    supported = [i for i in range(center) if float(profile[center + i]) > 1e-3]
    assert max(supported) == pytest.approx(_cutoff_indices(otf)[2], rel=0.15)


def test_wiener_butterworth_flattens_the_spectral_product(otf):
    """The mechanism behind the speed-up: a flatter ``|DFT(f) DFT(b)|``.

    Flatness is measured only where the OTF has real support. Outside it the
    product is zero whatever the back projector, and the missing cone would
    otherwise dominate the statistic.
    """
    support = torch.abs(otf) > 0.05

    def coefficient_of_variation(kind):
        product = torch.abs(otf * calculate_back_projector(otf, kind, **_FILTER_KWARGS))[support]
        return float(product.std() / product.mean())

    matched = coefficient_of_variation("matched")
    wiener_butterworth = coefficient_of_variation("wiener_butterworth")
    assert wiener_butterworth < matched / 10
    # Every unmatched choice should improve on the matched one.
    for kind in _UNMATCHED:
        assert coefficient_of_variation(kind) < matched


def test_butterworth_passes_beta_at_the_cutoff(otf):
    """``beta`` is defined as the gain at the resolution limit (Eq. 23)."""
    beta = 0.01
    back_projector = calculate_back_projector(otf, "butterworth", beta=beta, order=8)
    gain = backprojector._mean_gain_at_cutoff(_kx_profile(back_projector), _cutoff_indices(otf)[2])
    assert gain == pytest.approx(beta, rel=0.1)


def test_beta_conventions_differ_by_the_wiener_cutoff_gain(otf):
    """Guo et al.'s text and reference code calibrate ``beta`` differently.

    Under ``"paper"`` the Wiener-Butterworth gain at the cutoff is ``beta``
    exactly; under ``"reference"`` it is ``beta * sqrt(beta_w)``. Both are the
    same filter family, so this pins down which one we build.
    """
    cutoff = _cutoff_indices(otf)
    wiener = calculate_back_projector(otf, "wiener", alpha=_FILTER_KWARGS["alpha"])
    wiener_cutoff_gain = backprojector._wiener_cutoff_gain(wiener, cutoff)
    # The Wiener term amplifies near the cutoff, which is why the two
    # conventions never coincide in practice.
    assert wiener_cutoff_gain > 2.0

    def gain(convention):
        back_projector = calculate_back_projector(
            otf, "wiener_butterworth", beta_convention=convention, **_FILTER_KWARGS
        )
        return backprojector._mean_gain_at_cutoff(_kx_profile(back_projector), cutoff[2])

    beta = _FILTER_KWARGS["beta"]
    assert gain("paper") == pytest.approx(beta, rel=0.15)
    assert gain("reference") == pytest.approx(beta * wiener_cutoff_gain**0.5, rel=0.15)


def test_wiener_butterworth_factors_into_its_two_halves(otf):
    """WB is the elementwise product of the Wiener and Butterworth terms."""
    wiener = calculate_back_projector(otf, "wiener", alpha=_FILTER_KWARGS["alpha"])
    wiener_butterworth = calculate_back_projector(otf, "wiener_butterworth", **_FILTER_KWARGS)
    support = torch.abs(wiener) > 1e-6
    ratio = torch.abs(wiener_butterworth)[support] / torch.abs(wiener)[support]
    # The quotient is the Butterworth mask: unity at DC, never amplifying.
    assert float(torch.abs(wiener_butterworth[0, 0, 0]) / torch.abs(wiener[0, 0, 0])) == pytest.approx(1.0, abs=1e-4)
    assert float(ratio.max()) <= 1.0 + 1e-4


@pytest.mark.parametrize("kind", ["butterworth", "wiener_butterworth"])
def test_apodized_kernels_have_negative_lobes(otf, kind):
    """Butterworth apodization rings in real space (Supplementary Fig. 5).

    These negative lobes are expected and must survive: clipping them would
    turn the filter back into something matched-like. Richardson-Lucy handles
    them by clipping the *estimate* each iteration instead.
    """
    back_projector = calculate_back_projector(otf, kind, **_FILTER_KWARGS)
    kernel = torch.real(torch.fft.ifftn(back_projector))
    assert float(kernel.min()) < -1e-6 * float(kernel.max())


def test_gaussian_takes_no_parameters(otf):
    """The Gaussian back projector is fixed by the PSF FWHM alone."""
    baseline = calculate_back_projector(otf, "gaussian")
    for kwargs in ({"alpha": 0.5}, {"beta": 0.5}, {"order": 2}, {"resolution_mode": "fwhm_over_sqrt2"}):
        assert torch.equal(calculate_back_projector(otf, "gaussian", **kwargs), baseline)


def test_invalid_arguments_are_rejected(otf):
    for invalid_projector in ("nonsense", "traditional"):
        with pytest.raises(ValueError, match="back_projector"):
            calculate_back_projector(otf, invalid_projector)
    with pytest.raises(ValueError, match="requires resolution_zyx_px"):
        calculate_back_projector(otf, "butterworth", resolution_mode="manual")
    with pytest.raises(ValueError, match="only valid with resolution_mode"):
        calculate_back_projector(otf, "butterworth", resolution_zyx_px=(2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="3 entries"):
        calculate_back_projector(otf, "butterworth", resolution_mode="manual", resolution_zyx_px=(2.0, 2.0))
    with pytest.raises(ValueError, match="must be positive"):
        calculate_back_projector(otf, "butterworth", resolution_mode="manual", resolution_zyx_px=(2.0, 2.0, 0.0))
    with pytest.raises(ValueError, match="order"):
        calculate_back_projector(otf, "butterworth", order=0)
    with pytest.raises(ValueError, match="beta <= 1"):
        calculate_back_projector(otf, "butterworth", beta=2.0)
    with pytest.raises(ValueError, match="must be complex"):
        calculate_back_projector(torch.real(otf), "butterworth")
    with pytest.raises(ValueError, match="must be 3D"):
        calculate_back_projector(otf[0], "butterworth")
    with pytest.raises(ValueError, match="alpha must be positive"):
        calculate_back_projector(
            torch.zeros_like(otf),
            "wiener",
            resolution_mode="manual",
            resolution_zyx_px=(2.0, 2.0, 2.0),
        )


def test_undersampled_psf_reports_an_actionable_error():
    """A PSF that never falls to half maximum cannot yield a FWHM.

    The caller can recover by supplying the resolution explicitly, so the
    error says so rather than silently propagating a NaN cutoff.
    """
    # A Gaussian far wider than the array never crosses half maximum in z.
    z, y, x = 8, 32, 32
    indices = [torch.fft.fftfreq(n) * n for n in (z, y, x)]
    exponent = (indices[0].reshape(-1, 1, 1) / 100.0) ** 2 + (indices[1].reshape(1, -1, 1) / 3.0) ** 2
    exponent = exponent + (indices[2].reshape(1, 1, -1) / 3.0) ** 2
    otf = torch.fft.fftn(torch.exp(-0.5 * exponent).to(torch.complex64))

    with pytest.raises(ValueError, match="undersampled"):
        calculate_back_projector(otf, "wiener_butterworth", **_FILTER_KWARGS)
    # The documented escape hatch works.
    calculate_back_projector(
        otf, "wiener_butterworth", resolution_mode="manual", resolution_zyx_px=(4.0, 3.0, 3.0), **_FILTER_KWARGS
    )


def _bead_concentration(volume, beads, half=2):
    """Fraction of nonnegative energy inside small windows around ``beads``."""
    clamped = volume.clamp(min=0)
    total = float(clamped.sum())
    local = sum(
        float(clamped[z - half : z + half + 1, y - half : y + half + 1, x - half : x + half + 1].sum())
        for z, y, x in beads
    )
    return local / total


def test_one_wiener_butterworth_iteration_beats_many_matched_ones(otf):
    """The headline claim: an unmatched back projector collapses the iteration count.

    Energy concentration is the metric rather than bead FWHM, because the
    matched back projector's effective kernel is a narrow cusp sitting on a
    broad halo. Its FWHM looks respectable long before the halo clears, so FWHM
    understates how much work is left; concentration does not.
    """
    torch.manual_seed(0)
    beads = [(12, 20, 20), (12, 20, 44), (12, 44, 20), (12, 44, 44)]
    obj = torch.zeros(24, 64, 64)
    for bead in beads:
        obj[bead] = 5000.0
    data = torch.poisson(torch.clamp(thick.apply_transfer_function(obj, otf, 0, background=2), min=0))

    def reconstruct(kind, iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return thick.apply_inverse_transfer_function(
                data,
                otf,
                0,
                reconstruction_algorithm="RL",
                rl_iterations=iterations,
                rl_back_projector=kind,
                rl_bp_alpha=_FILTER_KWARGS["alpha"],
                rl_bp_beta=_FILTER_KWARGS["beta"],
                rl_bp_order=_FILTER_KWARGS["order"],
            )

    raw = _bead_concentration(data, beads)
    one_iteration = _bead_concentration(reconstruct("wiener_butterworth", 1), beads)
    forty_iterations = _bead_concentration(reconstruct("matched", 40), beads)

    assert one_iteration > 4 * raw
    assert one_iteration > 0.8 * forty_iterations


def test_rlgc_refuses_unmatched_back_projectors(otf):
    """RLGC's consensus test needs a true adjoint to be a valid inner product."""
    data = torch.rand(24, 64, 64)
    for kind in _UNMATCHED:
        with pytest.raises(NotImplementedError, match="RLGC"):
            thick.apply_inverse_transfer_function(data, otf, 0, reconstruction_algorithm="RLGC", rl_back_projector=kind)
    # The matched default is still allowed.
    thick.apply_inverse_transfer_function(data, otf, 0, reconstruction_algorithm="RLGC", rl_iterations=1)


def test_over_iterating_an_unmatched_back_projector_warns(otf):
    """Past a few iterations these degrade rather than converge."""
    data = torch.rand(24, 64, 64)
    with pytest.warns(UserWarning, match="rl_iterations"):
        thick.apply_inverse_transfer_function(
            data, otf, 0, reconstruction_algorithm="RL", rl_iterations=25, rl_back_projector="wiener_butterworth"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thick.apply_inverse_transfer_function(
            data, otf, 0, reconstruction_algorithm="RL", rl_iterations=25, rl_back_projector="matched"
        )


def test_default_settings_leave_existing_configs_unchanged():
    """Adding these knobs must not alter any reconstruction already in use."""
    settings = fluorescence.ApplyInverseSettings()
    assert settings.rl_back_projector == "matched"
    assert settings.rl_bp_alpha is None
    assert settings.rl_bp_beta is None


def test_settings_round_trip():
    settings = fluorescence.ApplyInverseSettings(
        reconstruction_algorithm="RL",
        rl_iterations=1,
        rl_back_projector="wiener_butterworth",
        rl_bp_alpha=0.001,
        rl_bp_beta=0.001,
        rl_bp_order=10,
        rl_bp_resolution_mode="fwhm_over_sqrt2",
    )
    dumped = settings.model_dump()
    assert dumped["rl_back_projector"] == "wiener_butterworth"
    assert dumped["rl_bp_order"] == 10
    assert dumped["rl_bp_resolution_mode"] == "fwhm_over_sqrt2"
    # The dump is splatted straight into the model function, so the keys must match.
    zyx_shape = (16, 32, 32)
    thick.apply_inverse_transfer_function(
        torch.rand(*zyx_shape), thick.calculate_transfer_function(zyx_shape, **_OTF_KWARGS), 0, **dumped
    )


def test_rejects_invalid_settings():
    for invalid_projector in ("nonsense", "traditional"):
        with pytest.raises(ValueError):
            fluorescence.ApplyInverseSettings(rl_back_projector=invalid_projector)
    with pytest.raises(ValueError):
        fluorescence.ApplyInverseSettings(rl_bp_resolution_mode="manual")

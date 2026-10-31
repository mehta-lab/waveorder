"""PSF/OTF synthesis matches existing numerics; evidence recovers the truth."""

import torch

from waveorder.calibrate.scores import calibration_penalty, whittle_evidence
from waveorder.calibrate.synthesis import fluorescence_otf, fluorescence_psf
from waveorder.models.shift_variant_fluorescent_3d import calculate_psf_at_field_position
from waveorder.zernike import SpatialPolynomialPupil

ZYX_SHAPE = (24, 48, 48)
OPTICS = dict(
    yx_pixel_size=0.2,
    z_pixel_size=0.4,
    wavelength_emission=0.532,
    index_of_refraction_media=1.3,
    numerical_aperture_detection=1.2,
)


def _reference_psf(coefficients: dict[tuple[int, int, int], float]) -> torch.Tensor:
    """PSF from the existing shift-variant model at the field center."""
    pupil = SpatialPolynomialPupil(coefficients, field_extent_um=(10.0, 10.0))
    return calculate_psf_at_field_position(
        ZYX_SHAPE,
        OPTICS["yx_pixel_size"],
        OPTICS["z_pixel_size"],
        OPTICS["wavelength_emission"],
        OPTICS["index_of_refraction_media"],
        OPTICS["numerical_aperture_detection"],
        pupil,
        x_o_um=0.0,
        y_o_um=0.0,
    )


def test_psf_matches_shift_variant_model_unaberrated():
    psf = fluorescence_psf(ZYX_SHAPE, **OPTICS)
    reference = _reference_psf({})
    assert torch.allclose(psf, reference, atol=1e-6)


def test_psf_matches_shift_variant_model_aberrated():
    coefficients = {"spherical": 0.1, "coma_x": 0.05, "astigmatism_45": -0.03}
    psf = fluorescence_psf(ZYX_SHAPE, zernike_coefficients=coefficients, **OPTICS)
    reference = _reference_psf({(11, 0, 0): 0.1, (8, 0, 0): 0.05, (5, 0, 0): -0.03})
    assert torch.allclose(psf, reference, atol=1e-6)


def test_noll_and_named_keys_agree():
    psf_named = fluorescence_psf(ZYX_SHAPE, zernike_coefficients={"defocus": 0.2}, **OPTICS)
    psf_noll = fluorescence_psf(ZYX_SHAPE, zernike_coefficients={4: 0.2}, **OPTICS)
    assert torch.equal(psf_named, psf_noll)


def test_psf_normalized_and_centered():
    psf = fluorescence_psf(ZYX_SHAPE, **OPTICS)
    assert abs(float(psf.sum()) - 1.0) < 1e-5
    # In-focus (z index 0 in FFT order) and origin-centered in y, x.
    peak = torch.argmax(psf)
    z, y, x = torch.unravel_index(peak, psf.shape)
    assert int(z) == 0 and int(y) == 0 and int(x) == 0


def test_otf_peak_normalized():
    otf = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"spherical": 0.1}, **OPTICS)
    assert abs(float(torch.abs(otf).max()) - 1.0) < 1e-5


def test_gradients_flow_through_coefficients():
    coefficient = torch.tensor(0.05, requires_grad=True)
    psf = fluorescence_psf(ZYX_SHAPE, zernike_coefficients={"spherical": coefficient}, **OPTICS)
    psf.sum().backward()  # sum is constant 1, but graph must exist
    assert coefficient.grad is not None


def _simulate(otf_truth: torch.Tensor, noise_sigma: float, regularization_strength: float, seed: int = 0):
    """Draw g = H f + n with f from the Tikhonov prior S_k = sigma^2 / lambda."""
    generator = torch.Generator().manual_seed(seed)
    shape = otf_truth.shape
    prior_std = noise_sigma / regularization_strength**0.5
    f = torch.randn(shape, generator=generator) * prior_std
    g = torch.real(torch.fft.ifftn(torch.fft.fftn(f) * otf_truth))
    g = g + torch.randn(shape, generator=generator) * noise_sigma
    return g


def test_evidence_prefers_true_astigmatism_magnitude():
    # Astigmatism changes |H|, which the Whittle evidence sees. Two known
    # invariances temper what can be asserted:
    # * pure defocus is close to an axial shift, which leaves |H| nearly
    #   unchanged, so it is only weakly identifiable from the evidence;
    # * flipping the sign of the whole aberration vector reflects the
    #   intensity PSF through the origin, leaving |H| exactly unchanged
    #   (the classic blind-deconvolution PSF-flip ambiguity), so only the
    #   magnitude of a lone astigmatism term is identifiable.
    sigma, lam = 0.1, 1e-2
    truth = 0.15
    otf_truth = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"astigmatism_0": truth}, **OPTICS)
    g = _simulate(otf_truth, sigma, lam)
    g_fft = torch.fft.fftn(g, norm="ortho")

    candidates = torch.linspace(-0.3, 0.6, 10).tolist() + [truth, -truth]
    scores = {}
    for c in candidates:
        otf = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"astigmatism_0": c}, **OPTICS)
        scores[c] = float(whittle_evidence(data_fft=g_fft, otf=otf, noise_sigma=sigma, regularization_strength=lam))
    best = min(scores, key=scores.get)
    assert abs(abs(best) - truth) < 0.11, scores


def test_evidence_prefers_true_spherical():
    sigma, lam = 0.05, 1e-2
    truth = 0.2
    otf_truth = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"spherical": truth}, **OPTICS)
    g = _simulate(otf_truth, sigma, lam, seed=1)
    g_fft = torch.fft.fftn(g, norm="ortho")

    candidates = [0.0, 0.1, truth, 0.3, 0.4]
    scores = []
    for c in candidates:
        otf = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"spherical": c}, **OPTICS)
        scores.append(float(whittle_evidence(data_fft=g_fft, otf=otf, noise_sigma=sigma, regularization_strength=lam)))
    assert candidates[scores.index(min(scores))] == truth, dict(zip(candidates, scores))


def test_evidence_is_differentiable():
    sigma, lam = 0.1, 1e-2
    otf_truth = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"defocus": 0.1}, **OPTICS)
    g = _simulate(otf_truth, sigma, lam)
    coefficient = torch.tensor(0.0, requires_grad=True)
    otf = fluorescence_otf(ZYX_SHAPE, zernike_coefficients={"defocus": coefficient}, **OPTICS)
    score = whittle_evidence(data=g, otf=otf, noise_sigma=sigma, regularization_strength=lam)
    score.backward()
    assert coefficient.grad is not None
    assert torch.isfinite(coefficient.grad)


def test_calibration_penalty():
    values = {"na": 1.25, "spherical": 0.1, "defocus": 0.5}
    nominals = {"na": 1.2, "spherical": 0.0, "defocus": 0.0}
    tolerances = {"na": 0.05, "spherical": 0.1, "defocus": "free"}
    penalty = float(calibration_penalty(values, nominals, tolerances))
    assert abs(penalty - (0.5 + 0.5)) < 1e-9

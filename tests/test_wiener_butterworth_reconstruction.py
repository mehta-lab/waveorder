"""Observable Wiener–Butterworth RL behavior against the fluorescence model."""

import os

import pytest
import torch

from waveorder import backprojector
from waveorder.models import isotropic_fluorescent_thick_3d as thick
from waveorder.reconstruction._wiener_butterworth import WienerButterworthRL


def _sample(shape=(16, 24, 26)):
    # A real, positive circular PSF with nonzero axial/transverse support.
    grids = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij")
    coordinates = [torch.minimum(g, n - g) for g, n in zip(grids, shape)]
    psf = torch.exp(-0.5 * sum((axis / width) ** 2 for axis, width in zip(coordinates, (1.8, 1.3, 1.5))))
    psf /= psf.sum()
    otf = torch.fft.fftn(psf).contiguous()
    generator = torch.Generator().manual_seed(453)
    measured = (torch.rand((shape[0] - 4, shape[1], shape[2]), generator=generator) * 2 - 0.15).contiguous()
    return otf, measured


@pytest.mark.parametrize("convention", ["reference", "paper"])
@pytest.mark.parametrize("iterations", [1, 3])
def test_reconstructs_like_legacy_with_prebuilt_filter(convention, iterations):
    otf, measured = _sample()
    b = backprojector.calculate_back_projector(
        otf,
        "wiener_butterworth",
        alpha=0.005,
        beta=0.005,
        order=6,
        beta_convention=convention,
    )
    expected = thick.apply_inverse_transfer_function(
        measured,
        otf,
        2,
        reconstruction_algorithm="RL",
        rl_iterations=iterations,
        rl_background=0.07,
        rl_back_projector="wiener_butterworth",
        back_projector_otf=b,
    )
    with WienerButterworthRL(
        otf,
        z_padding=2,
        alpha=0.005,
        beta=0.005,
        order=6,
        beta_convention=convention,
        iterations=iterations,
        background=0.07,
    ) as solver:
        actual = solver(measured)
        assert torch.allclose(actual, expected, rtol=1e-4, atol=2e-5)
        # A result is caller-owned; running another tile must not overwrite it.
        frozen = actual.clone()
        buffer = torch.empty_like(measured)
        assert solver(measured * 0.5, out=buffer) is buffer
        assert torch.equal(actual, frozen)


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_nonunit_forward_dc_matches_legacy(backend, iterations):
    if backend == "cuda" and (os.getenv("WAVEORDER_TEST_NATIVE_CUDA") != "1" or not torch.cuda.is_available()):
        pytest.skip("Native CUDA needs an explicit opt-in and matching toolkit/compiler/Ninja")
    otf, measured = _sample()
    otf = (otf * 1.7).contiguous()
    measured = (measured + 0.2).clamp_min(0.05).contiguous()
    if backend == "cuda":
        otf, measured = otf.cuda(), measured.cuda()
    b = backprojector.calculate_back_projector(otf, "wiener_butterworth", alpha=0.005, beta=0.005, order=6)
    expected = thick.apply_inverse_transfer_function(
        measured,
        otf,
        2,
        reconstruction_algorithm="RL",
        rl_iterations=iterations,
        rl_background=0.13,
        rl_back_projector="wiener_butterworth",
        back_projector_otf=b,
    )
    with WienerButterworthRL(
        otf,
        z_padding=2,
        alpha=0.005,
        beta=0.005,
        order=6,
        iterations=iterations,
        background=0.13,
        backend=backend,
    ) as solver:
        actual = solver(measured)
    assert torch.allclose(actual, expected, rtol=3e-4 if backend == "cuda" else 1e-4, atol=3e-5)


def test_beta_conventions_change_the_reconstruction():
    otf, measured = _sample()
    results = []
    for convention in ("reference", "paper"):
        with WienerButterworthRL(
            otf, z_padding=2, alpha=0.005, beta=0.005, order=6, beta_convention=convention
        ) as solver:
            results.append(solver(measured))
    assert not torch.allclose(results[0], results[1], atol=1e-5, rtol=1e-5)


def test_manual_resolution_recovers_when_axial_fwhm_is_undersampled():
    shape = (8, 32, 32)
    axes = [torch.fft.fftfreq(size) * size for size in shape]
    exponent = (
        (axes[0][:, None, None] / 100.0) ** 2
        + (axes[1][None, :, None] / 3.0) ** 2
        + (axes[2][None, None, :] / 3.0) ** 2
    )
    otf = torch.fft.fftn(torch.exp(-0.5 * exponent).to(torch.complex64)).contiguous()
    measured = torch.rand(shape, generator=torch.Generator().manual_seed(813))
    with pytest.raises(ValueError, match="undersampled"):
        WienerButterworthRL(otf, alpha=0.001, beta=0.001)
    resolution = (4.0, 3.0, 3.0)
    b = backprojector.calculate_back_projector(
        otf,
        "wiener_butterworth",
        alpha=0.001,
        beta=0.001,
        resolution_mode="manual",
        resolution_zyx_px=resolution,
    )
    expected = thick.apply_inverse_transfer_function(
        measured,
        otf,
        0,
        reconstruction_algorithm="RL",
        rl_iterations=1,
        rl_back_projector="wiener_butterworth",
        back_projector_otf=b,
    )
    with WienerButterworthRL(
        otf,
        alpha=0.001,
        beta=0.001,
        resolution_mode="manual",
        resolution_zyx_px=resolution,
    ) as solver:
        assert torch.allclose(solver(measured), expected, rtol=1e-4, atol=2e-5)


def test_nonhermitian_otf_is_rejected_instead_of_silently_discarding_information():
    otf, _ = _sample()
    otf[1, 2, 3] += 0.1j
    with pytest.raises(ValueError, match="Hermitian"):
        WienerButterworthRL(otf)


@pytest.mark.parametrize("shape", [(16, 24, 26), (15, 23, 25)])
@pytest.mark.skipif(
    os.getenv("WAVEORDER_TEST_NATIVE_CUDA") != "1" or not torch.cuda.is_available(),
    reason="Native CUDA needs an explicit opt-in and matching toolkit/compiler/Ninja",
)
def test_native_matches_legacy_and_keeps_prior_output(shape):
    otf, measured = _sample(shape)
    otf, measured = otf.cuda(), measured.cuda()
    b = backprojector.calculate_back_projector(
        otf,
        "wiener_butterworth",
        alpha=0.005,
        beta=0.005,
        order=6,
    )
    expected = thick.apply_inverse_transfer_function(
        measured,
        otf,
        2,
        reconstruction_algorithm="RL",
        rl_iterations=3,
        rl_background=0.07,
        rl_back_projector="wiener_butterworth",
        back_projector_otf=b,
    )
    with WienerButterworthRL(
        otf, z_padding=2, alpha=0.005, beta=0.005, order=6, iterations=3, background=0.07, backend="cuda"
    ) as solver:
        actual = solver(measured)
        assert torch.allclose(actual, expected, rtol=3e-4, atol=3e-5)
        prior = actual.clone()
        solver(measured * 0.5)
        assert torch.equal(actual, prior)

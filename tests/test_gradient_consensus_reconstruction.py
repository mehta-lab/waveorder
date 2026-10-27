"""Observable matched-adjoint RLGC behavior on physical 3D fluorescence OTFs."""

import os

import pytest
import torch

from waveorder import rlgc, util
from waveorder.models import isotropic_fluorescent_thick_3d
from waveorder.reconstruction._gradient_consensus import GradientConsensusRL


def _otf(shape, device):
    coords = [torch.minimum(torch.arange(n, device=device), n - torch.arange(n, device=device)) for n in shape]
    z, y, x = torch.meshgrid(*coords, indexing="ij")
    psf = torch.exp(-(z.float() ** 2 / 1.2 + y.float() ** 2 / 1.7 + x.float() ** 2 / 1.4))
    return torch.fft.fftn(psf / psf.sum()).to(torch.complex64)


def _reference(measured, otf, *, padding, iterations, background, tolerance, generator):
    padded = util.pad_zyx_along_z(measured, padding).clamp_min(0)

    def forward(data):
        return torch.fft.ifftn(torch.fft.fftn(data) * otf).real

    def adjoint(data):
        return torch.fft.ifftn(torch.fft.fftn(data) * otf.conj()).real

    result = rlgc.richardson_lucy(
        padded,
        forward,
        adjoint,
        num_iterations=iterations,
        method="RLGC",
        background=background,
        stopping_tolerance=tolerance,
        generator=generator,
    )
    return result[padding:-padding] if padding else result


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param(
            "cuda",
            "torch",
            marks=pytest.mark.skipif(
                os.getenv("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                reason="Opt in to Torch CUDA reconstruction tests",
            ),
        ),
        pytest.param(
            "cuda",
            "cuda",
            marks=pytest.mark.skipif(
                os.getenv("WAVEORDER_TEST_NATIVE_CUDA") != "1" or not torch.cuda.is_available(),
                reason="Opt in to native CUDA extension tests",
            ),
        ),
    ],
)
def test_repeated_calls_match_reference_and_generator(device, backend):
    target = torch.device(device)
    otf = _otf((5, 4, 5), target)
    measured = torch.arange(3 * 4 * 5, device=target, dtype=torch.float32).reshape(3, 4, 5).remainder(9)
    measured[0, 0, 0] = -2.0
    seed = 493
    with GradientConsensusRL(otf, z_padding=1, iterations=3, background=0.2, backend=backend, device=target) as solver:
        first_generator = torch.Generator(device=device).manual_seed(seed)
        first = solver(measured, generator=first_generator)
        expected_generator = torch.Generator(device=device).manual_seed(seed)
        expected = _reference(
            measured, otf, padding=1, iterations=3, background=0.2, tolerance=None, generator=expected_generator
        )
        torch.testing.assert_close(first, expected, rtol=6e-4, atol=4e-5)
        assert torch.equal(first_generator.get_state(), expected_generator.get_state())
        snapshot = first.clone()
        second_generator = torch.Generator(device=device).manual_seed(seed)
        output = torch.empty_like(first)
        assert solver(measured, generator=second_generator, out=output) is output
        torch.testing.assert_close(output, expected, rtol=6e-4, atol=4e-5)
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
        assert torch.equal(second_generator.get_state(), expected_generator.get_state())
    torch.testing.assert_close(first, snapshot, rtol=0, atol=0)


def test_thick_model_matched_reference_and_tolerance_consumption():
    otf = _otf((5, 4, 5), torch.device("cpu"))
    measured = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).remainder(7)
    torch.manual_seed(317)
    with GradientConsensusRL(otf, z_padding=1, iterations=12, stopping_tolerance=100.0) as solver:
        actual = solver(measured)
        actual_rng = torch.random.get_rng_state()
    torch.manual_seed(317)
    expected = isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
        measured,
        otf,
        z_padding=1,
        reconstruction_algorithm="RLGC",
        rl_iterations=12,
        rl_stopping_tolerance=100.0,
        rl_back_projector="matched",
    )
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)
    assert torch.equal(actual_rng, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param(
            "cuda",
            "cuda",
            marks=pytest.mark.skipif(
                os.getenv("WAVEORDER_TEST_NATIVE_CUDA") != "1" or not torch.cuda.is_available(),
                reason="Opt in to native CUDA extension tests",
            ),
        ),
    ],
)
def test_all_frozen_stops_after_one_photon_split(device, backend):
    otf = torch.zeros((3, 3, 4), dtype=torch.complex64, device=device)
    measured = torch.full((3, 3, 4), 3.0, device=device)
    split = torch.Generator(device=device).manual_seed(86)
    expected_rng = torch.Generator(device=device).manual_seed(86)
    with GradientConsensusRL(otf, iterations=12, backend=backend) as solver:
        actual = solver(measured, generator=split)
    expected = _reference(
        measured, otf, padding=0, iterations=12, background=0.0, tolerance=None, generator=expected_rng
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(split.get_state(), expected_rng.get_state())


def test_nonhermitian_transfer_cannot_change_real_convolution():
    otf = _otf((3, 4, 5), torch.device("cpu"))
    otf[1, 2, 3] += 0.1j
    with pytest.raises(ValueError, match="Hermitian"):
        GradientConsensusRL(otf)


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param(
            "cuda",
            "torch",
            marks=pytest.mark.skipif(
                os.getenv("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                reason="Opt in to Torch CUDA reconstruction tests",
            ),
        ),
        pytest.param(
            "cuda",
            "cuda",
            marks=pytest.mark.skipif(
                os.getenv("WAVEORDER_TEST_NATIVE_CUDA") != "1" or not torch.cuda.is_available(),
                reason="Opt in to native CUDA extension tests",
            ),
        ),
    ],
)
def test_shifted_asymmetric_psf_uses_adjoint_power_spectrum(device, backend):
    target = torch.device(device)
    shape = (5, 4, 7)
    base = torch.fft.ifftn(_otf(shape, target)).real
    psf = base + 0.23 * torch.roll(base, shifts=(0, 1, -1), dims=(0, 1, 2))
    psf = torch.roll(psf, shifts=(1, -1, 2), dims=(0, 1, 2))
    otf = torch.fft.fftn(psf / psf.sum()).to(torch.complex64) * 1.3
    assert torch.allclose(otf[0, 0, 0].real, otf.real.new_tensor(1.3), rtol=1e-6, atol=1e-6)
    assert otf[0, 0, 1].imag.abs() > 0.05
    assert (otf[0, 0, 1].square() - otf[0, 0, 1].abs().square()).abs() > 0.05
    measured = torch.arange(5 * 4 * 7, device=target, dtype=torch.float32).reshape(shape).square().remainder(3)
    seed = 607
    with GradientConsensusRL(otf, iterations=4, background=0.4, backend=backend, device=target) as solver:
        actual = solver(measured, generator=torch.Generator(device=device).manual_seed(seed))
    expected = _reference(
        measured,
        otf,
        padding=0,
        iterations=4,
        background=0.4,
        tolerance=None,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    torch.testing.assert_close(actual, expected, rtol=6e-4, atol=4e-5)

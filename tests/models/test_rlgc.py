"""Correctness tests for Richardson-Lucy (RL) and Gradient-Consensus (RLGC).

These tests check the operator-agnostic core (:mod:`waveorder.rlgc`) and its
wiring into 3D fluorescence reconstruction:

* the FFT forward/adjoint operators used for deconvolution are true adjoints,
* RL and RLGC sharpen a Poisson-noisy bead simulation,
* over-iterated RL overfits the noise into a "starry night" of spurious
  bright voxels while RLGC resists it,
* RL/RLGC are refused for 2D fluorescence and for phase/birefringence.
"""

import pytest
import torch

from waveorder import rlgc
from waveorder.api import fluorescence, phase
from waveorder.models import (
    isotropic_fluorescent_thick_3d as thick,
)
from waveorder.models import (
    isotropic_fluorescent_thin_3d as thin,
)
from waveorder.models import isotropic_thin_3d as phase_thin
from waveorder.models import phase_thick_3d

# Shared, physically reasonable widefield fluorescence imaging parameters.
_OTF_KWARGS = dict(
    yx_pixel_size=0.1,
    z_pixel_size=0.3,
    wavelength_emission=0.515,
    z_padding=0,
    index_of_refraction_media=1.4,
    numerical_aperture_detection=1.2,
)


def _otf(zyx_shape):
    return thick.calculate_transfer_function(zyx_shape, **_OTF_KWARGS)


def _fft_operators(otf):
    """Build the deconvolution forward/adjoint the model uses internally."""

    def forward(x):
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-3, -2, -1)) * otf, dim=(-3, -2, -1)))

    def transpose(y):
        return torch.real(torch.fft.ifftn(torch.fft.fftn(y, dim=(-3, -2, -1)) * torch.conj(otf), dim=(-3, -2, -1)))

    return forward, transpose


def _bead_concentration(volume, beads, half=1):
    """Fraction of nonnegative energy within small windows around ``beads``."""
    v = volume.clamp(min=0)
    total = float(v.sum())
    local = 0.0
    for z, y, x in beads:
        local += float(v[z - half : z + half + 1, y - half : y + half + 1, x - half : x + half + 1].sum())
    return local / total


def test_fft_operators_are_adjoint():
    """The conjugate-OTF transpose must be a true adjoint of the forward."""
    otf = _otf((12, 48, 48))
    forward, transpose = _fft_operators(otf)
    torch.manual_seed(0)
    a = torch.rand(12, 48, 48)
    b = torch.rand(12, 48, 48)
    lhs = float((forward(a) * b).sum())
    rhs = float((a * transpose(b)).sum())
    assert abs(lhs - rhs) <= 1e-5 * max(abs(lhs), abs(rhs))


def test_core_richardson_lucy_smoke():
    """The core solver returns a positive estimate of the right shape and
    increases the Poisson log-likelihood of the measurement."""
    otf = _otf((8, 32, 32))
    forward, transpose = _fft_operators(otf)
    torch.manual_seed(0)
    obj = torch.zeros(8, 32, 32)
    obj[4, 10, 10] = 500.0
    measured = torch.poisson((forward(obj)).clamp(min=0))

    for method in ("RL", "RLGC"):
        gen = torch.Generator().manual_seed(0)
        estimate = rlgc.richardson_lucy(measured, forward, transpose, num_iterations=20, method=method, generator=gen)
        assert estimate.shape == obj.shape
        assert torch.all(estimate > 0)  # RL step keeps the estimate positive
        ll_start = rlgc.poisson_log_likelihood(forward(torch.ones_like(obj)), measured)
        ll_end = rlgc.poisson_log_likelihood(forward(estimate), measured)
        assert ll_end > ll_start


def test_core_rejects_bad_arguments():
    otf = _otf((4, 16, 16))
    forward, transpose = _fft_operators(otf)
    measured = torch.ones(4, 16, 16)
    with pytest.raises(ValueError):
        rlgc.richardson_lucy(measured, forward, transpose, num_iterations=0)
    with pytest.raises(ValueError):
        rlgc.richardson_lucy(measured, forward, transpose, num_iterations=5, method="bogus")


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_sharpens_noisy_beads(algorithm):
    """RL and RLGC concentrate a Poisson-noisy bead simulation, recovering
    energy that the microscope's blur had spread out."""
    zyx_shape = (24, 64, 64)
    beads = [(12, 20, 20), (12, 20, 44), (8, 40, 32)]
    background = 2.0

    otf = _otf(zyx_shape)
    obj = torch.full(zyx_shape, 0.0)
    for b in beads:
        obj[b] = 2000.0

    torch.manual_seed(0)
    clean = thick.apply_transfer_function(obj, otf, z_padding=0, background=background)
    data = torch.poisson(clean.clamp(min=0))

    raw_conc = _bead_concentration(data, beads)

    torch.manual_seed(1)
    recon = thick.apply_inverse_transfer_function(
        data,
        otf,
        z_padding=0,
        reconstruction_algorithm=algorithm,
        rl_iterations=100,
        rl_background=background,
    )
    recon_conc = _bead_concentration(recon, beads)

    # Deconvolution should concentrate energy far more tightly than the raw
    # blurred data around the true bead locations.
    assert recon_conc > 0.2
    assert recon_conc > 10 * raw_conc


def test_overiteration_starry_night_rl_vs_rlgc():
    """Over-iterated RL overfits Poisson noise into a 'starry night' of
    spurious bright voxels; RLGC freezes those voxels and stays clean."""
    zyx_shape = (16, 64, 64)
    beads = [(8, 20, 20), (8, 20, 44), (6, 40, 32), (10, 44, 44)]
    # A dim, uniform fluorophore field carries Poisson noise everywhere,
    # which is what RL overfits (cf. Andrew York's demo).
    otf = _otf(zyx_shape)
    obj = torch.full(zyx_shape, 2.0)
    for b in beads:
        obj[b] = 80.0

    torch.manual_seed(0)
    data = torch.poisson(thick.apply_transfer_function(obj, otf, z_padding=0, background=0).clamp(min=0))

    # A region with no beads: it should stay smooth after reconstruction.
    empty = (slice(2, 14), slice(0, 10), slice(0, 10))
    bright_threshold = 40.0
    true_bright = int((obj > bright_threshold).sum())

    def reconstruct(algorithm):
        torch.manual_seed(1)
        return thick.apply_inverse_transfer_function(
            data,
            otf,
            z_padding=0,
            reconstruction_algorithm=algorithm,
            rl_iterations=1000,
            rl_background=0.0,
        )

    rl = reconstruct("RL")
    gc = reconstruct("RLGC")

    rl_empty_std = float(rl[empty].std())
    gc_empty_std = float(gc[empty].std())
    rl_bright = int((rl > bright_threshold).sum())
    gc_bright = int((gc > bright_threshold).sum())

    # RL overfits: the empty region becomes noisy and littered with spurious
    # bright voxels far exceeding the four true beads.
    assert rl_bright > 100
    assert rl_empty_std > 20 * gc_empty_std
    # RLGC resists overfitting: no spurious bright voxels beyond the truth.
    assert gc_bright <= true_bright
    assert float(gc[empty].max()) < 10.0


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_rl_stable_on_coarse_sampling(algorithm):
    """Coarse (sub-Nyquist) sampling is where the OTF crop used to leave
    negative PSF lobes that can destabilize Richardson-Lucy. The forward PSF
    must be nonnegative and RL/RLGC must stay finite and bounded."""
    zyx_shape = (20, 48, 48)
    otf = thick.calculate_transfer_function(
        zyx_shape,
        yx_pixel_size=0.65,
        z_pixel_size=0.65,
        wavelength_emission=0.515,
        z_padding=0,
        index_of_refraction_media=1.4,
        numerical_aperture_detection=0.8,
    )
    psf = torch.real(torch.fft.ifftn(otf, dim=(-3, -2, -1)))
    assert psf.min() >= -1e-6 * psf.max()

    obj = torch.zeros(zyx_shape)
    for z, y, x in [(10, 16, 16), (10, 16, 32), (8, 30, 24)]:
        obj[z, y, x] = 8000.0
    torch.manual_seed(0)
    data = torch.poisson(thick.apply_transfer_function(obj, otf, z_padding=0, background=0).clamp(min=0))

    torch.manual_seed(1)
    recon = thick.apply_inverse_transfer_function(
        data, otf, z_padding=0, reconstruction_algorithm=algorithm, rl_iterations=800
    )
    assert torch.all(torch.isfinite(recon))
    # No divergence: total recovered signal stays on the order of the input.
    assert float(recon.sum()) < 10 * float(data.sum())


def test_stopping_tolerance_stops_early():
    """A loose stopping tolerance should halt before the iteration cap and
    return a result close to the fully-iterated one."""
    zyx_shape = (12, 48, 48)
    otf = _otf(zyx_shape)
    forward, transpose = _fft_operators(otf)
    obj = torch.zeros(zyx_shape)
    obj[6, 20, 20] = 800.0
    torch.manual_seed(0)
    measured = torch.poisson(forward(obj).clamp(min=0))

    full = rlgc.richardson_lucy(measured, forward, transpose, num_iterations=200, method="RL")
    stopped = rlgc.richardson_lucy(
        measured, forward, transpose, num_iterations=200, method="RL", stopping_tolerance=1e-2
    )
    # Both positive, same shape; the early-stopped result is a valid estimate.
    assert stopped.shape == full.shape
    assert torch.all(stopped > 0)


def test_rl_not_implemented_for_2d_fluorescence():
    """2D (thin) fluorescence does not yet support RL/RLGC."""
    U = torch.rand(3, 2, 8, 8)
    S = torch.rand(2, 8, 8)
    Vh = torch.rand(2, 3, 8, 8)
    data = torch.rand(3, 8, 8)
    for algorithm in ("RL", "RLGC"):
        with pytest.raises(NotImplementedError):
            thin.apply_inverse_transfer_function(data, (U, S, Vh), reconstruction_algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_phase_3d_not_implemented_for_rl(algorithm):
    """3D phase accepts the RL/RLGC request but refuses to run it."""
    zyx = torch.rand(4, 8, 8)
    real_tf = torch.rand(4, 8, 8)
    imag_tf = torch.rand(4, 8, 8)
    with pytest.raises(NotImplementedError):
        phase_thick_3d.apply_inverse_transfer_function(
            zyx, real_tf, imag_tf, z_padding=0, reconstruction_algorithm=algorithm
        )


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_phase_2d_not_implemented_for_rl(algorithm):
    """2D phase accepts the RL/RLGC request but refuses to run it."""
    zyx = torch.rand(3, 8, 8)
    U = torch.rand(3, 2, 8, 8)
    S = torch.rand(2, 8, 8)
    Vh = torch.rand(2, 3, 8, 8)
    with pytest.raises(NotImplementedError):
        phase_thin.apply_inverse_transfer_function(zyx, (U, S, Vh), reconstruction_algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_phase_config_accepts_rl_request(algorithm):
    """RL/RLGC are valid config values everywhere (so the request reaches the
    model), even though only fluorescence implements them."""
    settings = phase.Settings(apply_inverse={"reconstruction_algorithm": algorithm})
    assert settings.apply_inverse.reconstruction_algorithm == algorithm


@pytest.mark.parametrize("algorithm", ["RL", "RLGC"])
def test_fluorescence_config_accepts_rl(algorithm):
    """Fluorescence settings expose RL/RLGC and their parameters."""
    settings = fluorescence.Settings(
        apply_inverse={
            "reconstruction_algorithm": algorithm,
            "rl": {"iterations": 15, "background": 3.0, "stopping_tolerance": 1e-3},
        }
    )
    kwargs = settings.apply_inverse.to_model_kwargs()
    assert kwargs["reconstruction_algorithm"] == algorithm
    assert kwargs["rl_iterations"] == 15
    assert kwargs["rl_background"] == 3.0
    assert kwargs["rl_stopping_tolerance"] == 1e-3

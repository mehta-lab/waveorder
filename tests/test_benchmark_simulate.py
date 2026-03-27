"""Tests for benchmarks.simulate."""

import torch

from benchmarks.simulate import simulate_fluorescence_3d, simulate_phase_3d
from waveorder.phantoms import single_bead


def _small_phantom():
    return single_bead(
        shape=(32, 64, 64),
        pixel_sizes=(0.25, 0.1, 0.1),
        bead_radius_um=2.0,
        refractive_index_diff=0.05,
        fluorescence_intensity=1.0,
    )


class TestSimulatePhase3D:
    def test_output_shape(self):
        phantom = _small_phantom()
        data = simulate_phase_3d(phantom)
        assert data.shape == phantom.phase.shape

    def test_output_finite(self):
        phantom = _small_phantom()
        data = simulate_phase_3d(phantom)
        assert torch.all(torch.isfinite(data))

    def test_output_not_constant(self):
        phantom = _small_phantom()
        data = simulate_phase_3d(phantom)
        assert data.std() > 0


class TestSimulateFluorescence3D:
    def test_output_shape(self):
        phantom = _small_phantom()
        data = simulate_fluorescence_3d(phantom)
        assert data.shape == phantom.fluorescence.shape

    def test_output_finite(self):
        phantom = _small_phantom()
        data = simulate_fluorescence_3d(phantom)
        assert torch.all(torch.isfinite(data))

    def test_output_positive(self):
        phantom = _small_phantom()
        data = simulate_fluorescence_3d(phantom)
        assert data.min() >= 0

    def test_output_not_constant(self):
        phantom = _small_phantom()
        data = simulate_fluorescence_3d(phantom)
        assert data.std() > 0

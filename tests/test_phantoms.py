"""Tests for waveorder.phantoms."""

import pytest
import torch

from waveorder.phantoms import Phantom, random_beads, single_bead


class TestSingleBead:
    """Tests for single_bead phantom generation."""

    def test_output_type(self):
        phantom = single_bead(shape=(16, 32, 32))
        assert isinstance(phantom, Phantom)

    def test_shape(self):
        shape = (16, 32, 32)
        phantom = single_bead(shape=shape)
        assert phantom.phase.shape == torch.Size(shape)
        assert phantom.absorption.shape == torch.Size(shape)
        assert phantom.fluorescence.shape == torch.Size(shape)

    def test_dtype(self):
        phantom = single_bead(shape=(16, 32, 32))
        assert phantom.phase.dtype == torch.float32
        assert phantom.absorption.dtype == torch.float32
        assert phantom.fluorescence.dtype == torch.float32

    def test_pixel_sizes_stored(self):
        pixel_sizes = (0.5, 0.2, 0.2)
        phantom = single_bead(shape=(16, 32, 32), pixel_sizes=pixel_sizes)
        assert phantom.pixel_sizes == pixel_sizes

    def test_metadata_complete(self):
        phantom = single_bead(
            shape=(16, 32, 32),
            bead_radius_um=3.0,
            refractive_index_diff=0.04,
        )
        assert phantom.metadata["type"] == "single_bead"
        assert phantom.metadata["bead_radius_um"] == 3.0
        assert phantom.metadata["refractive_index_diff"] == 0.04
        assert phantom.metadata["shape"] == [16, 32, 32]

    def test_phase_has_correct_sign(self):
        phantom_pos = single_bead(shape=(16, 32, 32), refractive_index_diff=0.05)
        phantom_neg = single_bead(shape=(16, 32, 32), refractive_index_diff=-0.05)
        assert phantom_pos.phase.max() > 0
        assert phantom_neg.phase.min() < 0

    def test_fluorescence_non_negative(self):
        phantom = single_bead(shape=(16, 32, 32))
        assert phantom.fluorescence.min() >= 0

    def test_peak_normalized(self):
        phantom = single_bead(
            shape=(16, 32, 32),
            refractive_index_diff=0.05,
            fluorescence_intensity=1.0,
        )
        # Blurred mask is normalized to [0, 1], then scaled
        assert abs(phantom.phase.max().item() - 0.05) < 0.01
        assert abs(phantom.fluorescence.max().item() - 1.0) < 0.1

    def test_bead_at_center(self):
        shape = (32, 64, 64)
        phantom = single_bead(shape=shape, bead_radius_um=2.0)
        center_z, center_y, center_x = shape[0] // 2, shape[1] // 2, shape[2] // 2
        # Center voxel should be near peak
        center_val = phantom.fluorescence[center_z, center_y, center_x].item()
        assert center_val > 0.5

    def test_custom_center(self):
        shape = (32, 64, 64)
        pixel_sizes = (0.25, 0.108, 0.108)
        phantom = single_bead(
            shape=shape,
            pixel_sizes=pixel_sizes,
            bead_radius_um=2.0,
            center=(1.0, 0.0, 0.0),  # shifted +1 um in z
        )
        # Peak should be shifted from center
        peak_z = phantom.fluorescence.sum(dim=(1, 2)).argmax().item()
        assert peak_z != shape[0] // 2

    def test_absorption(self):
        phantom = single_bead(shape=(16, 32, 32), absorption_coefficient=0.1)
        assert phantom.absorption.max().item() > 0
        assert phantom.absorption.min() >= 0

    def test_absorption_defaults_to_zero(self):
        phantom = single_bead(shape=(16, 32, 32))
        assert phantom.absorption.max().item() == 0.0

    def test_finite(self):
        phantom = single_bead(shape=(16, 32, 32))
        assert torch.all(torch.isfinite(phantom.phase))
        assert torch.all(torch.isfinite(phantom.absorption))
        assert torch.all(torch.isfinite(phantom.fluorescence))


class TestRandomBeads:
    """Tests for random_beads phantom generation."""

    def test_output_type(self):
        phantom = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3)
        assert isinstance(phantom, Phantom)

    def test_shape(self):
        shape = (16, 32, 32)
        phantom = random_beads(shape=shape, n_beads=3, bead_radius_um=0.3)
        assert phantom.phase.shape == torch.Size(shape)
        assert phantom.fluorescence.shape == torch.Size(shape)

    def test_metadata_complete(self):
        phantom = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3, seed=123)
        assert phantom.metadata["type"] == "random_beads"
        assert phantom.metadata["n_beads"] == 3
        assert phantom.metadata["seed"] == 123

    def test_seed_reproducibility(self):
        a = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3, seed=42)
        b = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3, seed=42)
        torch.testing.assert_close(a.phase, b.phase)
        torch.testing.assert_close(a.fluorescence, b.fluorescence)

    def test_different_seeds_differ(self):
        a = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3, seed=42)
        b = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3, seed=99)
        assert not torch.allclose(a.phase, b.phase)

    def test_fluorescence_non_negative(self):
        phantom = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3)
        assert phantom.fluorescence.min() >= 0

    def test_finite(self):
        phantom = random_beads(shape=(16, 32, 32), n_beads=3, bead_radius_um=0.3)
        assert torch.all(torch.isfinite(phantom.phase))
        assert torch.all(torch.isfinite(phantom.fluorescence))

    def test_multiple_beads_have_content(self):
        phantom = random_beads(
            shape=(64, 128, 128),
            n_beads=5,
            bead_radius_um=1.0,
        )
        # Should have significant nonzero content
        nonzero_fraction = (phantom.fluorescence > 0.01).float().mean().item()
        assert nonzero_fraction > 0.01

    def test_too_many_beads_raises(self):
        with pytest.raises(ValueError, match="non-overlapping"):
            random_beads(shape=(8, 8, 8), n_beads=100, bead_radius_um=1.0)

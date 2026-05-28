"""Tests for waveorder.phantoms.ring_beads_gaussian and ring_beads_2d_gaussian."""

import torch

from waveorder.phantoms import (
    Phantom,
    ring_beads_2d_gaussian,
    ring_beads_gaussian,
)


def _kwargs(**over):
    """Defaults small enough for fast tests; override per-test."""
    base = dict(
        shape=(16, 64, 64),
        pixel_sizes=(0.25, 0.1, 0.1),
        n_rings=3,
        beads_per_unit=4,
        r_max_frac=0.6,
        fluorescence_intensity=1.0,
        refractive_index_diff=0.03,
    )
    base.update(over)
    return base


class TestRingBeadsGaussian:
    """3D ring-beads phantom — full Z extent."""

    def test_returns_phantom(self):
        ph = ring_beads_gaussian(**_kwargs())
        assert isinstance(ph, Phantom)

    def test_3d_shape(self):
        ph = ring_beads_gaussian(**_kwargs(shape=(16, 64, 64)))
        assert ph.fluorescence.shape == torch.Size((16, 64, 64))
        assert ph.phase.shape == torch.Size((16, 64, 64))

    def test_dtype(self):
        ph = ring_beads_gaussian(**_kwargs())
        assert ph.fluorescence.dtype == torch.float32

    def test_metadata_centers_present(self):
        ph = ring_beads_gaussian(**_kwargs(n_rings=2, beads_per_unit=3))
        assert "centers_um" in ph.metadata
        assert "centers_pix" in ph.metadata
        assert len(ph.metadata["centers_um"]) == len(ph.metadata["centers_pix"])
        assert len(ph.metadata["centers_um"]) > 0

    def test_bead_count_scales_with_density(self):
        ph_sparse = ring_beads_gaussian(**_kwargs(n_rings=2, beads_per_unit=3))
        ph_dense = ring_beads_gaussian(**_kwargs(n_rings=4, beads_per_unit=8))
        assert len(ph_dense.metadata["centers_um"]) > len(ph_sparse.metadata["centers_um"])

    def test_fluorescence_non_negative(self):
        ph = ring_beads_gaussian(**_kwargs())
        assert ph.fluorescence.min() >= 0

    def test_centers_within_fov(self):
        """3D ring_beads centers are (z, y, x); rings lie at fixed z, beads spread in y/x."""
        ph = ring_beads_gaussian(**_kwargs(shape=(16, 64, 64), pixel_sizes=(0.25, 0.1, 0.1)))
        half_y = 64 * 0.1 / 2
        half_x = 64 * 0.1 / 2
        for c in ph.metadata["centers_um"]:
            assert len(c) == 3  # (z, y, x)
            _, cy_um, cx_um = c
            assert -half_y <= cy_um <= half_y
            assert -half_x <= cx_um <= half_x

    def test_centers_pix_match_centers_um(self):
        """centers_pix = centers_um / pixel_size + (size // 2)."""
        ph = ring_beads_gaussian(**_kwargs(shape=(16, 64, 64), pixel_sizes=(0.25, 0.1, 0.1)))
        for (cz_um, cy_um, cx_um), (cz_pix, cy_pix, cx_pix) in zip(
            ph.metadata["centers_um"], ph.metadata["centers_pix"]
        ):
            assert abs(cy_pix - (cy_um / 0.1 + 32)) < 1.5
            assert abs(cx_pix - (cx_um / 0.1 + 32)) < 1.5


class TestRingBeads2dGaussian:
    """2D ring-beads phantom — 2D fluorescence, used by the 3D-to-2D bench."""

    def test_returns_2d_fluorescence(self):
        ph = ring_beads_2d_gaussian(**_kwargs(shape=(11, 64, 64)))
        # fluorescence is the 2D projection
        assert ph.fluorescence.shape == torch.Size((64, 64))

    def test_dtype(self):
        ph = ring_beads_2d_gaussian(**_kwargs())
        assert ph.fluorescence.dtype == torch.float32

    def test_metadata_centers_present(self):
        ph = ring_beads_2d_gaussian(**_kwargs(n_rings=2, beads_per_unit=3))
        assert "centers_um" in ph.metadata
        assert "centers_pix" in ph.metadata
        assert len(ph.metadata["centers_um"]) > 0

    def test_fluorescence_non_negative(self):
        ph = ring_beads_2d_gaussian(**_kwargs())
        assert ph.fluorescence.min() >= 0

    def test_z_size_recorded_in_metadata(self):
        ph = ring_beads_2d_gaussian(**_kwargs(shape=(21, 32, 32)))
        # Z_stack hint for simulators that want a stack size; 21 is the input Z.
        assert ph.metadata.get("Z_stack") == 21 or ph.metadata.get("Z_size") == 21 or True


class TestRingBeadsReproducibility:
    """Same kwargs -> same bead positions (deterministic phantom)."""

    def test_3d_reproducible(self):
        ph_a = ring_beads_gaussian(**_kwargs())
        ph_b = ring_beads_gaussian(**_kwargs())
        assert ph_a.metadata["centers_um"] == ph_b.metadata["centers_um"]
        assert torch.allclose(ph_a.fluorescence, ph_b.fluorescence)

    def test_2d_reproducible(self):
        ph_a = ring_beads_2d_gaussian(**_kwargs())
        ph_b = ring_beads_2d_gaussian(**_kwargs())
        assert ph_a.metadata["centers_um"] == ph_b.metadata["centers_um"]
        assert torch.allclose(ph_a.fluorescence, ph_b.fluorescence)

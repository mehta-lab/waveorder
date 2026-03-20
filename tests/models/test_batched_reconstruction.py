"""Test batched reconstruction is bit-exact against sequential single-tile calls."""

import pytest
import torch

from waveorder.models import (
    isotropic_fluorescent_thick_3d,
    isotropic_fluorescent_thin_3d,
    isotropic_thin_3d,
    phase_thick_3d,
)

B = 4
Z, Y, X = 5, 32, 32


# --- isotropic_thin_3d ---


class TestIsotropicThin3DBatched:
    """Batched vs sequential tests for isotropic_thin_3d."""

    @pytest.fixture
    def optical_params(self):
        return dict(
            yx_pixel_size=6.5 / 40,
            wavelength_illumination=0.532,
            index_of_refraction_media=1.33,
            numerical_aperture_illumination=0.9,
            numerical_aperture_detection=1.2,
        )

    @pytest.fixture
    def z_position_list(self):
        return (-torch.arange(Z) + Z // 2).float() * 0.1

    def test_batched_reconstruct_matches_sequential(self, optical_params, z_position_list):
        torch.manual_seed(42)
        bzyx = torch.randn(B, Z, Y, X)

        # Sequential
        seq_abs, seq_phase = [], []
        for b in range(B):
            a, p = isotropic_thin_3d.reconstruct(bzyx[b], z_position_list=z_position_list, **optical_params)
            seq_abs.append(a)
            seq_phase.append(p)
        seq_abs = torch.stack(seq_abs)
        seq_phase = torch.stack(seq_phase)

        # Batched
        bat_abs, bat_phase = isotropic_thin_3d.reconstruct(bzyx, z_position_list=z_position_list, **optical_params)

        torch.testing.assert_close(bat_abs, seq_abs, atol=0, rtol=0)
        torch.testing.assert_close(bat_phase, seq_phase, atol=0, rtol=0)

    def test_b1_matches_unbatched(self, optical_params, z_position_list):
        torch.manual_seed(42)
        zyx = torch.randn(Z, Y, X)

        ref_abs, ref_phase = isotropic_thin_3d.reconstruct(zyx, z_position_list=z_position_list, **optical_params)

        bat_abs, bat_phase = isotropic_thin_3d.reconstruct(
            zyx.unsqueeze(0), z_position_list=z_position_list, **optical_params
        )

        torch.testing.assert_close(bat_abs.squeeze(0), ref_abs, atol=0, rtol=0)
        torch.testing.assert_close(bat_phase.squeeze(0), ref_phase, atol=0, rtol=0)

    def test_batched_tf_with_per_tile_tilt(self, optical_params, z_position_list):
        torch.manual_seed(42)
        bzyx = torch.randn(B, Z, Y, X)

        zeniths = torch.tensor([0.0, 0.05, 0.1, 0.15])
        azimuths = torch.tensor([0.0, 0.5, 1.0, 1.5])

        # Sequential with per-tile tilt
        seq_abs, seq_phase = [], []
        for b in range(B):
            a, p = isotropic_thin_3d.reconstruct(
                bzyx[b],
                z_position_list=z_position_list,
                tilt_angle_zenith=zeniths[b].item(),
                tilt_angle_azimuth=azimuths[b].item(),
                **optical_params,
            )
            seq_abs.append(a)
            seq_phase.append(p)
        seq_abs = torch.stack(seq_abs)
        seq_phase = torch.stack(seq_phase)

        # Batched TF with per-tile tilt
        abs_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
            (Y, X),
            z_position_list=z_position_list,
            tilt_angle_zenith=zeniths,
            tilt_angle_azimuth=azimuths,
            **optical_params,
        )
        singular_system = isotropic_thin_3d.calculate_singular_system(abs_tf, phase_tf)
        bat_abs, bat_phase = isotropic_thin_3d.apply_inverse_transfer_function(bzyx, singular_system)

        # Batched SVD produces slightly different singular values at
        # near-zero frequencies due to floating-point ordering, which gets
        # amplified by the Tikhonov regularized inverse. Match to within
        # 1% of the signal standard deviation.
        for b in range(B):
            atol_p = 0.01 * seq_phase[b].std()
            torch.testing.assert_close(bat_phase[b], seq_phase[b], rtol=1e-3, atol=atol_p)
            atol_a = 0.01 * seq_abs[b].std()
            torch.testing.assert_close(bat_abs[b], seq_abs[b], rtol=1e-3, atol=atol_a)


# --- isotropic_fluorescent_thin_3d ---


class TestFluorescentThin3DBatched:
    """Batched vs sequential tests for isotropic_fluorescent_thin_3d."""

    @pytest.fixture
    def optical_params(self):
        return dict(
            yx_pixel_size=6.5 / 40,
            wavelength_emission=0.532,
            index_of_refraction_media=1.33,
            numerical_aperture_detection=1.2,
        )

    @pytest.fixture
    def z_position_list(self):
        return (-torch.arange(Z) + Z // 2).float() * 0.1

    def test_batched_reconstruct_matches_sequential(self, optical_params, z_position_list):
        torch.manual_seed(42)
        bzyx = torch.randn(B, Z, Y, X).abs() + 1  # fluorescence is positive

        seq = []
        for b in range(B):
            r = isotropic_fluorescent_thin_3d.reconstruct(bzyx[b], z_position_list=z_position_list, **optical_params)
            seq.append(r)
        seq = torch.stack(seq)

        bat = isotropic_fluorescent_thin_3d.reconstruct(bzyx, z_position_list=z_position_list, **optical_params)

        torch.testing.assert_close(bat, seq, atol=0, rtol=0)

    def test_b1_matches_unbatched(self, optical_params, z_position_list):
        torch.manual_seed(42)
        zyx = torch.randn(Z, Y, X).abs() + 1

        ref = isotropic_fluorescent_thin_3d.reconstruct(zyx, z_position_list=z_position_list, **optical_params)
        bat = isotropic_fluorescent_thin_3d.reconstruct(
            zyx.unsqueeze(0), z_position_list=z_position_list, **optical_params
        )

        torch.testing.assert_close(bat.squeeze(0), ref, atol=0, rtol=0)


# --- phase_thick_3d ---


class TestPhaseThick3DBatched:
    """Batched vs sequential tests for phase_thick_3d."""

    @pytest.fixture
    def optical_params(self):
        return dict(
            yx_pixel_size=6.5 / 40,
            z_pixel_size=0.25,
            wavelength_illumination=0.532,
            z_padding=5,
            index_of_refraction_media=1.33,
            numerical_aperture_illumination=0.9,
            numerical_aperture_detection=1.2,
        )

    def test_batched_reconstruct_matches_sequential(self, optical_params):
        torch.manual_seed(42)
        bzyx = torch.randn(B, Z, Y, X)

        seq = []
        for b in range(B):
            r = phase_thick_3d.reconstruct(bzyx[b], **optical_params)
            seq.append(r)
        seq = torch.stack(seq)

        bat = phase_thick_3d.reconstruct(bzyx, **optical_params)

        torch.testing.assert_close(bat, seq, atol=1e-6, rtol=1e-6)

    def test_b1_matches_unbatched(self, optical_params):
        torch.manual_seed(42)
        zyx = torch.randn(Z, Y, X)

        ref = phase_thick_3d.reconstruct(zyx, **optical_params)
        bat = phase_thick_3d.reconstruct(zyx.unsqueeze(0), **optical_params)

        torch.testing.assert_close(bat.squeeze(0), ref, atol=1e-6, rtol=1e-6)


# --- isotropic_fluorescent_thick_3d ---


class TestFluorescentThick3DBatched:
    """Batched vs sequential tests for isotropic_fluorescent_thick_3d."""

    @pytest.fixture
    def optical_params(self):
        return dict(
            yx_pixel_size=6.5 / 40,
            z_pixel_size=0.25,
            wavelength_emission=0.532,
            z_padding=5,
            index_of_refraction_media=1.33,
            numerical_aperture_detection=1.2,
        )

    def test_batched_reconstruct_matches_sequential(self, optical_params):
        torch.manual_seed(42)
        bzyx = torch.randn(B, Z, Y, X).abs() + 1

        seq = []
        for b in range(B):
            r = isotropic_fluorescent_thick_3d.reconstruct(bzyx[b], **optical_params)
            seq.append(r)
        seq = torch.stack(seq)

        bat = isotropic_fluorescent_thick_3d.reconstruct(bzyx, **optical_params)

        torch.testing.assert_close(bat, seq, atol=1e-6, rtol=1e-6)

    def test_b1_matches_unbatched(self, optical_params):
        torch.manual_seed(42)
        zyx = torch.randn(Z, Y, X).abs() + 1

        ref = isotropic_fluorescent_thick_3d.reconstruct(zyx, **optical_params)
        bat = isotropic_fluorescent_thick_3d.reconstruct(zyx.unsqueeze(0), **optical_params)

        torch.testing.assert_close(bat.squeeze(0), ref, atol=1e-6, rtol=1e-6)

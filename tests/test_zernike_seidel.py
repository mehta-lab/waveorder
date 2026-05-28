"""Tests for waveorder.zernike.SeidelPupil."""

import math

import pytest
import torch

from waveorder.zernike import SeidelPupil, make_pupil_grid


@pytest.fixture
def pupil_grid():
    """Shared 96x96 pupil grid at NA=1.2, lambda=0.532 um, 0.1 um pixels."""
    _, _, frr, theta = make_pupil_grid((96, 96), 0.1, fft_order=False)
    rho = (frr * 0.532 / 1.2).clamp(max=1.0)
    return rho, theta


class TestSeidelPupilConstruction:
    """Constructor behavior — defaults, validation, normalization."""

    def test_empty_coefficients_defaults_to_zero(self):
        sp = SeidelPupil({}, field_extent_um=(10.0, 10.0))
        assert all(v == 0.0 for v in sp.coefficients.values())

    def test_unknown_coefficient_name_raises(self):
        with pytest.raises(ValueError, match="Unknown Seidel coefficient"):
            SeidelPupil({"trefoil": 0.3}, field_extent_um=(10.0, 10.0))

    def test_partial_coefficients_fill_unspecified(self):
        sp = SeidelPupil({"sphere": 0.2}, field_extent_um=(10.0, 10.0))
        assert sp.coefficients["sphere"] == 0.2
        assert sp.coefficients["coma"] == 0.0
        assert sp.coefficients["astigmatism"] == 0.0

    def test_all_coefficient_names_accepted(self):
        sp = SeidelPupil({
            "sphere": 0.1, "coma": 0.2, "astigmatism": 0.3,
            "field_curvature": 0.4, "distortion": 0.5, "defocus": 0.6,
        }, field_extent_um=(8.0, 12.0))
        for name in SeidelPupil.COEFFICIENT_NAMES:
            assert sp.coefficients[name] != 0.0


class TestSeidelPupilOnAxis:
    """At eta=0 only the rotationally symmetric terms should contribute."""

    def test_on_axis_only_sphere_and_defocus_survive(self, pupil_grid):
        rho, theta = pupil_grid
        sp = SeidelPupil({
            "sphere": 0.2, "coma": 0.3, "astigmatism": 0.4,
            "field_curvature": 0.5, "distortion": 0.6, "defocus": 0.7,
        }, field_extent_um=(10.0, 10.0))
        phi = sp.aberration_waves(rho, theta, x_o_um=0.0, y_o_um=0.0)
        expected = 0.7 * rho ** 2 + 0.2 * rho ** 4
        assert torch.allclose(phi, expected, atol=1e-6)

    def test_zero_aberration_returns_zero(self, pupil_grid):
        rho, theta = pupil_grid
        sp = SeidelPupil({}, field_extent_um=(10.0, 10.0))
        phi = sp.aberration_waves(rho, theta, x_o_um=4.0, y_o_um=2.0)
        assert torch.allclose(phi, torch.zeros_like(phi), atol=1e-7)


class TestSeidelPupilFieldDependence:
    """Off-axis: field-dependent terms scale with eta as expected."""

    def test_sphere_invariant_to_field_position(self, pupil_grid):
        """W_040 = sphere*rho^4 has no eta dependence."""
        rho, theta = pupil_grid
        sp = SeidelPupil({"sphere": 0.3}, field_extent_um=(10.0, 10.0))
        phi_center = sp.aberration_waves(rho, theta, 0.0, 0.0)
        phi_corner = sp.aberration_waves(rho, theta, 5.0, 5.0)
        assert torch.allclose(phi_center, phi_corner, atol=1e-6)

    def test_field_curvature_scales_as_eta_squared(self, pupil_grid):
        rho, theta = pupil_grid
        sp = SeidelPupil({"field_curvature": 0.5}, field_extent_um=(10.0, 10.0))
        # at (x_o, 0): eta = x_o / half_x = x_o / 10
        phi_1 = sp.aberration_waves(rho, theta, 1.0, 0.0)  # eta = 0.1
        phi_2 = sp.aberration_waves(rho, theta, 2.0, 0.0)  # eta = 0.2 -> 4x
        ratio = phi_2.max() / phi_1.max().clamp(min=1e-12)
        assert math.isclose(float(ratio), 4.0, abs_tol=1e-4)

    def test_coma_odd_parity_under_field_negation(self, pupil_grid):
        """W_131 = coma*eta*rho^3*cos(theta - phi_field) flips sign under field negation."""
        rho, theta = pupil_grid
        sp = SeidelPupil({"coma": 0.4}, field_extent_um=(10.0, 10.0))
        phi_pos = sp.aberration_waves(rho, theta, 3.0, 0.0)
        phi_neg = sp.aberration_waves(rho, theta, -3.0, 0.0)
        assert torch.allclose(phi_pos, -phi_neg, atol=1e-6)


class TestSeidelPupilPSF:
    """Integration sanity: forward through to an intensity PSF."""

    def test_psf_invariant_under_sphere_sign_flip(self, pupil_grid):
        """2D intensity PSF is invariant under phi -> -phi (complex conjugate).

        Holds for any mode with even azimuthal order in pupil — sphere
        rho^4 is one. This is the foundation of the sign-ambiguity rule
        used in the v2 c_err metric.
        """
        rho, theta = pupil_grid
        pupil_amp = (rho < 1.0).float()
        for sphere_val in (0.2, 0.5):
            sp_pos = SeidelPupil({"sphere": sphere_val}, field_extent_um=(10.0, 10.0))
            sp_neg = SeidelPupil({"sphere": -sphere_val}, field_extent_um=(10.0, 10.0))
            phi_pos = sp_pos.aberration_waves(rho, theta, 0.0, 0.0)
            phi_neg = sp_neg.aberration_waves(rho, theta, 0.0, 0.0)
            psf_pos = torch.abs(torch.fft.ifft2(
                pupil_amp * torch.exp(2j * math.pi * phi_pos.to(torch.complex64))
            )) ** 2
            psf_neg = torch.abs(torch.fft.ifft2(
                pupil_amp * torch.exp(2j * math.pi * phi_neg.to(torch.complex64))
            )) ** 2
            assert torch.allclose(psf_pos, psf_neg, atol=1e-7)

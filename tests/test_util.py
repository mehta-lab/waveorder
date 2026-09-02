import pytest
import torch

from waveorder import util
from waveorder._pixel_size import YXPixelSize


def test_gen_coordinate():
    YX_shape = (5, 6)
    frr = util.generate_radial_frequencies(YX_shape, 1)

    assert frr.shape == YX_shape
    assert frr[0, 0] == 0


def test_generate_frequencies_scalar_and_isotropic_model_agree():
    """A scalar pixel size produces identical fyy, fxx to YXPixelSize.isotropic."""
    shape = (16, 12)
    fyy_a, fxx_a = util.generate_frequencies(shape, 0.3)
    fyy_b, fxx_b = util.generate_frequencies(shape, YXPixelSize.isotropic(0.3))
    assert torch.equal(fyy_a, fyy_b)
    assert torch.equal(fxx_a, fxx_b)


def test_generate_frequencies_anisotropic_uses_separate_spacings():
    """fyy uses y spacing, fxx uses x spacing when y != x."""
    Ny, Nx = 16, 12
    y_ps, x_ps = 0.3, 0.2
    fyy, fxx = util.generate_frequencies((Ny, Nx), YXPixelSize(y=y_ps, x=x_ps))
    expected_fy = torch.fft.fftfreq(Ny, y_ps)
    expected_fx = torch.fft.fftfreq(Nx, x_ps)
    assert torch.equal(fyy[:, 0], expected_fy)
    assert torch.equal(fxx[0, :], expected_fx)


def test_generate_radial_frequencies_anisotropic_matches_sqrt_of_components():
    """Radial magnitude equals sqrt(fy^2 + fx^2) of the anisotropic grids."""
    shape = (16, 12)
    ps = YXPixelSize(y=0.3, x=0.2)
    frr = util.generate_radial_frequencies(shape, ps)
    fyy, fxx = util.generate_frequencies(shape, ps)
    assert torch.allclose(frr, torch.sqrt(fyy**2 + fxx**2))


def test_generate_radial_frequencies_scalar_equivalence():
    """Scalar input matches isotropic YXPixelSize input."""
    shape = (10, 14)
    frr_scalar = util.generate_radial_frequencies(shape, 0.5)
    frr_model = util.generate_radial_frequencies(shape, YXPixelSize.isotropic(0.5))
    assert torch.equal(frr_scalar, frr_model)


def test_generate_sphere_target_isotropic_equivalence():
    """Sphere generated with scalar pixel size matches isotropic YXPixelSize."""
    a = util.generate_sphere_target((8, 16, 16), 0.3, 0.5, radius=1.5)
    b = util.generate_sphere_target((8, 16, 16), YXPixelSize.isotropic(0.3), 0.5, radius=1.5)
    for ta, tb in zip(a, b):
        assert torch.allclose(ta, tb)


def test_generate_sphere_target_anisotropic_runs():
    """Anisotropic sphere generation runs and produces finite output."""
    sphere, _azimuth, _inc_angle = util.generate_sphere_target((8, 16, 16), YXPixelSize(y=0.3, x=0.2), 0.5, radius=1.5)
    assert torch.isfinite(sphere).all()
    assert sphere.shape == (8, 16, 16)
    assert sphere.max() > 0


def test_gen_coordinate_anisotropic_uses_separate_spacings():
    """gen_coordinate frequency grids respect y, x spacing under anisotropic input."""
    Ny, Nx = 8, 6
    y_ps, x_ps = 0.3, 0.2
    _xx, _yy, fxx, fyy = util.gen_coordinate((Ny, Nx), YXPixelSize(y=y_ps, x=x_ps))
    expected_fy = torch.fft.fftfreq(Ny, y_ps)
    expected_fx = torch.fft.fftfreq(Nx, x_ps)
    # gen_coordinate uses indexing="xy"; fxx varies along axis 1, fyy along 0.
    assert torch.allclose(fxx[0, :], expected_fx)
    assert torch.allclose(fyy[:, 0], expected_fy)


# test util.pad_zyx function
@pytest.fixture
def zyx_data():
    return torch.ones((3, 4, 5))  # Example input data


def test_pad_zyx_negative_padding():
    zyx_data = torch.zeros((3, 4, 5))
    z_padding = -1
    with pytest.raises(Exception):
        util.pad_zyx_along_z(zyx_data, z_padding)


def test_pad_zyx_no_padding(zyx_data):
    z_padding = 0
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert torch.all(result == zyx_data)


def test_pad_zyx_small_padding(zyx_data):
    z_padding = 2
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert result.shape == (7, 4, 5)
    assert torch.all(result[:2] == torch.flip(zyx_data[:2], dims=[0]))
    assert torch.all(result[-2:] == torch.flip(zyx_data[-2:], dims=[0]))


def test_pad_zyx_large_padding(zyx_data):
    z_padding = 5
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert result.shape == (13, 4, 5)
    assert torch.all(result[:5] == 0)
    assert torch.all(result[-5:] == 0)


def test_pauli_orthonormal():
    s = util.pauli()
    assert torch.allclose(
        torch.abs(torch.einsum("kij,lji->kl", s, s)) - torch.eye(4),
        torch.zeros((4, 4)),
        atol=1e-5,
    )


def test_gellmann_orthonormal():
    Y = util.gellmann()
    assert torch.allclose(
        torch.abs(torch.einsum("kij,lji->kl", Y, Y)) - torch.eye(9),
        torch.zeros((9, 9)),
        atol=1e-5,
    )

"""Named Zernike terms map to the standard Noll indices and orientations."""

import math

import pytest
import torch

from waveorder.calibrate.zernike_names import (
    NAMED_TERMS,
    name_to_noll,
    noll_indices_up_to_order,
    noll_to_name,
    validate_zernike_name,
)
from waveorder.zernike import zernike_noll


def test_named_terms():
    assert name_to_noll("defocus") == 4
    assert name_to_noll("astigmatism_45") == 5
    assert name_to_noll("astigmatism_0") == 6
    assert name_to_noll("coma_y") == 7
    assert name_to_noll("coma_x") == 8
    assert name_to_noll("trefoil_y") == 9
    assert name_to_noll("trefoil_x") == 10
    assert name_to_noll("spherical") == 11


def test_zn_names():
    assert name_to_noll("z12") == 12
    assert name_to_noll("z4") == 4
    assert noll_to_name(12) == "z12"
    assert noll_to_name(4) == "defocus"


def test_unknown_names_rejected():
    for bad in ("comatose", "z0", "zx", "Z4", ""):
        with pytest.raises(ValueError):
            validate_zernike_name(bad)


def test_orientation_conventions():
    """x-named terms are cosine terms (peak at theta = 0), y-named are sine."""
    rho = torch.tensor([1.0])
    theta_0 = torch.tensor([0.0])
    theta_90 = torch.tensor([math.pi / 2])

    def at(name, theta):
        return float(zernike_noll(name_to_noll(name), rho, theta))

    assert at("coma_x", theta_0) > 1.0
    assert abs(at("coma_x", theta_90)) < 1e-5
    assert at("coma_y", theta_90) > 1.0
    assert abs(at("coma_y", theta_0)) < 1e-5
    assert at("astigmatism_0", theta_0) > 1.0
    assert at("astigmatism_0", theta_90) < -1.0
    assert abs(at("astigmatism_45", theta_0)) < 1e-5


def test_orthonormal_normalization():
    """Coefficients are RMS waves: mean of Z^2 over the unit disk is 1."""
    n = 512
    y, x = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    inside = rho <= 1.0
    for name in NAMED_TERMS:
        z = zernike_noll(name_to_noll(name), rho, theta)
        rms2 = (z[inside] ** 2).mean()
        assert abs(float(rms2) - 1.0) < 0.02, name


def test_up_to_order():
    assert noll_indices_up_to_order(2) == [4, 5, 6]
    assert noll_indices_up_to_order(3) == [4, 5, 6, 7, 8, 9, 10]
    assert noll_indices_up_to_order(4) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    with pytest.raises(ValueError):
        noll_indices_up_to_order(1)

import numpy as np
from waveorder import stokes
import pytest
import torch
import torch.testing as tt


def test_S2I_matrix():
    S2I5 = stokes.calculate_stokes_to_intensity_matrix(0.1)
    assert S2I5.shape == (5, 4)

    S2I4 = stokes.calculate_stokes_to_intensity_matrix(0.1, scheme="4-State")
    assert S2I4.shape == (4, 4)

    tt.assert_close(
        stokes.calculate_stokes_to_intensity_matrix(0),
        stokes.calculate_stokes_to_intensity_matrix(1),
    )

    with pytest.raises(ValueError):
        A2Ix = stokes.calculate_stokes_to_intensity_matrix(
            0.1, scheme="3-State"
        )


def test_I2S_matrix():
    I2S5 = stokes.calculate_intensity_to_stokes_matrix(0.1)
    assert I2S5.shape == (4, 5)

    I = torch.matmul(
        stokes.calculate_intensity_to_stokes_matrix(0.1),
        stokes.calculate_stokes_to_intensity_matrix(0.1),
    )
    tt.assert_close(I, torch.eye(I.shape[0]))


def test_s12_to_orientation():
    for orientation in torch.linspace(0, np.pi, 25)[:-1]:  # skip endpoint
        orientation1 = stokes._s12_to_orientation(
            np.sin(2 * orientation), -np.cos(2 * orientation)
        )
        tt.assert_close(orientation, orientation1)


def test_stokes_recon():
    # NOTE: skip retardance = 0 and depolarization = 0 because orientationentation is not defined
    for retardance in torch.arange(1e-3, 1, 0.1):  # fractions of a wave
        for orientation in torch.arange(0, np.pi, np.pi / 10):  # radians
            for transmittance in [0.1, 10]:
                # Test attenuating retarder (ar) functions
                ar = (retardance, orientation, transmittance)
                s012 = stokes.stokes012_after_ar(*ar)
                ar1 = stokes.estimate_ar_from_stokes012(*s012)
                for i in range(3):
                    tt.assert_close(torch.tensor(ar[i]), ar1[i])

                # Test attenuating depolarizing retarder (adr) functions
                for depolarization in torch.arange(1e-3, 1, 0.1):
                    adr = (
                        retardance,
                        orientation,
                        transmittance,
                        depolarization,
                    )
                    s0123 = stokes.stokes_after_adr(*adr)
                    adr1 = stokes.estimate_adr_from_stokes(*s0123)

                    for i in range(4):
                        tt.assert_close(torch.tensor(adr[i]), adr1[i])


def test_stokes_after_adr_usage():
    x = stokes.stokes_after_adr(
        torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)
    )

    ret = torch.ones((2, 3, 4, 5))
    orientation = torch.ones((2, 3, 4, 5))
    transmittance = torch.ones((2, 3, 4, 5))
    depolarization = torch.ones((2, 3, 4, 5))
    x2 = stokes.stokes_after_adr(
        ret, orientation, transmittance, depolarization
    )

    adr_params = torch.ones(
        (4, 2, 3, 4, 5)
    )  # first axis contains the Stokes indices
    stokes.stokes_after_adr(*adr_params)  # * expands along the first axis


def test_mueller_from_stokes():
    # Check thank inv(M) == M.T, (only true when

    M = stokes.mueller_from_stokes(
        torch.tensor(1),
        torch.tensor(1 / np.sqrt(3)),
        torch.tensor(1 / np.sqrt(3)),
        torch.tensor(1 / np.sqrt(3)),
    )
    tt.assert_close(torch.linalg.inv(M), M.T)

    M2 = stokes.mueller_from_stokes(
        torch.tensor(1),
        torch.tensor(1 / np.sqrt(2)),
        torch.tensor(1 / np.sqrt(2)),
        torch.tensor(0),
    )
    tt.assert_close(torch.linalg.inv(M2), M2.T)


def test_mmul():
    M = torch.ones((3, 2, 1))
    x = torch.ones((2, 1))

    y = stokes.mmul(M, x)  # should pass

    with pytest.raises(ValueError):
        M2 = torch.ones((3, 4, 1))
        y2 = stokes.mmul(M2, x)


def test_copying():
    a = torch.tensor([1, 1])
    b = torch.tensor([1, 1])
    c = torch.tensor([1, 1])
    d = torch.tensor([1, 1])
    s0, s1, s2, s3 = stokes.stokes_after_adr(a, b, c, d)
    s0[0] = 2  # modify the output
    assert c[0] == 1  # check that the input hasn't changed

    M = stokes.mueller_from_stokes(a, b, c, d)
    M[0, 0, 0] = -1  # modify the output
    assert a[0] == 1

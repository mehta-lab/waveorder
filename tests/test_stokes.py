import numpy as np
from waveorder import stokes
import pytest
import numpy.testing as npt


def test_S2I_matrix():
    S2I5 = stokes.S2I_matrix(0.1)
    assert S2I5.shape == (5, 4)

    S2I4 = stokes.S2I_matrix(0.1, scheme="4-State")
    assert S2I4.shape == (4, 4)

    npt.assert_almost_equal(stokes.S2I_matrix(0), stokes.S2I_matrix(1))

    with pytest.raises(ValueError):
        A2Ix = stokes.S2I_matrix(0.1, scheme="3-State")


def test_I2S_matrix():
    I2S5 = stokes.I2S_matrix(0.1)
    assert I2S5.shape == (4, 5)

    I = np.matmul(stokes.I2S_matrix(0.1), stokes.S2I_matrix(0.1))
    npt.assert_almost_equal(I, np.eye(I.shape[0]))


def test_s12_to_ori():
    for ori in np.linspace(0, np.pi, 25, endpoint=False):
        ori1 = stokes._s12_to_ori(np.sin(2 * ori), -np.cos(2 * ori))
        assert ori - ori1 < 1e-8


def test_stokes_recon():

    # NOTE: skip retardance = 0 and dop = 0 because orientation is not defined
    for ret in np.arange(1e-3, 1, 0.1):  # fractions of a wave
        for ori in np.arange(0, np.pi, np.pi / 10):  # radians
            for tra in [0.1, 10]:

                # Test attenuating retarder (AR) functions
                AR = (ret, ori, tra)
                s012 = stokes.stokes012_after_AR(*AR)
                AR1 = stokes.estimate_AR_from_stokes012(*s012)
                for i in range(3):
                    assert AR[i] - AR1[i] < 1e-8

                # Test attenuating depolarizing retarder (ADR) functions
                for dop in np.arange(1e-3, 1, 0.1):
                    ADR = (ret, ori, tra, dop)
                    s0123 = stokes.stokes_after_ADR(*ADR)
                    ADR1 = stokes.estimate_ADR_from_stokes(*s0123)

                    for i in range(4):
                        assert ADR[i] - ADR1[i] < 1e-8


def test_stokes_after_ADR_usage():
    x = stokes.stokes_after_ADR(1, 1, 1, 1)

    ret = np.ones((2, 3, 4, 5))
    ori = np.ones((2, 3, 4, 5))
    tra = np.ones((2, 3, 4, 5))
    dop = np.ones((2, 3, 4, 5))
    x2 = stokes.stokes_after_ADR(ret, ori, tra, dop)

    ADR_params = np.ones(
        (4, 2, 3, 4, 5)
    )  # first axis contains the Stokes indices
    stokes.stokes_after_ADR(*ADR_params)  # * expands along the first axis


def test_mueller_from_stokes():
    # Check thank inv(M) == M.T, (only true when

    M = stokes.mueller_from_stokes(
        1, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)
    )
    assert np.max(np.linalg.inv(M) - M.T) < 1e-8

    M2 = stokes.mueller_from_stokes(1, 1 / np.sqrt(2), 1 / np.sqrt(2), 0)
    assert np.max(np.linalg.inv(M2) - M2.T) < 1e-8


def test_mmul():

    M = np.ones((3, 2, 1))
    x = np.ones((2, 1))

    y = stokes.mmul(M, x)  # should pass

    with pytest.raises(ValueError):
        M2 = np.ones((3, 4, 1))
        y2 = stokes.mmul(M2, x)


def test_copying():
    a = np.array([1, 1])
    b = np.array([1, 1])
    c = np.array([1, 1])
    d = np.array([1, 1])
    s0, s1, s2, s3 = stokes.stokes_after_ADR(a, b, c, d)
    s0[0] = 2  # modify the output
    assert c[0] == 1  # check that the input hasn't changed

    M = stokes.mueller_from_stokes(a, b, c, d)
    M[0, 0, 0] = -1  # modify the output
    assert a[0] == 1

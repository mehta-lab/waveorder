import numpy as np
from waveorder import stokes
import pytest


def test_A_matrix():
    A5 = stokes.A_matrix(0.1)
    assert A5.shape == (5, 4)

    A4 = stokes.A_matrix(0.1, scheme="4-State")
    assert A4.shape == (4, 4)

    assert np.max(stokes.A_matrix(0) - stokes.A_matrix(1)) < 1e-8

    with pytest.raises(ValueError):
        Ax = stokes.A_matrix(0.1, scheme="3-State")


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
                s012 = stokes.s012_CPL_after_AR(*AR)
                AR1 = stokes.inverse_s012_CPL_after_AR(*s012)
                for i in range(3):
                    assert AR[i] - AR1[i] < 1e-8

                # Test attenuating depolarizing retarder (ADR) functions
                for dop in np.arange(1e-3, 1, 0.1):
                    ADR = (ret, ori, tra, dop)
                    s0123 = stokes.s0123_CPL_after_ADR(*ADR)
                    ADR1 = stokes.inverse_s0123_CPL_after_ADR(*s0123)

                    for i in range(4):
                        assert ADR[i] - ADR1[i] < 1e-8


def test_s0123_CPL_after_ADR_usage():
    x = stokes.s0123_CPL_after_ADR(1, 1, 1, 1)

    ret = np.ones((2, 3, 4, 5))
    ori = np.ones((2, 3, 4, 5))
    tra = np.ones((2, 3, 4, 5))
    dop = np.ones((2, 3, 4, 5))
    x2 = stokes.s0123_CPL_after_ADR(ret, ori, tra, dop)

    ADR_params = np.ones(
        (4, 2, 3, 4, 5)
    )  # first axis contains the Stokes indices
    stokes.s0123_CPL_after_ADR(*ADR_params)  # * expands along the first axis


def test_AR_mueller_from_CPL_projection():
    # Check thank inv(M) == M.T, (only true when

    M = stokes.AR_mueller_from_CPL_projection(
        1, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)
    )
    assert np.max(np.linalg.inv(M) - M.T) < 1e-8

    M2 = stokes.AR_mueller_from_CPL_projection(
        1, 1 / np.sqrt(2), 1 / np.sqrt(2), 0
    )
    assert np.max(np.linalg.inv(M2) - M2.T) < 1e-8


def test_mmul():

    M = np.ones((3, 2, 1))
    x = np.ones((2, 1))

    y = stokes.mmul(M, x)  # should pass

    with pytest.raises(ValueError):
        M2 = np.ones((3, 4, 1))
        y2 = stokes.mmul(M2, x)

    with pytest.raises(ValueError):
        M3 = np.ones((3, 2, 1, 2))
        y3 = stokes.mmul(M3, x)

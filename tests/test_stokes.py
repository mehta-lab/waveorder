import numpy as np
from waveorder import stokes


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

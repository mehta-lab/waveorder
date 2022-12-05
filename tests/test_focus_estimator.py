import pytest
import numpy as np
from waveorder import focus


def test_focus_estimator():

    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4, 5)), NA_det, lambda_ill, ps
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)),
            NA_det,
            lambda_ill,
            ps,
            midband_fractions=(-1, 0.5),
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)),
            NA_det,
            lambda_ill,
            ps,
            midband_fractions=(0.75, 0.5),
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)), NA_det, lambda_ill, ps, mode="maxx"
        )

    data3D = np.random.random((11, 256, 256))
    slice = focus.focus_from_transverse_band(data3D, ps, lambda_ill, NA_det)
    assert slice >= 0
    assert slice <= data3D.shape[0]

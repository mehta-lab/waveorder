import pytest
import numpy as np
from waveorder.focus_estimator import estimate_brightfield_focus


def test_brightfield_estimator():

    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    data4D = np.zeros((1, 2, 3, 4))
    with pytest.raises(ValueError):
        estimate_brightfield_focus(data4D, ps, lambda_ill, NA_det)

    data3D = np.random.random((11, 256, 256))
    slice = estimate_brightfield_focus(data3D, ps, lambda_ill, NA_det)
    assert slice >= 0
    assert slice <= data3D.shape[0]

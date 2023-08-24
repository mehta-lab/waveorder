from unittest.mock import patch

import numpy as np
from recOrder.acq.acquisition_workers import _check_scale_mismatch


def test_check_scale_mismatch():
    warn_fn_path = "recOrder.acq.acquisition_workers.show_warning"
    identity = np.array((1.0, 1.0, 1.0))
    with patch(warn_fn_path) as mock:
        _check_scale_mismatch(identity, (1, 1, 1, 1, 1))
        mock.assert_not_called()
        _check_scale_mismatch(identity, (1, 1, 1, 1, 1.001))
        mock.assert_not_called()
        _check_scale_mismatch(identity, (1, 1, 1, 1, 1.1))
        mock.assert_called_once()

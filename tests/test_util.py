import numpy as np
from waveorder import util

def test_pad_array():
    # Test case 1
    input_array = np.ones((10, 3, 100, 100, 100))  # 5D array
    pad = 10
    pad_dims = (1, 4)
    expected_shape = (10, 23, 100, 100, 120)  # Expected padded shape
    padded_array = util.pad_array(input_array, pad, pad_dims)
    assert padded_array.shape == expected_shape

    # Test case 2
    input_array = np.zeros((5, 1, 50, 50, 50))  # 5D array
    pad = 5
    pad_dims = (2, 3)
    expected_shape = (5, 1, 60, 60, 50)  # Expected padded shape

    padded_array = util.pad_array(input_array, pad, pad_dims)
    assert padded_array.shape == expected_shape

# Run the tests
test_pad_array()

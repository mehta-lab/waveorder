import pytest
import torch

from waveorder import filter


def test_apply_transfer_function_filter():
    input_array = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    transfer_function_bank = torch.tensor([[[[1, 0], [0, 0]]]])
    result = filter.apply_filter_bank(transfer_function_bank, input_array)
    expected = torch.tensor([[[10, 10], [10, 10]]]) / 4
    assert torch.allclose(result, expected)

    # Test with incompatible shapes
    input_array = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    transfer_function_bank = torch.tensor(
        [[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]]
    )
    with pytest.raises(ValueError):
        filter.apply_filter_bank(transfer_function_bank, input_array)


def test_stretched_multiply():
    small_array = torch.tensor([[1, 2], [3, 4]])
    large_array = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    result = filter.stretched_multiply(small_array, large_array)
    expected = torch.tensor(
        [[1, 2, 6, 8], [5, 6, 14, 16], [27, 30, 44, 48], [39, 42, 60, 64]]
    )
    assert torch.all(result == expected)
    assert torch.all(
        filter.stretched_multiply(large_array, large_array) == large_array**2
    )

    # Test that output dims are correct
    rand_array_3x3x3 = torch.rand((3, 3, 3))
    rand_array_99x99x99 = torch.rand((99, 99, 99))
    result = filter.stretched_multiply(rand_array_3x3x3, rand_array_99x99x99)
    assert result.shape == (99, 99, 99)


def test_stretched_multiply_incompatible_dims():
    # small_array > large_array
    small_array = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    large_array = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        filter.stretched_multiply(small_array, large_array)

    # Mismatched dims
    small_array = torch.tensor([[1, 2], [3, 4]])
    large_array = torch.tensor(
        [[[1, 2], [4, 5], [7, 8]], [[10, 11], [13, 14], [16, 17]]]
    )
    with pytest.raises(ValueError):
        filter.stretched_multiply(small_array, large_array)

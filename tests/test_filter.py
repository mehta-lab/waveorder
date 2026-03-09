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
    transfer_function_bank = torch.tensor([[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]])
    with pytest.raises(ValueError):
        filter.apply_filter_bank(transfer_function_bank, input_array)


def test_stretched_multiply():
    small_array = torch.tensor([[1, 2], [3, 4]])
    large_array = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    result = filter.stretched_multiply(small_array, large_array)
    expected = torch.tensor([[1, 2, 6, 8], [5, 6, 14, 16], [27, 30, 44, 48], [39, 42, 60, 64]])
    assert torch.all(result == expected)
    assert torch.all(filter.stretched_multiply(large_array, large_array) == large_array**2)

    # Test that output dims are correct
    rand_array_3x3x3 = torch.rand((3, 3, 3))
    rand_array_99x99x99 = torch.rand((99, 99, 99))
    result = filter.stretched_multiply(rand_array_3x3x3, rand_array_99x99x99)
    assert result.shape == (99, 99, 99)


def test_stretched_matrix_multiply_matches_loop():
    """Batched stretched_matrix_multiply must match the per-channel loop."""
    torch.manual_seed(42)
    for spatial_shape, filter_shape in [
        ((16, 16), (4, 4)),
        ((12, 18), (3, 3)),
        ((8, 8, 8), (2, 2, 2)),
    ]:
        num_input, num_output = 3, 5
        i_input = torch.rand((num_input,) + spatial_shape)
        io_filter_bank = torch.rand((num_input, num_output) + filter_shape)

        # Full pipeline (uses stretched_matrix_multiply internally)
        result = filter.apply_filter_bank(io_filter_bank, i_input)

        # Reference: original per-channel loop
        import itertools

        fft_dims = list(range(1, i_input.ndim))
        pad_sizes = [(0, (t - (s % t)) % t) for t, s in zip(filter_shape[::-1], spatial_shape[::-1])]
        padded = torch.nn.functional.pad(i_input, list(itertools.chain(*pad_sizes)))
        spectrum = torch.fft.fftn(padded, dim=fft_dims)

        ref = torch.zeros((num_output,) + spectrum.shape[1:], dtype=spectrum.dtype)
        for i in range(num_input):
            for o in range(num_output):
                ref[o] += filter.stretched_multiply(io_filter_bank[i, o], spectrum[i])

        ref_out = torch.real(torch.fft.ifftn(ref, dim=fft_dims))
        slices = (slice(None),) + tuple(slice(0, n) for n in spatial_shape)
        ref_out = ref_out[slices]

        assert result.shape == ref_out.shape
        assert torch.allclose(result, ref_out, atol=1e-5)


def test_stretched_multiply_incompatible_dims():
    # small_array > large_array
    small_array = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    large_array = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        filter.stretched_multiply(small_array, large_array)

    # Mismatched dims
    small_array = torch.tensor([[1, 2], [3, 4]])
    large_array = torch.tensor([[[1, 2], [4, 5], [7, 8]], [[10, 11], [13, 14], [16, 17]]])
    with pytest.raises(ValueError):
        filter.stretched_multiply(small_array, large_array)

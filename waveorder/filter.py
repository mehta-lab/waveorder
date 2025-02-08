import itertools
import numpy as np
import torch


def apply_transfer_function_filter(
    transfer_function: torch.Tensor,
    input_array: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a transfer function filter to an input array.

    transfer_function.shape must be smaller or equal to input_array.shape in all
    dimensions. When transfer_function is smaller, it is effectively "stretched"
    to apply the filter.

    transfer_function is in "wrapped" format, i.e., the zero frequency is the
    zeroth element.

    input_array and transfer_function must have inverse sample spacing, i.e.,
    is input_array contains samples spaced by dx, then transfer_function must
    have extent 1/dx. Note that there is no need for transfer_function to have
    sample spacing 1/(n*dx) because transfer_function will be stretched.

    Parameters
    ----------
    transfer_function : torch.Tensor
        The transfer function to be applied in the frequency domain.
        Extent of transfer_function must be 1/dx, where dx is the sample spacing
        of input_array.
    input_array : torch.Tensor
        The input array to be filtered.

    Returns
    -------
    torch.Tensor
        The filtered output array with the same shape and dtype as input_array.
    """
    # Ensure all dimensions of transfer_function are smaller than or equal to input_array
    if any(t > i for t, i in zip(transfer_function.shape, input_array.shape)):
        raise ValueError(
            "All dimensions of transfer_function must be <= input_array"
        )

    # Ensure the number of dimensions match
    if transfer_function.ndim != input_array.ndim:
        raise ValueError(
            "transfer_function and input_array must have the same number of dimensions"
        )

    # Pad input_array until each dimension is divisible by transfer_function
    pad_sizes = [
        (0, (t - (i % t)) % t)
        for t, i in zip(transfer_function.shape[::-1], input_array.shape[::-1])
    ]
    flat_pad_sizes = list(itertools.chain(*pad_sizes))
    padded_input_array = torch.nn.functional.pad(input_array, flat_pad_sizes)

    # Apply the transfer function in the frequency domain
    padded_input_spectrum = torch.fft.fftn(padded_input_array)
    padded_output_spectrum = stretched_multiply(
        transfer_function, padded_input_spectrum
    )

    # Casts to input_array dtype, which typically ignores imaginary part
    padded_result = torch.fft.ifftn(padded_output_spectrum).type(
        input_array.dtype
    )

    # Remove padding and return
    slices = tuple(slice(0, i) for i in input_array.shape)
    return padded_result[slices]


def stretched_multiply(
    small_array: torch.Tensor, large_array: torch.Tensor
) -> torch.Tensor:
    """
    Effectively "stretches" small_array onto large_array before multiplying.

    Each dimension of large_array must be divisible by each dimension of small_array.

    Instead of upsampling small_array, this function uses a "block element-wise"
    multiplication by breaking the large_array into blocks before element-wise
    multiplication with the small_array.

    For example, a `stretched_multiply` of a 3x3 array by a 99x99 array will
    divide the 99x99 array into 33x33 blocks
    [[33x33, 33x33, 33x33],
     [33x33, 33x33, 33x33],
     [33x33, 33x33, 33x33]]
     and multiply each block by the corresponding element in the 3x3 array.

    Returns an array with the same shape as large_array.

    Works for arbitrary dimensions.

    Parameters
    ----------
    small_array : torch.Tensor
        A smaller array whose elements will be "stretched" onto blocks in the large array.
    large_array : torch.Tensor
        A larger array that will be divided into blocks and multiplied by the small array.

    Returns
    -------
    torch.Tensor
        Resulting tensor with shape matching large_array.

    Example
    -------
    small_array = torch.tensor([[1, 2],
                                [3, 4]])

    large_array = torch.tensor([[1,   2,  3,  4],
                                [5,   6,  7,  8],
                                [9,  10, 11, 12],
                                [13, 14, 15, 16]])

    stretched_multiply(small_array, large_array) returns

    [[  1,  2,  6,  8],
     [  5,  6, 14, 16],
     [ 27, 30, 44, 48],
     [ 39, 42, 60, 64]]
    """

    # Ensure each dimension of large_array is divisible by each dimension of small_array
    if any(l % s != 0 for s, l in zip(small_array.shape, large_array.shape)):
        raise ValueError(
            "Each dimension of large_array must be divisible by each dimension of small_array"
        )

    # Ensure the number of dimensions match
    if small_array.ndim != large_array.ndim:
        raise ValueError(
            "small_array and large_array must have the same number of dimensions"
        )

    # Get shapes
    s_shape = small_array.shape
    l_shape = large_array.shape

    # Reshape both array into blocks
    block_shape = tuple(p // s for p, s in zip(l_shape, s_shape))
    new_large_shape = tuple(itertools.chain(*zip(s_shape, block_shape)))
    new_small_shape = tuple(
        itertools.chain(*zip(s_shape, small_array.ndim * (1,)))
    )
    reshaped_large_array = large_array.reshape(new_large_shape)
    reshaped_small_array = small_array.reshape(new_small_shape)

    # Multiply the reshaped arrays
    reshaped_result = reshaped_large_array * reshaped_small_array

    # Reshape the result back to the large array shape
    result = reshaped_result.reshape(l_shape)

    return result

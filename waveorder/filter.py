import numpy as np
import torch

def apply_transfer_function_filter(
    transfer_function: torch.Tensor, input_array: torch.Tensor, 
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

    input_spectrum = torch.fft.fftn(input_array)
    output_spectrum = stretched_multiply(transfer_function, input_spectrum)

    # Casts to input_array dtype, which typically ignores imaginary part
    result = torch.fft.ifftn(output_spectrum).type(input_array.dtype)

    return result

def stretched_multiply(
    small_array: torch.Tensor, large_array: torch.Tensor
) -> torch.Tensor:
    """
    Effectively "stretches" small_array onto large_array before multiplying.

    Instead of upsampling small_array, this function uses a "block element-wise"
    multiplication by breaking the large_array into blocks before
    element-wise multiplication with the small_array.

    For example, a `stretched_multiply` of a 3x3 array by a 100x100 array will
    zero pad the 100x100 array so that it is divisible into 3x3 blocks with sizes
    [[34x34, 34x34, 34x34],
     [34x34, 34x34, 34x34],
     [34x34, 34x34, 34x34]]
    and multiply each block by the corresponding element in the 3x3 array. 

    Returns an array with the same shape as large_array (padding is cropped).

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
    # Ensure all dimensions of small_array are smaller than large_array
    if any(s > l for s, l in zip(small_array.shape, large_array.shape)):
        raise ValueError(
            "All dimensions of small_array must be <=  large_array"
        )

    # Ensure the number of dimensions match
    if small_array.ndim != large_array.ndim:
        raise ValueError(
            "small_array and large_array must have the same number of dimensions"
        )

    # Get shapes
    s_shape = small_array.shape
    l_shape = large_array.shape

    # Compute padding sizes to make large_array divisible by small_array
    pad_sizes = [(0, (s - (l % s)) % s) for l, s in zip(l_shape, s_shape)]
    pad_sizes_flat = [size for pair in pad_sizes for size in pair]

    # Pad the large array
    padded_large_array = torch.nn.functional.pad(large_array, pad_sizes_flat)

    # Reshape the padded large array into blocks
    new_shape = tuple(p // s for p, s in zip(padded_large_array.shape, s_shape)) + s_shape
    reshaped_large_array = padded_large_array.reshape(new_shape)

    # Multiply the reshaped large array with the small array
    result_blocks = reshaped_large_array * small_array

    # Reshape the result back to the padded large array shape
    result_padded = result_blocks.reshape(padded_large_array.shape)

    # Unpad the result to get back to the original large array shape
    slices = tuple(slice(0, l) for l in l_shape)
    result = result_padded[slices]

    return result

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
    input_spectrum = torch.fft.fftn(input_array)
    output_spectrum = stretched_multiply(transfer_function, input_spectrum)
    return torch.fft.ifftn(output_spectrum).type(input_array.dtype)

def stretched_multiply(
    small_array: torch.Tensor, large_array: torch.Tensor
) -> torch.Tensor:
    """
    Effectively "stretches" small_array onto large_array before multiplying.

    Instead of upsampling small_array, this function uses a "block element-wise"
    multiplication by breaking the large_array into blocks before
    element-wise multiplication with the small_array.

    For example, a `stretched_multiply` of a 3x3 array by a 100x100 array will
    break the 100x100 array into 3x3 blocks, with sizes
    [[34x34, 33x34, 33x33],
     [34x33, 33x33, 33x33],
     [34x33, 33x33, 33x33]]
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

    # Compute base block sizes using integer division
    # This gives the approximate size of each block before handling remainders
    base_block_size = tuple(l // s for l, s in zip(l_shape, s_shape))

    # Compute remainder (extra elements that don't fit evenly)
    remainder = tuple(l % s for l, s in zip(l_shape, s_shape))

    # Compute block boundaries by spreading remainder elements evenly
    indices = []
    for b, s, r in zip(base_block_size, s_shape, remainder):
        # Create an array where the first 'r' blocks get an extra +1
        # Example: if b=9, s=11, r=1 -> first remainder block gets size 10 instead of 9
        block_sizes = [(b + 1) if i < r else b for i in range(s)]

        # Compute cumulative sum to determine start and end indices
        # Example: [0, 10, 19, 28, ..., 100] gives the split points
        indices.append(np.cumsum([0] + block_sizes))

    # Create output array initialized as a copy of the large array
    result = large_array.clone()

    # Iterate over small_array indices using ndindex
    for idx in np.ndindex(s_shape):
        # Extract multi-dimensional indices
        slices = tuple(
            slice(indices[dim][idx[dim]], indices[dim][idx[dim] + 1])
            for dim in range(small_array.ndim)
        )

        # Multiply the corresponding block in the large array by the value from the small array
        result[slices] *= small_array[idx]

    return result

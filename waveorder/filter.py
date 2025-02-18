import itertools
import numpy as np
import torch


def apply_transfer_function_filter_tiled(
    transfer_function: torch.Tensor,
    input_array: torch.Tensor,
    tile_size: tuple,
    overlap_size: tuple,
) -> torch.Tensor:

    # Get the shape of the input array
    input_shape = input_array.shape

    # Calculate the number of tiles in each dimension
    num_tiles = tuple(
        (input_shape[dim] + tile_size[dim] - 1) // tile_size[dim]
        for dim in range(len(tile_size))
    )

    # Initialize the output array
    output_array = torch.zeros_like(input_array)

    # Iterate over each tile
    for tile_idx in np.ndindex(num_tiles):
        # Calculate the start and end indices for each dimension
        start_indices = tuple(
            max(0, tile_idx[dim] * tile_size[dim] - overlap_size[dim])
            for dim in range(len(tile_size))
        )
        end_indices = tuple(
            min(
                input_shape[dim],
                (tile_idx[dim] + 1) * tile_size[dim] + overlap_size[dim],
            )
            for dim in range(len(tile_size))
        )

        # Extract the tile from the input array
        tile_slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )
        input_tile = input_array[tile_slices]

        # Apply the transfer function filter to the tile
        filtered_tile = apply_filter_bank(transfer_function, input_tile)

        # Calculate the region to add the filtered tile to the output array
        output_start_indices = tuple(
            tile_idx[dim] * tile_size[dim] for dim in range(len(tile_size))
        )
        output_end_indices = tuple(
            min(
                input_shape[dim],
                output_start_indices[dim] + filtered_tile.shape[dim],
            )
            for dim in range(len(tile_size))
        )
        output_slices = tuple(
            slice(start, end)
            for start, end in zip(output_start_indices, output_end_indices)
        )
        import pdb

        pdb.set_trace()
        # Add the filtered tile to the output array
        output_array[output_slices] += filtered_tile

    return output_array


def apply_filter_bank(
    io_filter_bank: torch.Tensor,
    i_input_array: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a filter bank to an input array.

    io_filter_bank.shape must be smaller or equal to i_input_array.shape in all
    dimensions. When io_filter_bank is smaller, it is effectively "stretched"
    to apply the filter.

    io_filter_bank is in "wrapped" format, i.e., the zero frequency is the
    zeroth element.

    i_input_array and io_filter_bank must have inverse sample spacing, i.e.,
    is input_array contains samples spaced by dx, then io_filter_bank must
    have extent 1/dx. Note that there is no need for io_filter_bank to have
    sample spacing 1/(n*dx) because io_filter_bank will be stretched.

    Parameters
    ----------
    io_filter_bank : torch.Tensor
        The filter bank to be applied in the frequency domain.
        The spatial extent of io_filter_bank must be 1/dx, where dx is the
        sample spacing of i_input_array.

        Leading dimensions are the input and output dimensions.
        io_filter_bank.shape[:2] == (i, o)

        Trailing dimensions are spatial frequency dimensions.
        io_filter_bank.shape[2:] == (Z', Y', X') or (Y', X')

    i_input_array : torch.Tensor
        The input array with sample spacing dx to be filtered.

        Leading dimension is the input dimension, matching the filter bank.
        i_input_array.shape[0] == i

        Trailing dimensions are spatial dimensions.
        i_input_array.shape[1:] == (Z, Y, X) or (Y, X)

    Returns
    -------
    torch.Tensor
        The filtered output array with the same shape and dtype as input_array.
    """

    # Ensure all dimensions of transfer_function are smaller than or equal to input_array
    if any(
        t > i
        for t, i in zip(io_filter_bank.shape[2:], i_input_array.shape[1:])
    ):
        raise ValueError(
            "All spatial dimensions of io_filter_bank must be <= i_input_array."
        )

    # Ensure the number of spatial dimensions match
    if io_filter_bank.ndim - i_input_array.ndim != 1:
        raise ValueError(
            "io_filter_bank and i_input_array must have the same number of spatial dimensions."
        )

    # Ensure the input dimensions match
    if io_filter_bank.shape[0] != i_input_array.shape[0]:
        raise ValueError(
            "io_filter_bank.shape[0] and i_input_array.shape[0] must be the same."
        )

    I, O = io_filter_bank.shape[:2]
    spatial_dims = io_filter_bank.shape[2:]

    # Pad input_array until each dimension is divisible by transfer_function
    pad_sizes = [
        (0, (t - (i % t)) % t)
        for t, i in zip(
            io_filter_bank.shape[2:][::-1], i_input_array.shape[1:][::-1]
        )
    ]
    flat_pad_sizes = list(itertools.chain(*pad_sizes))
    padded_input_array = torch.nn.functional.pad(i_input_array, flat_pad_sizes)

    # Apply the transfer function in the frequency domain
    fft_dims = [d for d in range(1, i_input_array.ndim)]
    padded_input_spectrum = torch.fft.fftn(padded_input_array, dim=fft_dims)

    # Matrix-vector multiplication over f
    # If this is a bottleneck, consider extending `stretched_multiply` to
    # a `stretched_matrix_multiply` that uses an call like
    # torch.einsum('io..., i... -> o...', io_filter_bank, padded_input_spectrum)
    padded_output_spectrum = torch.zeros(
        (O,) + spatial_dims, dtype=padded_input_spectrum.dtype
    )
    for i_idx in range(I):
        for o_idx in range(O):
            padded_output_spectrum[o_idx] += stretched_multiply(
                io_filter_bank[i_idx, o_idx], padded_input_spectrum[i_idx]
            )

    # Casts to input_array dtype, which typically ignores imaginary part
    padded_result = torch.fft.ifftn(padded_output_spectrum, dim=fft_dims).type(
        i_input_array.dtype
    )

    # Remove padding and return
    slices = tuple(slice(0, i) for i in i_input_array.shape)
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

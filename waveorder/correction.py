"""Background correction methods"""

import torch
import torch.nn.functional as F
from torch import Tensor, Size


def _sample_block_medians(image: Tensor, block_size) -> Tensor:
    """
    Sample densely tiled square blocks from a 2D image and return their medians.
    Incomplete blocks (overhangs) will be ignored.

    Parameters
    ----------
    image : Tensor
        2D image
    block_size : int, optional
        Width and height of the blocks

    Returns
    -------
    Tensor
        Median intensity values for each block, flattened
    """
    if not image.dtype.is_floating_point:
        image.to(torch.float)
    blocks = F.unfold(image[None, None], block_size, stride=block_size)[0]
    return blocks.median(0)[0]


def _grid_coordinates(image: Tensor, block_size: int) -> Tensor:
    """Build image coordinates from the center points of square blocks"""
    coords = torch.meshgrid(
        [
            torch.arange(
                0 + block_size / 2,
                boundary - block_size / 2 + 1,
                block_size,
                device=image.device,
            )
            for boundary in image.shape
        ]
    )
    return torch.stack(coords, dim=-1).reshape(-1, 2)


def _fit_2d_polynomial_surface(
    coords: Tensor, values: Tensor, order: int, surface_shape: Size
) -> Tensor:
    """Fit a 2D polynomial to a set of coordinates and their values,
    and return the surface evaluated at every point."""
    n_coeffs = int((order + 1) * (order + 2) / 2)
    if n_coeffs >= len(values):
        raise ValueError(
            f"Cannot fit a {order} degree 2D polynomial "
            f"with {len(values)} sampled values"
        )
    orders = torch.arange(order + 1, device=coords.device)
    order_pairs = torch.stack(torch.meshgrid(orders, orders), -1)
    order_pairs = order_pairs[order_pairs.sum(-1) <= order].reshape(-1, 2)
    terms = torch.stack(
        [coords[:, 0] ** i * coords[:, 1] ** j for i, j in order_pairs], -1
    )
    # use "gels" driver for precision and GPU consistency
    coeffs = torch.linalg.lstsq(terms, values, driver="gels").solution
    dense_coords = torch.meshgrid(
        [
            torch.arange(s, dtype=values.dtype, device=values.device)
            for s in surface_shape
        ]
    )
    dense_terms = torch.stack(
        [dense_coords[0] ** i * dense_coords[1] ** j for i, j in order_pairs],
        -1,
    )
    return torch.matmul(dense_terms, coeffs)


def estimate_background(image: Tensor, order: int = 2, block_size: int = 32):
    """
    Combine sampling and polynomial surface fit for background estimation.
    To background correct an image, divide it by the background.

    Parameters
    ----------
    image : Tensor
        2D image
    order : int, optional
        Order of polynomial, by default 2
    block_size : int, optional
        Width and height of the blocks, by default 32

    Returns
    -------
    Tensor
        Background image
    """
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")
    height, width = image.shape
    if block_size > width:
        raise ValueError("Block size larger than image height")
    if block_size > height:
        raise ValueError("Block size larger than image width")
    medians = _sample_block_medians(image, block_size)
    coords = _grid_coordinates(image, block_size)
    return _fit_2d_polynomial_surface(coords, medians, order, image.shape)

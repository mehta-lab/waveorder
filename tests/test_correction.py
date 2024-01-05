import pytest
import torch

from tests.conftest import _DEVICE
from waveorder.correction import (
    _fit_2d_polynomial_surface,
    _grid_coordinates,
    _sample_block_medians,
    estimate_background,
)


def test_sample_block_medians():
    image = torch.arange(4 * 5, dtype=torch.float).reshape(4, 5)
    medians = _sample_block_medians(image, 2)
    assert torch.allclose(
        medians, torch.tensor([1, 3, 11, 13]).to(image.dtype)
    )


def test_grid_coordinates():
    image = torch.ones(15, 17)
    coords = _grid_coordinates(image, 4)
    assert coords.shape == (3 * 4, 2)


def test_fit_2d_polynomial_surface():
    coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    values = torch.tensor([0, 1, 2, 3], dtype=torch.float)
    surface = _fit_2d_polynomial_surface(coords, values, 1, (2, 2))
    assert torch.allclose(surface, values.reshape(surface.shape), atol=1e-2)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(*_DEVICE)
def test_estimate_background(order, device):
    image = torch.rand(200, 200).to(device)
    image[:100, :100] += 1
    background = estimate_background(image, order=order, block_size=32)
    assert 2.0 > background[50, 50] > 1.0
    assert 1.5 > background[0, 100] > 0.5
    assert 1.0 > background[150, 150] > 0.0

import torch

from waveorder.sampling import nd_fourier_central_cuboid


def test_nd_fourier_central_cuboid():
    source = torch.randn(8, 8)
    target_shape = (4, 4)
    result = nd_fourier_central_cuboid(source, target_shape)
    assert result.shape == target_shape

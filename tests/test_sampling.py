import torch

from waveorder.sampling import nd_fourier_central_cuboid, raised_cosine_window


def test_nd_fourier_central_cuboid():
    source = torch.randn(8, 8)
    target_shape = (4, 4)
    result = nd_fourier_central_cuboid(source, target_shape)
    assert result.shape == target_shape


def test_raised_cosine_window():
    window = raised_cosine_window(8, 0.5)
    assert window.shape == (8,)
    assert window[0] == 1.0  # DC untouched
    assert window[4] == 0.0  # zero at Nyquist
    # symmetric in +/- frequency
    assert torch.allclose(window[1:4], window.flip(0)[:3])
    # monotone roll-off
    assert torch.all(window[1:5] <= window[:4])

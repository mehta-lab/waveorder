import torch

from waveorder.sampling import apply_transverse_apodization, nd_fourier_central_cuboid, raised_cosine_window


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


def test_transverse_apodization_preserves_input_and_gradients():
    source = torch.full((2, 3, 8, 8), 1 + 2j, requires_grad=True)
    assert apply_transverse_apodization(source, 0.0) is source

    axis_window = torch.tensor([1, 1, 1, 0.5, 0, 0.5, 1, 1])
    window = (axis_window[:, None] * axis_window).to(source.dtype).expand_as(source)
    result = apply_transverse_apodization(source, 0.5)

    torch.testing.assert_close(result, (1 + 2j) * window)
    torch.testing.assert_close(source, torch.full_like(source, 1 + 2j))
    result.real.sum().backward()
    torch.testing.assert_close(source.grad, window)

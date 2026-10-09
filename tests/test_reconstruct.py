import pytest
import torch

from waveorder.reconstruct import tikhonov_regularized_inverse_filter


def test_inverse_filter_without_apodization():
    forward = torch.tensor([1 + 2j, 2 - 1j, 0j])
    expected = torch.tensor([1 - 2j, 2 + 1j, 0j]) / 5.25

    for kwargs in [{}, {"apodization_rolloff": 0.0}]:
        actual = tikhonov_regularized_inverse_filter(forward, 0.25, **kwargs)
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(forward, torch.tensor([1 + 2j, 2 - 1j, 0j]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "rolloff,y_window,x_window",
    [
        (0.5, [1, 1, 1, 0.5, 0, 0.5, 1, 1], [1, 1, 1, 1, 0.75, 0.25, 0, 0.25, 0.75, 1, 1, 1]),
        (
            1.0,
            [1, 0.654508497, 0.095491503, 0.095491503, 0.654508497],
            [1, 0.654508497, 0.095491503, 0.095491503, 0.654508497],
        ),
    ],
)
def test_inverse_filter_apodizes_only_transverse_axes(dtype, rolloff, y_window, x_window):
    value = 1 + 2j if dtype.is_complex else 2.0
    inverse_value = (1 - 2j) / 5.25 if dtype.is_complex else 2.0 / 4.25
    forward = torch.full((2, 3, len(y_window), len(x_window)), value, dtype=dtype)
    expected_window = torch.tensor(y_window)[:, None] * torch.tensor(x_window)
    expected = (inverse_value * expected_window.to(dtype)).expand_as(forward)

    actual = tikhonov_regularized_inverse_filter(forward, 0.25, apodization_rolloff=rolloff)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.dtype == forward.dtype
    assert actual.device == forward.device
    torch.testing.assert_close(forward, torch.full_like(forward, value))


def test_svd_apodization_preserves_inverse_and_gradients():
    generator = torch.Generator().manual_seed(584)
    U = torch.randn(2, 2, 2, 8, 12, dtype=torch.complex64, generator=generator)
    S = (torch.rand(2, 2, 8, 12, generator=generator) + 0.1).requires_grad_()
    Vh = torch.randn(2, 2, 4, 8, 12, dtype=torch.complex64, generator=generator)
    window = torch.tensor([1, 1, 1, 0.5, 0, 0.5, 1, 1])[:, None] * torch.tensor(
        [1, 1, 1, 1, 0.75, 0.25, 0, 0.25, 0.75, 1, 1, 1]
    )
    expected = torch.einsum("bsj...,bj...,bjf...->bfs...", U, S / (S**2 + 1e-3), Vh) * window
    S_reg = tikhonov_regularized_inverse_filter(S, 1e-3, apodization_rolloff=0.5)
    actual = torch.einsum("bsj...,bj...,bjf...->bfs...", U, S_reg, Vh)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    expected_grad = torch.autograd.grad(expected.abs().square().mean(), S, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.abs().square().mean(), S)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)

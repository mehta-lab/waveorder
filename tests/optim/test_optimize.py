"""Integration tests for the optimization loop."""

import torch

from waveorder.optim import optimize_reconstruction
from waveorder.optim.losses import midband_power_loss as mb_loss


def test_midband_power_loss_is_scalar():
    """Loss returns a scalar."""
    recon = torch.randn(64, 64)
    loss = mb_loss(recon, NA_det=1.2, lambda_ill=0.532, pixel_size=0.1)
    assert loss.ndim == 0


def test_midband_power_loss_is_differentiable():
    """Loss supports backward."""
    recon = torch.randn(64, 64, requires_grad=True)
    loss = mb_loss(recon, NA_det=1.2, lambda_ill=0.532, pixel_size=0.1)
    loss.backward()
    assert recon.grad is not None


def test_optimize_reconstruction_basic():
    """Basic optimization loop runs without error."""

    def reconstruct_fn(data, **params):
        # Simple: just scale data by the parameter
        scale = params.get("scale", torch.tensor(1.0))
        return data[0] * scale

    def loss_fn(recon):
        return -(recon**2).sum()

    data = torch.randn(3, 32, 32)
    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"scale": (0.5, 0.1)},
        num_iterations=3,
    )

    assert "scale" in result.optimized_values
    assert len(result.loss_history) == 3
    assert result.final_reconstruction is not None


def test_optimize_reconstruction_converges():
    """Optimization converges toward better loss."""

    def reconstruct_fn(data, **params):
        offset = params.get("offset", torch.tensor(0.0))
        return data[0] + offset

    target = torch.ones(16, 16) * 5.0
    data = torch.zeros(3, 16, 16)

    def loss_fn(recon):
        return ((recon - target) ** 2).sum()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        num_iterations=20,
    )

    # Loss should decrease
    assert result.loss_history[-1] < result.loss_history[0]
    # Parameter should move toward 5.0
    assert result.optimized_values["offset"] > 1.0

"""Integration tests for the optimization loop."""

import torch

from waveorder.focus import compute_midband_power
from waveorder.optim import optimize_reconstruction


def test_midband_power_is_scalar():
    """compute_midband_power returns a scalar for 2D input."""
    recon = torch.randn(64, 64)
    power = compute_midband_power(recon, NA_det=1.2, lambda_ill=0.532, pixel_size=0.1)
    assert power.ndim == 0


def test_midband_power_is_differentiable():
    """compute_midband_power supports backward."""
    recon = torch.randn(64, 64, requires_grad=True)
    loss = -compute_midband_power(recon, NA_det=1.2, lambda_ill=0.532, pixel_size=0.1)
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


def _make_quadratic_problem():
    """Helper: minimize (offset - 3)^2 over 8x8 grid."""
    target = torch.ones(8, 8) * 3.0
    data = torch.zeros(2, 8, 8)

    def reconstruct_fn(data, **params):
        offset = params.get("offset", torch.tensor(0.0))
        return data[0] + offset

    def loss_fn(recon):
        return ((recon - target) ** 2).sum()

    return data, reconstruct_fn, loss_fn


def test_convergence_early_stopping():
    """Early stopping triggers when loss plateaus."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (2.9, 0.5)},
        num_iterations=200,
        convergence_tol=1e-4,
        convergence_patience=5,
    )

    assert result.converged
    assert result.iterations_used < 200


def test_no_grad_mode():
    """use_gradients=False skips backward pass."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        num_iterations=5,
        use_gradients=False,
    )

    # Without gradients, Adam can't optimize, but it should still run
    assert len(result.loss_history) == 5
    assert result.final_reconstruction is not None


def test_lbfgs_backend():
    """L-BFGS optimizer runs and reduces loss."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        method="lbfgs",
        num_iterations=10,
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert result.optimized_values["offset"] > 1.0


def test_nelder_mead_backend():
    """Nelder-Mead optimizer runs without gradients."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        method="nelder_mead",
        num_iterations=50,
    )

    assert len(result.loss_history) > 0
    assert result.final_reconstruction is not None


def test_grid_search_backend():
    """Grid search finds the best value on the grid."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    # Grid centered at 3.0 with step 0.5 → [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (3.0, 0.5)},
        method="grid_search",
    )

    assert abs(result.optimized_values["offset"] - 3.0) < 0.6


def test_wall_times_recorded():
    """Wall times are recorded for each iteration."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        num_iterations=3,
    )

    assert len(result.wall_times) == 3
    assert all(t >= 0 for t in result.wall_times)

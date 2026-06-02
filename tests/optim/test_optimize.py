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
        max_iterations=3,
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
        max_iterations=20,
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
        max_iterations=200,
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
        max_iterations=5,
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
        max_iterations=10,
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
        max_iterations=50,
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
        max_iterations=3,
    )

    assert len(result.wall_times) == 3
    assert all(t >= 0 for t in result.wall_times)


def test_batched_optimization_independent_tiles():
    """Each tile in a batched run optimizes toward its own target."""
    B = 4
    target_per_tile = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = target_per_tile.view(B, 1, 1, 1).expand(B, 1, 8, 8)
    data = torch.zeros(B, 1, 8, 8)

    def reconstruct_fn(data, **params):
        offset = params["offset"]  # (B,) tensor
        return data + offset.view(B, 1, 1, 1)

    # loss_fn is called as loss_fn(recon[b]) per b and summed.
    # To make each tile see its own target, we index off target by matching
    # against a counter that resets per outer step.
    call_idx = [0]

    def loss_fn(recon_b):
        b = call_idx[0] % B
        call_idx[0] += 1
        return ((recon_b - target[b]) ** 2).sum()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        max_iterations=80,
    )

    # offset should be a list of B values, each moving toward its target
    assert isinstance(result.optimized_values["offset"], list)
    assert len(result.optimized_values["offset"]) == B
    for b, (got, want) in enumerate(zip(result.optimized_values["offset"], target_per_tile.tolist())):
        assert abs(got - want) < 0.3, f"tile {b}: got {got:.3f}, want {want:.3f}"


def test_per_tile_initial_value_tensor():
    """Tensor initial_value broadcasts/honors per-tile shape (B,)."""
    B = 3
    target_per_tile = torch.tensor([1.0, 2.0, 3.0])
    target = target_per_tile.view(B, 1, 1, 1).expand(B, 1, 4, 4)
    data = torch.zeros(B, 1, 4, 4)

    def reconstruct_fn(data, **params):
        offset = params["offset"]
        return data + offset.view(B, 1, 1, 1)

    call_idx = [0]

    def loss_fn(recon_b):
        b = call_idx[0] % B
        call_idx[0] += 1
        return ((recon_b - target[b]) ** 2).sum()

    # Per-tile warm-starts already very close to the targets — convergence is fast
    init = torch.tensor([0.9, 1.9, 2.9])
    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (init, 0.2)},
        max_iterations=30,
    )

    # All three tiles should land within 0.2 of their targets
    for b, (got, want) in enumerate(zip(result.optimized_values["offset"], target_per_tile.tolist())):
        assert abs(got - want) < 0.2, f"tile {b}: got {got:.3f}, want {want:.3f}"


def test_frozen_axis_does_not_move():
    """lr=0 marks a parameter as frozen — it stays at its initial value."""
    target = torch.ones(8, 8) * 5.0
    data = torch.zeros(2, 8, 8)

    def reconstruct_fn(data, **params):
        free = params["free"]
        frozen = params["frozen"]
        return data[0] + free + frozen  # frozen contributes but doesn't move

    def loss_fn(recon):
        return ((recon - target) ** 2).sum()

    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={
            "free": (0.0, 0.5),
            "frozen": (1.0, 0.0),  # lr=0 → frozen
        },
        max_iterations=40,
    )

    # frozen stays at initial value
    assert result.optimized_values["frozen"] == 1.0
    # free moves toward 4.0 so that free + frozen ≈ 5.0
    assert abs(result.optimized_values["free"] - 4.0) < 0.5


def test_all_frozen_raises():
    """Refuse a degenerate config where every param is frozen."""
    data, reconstruct_fn, loss_fn = _make_quadratic_problem()

    try:
        optimize_reconstruction(
            data=data,
            reconstruct_fn=reconstruct_fn,
            loss_fn=loss_fn,
            optimizable_params={"offset": (0.0, 0.0)},
            max_iterations=3,
        )
    except ValueError as e:
        assert "frozen" in str(e).lower()
        return
    raise AssertionError("expected ValueError when every param is frozen")


def test_per_tile_init_with_frozen_param():
    """Frozen parameter with per-tile init keeps each tile's initial value."""
    B = 3
    target_per_tile = torch.tensor([1.0, 2.0, 3.0])
    target = target_per_tile.view(B, 1, 1, 1).expand(B, 1, 4, 4)
    data = torch.zeros(B, 1, 4, 4)

    def reconstruct_fn(data, **params):
        free = params["free"]
        frozen = params["frozen"]
        return data + (free + frozen).view(B, 1, 1, 1)

    call_idx = [0]

    def loss_fn(recon_b):
        b = call_idx[0] % B
        call_idx[0] += 1
        return ((recon_b - target[b]) ** 2).sum()

    # Frozen per-tile prior; "free" optimizer makes up the difference
    frozen_prior = torch.tensor([0.5, 0.5, 0.5])
    result = optimize_reconstruction(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={
            "free": (0.0, 0.2),
            "frozen": (frozen_prior, 0.0),
        },
        max_iterations=120,
    )

    # frozen retained per-tile init
    assert result.optimized_values["frozen"] == [0.5, 0.5, 0.5]
    # free reaches target - 0.5 per tile
    for b, (got, want) in enumerate(
        zip(result.optimized_values["free"], (target_per_tile - 0.5).tolist())
    ):
        assert abs(got - want) < 0.3, f"tile {b}: got {got:.3f}, want {want:.3f}"


def test_per_tile_init_shape_mismatch_raises():
    """Wrong-shape per-tile init in batched mode is rejected."""
    B = 4
    data = torch.zeros(B, 1, 4, 4)

    def reconstruct_fn(data, **params):
        return data + params["offset"].view(B, 1, 1, 1)

    def loss_fn(recon_b):
        return (recon_b ** 2).sum()

    bad_init = torch.tensor([0.1, 0.2])  # shape (2,) but B=4
    try:
        optimize_reconstruction(
            data=data,
            reconstruct_fn=reconstruct_fn,
            loss_fn=loss_fn,
            optimizable_params={"offset": (bad_init, 0.1)},
            max_iterations=2,
        )
    except ValueError as e:
        assert "shape" in str(e).lower()
        return
    raise AssertionError("expected ValueError for shape mismatch")

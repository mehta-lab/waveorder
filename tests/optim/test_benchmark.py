"""Tests for benchmark_optimizers."""

import torch

from waveorder.optim.benchmark import benchmark_optimizers


def test_benchmark_returns_dataframe():
    """benchmark_optimizers returns a DataFrame with expected columns."""

    def reconstruct_fn(data, **params):
        offset = params.get("offset", torch.tensor(0.0))
        return data[0] + offset

    target = torch.ones(8, 8) * 3.0
    data = torch.zeros(2, 8, 8)

    def loss_fn(recon):
        return ((recon - target) ** 2).sum()

    df = benchmark_optimizers(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        methods=["adam", "nelder_mead"],
        max_iterations=5,
    )

    assert "method" in df.columns
    assert "iteration" in df.columns
    assert "loss" in df.columns
    assert "wall_time" in df.columns
    assert set(df["method"].unique()) == {"adam", "nelder_mead"}


def test_benchmark_with_ground_truth():
    """benchmark_optimizers computes param_error when ground truth given."""

    def reconstruct_fn(data, **params):
        offset = params.get("offset", torch.tensor(0.0))
        return data[0] + offset

    target = torch.ones(8, 8) * 3.0
    data = torch.zeros(2, 8, 8)

    def loss_fn(recon):
        return ((recon - target) ** 2).sum()

    df = benchmark_optimizers(
        data=data,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params={"offset": (0.0, 0.5)},
        methods=["adam"],
        max_iterations=5,
        ground_truth_params={"offset": 3.0},
    )

    assert df["param_error"].notna().all()

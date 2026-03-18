"""Benchmarking harness for comparing optimization strategies."""

from __future__ import annotations

import time
from typing import Callable

from torch import Tensor

from waveorder.optim.logging import PrintLogger
from waveorder.optim.optimize import optimize_reconstruction


def benchmark_optimizers(
    data: Tensor,
    reconstruct_fn: Callable[..., Tensor],
    loss_fn: Callable[[Tensor], Tensor],
    optimizable_params: dict[str, tuple[float, float]],
    methods: list[str] | None = None,
    max_iterations: int = 50,
    ground_truth_params: dict[str, float] | None = None,
    fixed_params: dict | None = None,
):
    """Run multiple optimizers and compare convergence.

    Parameters
    ----------
    data : Tensor
        Input data.
    reconstruct_fn : callable
        Reconstruction function.
    loss_fn : callable
        Loss function.
    optimizable_params : dict[str, tuple[float, float]]
        ``{param_name: (initial_value, learning_rate)}``.
    methods : list of str, optional
        Optimizer methods to compare. Default: ``["adam", "nelder_mead"]``.
    max_iterations : int
        Max iterations per method.
    ground_truth_params : dict[str, float], optional
        True parameter values for computing error on simulated data.
    fixed_params : dict, optional
        Fixed parameters.

    Returns
    -------
    pd.DataFrame
        Columns: method, iteration, loss, wall_time, param_values,
        param_error, converged, total_iterations, total_wall_time.
    """
    import pandas as pd

    if methods is None:
        methods = ["adam", "nelder_mead"]

    rows = []

    for method in methods:
        t_start = time.monotonic()

        result = optimize_reconstruction(
            data=data,
            reconstruct_fn=reconstruct_fn,
            loss_fn=loss_fn,
            optimizable_params=optimizable_params,
            fixed_params=fixed_params,
            method=method,
            max_iterations=max_iterations,
            logger=PrintLogger(),
        )

        total_time = time.monotonic() - t_start
        cumulative_time = 0.0

        for i, loss_val in enumerate(result.loss_history):
            wall_t = result.wall_times[i] if i < len(result.wall_times) else 0.0
            cumulative_time += wall_t

            param_error = None
            if ground_truth_params is not None:
                param_error = (
                    sum((result.optimized_values.get(k, 0) - v) ** 2 for k, v in ground_truth_params.items()) ** 0.5
                )

            rows.append(
                {
                    "method": method,
                    "iteration": i,
                    "loss": loss_val,
                    "wall_time": cumulative_time,
                    "param_values": dict(result.optimized_values),
                    "param_error": param_error,
                    "converged": result.converged,
                    "total_iterations": result.iterations_used,
                    "total_wall_time": total_time,
                }
            )

    return pd.DataFrame(rows)

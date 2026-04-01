"""Core optimization loop for parameter optimization."""

from __future__ import annotations

import contextlib
import io
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize as scipy_minimize
from torch import Tensor
from tqdm import tqdm

from waveorder.optim.logging import NullLogger, OptimLogger, PrintLogger


@dataclass
class OptimizationResult:
    """Result of parameter optimization.

    Attributes
    ----------
    optimized_values : dict[str, float]
        Final parameter values.
    loss_history : list[float]
        Loss at each evaluation.
    final_reconstruction : Tensor or None
        Last reconstruction tensor.
    converged : bool
        Whether early stopping triggered.
    iterations_used : int
        Number of evaluations performed.
    wall_times : list[float]
        Wall-clock time per evaluation in seconds.
    """

    optimized_values: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    final_reconstruction: Tensor | None = None
    converged: bool = False
    iterations_used: int = 0
    wall_times: list[float] = field(default_factory=list)


def optimize_reconstruction(
    data: Tensor,
    reconstruct_fn: Callable[..., Tensor],
    loss_fn: Callable[[Tensor], Tensor],
    optimizable_params: dict[str, tuple[float, float]],
    fixed_params: dict | None = None,
    method: str = "adam",
    max_iterations: int = 10,
    convergence_tol: float | None = None,
    convergence_patience: int = 5,
    use_gradients: bool | None = None,
    grid_points: int = 7,
    logger: OptimLogger | None = None,
    log_images: bool = False,
    log_extras_fn: Callable | None = None,
) -> OptimizationResult:
    """Run optimization loop over reconstruction parameters.

    Parameters
    ----------
    data : Tensor
        Input data (e.g., ZYX defocus stack).
    reconstruct_fn : callable
        Function that takes ``(data, **params)`` and returns a
        reconstruction.
    loss_fn : callable
        Function that takes a reconstruction and returns a scalar loss.
    optimizable_params : dict[str, tuple[float, float]]
        ``{param_name: (initial_value, learning_rate)}`` for each
        parameter. For grid_search, the learning_rate is the grid step.
    fixed_params : dict, optional
        Additional fixed parameters to pass to ``reconstruct_fn``.
    method : str
        Optimizer backend. Supported values:

        - ``"adam"`` — gradient-based (default). ``use_gradients``
          defaults to True. Per-iteration cost includes a backward pass.
        - ``"nadam"`` — Adam with Nesterov momentum. Same interface as
          ``"adam"`` but may converge faster on some problems.
        - ``"lbfgs"`` — gradient-based with line search.
          ``use_gradients`` defaults to True.
        - ``"nelder_mead"`` — gradient-free (scipy). Always runs under
          ``torch.no_grad()``. ``max_iterations`` caps the number of
          function evaluations.
        - ``"grid_search"`` — exhaustive evaluation over a parameter
          grid. Always runs under ``torch.no_grad()``.
          ``max_iterations`` is ignored; the grid is
          ``grid_points`` evenly spaced values centered on
          ``initial_value`` with spacing ``learning_rate``.
    max_iterations : int
        Maximum optimizer steps (or function evaluations for
        Nelder-Mead). Ignored by grid_search.
    convergence_tol : float, optional
        Stop early if loss does not improve by at least this amount.
    convergence_patience : int
        Number of iterations without improvement before early stopping.
    use_gradients : bool, optional
        Whether to compute gradients via ``.backward()``. If None,
        auto-detected from ``method``: True for adam/lbfgs, False for
        nelder_mead/grid_search. Setting True for a gradient-free
        method has no effect.
    grid_points : int
        Number of grid points per parameter for grid_search. The grid
        is centered on ``initial_value`` with spacing ``learning_rate``.
    logger : OptimLogger, optional
        Logger for tracking optimization progress.
    log_images : bool
        If True, log reconstruction images each iteration.
    log_extras_fn : callable, optional
        Extra logging function called each iteration.

    Returns
    -------
    OptimizationResult
        Optimized parameter values, loss history, and final
        reconstruction.
    """
    valid_methods = ("adam", "nadam", "lbfgs", "nelder_mead", "grid_search")
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Must be one of {valid_methods}.")

    if use_gradients is None:
        use_gradients = method in ("adam", "nadam", "lbfgs")

    if method == "nelder_mead":
        return _optimize_nelder_mead(
            data,
            reconstruct_fn,
            loss_fn,
            optimizable_params,
            fixed_params=fixed_params,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
            convergence_patience=convergence_patience,
            logger=logger,
            log_images=log_images,
            log_extras_fn=log_extras_fn,
        )

    if method == "grid_search":
        return _optimize_grid_search(
            data,
            reconstruct_fn,
            loss_fn,
            optimizable_params,
            fixed_params=fixed_params,
            grid_points=grid_points,
            logger=logger,
        )

    # Gradient-based methods: adam, lbfgs
    return _optimize_gradient(
        data,
        reconstruct_fn,
        loss_fn,
        optimizable_params,
        fixed_params=fixed_params,
        method=method,
        max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        convergence_patience=convergence_patience,
        use_gradients=use_gradients,
        logger=logger,
        log_images=log_images,
        log_extras_fn=log_extras_fn,
    )


def _optimize_gradient(
    data,
    reconstruct_fn,
    loss_fn,
    optimizable_params,
    fixed_params=None,
    method="adam",
    max_iterations=10,
    convergence_tol=None,
    convergence_patience=5,
    use_gradients=True,
    logger=None,
    log_images=False,
    log_extras_fn=None,
) -> OptimizationResult:
    """Gradient-based optimization (Adam, L-BFGS).

    When ``data.ndim == 4`` (batched), each parameter becomes a ``(B,)``
    tensor so that every tile is optimized independently. Standard Adam
    maintains per-element momentum and variance, so this is equivalent
    to running B independent Adam optimizers with a single backward pass.
    """
    if logger is None:
        logger = NullLogger()
    if fixed_params is None:
        fixed_params = {}

    batched = data.ndim == 4
    B = data.shape[0] if batched else 1

    param_tensors: dict[str, Tensor] = {}
    param_groups: list[dict] = []

    for name, (init_val, lr) in optimizable_params.items():
        if batched:
            # Per-tile parameter: (B,) tensor with independent gradients
            t = torch.full(
                (B,),
                init_val,
                dtype=torch.float32,
                requires_grad=use_gradients,
            )
        else:
            t = torch.tensor(
                init_val,
                dtype=torch.float32,
                requires_grad=use_gradients,
            )
        param_tensors[name] = t
        param_groups.append({"params": [t], "lr": lr})

    if method == "lbfgs":
        all_params = list(param_tensors.values())
        max_lr = max(lr for _, lr in optimizable_params.values())
        optimizer = torch.optim.LBFGS(
            all_params,
            lr=max_lr,
            max_iter=1,
        )
    elif method == "nadam":
        optimizer = torch.optim.NAdam(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)

    loss_history: list[float] = []
    wall_times: list[float] = []
    final_recon = None
    converged = False
    patience_counter = 0
    best_loss = float("inf")

    # Checkpoint initial params for rollback on NaN/SVD errors
    last_good = {name: t.detach().clone() for name, t in param_tensors.items()}

    pbar = tqdm(range(max_iterations), desc="Optimizing")
    for step in pbar:
        t_start = time.monotonic()

        def _compute_loss(recon):
            """Apply loss_fn; for batched data, sum per-tile losses."""
            if batched:
                return torch.stack([loss_fn(recon[b]) for b in range(B)]).sum()
            return loss_fn(recon)

        try:
            if method == "lbfgs":

                def closure():
                    optimizer.zero_grad()
                    kwargs = dict(fixed_params)
                    kwargs.update(param_tensors)
                    with contextlib.redirect_stdout(io.StringIO()):
                        recon = reconstruct_fn(data, **kwargs)
                    loss = _compute_loss(recon)
                    if use_gradients:
                        loss.backward()
                    return loss

                loss = optimizer.step(closure)
                kwargs = dict(fixed_params)
                kwargs.update(param_tensors)
                with torch.no_grad():
                    with contextlib.redirect_stdout(io.StringIO()):
                        recon = reconstruct_fn(data, **kwargs)
            else:
                optimizer.zero_grad()
                kwargs = dict(fixed_params)
                kwargs.update(param_tensors)

                grad_ctx = contextlib.nullcontext() if use_gradients else torch.no_grad()
                with grad_ctx, contextlib.redirect_stdout(io.StringIO()):
                    recon = reconstruct_fn(data, **kwargs)
                loss = _compute_loss(recon)

                if torch.isnan(loss) or torch.isnan(recon).any():
                    raise ValueError("NaN in reconstruction or loss")

                if use_gradients:
                    loss.backward()
                    optimizer.step()

        except (RuntimeError, ValueError):
            # SVD convergence failure or NaN — revert to last good params
            with torch.no_grad():
                for name, t in param_tensors.items():
                    t.copy_(last_good[name])
            break

        # Checkpoint good params
        last_good = {name: t.detach().clone() for name, t in param_tensors.items()}

        wall_times.append(time.monotonic() - t_start)
        loss_val = loss.item()
        loss_history.append(loss_val)

        postfix = {"loss": f"{loss_val:.4f}"}
        for name, t in param_tensors.items():
            postfix[name] = f"{t.mean().item():.4f}"
        pbar.set_postfix(postfix)

        logger.log_scalar("loss", loss_val, step)
        for name, t in param_tensors.items():
            logger.log_scalar(name, t.mean().item(), step)
            if use_gradients and t.grad is not None:
                logger.log_scalar(f"grad/{name}", t.grad.mean().item(), step)

        if log_images:
            img = recon.detach()
            if img.ndim == 4:
                img = img[0]  # Show first tile
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
            logger.log_image("reconstruction", img, step)

        if log_extras_fn is not None:
            log_extras_fn(step, logger, param_tensors)

        if convergence_tol is not None:
            if loss_val < best_loss - convergence_tol:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= convergence_patience:
                converged = True
                final_recon = recon.detach()
                break

        if step == max_iterations - 1:
            final_recon = recon.detach()

    logger.close()

    if batched:
        optimized_values = {name: t.detach().cpu().tolist() for name, t in param_tensors.items()}
    else:
        optimized_values = {name: t.item() for name, t in param_tensors.items()}

    return OptimizationResult(
        optimized_values=optimized_values,
        loss_history=loss_history,
        final_reconstruction=final_recon,
        converged=converged,
        iterations_used=len(loss_history),
        wall_times=wall_times,
    )


def _optimize_nelder_mead(
    data,
    reconstruct_fn,
    loss_fn,
    optimizable_params,
    fixed_params=None,
    max_iterations=10,
    convergence_tol=None,
    convergence_patience=5,
    logger=None,
    log_images=False,
    log_extras_fn=None,
) -> OptimizationResult:
    """Gradient-free Nelder-Mead optimization via scipy.

    Always runs under ``torch.no_grad()``. ``max_iterations`` caps
    both function evaluations and simplex iterations.
    """
    if logger is None:
        logger = PrintLogger()
    if fixed_params is None:
        fixed_params = {}

    param_names = list(optimizable_params.keys())
    x0 = np.array(
        [optimizable_params[n][0] for n in param_names],
        dtype=np.float64,
    )

    loss_history: list[float] = []
    wall_times: list[float] = []
    last_recon = [None]
    step_counter = [0]

    def objective(x):
        t_start = time.monotonic()
        param_tensors = {}
        for i, name in enumerate(param_names):
            param_tensors[name] = torch.tensor(
                float(x[i]),
                dtype=torch.float32,
            )

        kwargs = dict(fixed_params)
        kwargs.update(param_tensors)

        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                recon = reconstruct_fn(data, **kwargs)
        loss = loss_fn(recon)
        loss_val = loss.item()

        wall_times.append(time.monotonic() - t_start)
        loss_history.append(loss_val)
        last_recon[0] = recon.detach()

        step = step_counter[0]
        logger.log_scalar("loss", loss_val, step)
        for i, name in enumerate(param_names):
            logger.log_scalar(name, float(x[i]), step)

        if log_images and recon is not None:
            img = recon.detach()
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
            logger.log_image("reconstruction", img, step)

        if log_extras_fn is not None:
            log_extras_fn(step, logger, param_tensors)

        step_counter[0] += 1
        return loss_val

    options = {
        "maxfev": max_iterations,
        "maxiter": max_iterations,
    }
    if convergence_tol is not None:
        options["fatol"] = convergence_tol
        options["xatol"] = convergence_tol

    result = scipy_minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options=options,
    )

    logger.close()

    return OptimizationResult(
        optimized_values={name: float(result.x[i]) for i, name in enumerate(param_names)},
        loss_history=loss_history,
        final_reconstruction=last_recon[0],
        converged=result.success,
        iterations_used=len(loss_history),
        wall_times=wall_times,
    )


def _optimize_grid_search(
    data,
    reconstruct_fn,
    loss_fn,
    optimizable_params,
    fixed_params=None,
    grid_points=7,
    logger=None,
) -> OptimizationResult:
    """Exhaustive grid search over parameter ranges.

    Always runs under ``torch.no_grad()``. Each parameter gets
    ``grid_points`` evenly spaced values centered on ``initial_value``
    with spacing ``learning_rate``.
    """
    if logger is None:
        logger = PrintLogger()
    if fixed_params is None:
        fixed_params = {}

    param_names = list(optimizable_params.keys())
    grids = []
    for name in param_names:
        center, step = optimizable_params[name]
        half = (grid_points // 2) * step
        grids.append(np.arange(center - half, center + half + step / 2, step))

    loss_history: list[float] = []
    wall_times: list[float] = []
    best_loss = float("inf")
    best_values = {name: optimizable_params[name][0] for name in param_names}
    best_recon = None

    all_combos = list(itertools.product(*grids))
    pbar = tqdm(all_combos, desc="Grid search")
    for step_idx, combo in enumerate(pbar):
        t_start = time.monotonic()
        param_tensors = {}
        for i, name in enumerate(param_names):
            param_tensors[name] = torch.tensor(
                float(combo[i]),
                dtype=torch.float32,
            )

        kwargs = dict(fixed_params)
        kwargs.update(param_tensors)

        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                recon = reconstruct_fn(data, **kwargs)
        loss = loss_fn(recon)
        loss_val = loss.item()

        wall_times.append(time.monotonic() - t_start)
        loss_history.append(loss_val)

        logger.log_scalar("loss", loss_val, step_idx)
        for i, name in enumerate(param_names):
            logger.log_scalar(name, float(combo[i]), step_idx)

        if loss_val < best_loss:
            best_loss = loss_val
            best_values = {name: float(combo[i]) for i, name in enumerate(param_names)}
            best_recon = recon.detach()

        postfix = {"best_loss": f"{best_loss:.4f}"}
        postfix.update({name: f"{best_values[name]:.4f}" for name in param_names})
        pbar.set_postfix(postfix)

    logger.close()

    return OptimizationResult(
        optimized_values=best_values,
        loss_history=loss_history,
        final_reconstruction=best_recon,
        converged=True,
        iterations_used=len(loss_history),
        wall_times=wall_times,
    )

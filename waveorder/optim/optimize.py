"""Core optimization loop for parameter optimization."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from waveorder.optim.logging import OptimLogger, PrintLogger


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    optimized_values: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    final_reconstruction: Tensor | None = None


def optimize_reconstruction(
    data: Tensor,
    reconstruct_fn: Callable[..., Tensor],
    loss_fn: Callable[[Tensor], Tensor],
    optimizable_params: dict[str, tuple[float, float]],
    fixed_params: dict | None = None,
    num_iterations: int = 10,
    logger: OptimLogger | None = None,
    log_images: bool = False,
    log_extras_fn: Callable | None = None,
    device: str | torch.device = "cpu",
) -> OptimizationResult:
    """Run optimization loop over reconstruction parameters.

    Parameters
    ----------
    data : Tensor
        Input data (e.g., ZYX defocus stack)
    reconstruct_fn : callable
        Function that takes (data, **params) and returns a reconstruction.
        All optimizable params are passed as tensors.
    loss_fn : callable
        Function that takes a reconstruction and returns a scalar loss.
    optimizable_params : dict[str, tuple[float, float]]
        {param_name: (initial_value, learning_rate)} for each optimizable parameter.
    fixed_params : dict, optional
        Additional fixed parameters to pass to reconstruct_fn.
    num_iterations : int
        Number of Adam optimizer steps.
    logger : OptimLogger, optional
        Logger for tracking optimization progress.
    log_images : bool
        If True, log reconstruction images each iteration.
    device : str or torch.device
        Device for tensors (e.g., "cpu", "cuda"). Default: "cpu".

    Returns
    -------
    OptimizationResult
        Contains optimized parameter values, loss history, and final reconstruction.
    """
    if logger is None:
        logger = PrintLogger()

    if fixed_params is None:
        fixed_params = {}

    data = data.to(device)

    # Create optimizable tensors with per-parameter learning rates
    param_tensors: dict[str, Tensor] = {}
    param_groups: list[dict] = []

    for name, (init_val, lr) in optimizable_params.items():
        t = torch.tensor(init_val, dtype=torch.float32, device=device, requires_grad=True)
        param_tensors[name] = t
        param_groups.append({"params": [t], "lr": lr})

    optimizer = torch.optim.Adam(param_groups)

    loss_history: list[float] = []
    final_recon = None

    pbar = tqdm(range(num_iterations), desc="Optimizing")
    for step in pbar:
        optimizer.zero_grad()

        # Build kwargs: merge fixed params with current tensor values
        kwargs = dict(fixed_params)
        kwargs.update(param_tensors)

        # Forward pass (suppress prints from inner functions)
        with contextlib.redirect_stdout(io.StringIO()):
            recon = reconstruct_fn(data, **kwargs)
        loss = loss_fn(recon)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Update progress bar
        postfix = {"loss": f"{loss_val:.4f}"}
        for name, t in param_tensors.items():
            postfix[name] = f"{t.item():.4f}"
        pbar.set_postfix(postfix)

        # Log
        logger.log_scalar("loss", loss_val, step)
        for name, t in param_tensors.items():
            logger.log_scalar(name, t.item(), step)
            grad_val = t.grad.item() if t.grad is not None else 0.0
            logger.log_scalar(f"grad/{name}", grad_val, step)

        if log_images:
            img = recon.detach()
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
            logger.log_image("reconstruction", img, step)

        if log_extras_fn is not None:
            log_extras_fn(step, logger, param_tensors)

        if step == num_iterations - 1:
            final_recon = recon.detach()

    logger.close()

    optimized_values = {name: t.item() for name, t in param_tensors.items()}

    return OptimizationResult(
        optimized_values=optimized_values,
        loss_history=loss_history,
        final_reconstruction=final_recon,
    )

"""Logging utilities for optimization."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class OptimLogger(Protocol):
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_image(self, tag: str, image: Tensor, step: int) -> None: ...
    def close(self) -> None: ...


class NullLogger:
    """Silent logger (default)."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_image(self, tag: str, image: Tensor, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class PrintLogger:
    """Logger that prints to stdout."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        print(f"  [{step}] {tag}: {value:.6f}")

    def log_image(self, tag: str, image: Tensor, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger:
    """Logger that writes to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image: Tensor, step: int) -> None:
        # Normalize for visualization
        img = image.detach().cpu()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        self._writer.add_image(tag, img.unsqueeze(0), step)

    def close(self) -> None:
        self._writer.close()

"""Device selection for waveorder computations."""

from __future__ import annotations

import torch


def auto_device() -> torch.device:
    """Select the best available device.

    Returns
    -------
    torch.device
        ``cuda`` if available, then ``mps``, else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device specification to a torch.device.

    Parameters
    ----------
    device : str, torch.device, or None
        Device specification. ``None`` means CPU. ``"auto"`` selects
        the best available device. Strings like ``"cuda:0"``,
        ``"cuda:1"``, ``"mps"``, ``"cpu"`` are passed through.

    Returns
    -------
    torch.device
    """
    if device is None or device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        return auto_device()
    return torch.device(device)

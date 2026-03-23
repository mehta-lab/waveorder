"""OptimizableFloat type and helpers for parameter optimization."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, model_validator


class OptimizableFloat(BaseModel):
    """A float parameter that can optionally be optimized.

    Accepts a bare float/int (backwards compatible, not optimizable)
    or a mapping with `init`/`initial_value` and `lr`/`learning_rate`.

    Examples
    --------
    >>> OptimizableFloat(init=0.0, lr=0.01)
    OptimizableFloat(initial_value=0.0, learning_rate=0.01)
    >>> OptimizableFloat(initial_value=3.0, learning_rate=0.5)
    OptimizableFloat(initial_value=3.0, learning_rate=0.5)
    """

    initial_value: float = 0.0
    learning_rate: float = 0.01

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, (int, float)):
            return {"initial_value": float(data), "learning_rate": 0.0}
        if isinstance(data, dict):
            out = dict(data)
            if "init" in out:
                out.setdefault("initial_value", out.pop("init"))
            if "lr" in out:
                out.setdefault("learning_rate", out.pop("lr"))
            return out
        return data

    @property
    def value(self) -> float:
        return self.initial_value

    @property
    def lr(self) -> float:
        return self.learning_rate

    @property
    def is_optimizable(self) -> bool:
        return self.learning_rate > 0

    def __float__(self) -> float:
        return self.initial_value

    def __repr__(self) -> str:
        return f"OptimizableFloat(initial_value={self.initial_value}, learning_rate={self.learning_rate})"


def has_optimizable_params(obj: Any) -> bool:
    """Recursively check if a Pydantic model contains any optimizable parameters."""
    if isinstance(obj, OptimizableFloat):
        return obj.is_optimizable
    if isinstance(obj, BaseModel):
        for name in obj.__class__.model_fields:
            val = getattr(obj, name)
            if has_optimizable_params(val):
                return True
    return False


def extract_optimizable_params(obj: Any, prefix: str = "") -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Extract optimizable and fixed parameters from a Pydantic model.

    Returns
    -------
    optimizable : dict[str, tuple[float, float]]
        {dotted.name: (initial_value, learning_rate)} for optimizable params
    fixed : dict[str, float]
        {dotted.name: value} for fixed OptimizableFloat params
    """
    optimizable: dict[str, tuple[float, float]] = {}
    fixed: dict[str, float] = {}

    if isinstance(obj, OptimizableFloat):
        key = prefix
        if obj.is_optimizable:
            optimizable[key] = (obj.initial_value, obj.learning_rate)
        else:
            fixed[key] = obj.initial_value
        return optimizable, fixed

    if isinstance(obj, BaseModel):
        for name in obj.__class__.model_fields:
            val = getattr(obj, name)
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(val, (OptimizableFloat, BaseModel)):
                o, f = extract_optimizable_params(val, path)
                optimizable.update(o)
                fixed.update(f)

    return optimizable, fixed


# Type alias for use in Union annotations
OptimizableValue = Union[float, OptimizableFloat]

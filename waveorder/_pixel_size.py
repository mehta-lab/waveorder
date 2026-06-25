"""Lateral pixel size value type, supporting isotropic and anisotropic yx."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_serializer


class YXPixelSize(BaseModel):
    """Lateral pixel size in micrometers, possibly anisotropic.

    Use :meth:`isotropic` for square pixels, or :meth:`from_value` to
    normalize either a scalar, a mapping, or an existing YXPixelSize
    into a :class:`YXPixelSize`. The ``from_value`` constructor is the
    intended entry point for internal functions that accept either a
    plain float (treated as isotropic) or this model.

    Attributes
    ----------
    y : float
        Pixel size along the y axis in micrometers.
    x : float
        Pixel size along the x axis in micrometers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    y: PositiveFloat = Field(description="pixel size along y in micrometers")
    x: PositiveFloat = Field(description="pixel size along x in micrometers")

    @classmethod
    def isotropic(cls, value: float) -> YXPixelSize:
        """Construct a YXPixelSize with equal y and x spacing.

        Parameters
        ----------
        value : float
            Pixel size in micrometers, applied to both y and x.
        """
        return cls(y=value, x=value)

    @classmethod
    def from_value(cls, value: Any) -> YXPixelSize:
        """Normalize a scalar, mapping, or YXPixelSize into a YXPixelSize.

        Accepts any float-castable scalar (Python ``float`` / ``int``, numpy
        scalars, 0-d ``ndarray``, 0-d tensors with ``__float__``), a mapping
        with keys ``y`` and ``x``, or an existing :class:`YXPixelSize`.

        Parameters
        ----------
        value : float-like, dict, or YXPixelSize
            Scalar (isotropic shorthand), mapping with keys ``y`` and ``x``,
            or an existing YXPixelSize (returned unchanged).

        Returns
        -------
        YXPixelSize
            The normalized value.

        Raises
        ------
        TypeError
            If ``value`` is not one of the accepted forms.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise TypeError("cannot convert bool to YXPixelSize")
        if isinstance(value, str):
            raise TypeError("cannot convert str to YXPixelSize")
        if isinstance(value, dict):
            return cls(**value)
        # Accept any float-castable scalar: Python numbers, numpy scalars,
        # 0-d ndarrays, 0-d tensors. ``float()`` raises TypeError for
        # multi-element arrays and non-numeric inputs.
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"cannot convert {type(value).__name__} to YXPixelSize; "
                "expected float, mapping with keys 'y' and 'x', or YXPixelSize"
            ) from None
        return cls.isotropic(as_float)

    @property
    def is_isotropic(self) -> bool:
        """True if y and x spacings are equal."""
        return self.y == self.x

    @model_serializer
    def _serialize(self):
        """Serialize as a scalar when isotropic, else as ``{'y': ..., 'x': ...}``.

        Both forms round-trip through :meth:`from_value`, so YAML configs
        for square pixels stay readable as a single number.
        """
        if self.y == self.x:
            return self.y
        return {"y": self.y, "x": self.x}

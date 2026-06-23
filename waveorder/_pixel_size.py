"""Lateral pixel size value type, supporting isotropic and anisotropic yx."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


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

        Parameters
        ----------
        value : float or dict or YXPixelSize
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
        if isinstance(value, (int, float)):
            return cls.isotropic(float(value))
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"cannot convert {type(value).__name__} to YXPixelSize; "
            "expected float, mapping with keys 'y' and 'x', or YXPixelSize"
        )

    @property
    def is_isotropic(self) -> bool:
        """True if y and x spacings are equal."""
        return self.y == self.x

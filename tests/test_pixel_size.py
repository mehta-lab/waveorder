"""Tests for the YXPixelSize value type."""

import pytest
from pydantic import ValidationError

from waveorder._pixel_size import YXPixelSize


def test_isotropic_constructor_sets_equal_y_and_x():
    """Isotropic constructor produces a square YXPixelSize."""
    p = YXPixelSize.isotropic(0.3)
    assert p.y == 0.3
    assert p.x == 0.3
    assert p.is_isotropic


def test_explicit_constructor_supports_anisotropic():
    """Direct construction with separate y and x produces an anisotropic value."""
    p = YXPixelSize(y=0.3, x=0.25)
    assert p.y == 0.3
    assert p.x == 0.25
    assert not p.is_isotropic


def test_from_value_passes_through_existing_yx_pixel_size():
    """from_value returns an existing YXPixelSize unchanged."""
    p = YXPixelSize(y=0.3, x=0.25)
    assert YXPixelSize.from_value(p) is p


def test_from_value_accepts_scalar_as_isotropic():
    """from_value treats a float as isotropic spacing."""
    p = YXPixelSize.from_value(0.3)
    assert p == YXPixelSize.isotropic(0.3)


def test_from_value_accepts_int_as_isotropic():
    """from_value accepts an int scalar (numpy/yaml may yield ints)."""
    p = YXPixelSize.from_value(1)
    assert p == YXPixelSize.isotropic(1.0)


def test_from_value_accepts_mapping():
    """from_value accepts a {'y': ..., 'x': ...} mapping."""
    p = YXPixelSize.from_value({"y": 0.3, "x": 0.25})
    assert p == YXPixelSize(y=0.3, x=0.25)


def test_from_value_rejects_bool():
    """Bools are not valid pixel sizes even though they're a subclass of int."""
    with pytest.raises(TypeError):
        YXPixelSize.from_value(True)


def test_from_value_rejects_unknown_type():
    """Unknown types raise a clear TypeError."""
    with pytest.raises(TypeError, match="cannot convert"):
        YXPixelSize.from_value("0.3")


def test_from_value_rejects_negative_via_mapping():
    """Negative spacings via the mapping form fail pydantic validation."""
    with pytest.raises(ValidationError):
        YXPixelSize.from_value({"y": -0.1, "x": 0.1})


def test_from_value_rejects_zero_via_scalar():
    """Zero spacing is not a valid isotropic value."""
    with pytest.raises(ValidationError):
        YXPixelSize.from_value(0.0)


def test_from_value_rejects_extra_keys_in_mapping():
    """Unknown mapping keys are rejected by extra='forbid'."""
    with pytest.raises(ValidationError):
        YXPixelSize.from_value({"y": 0.3, "x": 0.25, "z": 0.5})


def test_yx_pixel_size_is_frozen():
    """Instances are immutable; cannot mutate y or x after construction."""
    p = YXPixelSize(y=0.3, x=0.25)
    with pytest.raises(ValidationError):
        p.y = 0.5  # type: ignore[misc]


def test_equality_is_value_based():
    """Two YXPixelSize with the same y and x compare equal."""
    assert YXPixelSize(y=0.3, x=0.25) == YXPixelSize(y=0.3, x=0.25)
    assert YXPixelSize.isotropic(0.3) == YXPixelSize(y=0.3, x=0.3)
    assert YXPixelSize.isotropic(0.3) != YXPixelSize(y=0.3, x=0.25)


def test_is_isotropic_distinguishes_equal_and_unequal():
    """is_isotropic returns True only when y == x exactly."""
    assert YXPixelSize.isotropic(0.3).is_isotropic
    assert not YXPixelSize(y=0.3, x=0.3 + 1e-12).is_isotropic

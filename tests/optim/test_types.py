"""Tests for OptimizableFloat type and helpers."""

import yaml
from pydantic import Field

from waveorder.api._settings import MyBaseModel
from waveorder.optim._types import (
    OptimizableFloat,
    extract_optimizable_params,
    has_optimizable_params,
)


def test_bare_float_creates_non_optimizable():
    of = OptimizableFloat.model_validate(3.14)
    assert of.value == 3.14
    assert of.lr == 0.0
    assert not of.is_optimizable
    assert float(of) == 3.14


def test_bare_int_creates_non_optimizable():
    of = OptimizableFloat.model_validate(0)
    assert of.value == 0.0
    assert not of.is_optimizable


def test_init_lr_creates_optimizable():
    of = OptimizableFloat(init=0.0, lr=0.01)
    assert of.value == 0.0
    assert of.lr == 0.01
    assert of.is_optimizable


def test_initial_value_learning_rate_aliases():
    of = OptimizableFloat(initial_value=3.0, learning_rate=0.5)
    assert of.value == 3.0
    assert of.lr == 0.5
    assert of.is_optimizable


def test_dict_with_init_lr():
    of = OptimizableFloat.model_validate({"init": 1.5, "lr": 0.1})
    assert of.value == 1.5
    assert of.lr == 0.1


def test_inside_pydantic_model():
    class DemoSettings(MyBaseModel):
        z_focus_offset: float | OptimizableFloat = Field(default=0.0)

    # Plain float
    s1 = DemoSettings(z_focus_offset=2.5)
    assert s1.z_focus_offset == 2.5

    # OptimizableFloat dict
    s2 = DemoSettings(z_focus_offset={"init": 0, "lr": 0.01})
    assert isinstance(s2.z_focus_offset, OptimizableFloat)
    assert s2.z_focus_offset.is_optimizable


def test_yaml_roundtrip():
    data = {"z_focus_offset": {"init": 0, "lr": 0.01}}
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    of = OptimizableFloat.model_validate(loaded["z_focus_offset"])
    assert of.value == 0.0
    assert of.lr == 0.01


def test_yaml_bare_float_roundtrip():
    data = {"z_focus_offset": 3.0}
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    of = OptimizableFloat.model_validate(loaded["z_focus_offset"])
    assert of.value == 3.0
    assert not of.is_optimizable


def test_extract_optimizable_params():
    class Inner(MyBaseModel):
        na: float | OptimizableFloat = 0.9
        offset: float | OptimizableFloat = Field(default=0.0)

    class Outer(MyBaseModel):
        tf: Inner = Inner()

    settings = Outer(
        tf=Inner(
            na={"init": 0.9, "lr": 0.01},
            offset=5.0,
        )
    )
    opt, fixed = extract_optimizable_params(settings)
    assert "tf.na" in opt
    assert opt["tf.na"] == (0.9, 0.01)
    # bare float fields are not in fixed (only OptimizableFloat with lr=0)
    assert "tf.offset" not in fixed


def test_extract_optimizable_params_with_optimizable_float():
    class Inner(MyBaseModel):
        na: float | OptimizableFloat = 0.9
        offset: float | OptimizableFloat = Field(default=0.0)

    class Outer(MyBaseModel):
        tf: Inner = Inner()

    settings = Outer(
        tf=Inner(
            na=OptimizableFloat(init=0.9, lr=0.01),
            offset=OptimizableFloat(init=5.0, lr=0.0),
        )
    )
    opt, fixed = extract_optimizable_params(settings)
    assert "tf.na" in opt
    assert "tf.offset" in fixed
    assert fixed["tf.offset"] == 5.0


def test_has_optimizable_params_true():
    class Inner(MyBaseModel):
        val: float | OptimizableFloat = 0.0

    class Outer(MyBaseModel):
        inner: Inner = Inner()

    s = Outer(inner=Inner(val={"init": 0, "lr": 0.1}))
    assert has_optimizable_params(s)


def test_has_optimizable_params_false():
    class Inner(MyBaseModel):
        val: float | OptimizableFloat = 0.0

    class Outer(MyBaseModel):
        inner: Inner = Inner()

    s = Outer(inner=Inner(val=3.0))
    assert not has_optimizable_params(s)


def test_backwards_compat_with_existing_configs():
    """Existing configs with plain floats should work unchanged."""
    config = {
        "z_focus_offset": 0,
        "numerical_aperture_detection": 1.2,
    }

    class FakeSettings(MyBaseModel):
        z_focus_offset: float | OptimizableFloat = 0.0
        numerical_aperture_detection: float | OptimizableFloat = 1.2

    s = FakeSettings(**config)
    assert s.z_focus_offset == 0
    assert s.numerical_aperture_detection == 1.2
    assert not has_optimizable_params(s)

"""The form must not switch a feature on just because its settings block is Optional.

``auto_regularization: null`` in a config means "off". These tests drive the two
pydantic-to-widget traversals directly, without a napari viewer.
"""

import pytest

from waveorder.cli import settings
from waveorder.optim.autoreg import AutoRegularizationSettings

widgets = pytest.importorskip("magicgui.widgets")
tab_recon = pytest.importorskip("waveorder.plugin.tab_recon")

EXCLUDES = ["birefringence", "fluorescence"]


class _Form:
    """Just the two traversal methods, which use ``self`` only to recurse."""

    add_pydantic_to_container = tab_recon.Ui_ReconTab_Form.add_pydantic_to_container
    get_pydantic_kwargs = tab_recon.Ui_ReconTab_Form.get_pydantic_kwargs


def _phase_model():
    """What build_model produces when only the phase mode is ticked."""
    return settings.ReconstructionSettings(input_channel_names=["BF"], phase=settings.PhaseSettings())


def _as_loaded(model):
    """The dict a config file or a previous model hands the form.

    ``yx_pixel_size`` is spelled out as a mapping: its isotropic shorthand serializes
    to a bare float, which the form's nested-model recursion cannot take (an existing
    limitation, separate from what is tested here).
    """
    json_dict = model.model_dump(mode="json")
    pixel = json_dict["phase"]["transfer_function"]["yx_pixel_size"]
    if not isinstance(pixel, dict):
        json_dict["phase"]["transfer_function"]["yx_pixel_size"] = {"y": pixel, "x": pixel}
    return json_dict


def _round_trip(model, json_dict=None):
    """Build the form for ``model`` and collect what it would submit, untouched."""
    form = _Form()
    container = widgets.Container()
    form.add_pydantic_to_container(model, container, EXCLUDES, json_dict)
    kwargs = {}
    form.get_pydantic_kwargs(container, model, kwargs, EXCLUDES)
    return container, kwargs


def _toggle(container):
    return getattr(container.phase.apply_inverse.auto_regularization, tab_recon.OPTIONAL_BLOCK_TOGGLE)


def test_untouched_optional_block_is_submitted_as_none():
    """Neither a fresh form nor a config loaded with ``null`` may turn the block on."""
    for json_dict in (None, _as_loaded(_phase_model())):
        container, kwargs = _round_trip(_phase_model(), json_dict)
        assert _toggle(container).value is False
        assert kwargs["phase"]["apply_inverse"]["auto_regularization"] is None

    # And the submission parses to a model with the block off, not to a sweep.
    kwargs.update(input_channel_names=["BF"], time_indices="all")
    parsed = settings.ReconstructionSettings.model_validate(kwargs)
    assert parsed.phase.apply_inverse.auto_regularization is None


def test_optional_block_present_in_the_config_comes_back_populated():
    json_dict = _as_loaded(_phase_model())
    json_dict["phase"]["apply_inverse"]["auto_regularization"] = AutoRegularizationSettings(
        rule="l_curve", num_samples=11
    ).model_dump(mode="json")

    container, kwargs = _round_trip(_phase_model(), json_dict)

    assert _toggle(container).value is True
    block = kwargs["phase"]["apply_inverse"]["auto_regularization"]
    assert block["rule"] == "l_curve"
    assert block["num_samples"] == 11

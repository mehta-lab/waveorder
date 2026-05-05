"""Unit tests for ``waveorder.api.tile_stitch`` settings + the kind discriminator."""

import pytest
from pydantic import ValidationError

from waveorder.api.birefringence import Settings as BirefringenceSettings
from waveorder.api.fluorescence import Settings as FluorescenceSettings
from waveorder.api.phase import Settings as PhaseSettings
from waveorder.api.tile_stitch import (
    BlendSettings,
    TileSettings,
    TileStitchSettings,
)
from waveorder.tile_stitch.blend import Blend

# --- BlendSettings ---


def test_blend_settings_defaults():
    s = BlendSettings()
    assert s.kind == "uniform_mean"
    assert s.sigma_fraction is None
    assert s.accumulator_dtype == "float32"


def test_blend_settings_builds_blend():
    assert isinstance(BlendSettings(kind="uniform_mean").build(), Blend)
    assert isinstance(BlendSettings(kind="gaussian_mean", sigma_fraction=0.1).build(), Blend)
    assert isinstance(BlendSettings(kind="max").build(), Blend)
    assert isinstance(BlendSettings(kind="min").build(), Blend)


def test_blend_settings_gaussian_default_sigma():
    b = BlendSettings(kind="gaussian_mean").build()
    assert "gaussian_mean" in b.name


def test_blend_settings_sigma_only_when_gaussian():
    with pytest.raises(ValidationError, match="sigma_fraction is only valid"):
        BlendSettings(kind="uniform_mean", sigma_fraction=0.1)
    with pytest.raises(ValidationError, match="sigma_fraction is only valid"):
        BlendSettings(kind="max", sigma_fraction=0.1)
    BlendSettings(kind="gaussian_mean", sigma_fraction=0.1)


def test_blend_settings_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        BlendSettings(kind="median")


def test_blend_settings_extra_fields_rejected():
    with pytest.raises(ValidationError):
        BlendSettings(kind="uniform_mean", typo_field=True)


# --- TileSettings ---


def test_tile_settings_overlap_dim_must_subset_tile_dims():
    with pytest.raises(ValidationError, match="not present in tile_size"):
        TileSettings(tile_size={"y": 512, "x": 512}, overlap={"z": 4, "y": 32, "x": 32})


def test_tile_settings_overlap_must_be_strictly_less_than_tile_size():
    with pytest.raises(ValidationError, match="must be strictly less than tile_size"):
        TileSettings(tile_size={"y": 64}, overlap={"y": 64})
    with pytest.raises(ValidationError, match="must be strictly less than tile_size"):
        TileSettings(tile_size={"y": 64}, overlap={"y": 100})
    TileSettings(tile_size={"y": 64}, overlap={"y": 32})


def test_tile_settings_no_output_chunk_field():
    """output_chunk is not a user-adjustable parameter; it always shadows tile_size."""
    s = TileSettings(tile_size={"y": 512, "x": 512})
    assert not hasattr(s, "output_chunk")
    # Reject if a user tries to pass it in.
    with pytest.raises(ValidationError):
        TileSettings(tile_size={"y": 512}, output_chunk={"y": 256})


def test_tile_settings_overlap_defaults_to_zero_per_dim():
    s = TileSettings(tile_size={"y": 512, "x": 512})
    assert s.overlap == {}


# --- TileStitchSettings + recon discriminator ---


def test_tilestitch_schema_version_pinned_to_one():
    s = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        recon=PhaseSettings(),
    )
    assert s.schema_version == "1"
    with pytest.raises(ValidationError):
        TileStitchSettings(
            schema_version="2",
            tile=TileSettings(tile_size={"y": 64, "x": 64}),
            recon=PhaseSettings(),
        )


def test_tilestitch_recon_phase():
    s = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        recon=PhaseSettings(),
    )
    assert isinstance(s.recon, PhaseSettings)
    assert s.recon.kind == "phase"


def test_tilestitch_recon_fluorescence():
    s = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        recon=FluorescenceSettings(),
    )
    assert isinstance(s.recon, FluorescenceSettings)
    assert s.recon.kind == "fluorescence"


def test_tilestitch_recon_birefringence():
    s = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        recon=BirefringenceSettings(),
    )
    assert isinstance(s.recon, BirefringenceSettings)
    assert s.recon.kind == "birefringence"


def test_tilestitch_recon_discriminator_round_trip():
    """Dict-form recon (as it would arrive from YAML) selects the right Settings via 'kind'."""
    payload = {
        "tile": {"tile_size": {"y": 64, "x": 64}},
        "recon": {"kind": "phase"},
    }
    s = TileStitchSettings.model_validate(payload)
    assert isinstance(s.recon, PhaseSettings)

    payload["recon"]["kind"] = "fluorescence"
    s = TileStitchSettings.model_validate(payload)
    assert isinstance(s.recon, FluorescenceSettings)

    payload["recon"]["kind"] = "birefringence"
    s = TileStitchSettings.model_validate(payload)
    assert isinstance(s.recon, BirefringenceSettings)


def test_tilestitch_recon_discriminator_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        TileStitchSettings.model_validate(
            {"tile": {"tile_size": {"y": 64, "x": 64}}, "recon": {"kind": "diffraction-tomography"}}
        )


def test_tilestitch_blend_default_then_override():
    s = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        recon=PhaseSettings(),
    )
    assert s.blend.kind == "uniform_mean"

    s2 = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 64, "x": 64}),
        blend=BlendSettings(kind="gaussian_mean", sigma_fraction=0.05),
        recon=PhaseSettings(),
    )
    assert s2.blend.kind == "gaussian_mean"
    assert s2.blend.sigma_fraction == 0.05

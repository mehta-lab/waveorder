from pathlib import Path

import pytest
from pydantic import ValidationError

from waveorder.api import birefringence, fluorescence, phase
from waveorder.cli import settings
from waveorder.io import utils


def test_reconstruction_settings():
    # Test defaults
    s = settings.ReconstructionSettings(
        birefringence=settings.BirefringenceSettings()
    )
    assert len(s.input_channel_names) == 4
    assert s.birefringence.apply_inverse.background_path == ""
    assert s.phase == None
    assert s.fluorescence == None

    # Test logic that "fluorescence" or ("phase" and/or "birefringence")
    s = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        birefringence=None,
        phase=None,
        fluorescence=settings.FluorescenceSettings(),
    )

    assert s.fluorescence.apply_inverse.reconstruction_algorithm == "Tikhonov"

    # Not allowed to supply both phase/biref and fluorescence
    with pytest.raises(ValidationError):
        settings.ReconstructionSettings(
            phase=settings.PhaseSettings(),
            fluorescence=settings.FluorescenceSettings(),
        )

    # Test incorrect settings
    with pytest.raises(ValidationError):
        settings.ReconstructionSettings(input_channel_names=3)

    with pytest.raises(ValidationError):
        settings.ReconstructionSettings(reconstruction_dimension=1)

    # Test typo
    with pytest.raises(ValidationError):
        settings.ReconstructionSettings(
            flurescence=settings.FluorescenceSettings()
        )


def test_biref_tf_settings():
    birefringence.TransferFunctionSettings(swing=0.1)

    with pytest.raises(ValidationError):
        birefringence.TransferFunctionSettings(swing=1.1)

    with pytest.raises(ValidationError):
        birefringence.TransferFunctionSettings(scheme="Test")


def test_phase_tf_settings():
    phase.TransferFunctionSettings(
        index_of_refraction_media=1.0, numerical_aperture_detection=0.8
    )

    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(
            index_of_refraction_media=1.0, numerical_aperture_detection=1.1
        )

    # Inconsistent units
    with pytest.warns(UserWarning):
        phase.TransferFunctionSettings(yx_pixel_size=650, z_pixel_size=0.3)

    # Extra parameter
    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(zyx_pixel_size=650)


def test_fluor_tf_settings():
    fluorescence.TransferFunctionSettings(
        wavelength_emission=0.500, yx_pixel_size=0.2
    )

    with pytest.warns(UserWarning):
        fluorescence.TransferFunctionSettings(
            wavelength_emission=0.500, yx_pixel_size=2000
        )


def test_generate_example_settings():
    project_root = Path(__file__).parent.parent.parent
    example_path = project_root / "docs" / "examples" / "cli" / "configs"

    # 2D configs override regularization_strength for better 2D defaults
    phase_2d_apply_inverse = phase.ApplyInverseSettings(
        regularization_strength=1e-2
    )
    fluor_2d_apply_inverse = fluorescence.ApplyInverseSettings(
        regularization_strength=1e-2
    )

    configs = {
        "birefringence_3d.yml": settings.ReconstructionSettings(
            birefringence=settings.BirefringenceSettings(),
        ),
        "phase_3d.yml": settings.ReconstructionSettings(
            input_channel_names=["Brightfield"],
            phase=settings.PhaseSettings(),
        ),
        "phase_2d.yml": settings.ReconstructionSettings(
            input_channel_names=["Brightfield"],
            reconstruction_dimension=2,
            phase=phase.Settings(apply_inverse=phase_2d_apply_inverse),
        ),
        "fluorescence_3d.yml": settings.ReconstructionSettings(
            input_channel_names=["GFP"],
            fluorescence=settings.FluorescenceSettings(),
        ),
        "fluorescence_2d.yml": settings.ReconstructionSettings(
            input_channel_names=["GFP"],
            reconstruction_dimension=2,
            fluorescence=fluorescence.Settings(
                apply_inverse=fluor_2d_apply_inverse
            ),
        ),
        "birefringence-and-phase_3d.yml": settings.ReconstructionSettings(
            birefringence=settings.BirefringenceSettings(),
            phase=settings.PhaseSettings(),
        ),
    }

    for file_name, settings_obj in configs.items():
        config_path = example_path / file_name
        utils.model_to_commented_yaml(settings_obj, config_path)
        settings_roundtrip = utils.yaml_to_model(
            config_path, settings.ReconstructionSettings
        )
        assert settings_obj.model_dump() == settings_roundtrip.model_dump()

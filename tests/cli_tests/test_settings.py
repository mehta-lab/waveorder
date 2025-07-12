from pathlib import Path

import pytest
from pydantic.v1 import ValidationError

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
    settings.BirefringenceTransferFunctionSettings(swing=0.1)

    with pytest.raises(ValidationError):
        settings.BirefringenceTransferFunctionSettings(swing=1.1)

    with pytest.raises(ValidationError):
        settings.BirefringenceTransferFunctionSettings(scheme="Test")


def test_phase_tf_settings():
    settings.PhaseTransferFunctionSettings(
        index_of_refraction_media=1.0, numerical_aperture_detection=0.8
    )

    with pytest.raises(ValidationError):
        settings.PhaseTransferFunctionSettings(
            index_of_refraction_media=1.0, numerical_aperture_detection=1.1
        )

    # Inconsistent units
    with pytest.warns(UserWarning):
        settings.PhaseTransferFunctionSettings(
            yx_pixel_size=650, z_pixel_size=0.3
        )

    # Extra parameter
    with pytest.raises(ValidationError):
        settings.PhaseTransferFunctionSettings(zyx_pixel_size=650)


def test_fluor_tf_settings():
    settings.FluorescenceTransferFunctionSettings(
        wavelength_emission=0.500, yx_pixel_size=0.2
    )

    with pytest.warns(UserWarning):
        settings.FluorescenceTransferFunctionSettings(
            wavelength_emission=0.500, yx_pixel_size=2000
        )


def test_generate_example_settings():
    project_root = Path(__file__).parent.parent.parent
    example_path = project_root / "docs" / "examples" / "configs"

    s0 = settings.ReconstructionSettings(
        birefringence=settings.BirefringenceSettings(),
        phase=settings.PhaseSettings(),
    )
    s1 = settings.ReconstructionSettings(
        input_channel_names=["BF"],
        phase=settings.PhaseSettings(),
    )
    s2 = settings.ReconstructionSettings(
        birefringence=settings.BirefringenceSettings(),
    )
    s3 = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        fluorescence=settings.FluorescenceSettings(),
    )
    file_names = [
        "birefringence-and-phase.yml",
        "phase.yml",
        "birefringence.yml",
        "fluorescence.yml",
    ]
    settings_list = [s0, s1, s2, s3]

    # Save to examples folder and test roundtrip
    for file_name, settings_obj in zip(file_names, settings_list):
        config_path = example_path / file_name
        utils.model_to_yaml(settings_obj, config_path)
        settings_roundtrip = utils.yaml_to_model(
            config_path, settings.ReconstructionSettings
        )
        assert settings_obj.dict() == settings_roundtrip.dict()

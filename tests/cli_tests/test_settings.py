import pytest
import yaml
from pydantic import ValidationError

from waveorder._pixel_size import YXPixelSize
from waveorder.api import birefringence, fluorescence, phase
from waveorder.cli import settings
from waveorder.io import utils


def test_reconstruction_settings():
    # Test defaults
    s = settings.ReconstructionSettings(birefringence=settings.BirefringenceSettings())
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
        settings.ReconstructionSettings(flurescence=settings.FluorescenceSettings())


def test_biref_tf_settings():
    birefringence.TransferFunctionSettings(swing=0.1)

    with pytest.raises(ValidationError):
        birefringence.TransferFunctionSettings(swing=1.1)

    with pytest.raises(ValidationError):
        birefringence.TransferFunctionSettings(scheme="Test")


def test_phase_tf_settings():
    phase.TransferFunctionSettings(index_of_refraction_media=1.0, numerical_aperture_detection=0.8)

    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(index_of_refraction_media=1.0, numerical_aperture_detection=1.1)

    # Inconsistent units
    with pytest.warns(UserWarning):
        phase.TransferFunctionSettings(yx_pixel_size=650, z_pixel_size=0.3)

    # Extra parameter
    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(zyx_pixel_size=650)


def test_fluor_tf_settings():
    fluorescence.TransferFunctionSettings(wavelength_emission=0.500, yx_pixel_size=0.2)

    with pytest.warns(UserWarning):
        fluorescence.TransferFunctionSettings(wavelength_emission=0.500, yx_pixel_size=2000)


def test_generate_example_settings(pytestconfig):
    example_path = pytestconfig.rootpath / "docs" / "examples" / "cli" / "configs"

    # 2D configs override regularization_strength for better 2D defaults
    phase_2d_apply_inverse = phase.ApplyInverseSettings(regularization_strength=1e-2)
    fluor_2d_apply_inverse = fluorescence.ApplyInverseSettings(regularization_strength=1e-2)

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
            fluorescence=fluorescence.Settings(apply_inverse=fluor_2d_apply_inverse),
        ),
        "birefringence-and-phase_3d.yml": settings.ReconstructionSettings(
            birefringence=settings.BirefringenceSettings(),
            phase=settings.PhaseSettings(),
        ),
    }

    for file_name, settings_obj in configs.items():
        config_path = example_path / file_name
        utils.model_to_commented_yaml(settings_obj, config_path)
        settings_roundtrip = utils.yaml_to_model(config_path, settings.ReconstructionSettings)
        assert settings_obj.model_dump() == settings_roundtrip.model_dump()


def test_phase_yx_pixel_size_scalar_form():
    """The legacy scalar form parses into an isotropic YXPixelSize."""
    tf = phase.TransferFunctionSettings(yx_pixel_size=0.3)
    assert isinstance(tf.yx_pixel_size, YXPixelSize)
    assert tf.yx_pixel_size.y == 0.3
    assert tf.yx_pixel_size.x == 0.3


def test_phase_yx_pixel_size_dict_form():
    """The {y, x} mapping form parses into an anisotropic YXPixelSize."""
    tf = phase.TransferFunctionSettings(yx_pixel_size={"y": 0.3, "x": 0.25})
    assert isinstance(tf.yx_pixel_size, YXPixelSize)
    assert tf.yx_pixel_size.y == 0.3
    assert tf.yx_pixel_size.x == 0.25


def test_phase_yx_pixel_size_scalar_equals_isotropic_mapping():
    """Scalar and isotropic mapping forms produce equal settings."""
    a = phase.TransferFunctionSettings(yx_pixel_size=0.3)
    b = phase.TransferFunctionSettings(yx_pixel_size={"y": 0.3, "x": 0.3})
    assert a.yx_pixel_size == b.yx_pixel_size


def test_yaml_roundtrip_anisotropic_yx_pixel_size(tmp_path):
    """An anisotropic config round-trips through YAML losslessly."""
    s = phase.TransferFunctionSettings(yx_pixel_size={"y": 0.3, "x": 0.25})
    path = tmp_path / "phase_aniso.yml"
    utils.model_to_yaml(s, path)
    parsed = yaml.safe_load(path.read_text())
    assert parsed["yx_pixel_size"] == {"y": 0.3, "x": 0.25}
    reparsed = utils.yaml_to_model(path, phase.TransferFunctionSettings)
    assert reparsed.yx_pixel_size.y == 0.3
    assert reparsed.yx_pixel_size.x == 0.25


def test_yaml_roundtrip_isotropic_yx_pixel_size_stays_scalar(tmp_path):
    """An isotropic config serializes back as a scalar (not as {y: 0.3, x: 0.3})."""
    s = phase.TransferFunctionSettings(yx_pixel_size=0.3)
    path = tmp_path / "phase_iso.yml"
    utils.model_to_yaml(s, path)
    parsed = yaml.safe_load(path.read_text())
    assert parsed["yx_pixel_size"] == 0.3


def test_invalid_yx_pixel_size_mapping_rejected():
    """A typo in the mapping form fails validation, not silently."""
    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(yx_pixel_size={"Y": 0.3, "x": 0.25})


def test_negative_yx_pixel_size_rejected():
    """Negative spacings are rejected."""
    with pytest.raises(ValidationError):
        phase.TransferFunctionSettings(yx_pixel_size={"y": -0.3, "x": 0.25})

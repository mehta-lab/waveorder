import os
import pytest
import yaml

from recOrder.cli import settings
from recOrder.io.utils import model_to_yaml, yaml_to_model


@pytest.fixture
def model():
    # Create a sample model object
    return settings.ReconstructionSettings(
        birefringence=settings.BirefringenceSettings()
    )


@pytest.fixture
def yaml_path(tmpdir):
    # Create a temporary YAML file path
    return os.path.join(tmpdir, "model.yaml")


def test_model_to_yaml(model, yaml_path):
    # Call the function under test
    model_to_yaml(model, yaml_path)

    # Check if the YAML file is created
    assert os.path.exists(yaml_path)

    # Load the YAML file and verify its contents
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Check if the YAML data is a dictionary
    assert isinstance(yaml_data, dict)

    # Check YAML data
    assert "input_channel_names" in yaml_data


def test_model_to_yaml_invalid_model():
    # Create an object that does not have a 'dict()' method
    invalid_model = "not a model object"

    # Call the function and expect a TypeError
    with pytest.raises(TypeError):
        model_to_yaml(invalid_model, "model.yaml")

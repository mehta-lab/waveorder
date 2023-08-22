import os
from pathlib import Path

import pytest
import yaml

from recOrder.cli import settings
from recOrder.io.utils import add_index_to_path, model_to_yaml


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


def test_add_index_to_path(tmp_path):
    test_cases = [
        ("output.txt", "output_0.txt"),
        ("output.txt", "output_1.txt"),
        ("output.txt", "output_2.txt"),
        ("output.png", "output_0.png"),
        ("output.png", "output_1.png"),
        ("output.png", "output_2.png"),
        ("folder", "folder_0"),
        ("folder", "folder_1"),
        ("folder", "folder_2"),
    ]

    for input_path_str, expected_output_str in test_cases:
        input_path = tmp_path / Path(input_path_str)
        expected_output = tmp_path / Path(expected_output_str)

        output_path = add_index_to_path(input_path)
        assert output_path == expected_output

        output_path.touch()  # Create a file/folder at the expected output path for testing

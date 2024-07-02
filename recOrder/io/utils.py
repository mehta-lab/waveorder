import os
import textwrap
from pathlib import Path

import psutil
import torch
import yaml
from iohub import open_ome_zarr



def add_index_to_path(path: Path):
    """Takes a path to a file or folder and appends the smallest index that does
    not already exist in that folder.

    For example:
    './output.txt' -> './output_0.txt' if no other files named './output*.txt' exist.
    './output.txt' -> './output_2.txt' if './output_0.txt' and './output_1.txt' already exist.

    Parameters
    ----------
    path: Path
        Base path to add index to

    Returns
    -------
    Path
    """
    index = 0
    new_stem = f"{path.stem}_{index}"

    while (path.parent / (new_stem + path.suffix)).exists():
        index += 1
        new_stem = f"{path.stem}_{index}"

    return path.parent / (new_stem + path.suffix)


def load_background(background_path):
    with open_ome_zarr(
        os.path.join(background_path, "background.zarr", "0", "0", "0")
    ) as dataset:
        cyx_data = dataset["0"][0, :, 0]
        return torch.tensor(cyx_data, dtype=torch.float32)


class MockEmitter:
    def emit(self, value):
        pass


def ram_message():
    """
    Determine if the system's RAM capacity is sufficient for running reconstruction.
    The message should be treated as a warning if the RAM detected is less than 32 GB.

    Returns
    -------
    ram_report    (is_warning, message)
    """
    BYTES_PER_GB = 2**30
    gb_available = psutil.virtual_memory().total / BYTES_PER_GB
    is_warning = gb_available < 32

    if is_warning:
        message = " \n".join(
            textwrap.wrap(
                f"recOrder reconstructions often require more than the {gb_available:.1f} "
                f"GB of RAM that this computer is equipped with. We recommend starting with reconstructions of small "
                f"volumes ~1000 x 1000 x 10 and working up to larger volumes while monitoring your RAM usage with "
                f"Task Manager or htop.",
            )
        )
    else:
        message = f"{gb_available:.1f} GB of RAM is available."

    return (is_warning, message)


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, 'model.yaml')

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.dict()

    # Remove None-valued fields
    clean_model_dict = {
        key: value for key, value in model_dict.items() if value is not None
    }

    with open(yaml_path, "w+") as f:
        yaml.dump(
            clean_model_dict, f, default_flow_style=False, sort_keys=False
        )


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError(
            "The provided model must be a class with a callable constructor."
        )

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)

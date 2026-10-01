import os
import textwrap
from pathlib import Path

import psutil
import torch
import yaml
from iohub import open_ome_zarr

from waveorder.api._settings import MyBaseModel


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
    with open_ome_zarr(os.path.join(background_path, "background.zarr", "0", "0", "0")) as dataset:
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
                f"waveorder reconstructions often require more than the {gb_available:.1f} "
                f"GB of RAM that this computer is equipped with. We recommend starting with reconstructions of small "
                f"volumes ~1000 x 1000 x 10 and working up to larger volumes while monitoring your RAM usage with "
                f"Task Manager or htop.",
            )
        )
    else:
        message = f"{gb_available:.1f} GB of RAM is available."

    return (is_warning, message)


def model_to_yaml(model: MyBaseModel, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Parameters
    ----------
    model : MyBaseModel
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
    >>> model_to_yaml(model, "model.yaml")

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.model_dump()

    # Remove None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def _collect_field_descriptions(model, prefix=""):
    """Build {dotted.path: description} from a pydantic model instance."""
    from pydantic import BaseModel

    descriptions = {}
    for name, field_info in model.__class__.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        if field_info.description:
            descriptions[path] = field_info.description
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            descriptions.update(_collect_field_descriptions(value, path))
    return descriptions


def _optional_model_type(annotation):
    """The BaseModel class inside an ``Optional[Model]`` annotation, else None."""
    import types
    import typing

    from pydantic import BaseModel

    origin = typing.get_origin(annotation)
    candidates = typing.get_args(annotation) if origin in (typing.Union, types.UnionType) else (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _collect_null_blocks(model, prefix=""):
    """{dotted.path: default instance} for every ``Optional[Model]`` field that is None.

    A config shows such a block as ``field: null`` followed by the block's fields
    commented out, so a reader can see how to turn it on without it being on.
    """
    from pydantic import BaseModel, ValidationError

    blocks = {}
    for name, field_info in model.__class__.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            blocks.update(_collect_null_blocks(value, path))
        elif value is None:
            model_type = _optional_model_type(field_info.annotation)
            if model_type is not None:
                try:
                    blocks[path] = model_type()
                except ValidationError:
                    continue  # no usable defaults, so nothing to show
    return blocks


def _add_yaml_comments(yaml_str, descriptions, comment_column=44, null_blocks=None):
    """Add inline comments to YAML lines from a {dotted.path: description} map.

    ``null_blocks`` maps a dotted path whose value is ``null`` to a default model
    instance; its fields are written beneath that line, commented out and indented
    as the block's children would be, so uncommenting them turns the block on.
    """
    null_blocks = null_blocks or {}
    lines = yaml_str.splitlines()
    result = []
    path_stack = []  # [(indent_level, key)]

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            result.append(line)
            continue

        indent = len(line) - len(stripped)

        block = None
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()

            # Pop entries at same or deeper indent
            while path_stack and path_stack[-1][0] >= indent:
                path_stack.pop()

            path_stack.append((indent, key))
            field_path = ".".join(item[1] for item in path_stack)

            desc = descriptions.get(field_path)
            if desc:
                padding = max(1, comment_column - len(line))
                line = line + " " * padding + "# " + desc
            if value.strip() == "null":
                block = null_blocks.get(field_path)

        result.append(line)

        if block is not None:
            prefix = " " * indent + "# "
            result.append(prefix + "or, instead of null:")
            block_yaml = yaml.dump(block.model_dump(), default_flow_style=False, sort_keys=False)
            block_text = _add_yaml_comments(
                block_yaml,
                _collect_field_descriptions(block),
                comment_column - len(prefix) - 2,
                _collect_null_blocks(block),
            )
            result.extend(prefix + "  " + block_line for block_line in block_text.splitlines())

    return "\n".join(result) + "\n"


def model_to_commented_yaml(model: MyBaseModel, yaml_path: Path, comment_column: int = 44) -> None:
    """Save a model to YAML with inline comments from Field descriptions."""
    yaml_path = Path(yaml_path)

    model_dict = model.model_dump()

    # Remove top-level None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    yaml_str = yaml.dump(clean_model_dict, default_flow_style=False, sort_keys=False)

    descriptions = _collect_field_descriptions(model)
    null_blocks = _collect_null_blocks(model)
    commented_yaml = _add_yaml_comments(yaml_str, descriptions, comment_column, null_blocks)

    with open(yaml_path, "w") as f:
        f.write(commented_yaml)


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
    # >>> from my_model import MyModel
    # >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)

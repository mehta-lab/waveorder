import os
import textwrap
from pathlib import Path
from typing import Literal, Union

import numpy as np
import psutil
import torch
import yaml
from colorspacious import cspace_convert
from iohub import open_ome_zarr
from matplotlib.colors import hsv_to_rgb
from waveorder.waveorder_reconstructor import waveorder_microscopy


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


def generic_hsv_overlay(
    H, S, V, H_scale=None, S_scale=None, V_scale=None, mode="2D"
):
    """
    Generates a generic HSV overlay in either 2D or 3D

    Parameters
    ----------
    H:          (nd-array) data to use in the Hue channel
    S:          (nd-array) data to use in the Saturation channel
    V:          (nd-array) data to use in the Value channel
    H_scale:    (tuple) values at which to clip the hue data for display
    S_scale:    (tuple) values at which to clip the saturation data for display
    V_scale:    (tuple) values at which to clip the value data for display
    mode:       (str) '3D' or '2D'

    Returns
    -------
    overlay:    (nd-array) RGB overlay array of shape (Z, Y, X, 3) or (Y, X, 3)

    """

    if H.shape != S.shape or H.shape != S.shape or S.shape != V.shape:
        raise ValueError(
            f"Channel shapes do not match: {H.shape} vs. {S.shape} vs. {V.shape}"
        )

    if mode == "3D":
        overlay_final = np.zeros((H.shape[0], H.shape[1], H.shape[2], 3))
        slices = H.shape[0]
    else:
        overlay_final = np.zeros((1, H.shape[-2], H.shape[-1], 3))
        H = np.expand_dims(H, axis=0)
        S = np.expand_dims(S, axis=0)
        V = np.expand_dims(V, axis=0)
        slices = 1

    for i in range(slices):
        H_ = np.interp(H[i], H_scale, (0, 1))
        S_ = np.interp(S[i], S_scale, (0, 1))
        V_ = np.interp(V[i], V_scale, (0, 1))

        hsv = np.transpose(np.stack([H_, S_, V_]), (1, 2, 0))
        overlay_final[i] = hsv_to_rgb(hsv)

    return overlay_final[0] if mode == "2D" else overlay_final


def ret_ori_overlay(
    retardance,
    orientation,
    ret_max: Union[float, Literal["auto"]] = 10,
    cmap: Literal["JCh", "HSV"] = "JCh",
):
    """
    This function will create an overlay of retardance and orientation with two different colormap options.
    HSV is the standard Hue, Saturation, Value colormap while JCh is a similar colormap but is perceptually uniform.

    Parameters
    ----------
    retardance:             (nd-array) retardance array in nanometers (shape must match orientation)
    orientation:            (nd-array) orientation array in radian [0, pi] (shape must match retardance)
    ret_max:                (float) maximum displayed retardance. Typically use adjusted contrast limits.
    cmap:                   (str) 'JCh' or 'HSV'

    Returns
    -------
    overlay                 (nd-array) RGB image with shape retardance.shape + (3,)

    """
    if retardance.shape != orientation.shape:
        raise ValueError(
            "Retardance and Orientation shapes do not match: "
            f"{retardance.shape} vs. {orientation.shape}"
        )

    if ret_max == "auto":
        ret_max = np.percentile(np.ravel(retardance), 99.99)

    # Prepare input and output arrays
    ret_ = np.clip(retardance, 0, ret_max)  # clip and copy
    # Convert 180 degree range into 360 to match periodicity of hue.
    ori_ = orientation * 360 / np.pi
    overlay_final = np.zeros_like(retardance)

    # FIX ME: this binning code leads to artifacts.
    # levels = 32
    # ori_binned = (
    #     np.round(orientation[i] / 180 * levels + 0.5) / levels - 1 / levels
    # ) # bin orientation into 32 levels.
    # ori_ = np.interp(ori_binned, (0, 1), (0, 360))

    if cmap == "JCh":
        noise_level = 1

        J = ret_
        C = np.ones_like(J) * 60
        C[ret_ < noise_level] = 0
        h = ori_

        JCh = np.stack((J, C, h), axis=-1)
        JCh_rgb = cspace_convert(JCh, "JCh", "sRGB1")

        JCh_rgb[JCh_rgb < 0] = 0
        JCh_rgb[JCh_rgb > 1] = 1

        overlay_final = JCh_rgb
    elif cmap == "HSV":
        I_hsv = np.moveaxis(
            np.stack(
                [
                    ori_ / 360,
                    np.ones_like(ori_),
                    ret_ / np.max(ret_),
                ]
            ),
            source=0,
            destination=-1,
        )
        overlay_final = hsv_to_rgb(I_hsv)
    else:
        raise ValueError(f"Colormap {cmap} not understood")

    return overlay_final


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

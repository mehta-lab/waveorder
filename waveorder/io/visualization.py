from typing import Literal, Union

import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import hsv_to_rgb
from skimage.color import hsv2rgb
from skimage.exposure import rescale_intensity


def ret_ori_overlay(
    czyx,
    ret_min: float = 1,
    ret_max: Union[float, Literal["auto"]] = 10,
    cmap: Literal["JCh", "HSV"] = "HSV",
):
    """
    Creates an overlay of retardance and orientation with two different colormap options.
    "HSV" maps orientation to hue and retardance to value with maximum saturation.
    "JCh" is a similar colormap but is perceptually uniform.

    Parameters
    ----------
    czyx:                   (nd-array) czyx[0] is retardance in nanometers, czyx[1] is orientation in radians [0, pi],
                            czyx.shape = (2, ...)

    ret_min:                (float) minimum displayed retardance. Typically a noise floor.
    ret_max:                (float) maximum displayed retardance. Typically used to adjust contrast limits.

    cmap:                   (str) 'JCh' or 'HSV'

    Returns
    -------
    overlay                 (nd-array) RGB image with shape (3, ...)

    """
    if czyx.shape[0] != 2:
        raise ValueError(
            f"Input must have shape (2, ...) instead of ({czyx.shape[0]}, ...)"
        )

    retardance = czyx[0]
    orientation = czyx[1]

    if ret_max == "auto":
        ret_max = np.percentile(np.ravel(retardance), 99.99)

    # Prepare input and output arrays
    ret_ = np.clip(retardance, 0, ret_max)  # clip and copy
    # Convert 180 degree range into 360 to match periodicity of hue.
    ori_ = orientation * 360 / np.pi
    overlay_final = np.zeros_like(retardance)

    if cmap == "JCh":
        J = ret_
        C = np.ones_like(J) * 60
        C[ret_ < ret_min] = 0
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

    return np.moveaxis(
        overlay_final, source=-1, destination=0
    )  # .shape = (3, ...)


def ret_ori_phase_overlay(
    czyx, max_val_V: float = 1.0, max_val_S: float = 1.0
):
    """
    Creates an overlay of retardance, orientation, and phase.
    Maps orientation to hue, retardance to saturation, and phase to value.

    HSV encoding of retardance + orientation + phase image with hsv colormap
    (orientation in h, retardance in s, phase in v)
    Parameters
    ----------
        czyx        : numpy.ndarray
                    czyx[0] corresponds to the retardance image
                    czyx[1]is the orientation image (range from 0 to pi)
                    czyx[2] is the the phase image

        max_val_V   : float
                      raise the brightness of the phase channel by 1/max_val_V

        max_val_S   : float
                      raise the brightness of the retardance channel by 1/max_val_S

    Returns
    -------
    overlay                 (nd-array) RGB image with shape (3, ...)

    Returns:
        RGB with HSV
    """

    if czyx.shape[0] != 3:
        raise ValueError(
            f"Input must have shape (3, ...) instead of ({czyx.shape[0]}, ...)"
        )

    czyx_out = np.zeros_like(czyx, dtype=np.float32)

    retardance = czyx[0]
    orientation = czyx[1]
    phase = czyx[2]

    # Normalize the stack
    ordered_stack = np.stack(
        (
            # Normalize the first channel by dividing by pi
            orientation / np.pi,
            # Normalize the second channel and rescale intensity
            rescale_intensity(
                retardance,
                in_range=(
                    np.min(retardance),
                    np.max(retardance),
                ),
                out_range=(0, 1),
            )
            / max_val_S,
            # Normalize the third channel and rescale intensity
            rescale_intensity(
                phase,
                in_range=(
                    np.min(phase),
                    np.max(phase),
                ),
                out_range=(0, 1),
            )
            / max_val_V,
        ),
        axis=0,
    )
    czyx_out = hsv2rgb(ordered_stack, channel_axis=0)
    return czyx_out

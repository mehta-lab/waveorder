from typing import Literal
from waveorder import util
import numpy as np


def focus_from_transverse_band(
    zyx_array,
    NA_det,
    lambda_ill,
    pixel_size,
    midband_fractions=(0.125, 0.25),
    mode: Literal["min" "max"] = "max",
):
    """Estimates the in-focus slice from a 3D stack by optimizing a transverse spatial frequency band.

    Parameters
    ----------
    zyx_array : np.array
        Data stack in (Z, Y, X) order.
        Requires len(3d_array.shape) == 3.
    NA_det : float
        Detection NA.
    lambda_ill : float
        Illumination wavelength
        Units are arbitrary, but must match [pixel_size]
    pixel_size : float
        Object-space pixel size = camera pixel size / magnification.
        Units are arbitrary, but must match [lambda_ill]
    midband_fractions: Tuple[float, float], optional
        The minimum and maximum fraction of the cutoff frequency that define the midband.
        Requires: 0 <= midband_fractions[0] < midband_fractions[1] <= 1.
    mode: {'max', 'min'}, optional
        Option to choose the in-focus slice my minimizing or maximizing the midband frequency.

    Returns
    ------
    slice : int
        The index of the in-focus slice.

    Example
    ------
    >>> zyx_array.shape
    (11, 2048, 2048)
    >>> from waveorder.focus import focus_from_transverse_band
    >>> slice = focus_from_transverse_band(zyx_array, NA_det=0.55, lambda_ill=0.532, pixel_size=6.5/20)
    >>> in_focus_data = data[slice,:,:]
    """

    # Check inputs
    N = len(zyx_array.shape)
    if N != 3:
        raise ValueError(
            f"{N}D array supplied. `estimate_brightfield_focus` only accepts 3D arrays."
        )
    if zyx_array.shape[0] == 1:
        print(
            "WARNING: The dataset only contained a single slice. Returning trivial slice index = 0."
        )
        return 0

    if NA_det < 0:
        raise ValueError("NA must be > 0")
    if lambda_ill < 0:
        raise ValueError("lambda_ill must be > 0")
    if pixel_size < 0:
        raise ValueError("pixel_size must be > 0")
    if not 0.4 < lambda_ill / pixel_size < 10:
        print(
            f"WARNING: lambda_ill/pixel_size = {lambda_ill/pixel_size}."
            f"Did you use the same units?"
            f"Did you enter the pixel size in (demagnified) object-space units?"
        )
    if not midband_fractions[0] < midband_fractions[1]:
        raise ValueError(
            "midband_fractions[0] must be less than midband_fractions[1]"
        )
    if not (0 <= midband_fractions[0] <= 1):
        raise ValueError("midband_fractions[0] must be between 0 and 1")
    if not (0 <= midband_fractions[1] <= 1):
        raise ValueError("midband_fractions[1] must be between 0 and 1")
    if mode == "min":
        minmaxfunc = np.argmin
    elif mode == "max":
        minmaxfunc = np.argmax
    else:
        raise ValueError("mode must be either `min` or `max`")

    # Calculate coordinates
    _, Y, X = zyx_array.shape
    _, _, fxx, fyy = util.gen_coordinate((Y, X), pixel_size)
    frr = np.sqrt(fxx**2 + fyy**2)

    # Calculate fft
    xy_abs_fft = np.abs(np.fft.fftn(zyx_array, axes=(1, 2)))

    # Calculate midband mask
    cutoff = 2 * NA_det / lambda_ill
    midband_mask = np.logical_and(
        frr > cutoff * midband_fractions[0],
        frr < cutoff * midband_fractions[1],
    )

    # Return slice index with min/max power in midband
    midband_sum = np.sum(xy_abs_fft[:, midband_mask], axis=1)
    return minmaxfunc(midband_sum)

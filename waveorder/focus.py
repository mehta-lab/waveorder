from scipy.signal import peak_widths
from typing import Literal, Optional
from waveorder import util
import matplotlib.pyplot as plt
import numpy as np


def focus_from_transverse_band(
    zyx_array,
    NA_det,
    lambda_ill,
    pixel_size,
    midband_fractions=(0.125, 0.25),
    mode: Literal["min", "max"] = "max",
    plot_path: Optional[str] = None,
    threshold_FWHM: float = 0,
):
    """Estimates the in-focus slice from a 3D stack by optimizing a transverse spatial frequency band.

    Parameters
    ----------
    zyx_array: np.array
        Data stack in (Z, Y, X) order.
        Requires len(zyx_array.shape) == 3.
    NA_det: float
        Detection NA.
    lambda_ill: float
        Illumination wavelength
        Units are arbitrary, but must match [pixel_size]
    pixel_size: float
        Object-space pixel size = camera pixel size / magnification.
        Units are arbitrary, but must match [lambda_ill]
    midband_fractions: Tuple[float, float], optional
        The minimum and maximum fraction of the cutoff frequency that define the midband.
        Requires: 0 <= midband_fractions[0] < midband_fractions[1] <= 1.
    mode: {'max', 'min'}, optional
        Option to choose the in-focus slice by minimizing or maximizing the midband frequency.
    plot_path: str or None, optional
        File name for a diagnostic plot (supports matplotlib filetypes .png, .pdf, .svg, etc.).
        Use None to skip.
    threshold_FWHM: float, optional
        Threshold full-width half max for a peak to be considered in focus.
        The default value, 0, applies no threshold, and the maximum midband power is always considered in focus.
        For values > 0, the peak's FWHM must be greater than the threshold for the slice to be considered in focus.
        If the peak does not meet this threshold, the function returns None.

    Returns
    ------
    slice : int or None
        If peak's FWHM > peak_width_threshold:
            return the index of the in-focus slice
        else:
            return None

    Example
    ------
    >>> zyx_array.shape
    (11, 2048, 2048)
    >>> from waveorder.focus import focus_from_transverse_band
    >>> slice = focus_from_transverse_band(zyx_array, NA_det=0.55, lambda_ill=0.532, pixel_size=6.5/20)
    >>> in_focus_data = data[slice,:,:]
    """
    minmaxfunc = _check_focus_inputs(
        zyx_array, NA_det, lambda_ill, pixel_size, midband_fractions, mode
    )

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

    # Find slice index with min/max power in midband
    midband_sum = np.sum(xy_abs_fft[:, midband_mask], axis=1)
    peak_index = minmaxfunc(midband_sum)

    peak_results = peak_widths(midband_sum, [peak_index])
    peak_FWHM = peak_results[0][0]

    if peak_FWHM >= threshold_FWHM:
        in_focus_index = peak_index
    else:
        in_focus_index = None

    # Plot
    if plot_path is not None:
        _plot_focus_metric(
            plot_path, midband_sum, peak_index, in_focus_index, peak_results, threshold_FWHM
        )

    return in_focus_index


def _check_focus_inputs(
    zyx_array, NA_det, lambda_ill, pixel_size, midband_fractions, mode
):
    N = len(zyx_array.shape)
    if N != 3:
        raise ValueError(
            f"{N}D array supplied. `focus_from_transverse_band` only accepts 3D arrays."
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
    return minmaxfunc


def _plot_focus_metric(
    plot_path, midband_sum, peak_index, in_focus_index, peak_results, threshold_FWHM
):
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(midband_sum, "-k")
    ax.plot(
        peak_index,
        midband_sum[peak_index],
        "go" if in_focus_index is not None else "ro",
    )
    ax.hlines(*peak_results[1:], color="k", linestyles="dashed")

    ax.set_xlabel("Slice index")
    ax.set_ylabel("Midband power")

    ax.annotate(
        f"In-focus slice = {in_focus_index}\n Peak width = {peak_results[0][0]:.2f}\n Peak width threshold = {threshold_FWHM}",
        xy=(1, 1),
        xytext=(1.0, 1.1),
        textcoords="axes fraction",
        xycoords="axes fraction",
        ha="right",
        va="center",
        annotation_clip=False,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))

    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()

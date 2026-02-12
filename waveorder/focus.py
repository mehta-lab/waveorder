import warnings
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import peak_widths

from waveorder import util


def compute_midband_power(
    input_tensor: torch.Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> torch.Tensor:
    """Compute midband spatial frequency power by summing over a 2D midband donut.

    Accepts a 2D ``(Y, X)`` or 3D ``(Z, Y, X)`` tensor.  For 3D input the
    frequency mask is computed once and a batched 2D FFT is applied across all
    Z slices, which is significantly faster than looping.

    Parameters
    ----------
    input_tensor : torch.Tensor
        2D ``(Y, X)`` or 3D ``(Z, Y, X)`` tensor.
    NA_det : float
        Detection NA.
    lambda_ill : float
        Illumination wavelength.
        Units are arbitrary, but must match ``pixel_size``.
    pixel_size : float
        Object-space pixel size = camera pixel size / magnification.
        Units are arbitrary, but must match ``lambda_ill``.
    midband_fractions : tuple[float, float], optional
        The minimum and maximum fraction of the cutoff frequency that define the midband.
        Default is (0.125, 0.25).

    Returns
    -------
    torch.Tensor
        2D input: scalar tensor.
        3D input: tensor of shape ``(Z,)`` with per-slice midband power.
    """
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)  # (1, Y, X)
        squeeze = True
    elif input_tensor.ndim == 3:
        squeeze = False
    else:
        raise ValueError(
            f"{input_tensor.ndim}D tensor supplied. "
            "`compute_midband_power` only accepts 2D or 3D tensors."
        )

    Z, Y, X = input_tensor.shape
    device = input_tensor.device

    _, _, fxx, fyy = util.gen_coordinate((Y, X), pixel_size)
    frr = torch.tensor(np.sqrt(fxx**2 + fyy**2), device=device)
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(
        frr > cutoff * midband_fractions[0],
        frr < cutoff * midband_fractions[1],
    )

    abs_fft = torch.abs(torch.fft.fftn(input_tensor.float(), dim=(-2, -1)))
    result = abs_fft[:, mask].sum(dim=-1)  # (Z,)

    if squeeze:
        return result.squeeze(0)
    return result


def focus_from_transverse_band(
    zyx_array,
    NA_det,
    lambda_ill,
    pixel_size,
    midband_fractions=(0.125, 0.25),
    mode: Literal["min", "max"] = "max",
    polynomial_fit_order: Optional[int] = None,
    plot_path: Optional[str] = None,
    threshold_FWHM: float = 0,
    return_statistics: bool = False,
    enable_subpixel_precision: bool = False,
):
    """Estimates the in-focus slice from a 3D stack by optimizing a transverse spatial frequency band.

    Parameters
    ----------
    zyx_array: np.ndarray or torch.Tensor
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
    mode: {'min', 'max'}, optional
        Option to choose the in-focus slice by minimizing or maximizing the midband power. By default 'max'.
    polynomial_fit_order: int, optional
        Default None is no fit. If integer, a polynomial of that degree is fit to the midband power before choosing the extreme point as the in-focus slice.
    plot_path: str or None, optional
        File name for a diagnostic plot (supports matplotlib filetypes .png, .pdf, .svg, etc.).
        Use None to skip.
    threshold_FWHM: float, optional
        Threshold full-width half max for a peak to be considered in focus.
        The default value, 0, applies no threshold, and the maximum midband power is always considered in focus.
        For values > 0, the peak's FWHM must be greater than the threshold for the slice to be considered in focus.
        If the peak does not meet this threshold, the function returns None.
    return_statistics: bool, optional
        If True, returns a tuple (in_focus_index, peak_stats) instead of just in_focus_index.
        Default is False for backward compatibility.
    enable_subpixel_precision: bool, optional
        If True and polynomial_fit_order is provided, enables sub-pixel precision focus detection
        by finding the continuous extremum of the polynomial fit. Default is False for backward compatibility.

    Returns
    -------
    slice : int, float, None, or tuple
        If return_statistics is False (default):
            Returns in_focus_index (int if enable_subpixel_precision=False,
            float if enable_subpixel_precision=True and polynomial_fit_order is not None, or None).
        If return_statistics is True:
            Returns tuple (in_focus_index, peak_stats) where peak_stats is a dict
            containing 'peak_index' and 'peak_FWHM'.

    Example
    ------
    >>> zyx_array.shape
    (11, 2048, 2048)
    >>> from waveorder.focus import focus_from_transverse_band
    >>> slice = focus_from_transverse_band(zyx_array, NA_det=0.55, lambda_ill=0.532, pixel_size=6.5/20)
    >>> in_focus_data = data[slice,:,:]
    """
    minmaxfunc = _mode_to_minmaxfunc(mode)
    peak_stats = {"peak_index": None, "peak_FWHM": None}

    _check_focus_inputs(
        zyx_array, NA_det, lambda_ill, pixel_size, midband_fractions
    )

    # Check for single slice
    if zyx_array.shape[0] == 1:
        warnings.warn(
            "The dataset only contained a single slice. Returning trivial slice index = 0."
        )
        if return_statistics:
            return 0, peak_stats
        return 0

    # Convert to tensor if needed for vectorized midband power
    if isinstance(zyx_array, np.ndarray):
        zyx_tensor = torch.from_numpy(zyx_array).float()
    else:
        zyx_tensor = zyx_array.float()

    # Calculate midband power for each slice (vectorized over Z)
    midband_sum = (
        compute_midband_power(
            zyx_tensor, NA_det, lambda_ill, pixel_size, midband_fractions
        )
        .cpu()
        .numpy()
    )  # (Z,)

    if polynomial_fit_order is None:
        peak_index = minmaxfunc(midband_sum)
    else:
        x = np.arange(len(midband_sum))
        coeffs = np.polyfit(x, midband_sum, polynomial_fit_order)
        poly_func = np.poly1d(coeffs)

        if enable_subpixel_precision:
            # Find the continuous extremum using derivative
            poly_deriv = np.polyder(coeffs)
            # Find roots of the derivative (critical points)
            critical_points = np.roots(poly_deriv)

            # Filter for real roots within the data range
            real_critical_points = []
            for cp in critical_points:
                if np.isreal(cp) and 0 <= cp.real < len(midband_sum):
                    real_critical_points.append(cp.real)

            if real_critical_points:
                # Evaluate the polynomial at critical points to find extremum
                critical_values = [
                    poly_func(cp) for cp in real_critical_points
                ]
                if mode == "max":
                    best_idx = np.argmax(critical_values)
                else:  # mode == "min"
                    best_idx = np.argmin(critical_values)
                peak_index = real_critical_points[best_idx]
            else:
                # Fall back to discrete maximum if no valid critical points
                peak_index = float(minmaxfunc(poly_func(x)))
        else:
            peak_index = minmaxfunc(poly_func(x))

    # For peak width calculation, use integer peak index
    if enable_subpixel_precision and polynomial_fit_order is not None:
        # Use the closest integer index for peak width calculation
        integer_peak_index = int(np.round(peak_index))
    else:
        integer_peak_index = int(peak_index)

    peak_results = peak_widths(midband_sum, [integer_peak_index])
    peak_FWHM = peak_results[0][0]
    peak_stats.update(
        {"peak_index": int(peak_index), "peak_FWHM": float(peak_FWHM)}
    )

    if peak_FWHM >= threshold_FWHM:
        in_focus_index = peak_index
    else:
        in_focus_index = None

    # Plot
    if plot_path is not None:
        _plot_focus_metric(
            plot_path,
            midband_sum,
            peak_index,
            in_focus_index,
            peak_results,
            threshold_FWHM,
        )

    if return_statistics:
        return in_focus_index, peak_stats

    return in_focus_index


def _mode_to_minmaxfunc(mode):
    if mode == "min":
        minmaxfunc = np.argmin
    elif mode == "max":
        minmaxfunc = np.argmax
    else:
        raise ValueError("mode must be either `min` or `max`")
    return minmaxfunc


def _check_focus_inputs(
    zyx_array, NA_det, lambda_ill, pixel_size, midband_fractions
):
    N = len(zyx_array.shape)
    if N != 3:
        raise ValueError(
            f"{N}D array supplied. `focus_from_transverse_band` only accepts 3D arrays."
        )

    if NA_det < 0:
        raise ValueError("NA must be > 0")
    if lambda_ill < 0:
        raise ValueError("lambda_ill must be > 0")
    if pixel_size < 0:
        raise ValueError("pixel_size must be > 0")
    if not 0.4 < lambda_ill / pixel_size < 10:
        warnings.warn(
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


def _plot_focus_metric(
    plot_path,
    midband_sum,
    peak_index,
    in_focus_index,
    peak_results,
    threshold_FWHM,
):
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(midband_sum, "-k")

    # Handle floating-point peak_index for plotting
    if isinstance(peak_index, float) and not peak_index.is_integer():
        # Use interpolation to get the y-value at the floating-point x-position
        peak_y_value = np.interp(
            peak_index, np.arange(len(midband_sum)), midband_sum
        )
    else:
        peak_y_value = midband_sum[int(peak_index)]

    ax.plot(
        peak_index,
        peak_y_value,
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

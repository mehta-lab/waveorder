from waveorder import util
import numpy as np


def estimate_brightfield_focus(array3D, pixel_size, lambda_ill, NA_det):
    """Estimates the in-focus slice from a 3D brightfield stack by minimizing mid-band frequencies.

    Parameters
    ----------
    array3D : np.array
        Brightfield stack in (Z, Y, X) order.
        Requires len(3d_array.shape) == 3
    pixel_size : float
        um, object-space pixel size = camera pixel size / magnification
    lambda_ill : float
        um, illumination wavelength
    NA_det : float
        NA of the detection objective

    Returns
    ------
    slice : int
        The index of the in-focus slice.

    Example
    ------
    >>> data.shape
    (11, 2048, 2048)
    >>> from waveorder.focus_estimator import estimate_brightfield_focus
    >>> slice = estimate_brightfield_focus(data)
    >>> in_focus_data = data[slice,:,:]
    """

    # Check inputs
    N = len(array3D.shape)
    if N != 3:
        raise ValueError(
            f"{N}D array supplied. `estimate_brightfield_focus` only accepts 3D arrays."
        )

    # Calculate coordinates
    Z, Y, X = array3D.shape
    _, _, fxx, fyy = util.gen_coordinate((Y, X), pixel_size)
    frr = np.sqrt(fxx**2 + fyy**2)

    # Calculate fft
    xy_abs_fft = np.abs(np.fft.fftn(array3D, axes=(1, 2)))

    # Calculate midband mask
    cutoff = 2 * NA_det / lambda_ill
    midband_mask = np.logical_and(frr > cutoff / 8, frr < cutoff / 4)

    # Return slice with most power in midband
    midband_sum = np.sum(xy_abs_fft[:, midband_mask], axis=1)
    return np.argmax(midband_sum)

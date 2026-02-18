"""Loss functions for parameter optimization."""

from torch import Tensor

from waveorder.focus import compute_midband_power


def midband_power_loss(
    yx_reconstruction: Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> Tensor:
    """Negative midband spatial frequency power.

    Thin wrapper around :func:`waveorder.focus.compute_midband_power` that
    negates the result so that *minimizing* this loss *maximizes* midband
    power (encouraging sharp, well-focused reconstructions).

    Parameters
    ----------
    yx_reconstruction : Tensor
        2D reconstruction (Y, X)
    NA_det : float
        Detection numerical aperture
    lambda_ill : float
        Illumination/emission wavelength (same units as pixel_size)
    pixel_size : float
        Pixel size (same units as lambda_ill)
    midband_fractions : tuple[float, float]
        Fraction of cutoff frequency for (inner, outer) annulus.

    Returns
    -------
    Tensor
        Scalar loss (negative midband power)
    """
    return -compute_midband_power(
        yx_reconstruction,
        NA_det=NA_det,
        lambda_ill=lambda_ill,
        pixel_size=pixel_size,
        midband_fractions=midband_fractions,
    )

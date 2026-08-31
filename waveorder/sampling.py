import numpy as np
import torch


def transverse_nyquist(
    wavelength_emission,
    numerical_aperture_illumination,
    numerical_aperture_detection,
):
    """Transverse Nyquist sample spacing in `wavelength_emission` units.

    For widefield label-free imaging, the transverse Nyquist sample spacing is
    lambda / (2 * (NA_ill + NA_det)).

    Perhaps surprisingly, the transverse Nyquist sample spacing for widefield
    fluorescence is lambda / (4 * NA), which is equivalent to the above formula
    when NA_ill = NA_det.

    Parameters
    ----------
    wavelength_emission : float
        Output units match these units
    numerical_aperture_illumination : float
        For widefield fluorescence, set to numerical_aperture_detection
    numerical_aperture_detection : float

    Returns
    -------
    float
        Transverse Nyquist sample spacing

    """
    return wavelength_emission / (2 * (numerical_aperture_detection + numerical_aperture_illumination))


def axial_nyquist(
    wavelength_emission,
    numerical_aperture_detection,
    index_of_refraction_media,
):
    """Axial Nyquist sample spacing in `wavelength_emission` units.

    For widefield microscopes, the axial Nyquist cutoff frequency is:

    (n/lambda) - sqrt( (n/lambda)^2 - (NA_det/lambda)^2 ),

    and the axial Nyquist sample spacing is 1 / (2 * cutoff_frequency).

    Perhaps surprisingly, the axial Nyquist sample spacing is independent of
    the illumination numerical aperture.

    Parameters
    ----------
    wavelength_emission : float
        Output units match these units
    numerical_aperture_detection : float
    index_of_refraction_media: float

    Returns
    -------
    float
        Axial Nyquist sample spacing

    """
    n_on_lambda = index_of_refraction_media / wavelength_emission
    cutoff_frequency = n_on_lambda - np.sqrt(n_on_lambda**2 - (numerical_aperture_detection / wavelength_emission) ** 2)
    return 1 / (2 * cutoff_frequency)


def raised_cosine_window(size, rolloff, device=None):
    """1D raised-cosine apodization window in unshifted (fftfreq) order.

    The window is 1 over the inner ``1 - rolloff`` fraction of the band
    and rolls off smoothly (half-cosine) to 0 at the Nyquist frequency.

    Parameters
    ----------
    size : int
        Number of frequency samples.
    rolloff : float
        Fraction of the band (0 to 1] over which the window rolls off
        from 1 to 0 at the band edge.
    device : str, torch.device, or None
        Output tensor device.

    Returns
    -------
    torch.Tensor
        Window of shape ``(size,)`` in fftfreq (unshifted) order.
    """
    normalized_frequency = torch.abs(torch.fft.fftfreq(size, device=device)) * 2  # 1.0 at Nyquist
    rolloff_start = 1 - rolloff
    window = torch.ones(size, device=device)
    mask = normalized_frequency > rolloff_start
    window[mask] = 0.5 * (1 + torch.cos(torch.pi * (normalized_frequency[mask] - rolloff_start) / rolloff))
    return window


def nd_fourier_central_cuboid(source, target_shape):
    """Central cuboid of an N-D Fourier transform.

    If ``target_shape`` has fewer dimensions than ``source``, the
    central cuboid is taken from the last ``len(target_shape)``
    dimensions, preserving leading (batch) dimensions.

    Parameters
    ----------
    source : torch.Tensor
        Source tensor.
    target_shape : tuple of int
        Target spatial shape.

    Returns
    -------
    torch.Tensor
        Center cuboid in Fourier space.
    """
    n_dim = len(target_shape)
    dims = tuple(range(source.ndim))[-n_dim:]
    center_slices = tuple(
        slice((s - o) // 2, (s - o) // 2 + o) for s, o in zip(source.shape[-n_dim:], target_shape, strict=True)
    )
    center_slices = (slice(None),) * (source.ndim - n_dim) + center_slices
    return torch.fft.ifftshift(torch.fft.fftshift(source, dim=dims)[center_slices], dim=dims)

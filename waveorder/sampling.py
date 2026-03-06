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
    return wavelength_emission / (
        2 * (numerical_aperture_detection + numerical_aperture_illumination)
    )


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
    cutoff_frequency = n_on_lambda - torch.sqrt(
        torch.as_tensor(
            n_on_lambda**2
            - (numerical_aperture_detection / wavelength_emission) ** 2
        )
    )
    return 1 / (2 * cutoff_frequency)


def nd_fourier_central_cuboid(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Central cuboid of an N-D Fourier transform.
    If `target_shape` length is less than `source.ndim`, the central cuboid is
    taken from the last `len(target_shape)` dimensions of `source`.

    Parameters
    ----------
    source : torch.Tensor
        Source tensor
    target_shape : torch.Size
        Target shape

    Returns
    -------
    torch.Tensor
        Center cuboid in Fourier space

    """
    n_dim = len(target_shape)
    dims = tuple(range(source.ndim))[-n_dim:]
    center_slices = tuple(
        slice((s - o) // 2, (s - o) // 2 + o)
        for s, o in zip(source.shape[-n_dim:], target_shape, strict=True)
    )
    center_slices = (slice(None),) * (source.ndim - n_dim) + center_slices
    return torch.fft.ifftshift(torch.fft.fftshift(source, dim=dims)[center_slices], dim=dims)

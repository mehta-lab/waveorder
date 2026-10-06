import torch

from waveorder import sampling


def tikhonov_regularized_inverse_filter(
    forward_filter: torch.Tensor,
    regularization_strength: float,
    *,
    apodization_rolloff: float = 0.0,
) -> torch.Tensor:
    """Compute the Tikhonov regularized inverse filter from a forward filter.

    Parameters
    ----------
    forward_filter : torch.Tensor
        The forward filter tensor, or real singular values. With apodization
        enabled, the final two axes must be Y and X in unshifted FFT order.
    regularization_strength : float
        The strength of the regularization term.
    apodization_rolloff : float, optional
        Raised-cosine roll-off fraction between 0 and 1, by default 0.0
        (disabled). Applied after regularization along Y and X only;
        leading batch, channel, and Z axes are preserved.
        For an SVD inverse, regularized singular values may be apodized
        before contraction because the window is shared across modes.
    Returns
    -------
    torch.Tensor
        The Tikhonov regularized inverse filter.
    """

    forward_filter_conj = torch.conj(forward_filter)
    inverse_filter = forward_filter_conj / ((forward_filter_conj * forward_filter) + regularization_strength)
    return sampling.apply_transverse_apodization(inverse_filter, apodization_rolloff)

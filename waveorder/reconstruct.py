import torch


def tikhonov_regularized_inverse_filter(
    forward_filter: torch.Tensor, regularization_strength: float
):
    """Compute the Tikhonov regularized inverse filter from a forward filter.

    Parameters
    ----------
    forward_filter : torch.Tensor
        The forward filter tensor.
    regularization_strength : float
        The strength of the regularization term.
    Returns
    -------
    torch.Tensor
        The Tikhonov regularized inverse filter.
    """

    if forward_filter.ndim == 3:
        forward_filter_conj = torch.conj(forward_filter)
        return forward_filter_conj / (
            (forward_filter_conj * forward_filter) + regularization_strength
        )
    else:
        # TC TODO INTEGRATE THE 5D FILTER BANK CASE
        raise NotImplementedError("Only 3D tensors are supported.")

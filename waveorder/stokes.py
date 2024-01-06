"""
Overview
--------

This module collects Stokes- and Mueller-related calculations. 

The functions are roughly organized into groups:

1) Polarimeter instrument matrix functions
S2I = calculate_stokes_to_intensity_matrix(swing, scheme="5-State")
I2S = calculate_intensity_to_stokes_matrix(swing, scheme="5-State")

2) Forward functions (Stokes parameters following a optical element)
s0, s1, s2, s3 = stokes_after_adr(retardance, orientation, transmittance, depolarization, input="cpl")
s0, s1, s2 = stokes012_after_ar(retardance, orientation, transmittance, input="cpl")

3) Inverse functions (optical elements from Stokes parameters)
retardance, orientation, transmittance, depolarization = estimate_adr_from_stokes(s0, s1, s2, s3, input="cpl")
retardance, orientation, transmittance = estimate_ar_from_stokes012(s0, s1, s2, input="cpl")

4) A function for recovering Mueller matrices from Stokes vector
M = mueller_from_stokes(
    s0, s1, s2, s3, model="ar", direction="forward", input="cpl"
)

5) A convenience function for applying Mueller and instrument matrices
y = mmul(A, x)

Usage
-----

All functions are intended to be used with torch.Tensors with Stokes- or 
Mueller-indices as the first axes. 

For example, the following usage modes of stokes_after_adr are valid:

>>> stokes_after_adr(1, 1, 1, 1)

>>> retardance = torch.ones((2,3,4,5))
>>> orientation = torch.ones((2,3,4,5))
>>> transmittance = torch.ones((2,3,4,5))
>>> depolarization = torch.ones((2,3,4,5))
>>> stokes_after_adr(retardance, orientation, transmittance, depolarization)

>>> adr_params = torch.ones((4,2,3,4,5)) # first axis contains the Stokes indices
>>> stokes_after_adr(*adr_params) # * expands along the first axis

"""
import numpy as np
import torch


def calculate_stokes_to_intensity_matrix(swing, scheme="5-State"):
    """
    Calculate the polarimeter system matrix for a swing and calibration scheme.

    Parameters
    ----------
    swing : float
        Result is periodic on the integers,
        e.g. calculate_stokes_to_intensity_matrix(0.1) ==
             calculate_stokes_to_intensity_matrix(1.1)
    scheme : "4-State" or "5-State"
        Corresponds to the calibration scheme used to acquire data,
        by default "5-State"

    Returns
    -------
    torch.Tensor
        Returns different shapes depending on the scheme

        S2I.shape = (5, 4) for scheme = "5-State"
        S2I.shape = (4, 4) for scheme = "4-state"
    """
    chi = 2 * np.pi * swing
    if scheme == "5-State":
        stokes_to_intensity_matrix = torch.tensor(
            [
                [1, 0, 0, -1],
                [1, np.sin(chi), 0, -np.cos(chi)],
                [1, 0, np.sin(chi), -np.cos(chi)],
                [1, -np.sin(chi), 0, -np.cos(chi)],
                [1, 0, -np.sin(chi), -np.cos(chi)],
            ],
            dtype=torch.float32,
        )

    elif scheme == "4-State":
        stokes_to_intensity_matrix = torch.tensor(
            [
                [1, 0, 0, -1],
                [1, np.sin(chi), 0, -np.cos(chi)],
                [
                    1,
                    -0.5 * np.sin(chi),
                    np.sqrt(3) * np.cos(chi / 2) * np.sin(chi / 2),
                    -np.cos(chi),
                ],
                [
                    1,
                    -0.5 * np.sin(chi),
                    -np.sqrt(3) * np.cos(chi / 2) * np.sin(chi / 2),
                    -np.cos(chi),
                ],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(
            f"{scheme} is not implemented, use 4-State or 5-State"
        )
    return stokes_to_intensity_matrix


def calculate_intensity_to_stokes_matrix(swing, scheme="5-State"):
    """
    Calculate the inverse polarimeter system matrix for a swing and calibration
    scheme.

    Parameters
    ----------
    swing : float
        Result is periodic on the integers, e.g. I2S_matrix(0.1) = I2S_matrix(1.1)
    scheme : "4-State" or "5-State"
        Corresponds to the calibration scheme used to acquire data,
        by default "5-State"

    Returns
    -------
    torch.Tensor
        Returns different shapes depending on the scheme

        I2S.shape = (5, 4) for scheme = "5-State"
        I2S.shape = (4, 4) for scheme = "4-State"
    """
    return torch.linalg.pinv(
        calculate_stokes_to_intensity_matrix(swing, scheme=scheme)
    )


def stokes_after_adr(
    retardance, orientation, transmittance, depolarization, input="cpl"
):
    """
    Returns the Stokes parameters of the input polarization state (default =
    circularly polarized light = "cpl") that has passed through an attenuating
    depolarizing retarder (adr) parametrized by its retardance, slow-axis
    orientation, transmittance, and depolarization.

    Note: all four parameters can be torch.Tensor, but they must be the same size.
    If your parameters are in a tensor with shape = (4, ...), use
    the * operator to expand over the first dimension.

    e.g. stokes_after_adr(*array) is identical to
    stokes_after_adr(array[0], array[1], array[2], array[3]).

    Parameters
    ----------
    retardance, orientation, transmittance, depolarization : torch.Tensor, identical shapes
        retardance: retardance of adr, 2*pi periodic
        orientation: slow-axis orientation of adr, 2*pi periodic
        transmittance: transmittance of adr, 0 <= transmittance <= 1
        depolarization: depolarization of adr, 0 <= depolarization <= 1

    input : "cpl"
        Input polarization state

    Returns
    -------
    s0, s1, s2, s3: torch.Tensor, identical shapes
        Stokes parameters

    """
    if input != "cpl":
        raise NotImplementedError("input != cpl")

    # without copying transmittance, downstream changes to s0 will affect transmittance
    s0 = torch.tensor(transmittance).clone()
    s1 = (
        transmittance
        * depolarization
        * torch.sin(retardance)
        * torch.sin(2 * orientation)
    )
    s2 = (
        transmittance
        * depolarization
        * -torch.sin(retardance)
        * torch.cos(2 * orientation)
    )
    s3 = transmittance * depolarization * torch.cos(retardance)
    return s0, s1, s2, s3


def stokes012_after_ar(retardance, orientation, transmittance, input="cpl"):
    """
    Returns the first three Stokes parameters of the input polarization state
    (default = circularly polarized light = "cpl") that has passed through an
    attenuating retarder (ar) parametrized by its retardance, slow-axis
    orientation, and transmittance.

    This model is used to model non-depolarizing samples, or situations where
    depolarization information is unavailable...e.g. when only a linear Stokes
    polarimeter is available.

    Parameters
    ----------
    retardance, orientation, transmittance: torch.Tensor, identical shapes
        retardance: retardance of ar, 2*pi periodic
        orientation: slow-axis orientation of ar, 2*pi periodic
        transmittance: transmittance of ar, 0 <= transmittance <= 1

    input : "cpl"
        Input polarization state

    Returns
    -------
    s0, s1, s2: torch.Tensor
        First three Stokes parameters

    """
    if input != "cpl":
        raise NotImplementedError("input != cpl")

    # without copying transmittance, downstream changes to s0 will affect transmittance
    s0 = torch.tensor(transmittance).clone()
    s1 = transmittance * torch.sin(retardance) * torch.sin(2 * orientation)
    s2 = transmittance * -torch.sin(retardance) * torch.cos(2 * orientation)
    return s0, s1, s2


def _s12_to_orientation(s1, s2):
    """
    Converts s1 and s2 into a slow-axis orientation.

    This functions matches the sign convention used in s0123_cpl_after_adr and
    s012_cpl_after_ar (see tests for examples), and matches the orientation
    range convention used in comp-micro: 0 <= orientation < pi.

    Parameters
    ----------
    s1, s2: torch.Tensor, identical shapes
        Stokes parameters

    Returns
    -------
    torch.Tensor
        Slow-axis orientation with 0 <= orientation < pi.
    """
    return (torch.arctan2(s1, -s2) % (2 * np.pi)) / 2


def estimate_adr_from_stokes(s0, s1, s2, s3, input="cpl"):
    """
    Inverse of stokes_after_adr.

    When light with input polarization state (default = circularly polarized
    light = "cpl") has passed through an attenuating depolarizing retarder
    (adr), its Stokes parameters can be passed to this function to estimate
    the parameters of the adr, specifically its retardance, slow-axis
    orientation, transmittance, and depolarization.

    Note: this function is commonly used in QLIPP-type reconstructions. After
    converting raw intensities into Stokes parameters by applying an
    I2S_matrix, the Stokes parameters are used to estimate the parameters of
    the sample adr in a single step with this function.

    Parameters
    ----------
    s0, s1, s2, s3: torch.Tensor, identical shapes
        Stokes parameters

    input : "cpl"
        Input polarization state

    Returns
    ----------
    retardance, orientation, transmittance, depolarization: torch.Tensor
        retardance: retardance of adr, 2*pi periodic
        orientation: slow-axis orientation of adr, 2*pi periodic
        transmittance: transmittance of adr, 0 <= transmittance <= 1
        depolarization: depolarization of adr, 0 <= depolarization <= 1
    """
    if input != "cpl":
        raise NotImplementedError("input != cpl")

    len_pol = (s1**2 + s2**2 + s3**2) ** 0.5
    retardance = torch.arcsin(((s1**2 + s2**2) ** 0.5) / len_pol)
    orientation = _s12_to_orientation(s1, s2)
    # without copying s0, downstream changes to transmittance will affect s0
    transmittance = torch.tensor(s0).clone()
    depolarization = len_pol / s0
    return retardance, orientation, transmittance, depolarization


def estimate_ar_from_stokes012(s0, s1, s2, input="cpl"):
    """
    Inverse of stokes012_after_adr.

    When light with input polarization state (default = circularly polarized
    light = "cpl") has passed through an attenuating retarder (ar), its Stokes
    parameters can be passed to this function to estimate the parameters of
    the ar, specifically its retardance, slow-axis orientation, and
    transmittance.

    Parameters
    ----------
    s0, s1, s2: torch.Tensor, identical shapes
        First three Stokes parameters

    Returns
    ----------
    retardance, orientation, transmittance: torch.Tensor, identical shapes
        retardance: retardance of ar, 2*pi periodic
        orientation: slow-axis orientation of ar, 2*pi periodic
        transmittance: transmittance of ar, 0 <= transmittance <= 1
    """
    if input != "cpl":
        raise NotImplementedError("input != cpl")

    retardance = torch.arcsin(((s1**2 + s2**2) ** 0.5) / s0)
    orientation = _s12_to_orientation(s1, s2)
    # without copying s0, downstream changes to transmittance will affect s0
    transmittance = torch.tensor(s0).clone()
    return retardance, orientation, transmittance


def mueller_from_stokes(
    s0,
    s1,
    s2,
    s3,
    input="cpl",
    model="adr",
    direction="inverse",
):
    """
    When light with input polarization state (default = circularly polarized
    light = "cpl") has passed through a polarization element of a specific type
    (default = attenuating depolarizing retarder = "adr"), its Stokes parameters
    can be passed to this function to estimate the complete Mueller matrix of
    the polarization element.

    Parameters
    ----------
    s0, s1, s2, s3 : torch.Tensor, identical shapes
        Stokes parameters

    input : "cpl"
        Input polarization state

    model : "adr"
        The type of polarization element

    direction : "forward" or "inverse"
        Return the Mueller matrix (forward) or its inverse

    Returns
    -------
    torch.tensor, float, M.shape = (4, 4,) + s0.shape
        Mueller matrix on the same device as s0
    """
    if input != "cpl":
        raise NotImplementedError("input != cpl")

    if model != "adr":
        raise NotImplementedError("input != adr")

    if not (direction == "forward" or direction == "inverse"):
        raise NotImplementedError("direction must be `forward` or `inverse`")

    if direction == "forward":
        M = torch.zeros((4, 4) + torch.tensor(s0).shape, device=s0.device)
        denom = s1**2 + s2**2
        M[0, 0] = s0
        M[1, 1] = (s0 * s2**2 + s1**2 * s3) / denom
        M[1, 2] = s1 * s2 * (s3 - s0) / denom
        M[1, 3] = s1
        M[2, 1] = M[1, 2]
        M[2, 2] = (s0 * s1**2 + s2**2 * s3) / denom
        M[2, 3] = s2
        M[3, 1] = -M[1, 3]
        M[3, 2] = -M[2, 3]
        M[3, 3] = s3
        return M

    elif direction == "inverse":
        """
        NOTE: this implementation calculates the full matrix then inverts it.
        TODO: (medium) calculate the inverse matrix directly here and exploit
        the block-diagonal structure.
        TODO: (harder) instead of calculating the entire inverse matrix and
        using matrix multiplication to apply the correction, only calculate the
        unique terms (exploit the fact that the lower block is a scaled
        rotation matrix) and write a function that applies the correction
        directly.
        """
        M = mueller_from_stokes(
            s0, s1, s2, s3, input=input, model=model, direction="forward"
        )
        M_flip = torch.moveaxis(M, (0, 1), (-2, -1))
        M_inv_flip = torch.linalg.inv(M_flip)  # applied over the last two axes
        M_inv = torch.moveaxis(M_inv_flip, (-2, -1), (0, 1))
        return M_inv


def mmul(matrix, vector):
    """Convenient matrix-multiply used for
        - applying Mueller matrices to Stokes vectors
        - applying intensity_to_stokes matrices to intensities

    Parameters
    ----------
    matrix : torch.Tensor, shape = (N, M, ...)
    vector : torch.Tensor, shape = (M, ...)

    Returns
    -------
    torch.Tensor, shape = (N, ...)
    """
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("matrix.shape[1] is not equal to vector.shape[0]")

    return torch.einsum("NM...,M...->N...", matrix, vector)


def apply_orientation_offset(orientation, rotate, flip):
    """
    Applies a rotation and/or flip to each voxel of an orientation map while
    keeping the output range within 0 <= orientation < pi.

    Parameters
    ----------
    orientation : torch.Tensor
        Array of orientations measured in radians
    rotate : bool
        If True, rotate orientation pi/2 radians (90 degrees)
    flip : bool
        If True, flip the orientation

    Returns
    -------
    torch.Tensor with same shape as input

    Transformed array of orientations measured in radians
    with range 0 <= orientation < pi

    Note
    ----
    rotate=False and flip=False leaves the effective orientation unchanged
    while changing the output range to 0 <= orientation < pi
    """
    out_orientation = torch.clone(orientation)
    if rotate:
        out_orientation += torch.pi / 2
    if flip:
        out_orientation *= -1
    return torch.remainder(out_orientation, torch.pi)

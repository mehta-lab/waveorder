"""
Overview
--------

This module collects Stokes- and Mueller-related calculations. 

The functions are roughly organized into groups:

1) Polarimeter instrument matrix functions
S2I = S2I_matrix(swing, scheme="5-State")
I2S = I2S_matrix(swing, scheme="5-State")

2) Forward functions (Stokes parameters following a optical element)
s0, s1, s2, s3 = stokes_after_ADR(ret, ori, tra, dop, input="CPL")
s0, s1, s2 = stokes012_after_AR(ret, ori, tra, input="CPL")

3) Inverse functions (optical elements from Stokes parameters)
ret, ori, tra, dop = estimate_ADR_from_stokes(s0, s1, s2, s3, input="CPL")
ret, ori, tra = estimate_AR_from_stokes012(s0, s1, s2, input="CPL")

4) A function for recovering Mueller matrices from Stokes vector
M = mueller_from_stokes(
    s0, s1, s2, s3, model="AR", direction="forward", input="CPL"
)

5) A convenience function for applying Mueller and instrument matrices
y = mmul(A, x)

Usage
-----

All functions are intended to be used with ND-arrays with Stokes- or 
Mueller-indices as the first axes. 

For example, the following usage modes of stokes_after_ADR are valid:

>>> stokes_after_ADR(1, 1, 1, 1)

>>> ret = np.ones((2,3,4,5))
>>> ori = np.ones((2,3,4,5))
>>> tra = np.ones((2,3,4,5))
>>> dop = np.ones((2,3,4,5))
>>> stokes_after_ADR(ret, ori, tra, dop)

>>> ADR_params = np.ones((4,2,3,4,5)) # first axis contains the Stokes indices
>>> stokes_after_ADR(*ADR_params) # * expands along the first axis

"""
import numpy as np


def S2I_matrix(swing, scheme="5-State"):
    """
    Calculate the polarimeter system matrix for a swing and calibration scheme.

    Parameters
    ----------
    swing : float
        Result is periodic on the integers,
        e.g. S2I_matrix(0.1) = S2I_matrix(1.1)
    scheme : "4-State" or "5-State"
        Corresponds to the calibration scheme used to acquire data,
        by default "5-State"

    Returns
    -------
    NDArray
        Returns different shapes depending on the scheme

        S2I.shape = (5, 4) for scheme = "5-State"
        S2I.shape = (4, 4) for scheme = "4-state"
    """
    chi = 2 * np.pi * swing
    if scheme == "5-State":
        S2I = np.array(
            [
                [1, 0, 0, -1],
                [1, np.sin(chi), 0, -np.cos(chi)],
                [1, 0, np.sin(chi), -np.cos(chi)],
                [1, -np.sin(chi), 0, -np.cos(chi)],
                [1, 0, -np.sin(chi), -np.cos(chi)],
            ]
        )

    elif scheme == "4-State":
        S2I = np.array(
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
            ]
        )
    else:
        raise ValueError(
            f"{scheme} is not implemented, use 4-State or 5-State"
        )
    return S2I


def I2S_matrix(swing, scheme="5-State"):
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
    NDArray
        Returns different shapes depending on the scheme

        I2S.shape = (5, 4) for scheme = "5-State"
        I2S.shape = (4, 4) for scheme = "4-state"
    """
    return np.linalg.pinv(S2I_matrix(swing, scheme=scheme))


def stokes_after_ADR(ret, ori, tra, dop, input="CPL"):
    """
    Returns the Stokes parameters of the input polarization state (default =
    circularly polarized light = "CPL") that has passed through an attenuating
    depolarizing retarder (ADR) parametrized by its retardance (ret), slow-axis
    orientation (ori), transmittance (tra), and depolarization (dop).

    Note: all four parameters can be array_like, but they must be the same size.
    If your parameters are in an ndarray with array.shape = (4, ...), use
    the * operator to expand over the first dimension.

    e.g. stokes_after_ADR(*array) is identical to
    stokes_after_ADR(array[0], array[1], array[2], array[3]).

    Parameters
    ----------
    ret, ori, tra, dop : array_like, identical shapes
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1
        dop: depolarization of ADR, 0 <= dop <= 1

    input : "CPL"
        Input polarization state

    Returns
    -------
    s0, s1, s2, s3: array_like, identical shapes
        Stokes parameters

    """
    if input != "CPL":
        NotImplementedError("input != CPL")

    s0 = tra
    s1 = tra * dop * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * dop * -np.sin(ret) * np.cos(2 * ori)
    s3 = tra * dop * np.cos(ret)
    return s0, s1, s2, s3


def stokes012_after_AR(ret, ori, tra, input="CPL"):
    """
    Returns the first three Stokes parameters of the input polarization state
    (default = circularly polarized light = "CPL") that has passed through an
    attenuating retarder (AR) parametrized by its retardance (ret), slow-axis
    orientation (ori), and transmittance (tra).

    This model is used to model non-depolarizing samples, or situations where
    depolarization information is unavailable...e.g. when only a linear Stokes
    polarimeter is available.

    Parameters
    ----------
    ret, ori, tra: array_like, identical shapes
        ret: retardance of AR, 2*pi periodic
        ori: slow-axis orientation of AR, 2*pi periodic
        tra: transmittance of AR, 0 <= tra <= 1

    input : "CPL"
        Input polarization state

    Returns
    -------
    s0, s1, s2: array_like
        First three Stokes parameters

    """
    if input != "CPL":
        NotImplementedError("input != CPL")

    s0 = tra
    s1 = tra * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * -np.sin(ret) * np.cos(2 * ori)
    return s0, s1, s2


def _s12_to_ori(s1, s2):
    """
    Converts s1 and s2 into a slow-axis orientation.

    This functions matches the sign convention used in s0123_CPL_after_ADR and
    s012_CPL_after_AR (see tests for examples), and matches the orientation
    range convention used in comp-micro: 0 <= ori < pi.

    Parameters
    ----------
    s1, s2: array_like, identical shapes
        Stokes parameters

    Returns
    -------
    array_like
        Slow-axis orientation with 0 <= ori < pi.
    """
    return (np.arctan2(s1, -s2) % (2 * np.pi)) / 2


def estimate_ADR_from_stokes(s0, s1, s2, s3, input="CPL"):
    """
    Inverse of stokes_after_ADR.

    When light with input polarization state (default = circularly polarized
    light = "CPL") has passed through an attenuating depolarizing retarder
    (ADR), its Stokes parameters can be passed to this function to estimate
    the parameters of the ADR, specifically its retardance (ret), slow-axis
    orientation (ori), transmittance (tra), and depolarization (dop).

    Note: this function is commonly used in QLIPP-type reconstructions. After
    converting raw intensities into Stokes parameters by applying an
    I2S_matrix, the Stokes parameters are used to estimate the parameters of
    the sample ADR in a single step with this function.

    Parameters
    ----------
    s0, s1, s2, s3: array_like, identical shapes
        Stokes parameters

    input : "CPL"
        Input polarization state

    Returns
    ----------
    ret, ori, tra, dop: array_like
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1
        dop: depolarization of ADR, 0 <= dop <= 1
    """
    if input != "CPL":
        NotImplementedError("input != CPL")

    len_pol = (s1**2 + s2**2 + s3**2) ** 0.5
    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / len_pol)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    dop = len_pol / s0
    return ret, ori, tra, dop


def estimate_AR_from_stokes012(s0, s1, s2, input="CPL"):
    """
    Inverse of stokes012_after_ADR.

    When light with input polarization state (default = circularly polarized
    light = "CPL") has passed through an attenuating retarder (AR), its Stokes
    parameters can be passed to this function to estimate the parameters of
    the AR, specifically its retardance (ret), slow-axis orientation (ori), and
    transmittance (tra).

    Parameters
    ----------
    s0, s1, s2: array_like, identical shapes
        First three Stokes parameters

    Returns
    ----------
    ret, ori, tra: array_like, identical shapes
        ret: retardance of AR, 2*pi periodic
        ori: slow-axis orientation of AR, 2*pi periodic
        tra: transmittance of AR, 0 <= tra <= 1
    """
    if input != "CPL":
        NotImplementedError("input != CPL")

    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / s0)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    return ret, ori, tra


def mueller_from_stokes(
    s0,
    s1,
    s2,
    s3,
    input="CPL",
    model="AR",
    direction="forward",
):
    """
    When light with input polarization state (default = circularly polarized
    light = "CPL") has passed through a polarization element of a specific type
    (default = attenuating retarder = "AR"), its Stokes parameters can be
    passed to this function to estimate the complete Mueller matrix of the
    polarization element.

    Parameters
    ----------
    s0, s1, s2, s3 : array_like, identical shapes
        Stokes parameters

    input : "CPL"
        Input polarization state

    model : "AR"
        The type of polarization element

    direction : "forward" or "inverse"
        Return the Mueller matrix (forward) or its inverse

    Returns
    -------
    array_like, float, M.shape = (4, 4,) + s0.shape
        Mueller matrix
    """
    if input != "CPL":
        NotImplementedError("input != CPL")

    if model != "AR":
        NotImplementedError("input != AR")

    if not (direction == "forward" or direction == "inverse"):
        NotImplementedError("direction must be `forward` or `inverse`")

    if direction == "forward":
        M = np.zeros((4, 4) + np.array(s0).shape)
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
        TODO: (medium) Calculate the inverse matrix directly here and exploit
        the block-diagonal structure.
        TODO: (harder) Instead of calculating the entire inverse matrix and
        using matrix multiplication to apply the correction, only calculate the
        unique terms (exploit the fact that the lower block is a scaled
        rotation matrix) and write a function that applies the correction
        directly.
        """
        M = mueller_from_stokes(
            s0, s1, s2, s3, input=input, model=model, direction="forward"
        )
        M_flip = np.moveaxis(M, (0, 1), (-2, -1))
        M_inv_flip = np.linalg.inv(M_flip)  # applied over the last two axes
        M_inv = np.moveaxis(M_inv_flip, (-2, -1), (0, 1))
        return M_inv


def mmul(matrix, vector):
    """Convenient matrix-multiply used for
        - applying Mueller matrices to Stokes vectors
        - applying S2I matrices to intensities

    Parameters
    ----------
    matrix : array_like, shape = (N, M, ...)
    vector : array_like, shape = (M, ...)

    Returns
    -------
    array_like, shape = (N, ...)
    """
    if matrix.shape[1] != vector.shape[0]:
        ValueError("matrix.shape[1] is not equal to vector.shape[0]")

    return np.einsum("NM...,M...->N...", matrix, vector)

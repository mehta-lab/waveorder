"""
Overview
--------

This module collects Stokes- and Mueller-related calculations. 

The functions are organized into three groups:

1) A forward function group: 

A = A_matrix(swing, scheme="5-State")
s0, s1, s3, s3 = s0123_CPL_after_ADR(ret, ori, tra, dop)
s0, s1, s2 = s012_CPL_after_AR(ret, ori, dop)

2) An inverse function group:

ret, ori, tra, dop = inverse_s0123_CPL_after_ADR(s0, s1, s2, s3)
ret, ori, tra = inverse_s012_CPL_after_AR(s0, s1, s2)
M = AR_mueller_from_CPL_projection(s0, s1, s2, s3)

3) A convenience function group:

M = inv_AR_mueller_from_CPL_projection(s0, s1, s2, s3)
y = mmul(A, x)

Usage
-----

All functions (except A_matrix) are intended to be used with ND-arrays with
Stokes- or Mueller-indices as the first axes. 

For example, the following usage modes of s0123_CPL_after_ADR are valid:

>>> s0123_CPL_after_ADR(1, 1, 1, 1)

>>> ret = np.ones((2,3,4,5))
>>> ori = np.ones((2,3,4,5))
>>> tra = np.ones((2,3,4,5))
>>> dop = np.ones((2,3,4,5))
>>> s0123_CPL_after_ADR(ret, ori, tra, dop)

>>> ADR_params = np.ones((4,2,3,4,5)) # first axis contains the Stokes indices
>>> s0123_CPL_after_ADR(*ADR_params) # * expands along the first axis

"""
import numpy as np


# Forward function group


def A_matrix(swing, scheme="5-State"):
    """
    Calculate the polarimeter system matrix for a swing and calibration scheme.

    Parameters
    ----------
    swing : float
        Result is periodic on the integers, e.g. A_matrix(0.1) = A_matrix(1.1)
    scheme : "4-State" or "5-State"
        Corresponds to the calibration scheme used to acquire data,
        by default "5-State"

    Returns
    -------
    NDArray
        Returns different shapes depending on the scheme

        A.shape = (5, 4) for scheme = "5-State"
        A.shape = (4, 4) for scheme = "4-state"
    """
    chi = 2 * np.pi * swing
    if scheme == "5-State":
        A = np.array(
            [
                [1, 0, 0, -1],
                [1, np.sin(chi), 0, -np.cos(chi)],
                [1, 0, np.sin(chi), -np.cos(chi)],
                [1, -np.sin(chi), 0, -np.cos(chi)],
                [1, 0, -np.sin(chi), -np.cos(chi)],
            ]
        )

    elif scheme == "4-State":
        A = np.array(
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
    return A


def s0123_CPL_after_ADR(ret, ori, tra, dop):
    """
    Returns the Stokes parameters of circularly polarized light (CPL) that
    has passed through an attenuating depolarizing retarder (ADR) parametrized
    by its retardance (ret), slow-axis orientation (ori), transmittance (tra),
    and depolarization (dop).

    Note: all four parameters can be ndarrays, but they must be the same size.
    If your parameters are in an ndarray with array.shape = (4, ...), use
    the * operator to expand over the first dimension.

    e.g. s0123_CPL_after_ADR(*array) is identical to
    s0123_CPL_after_ADR(array[0], array[1], array[2], array[3]).

    Parameters
    ----------
    ret, ori, tra, dop : NDArray, identical shapes
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1
        dop: depolarization of ADR, 0 <= dop <= 1

    Returns
    -------
    s0, s1, s2, s3: NDArray, identical shapes
        Stokes parameters

    """
    s0 = tra
    s1 = tra * dop * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * dop * -np.sin(ret) * np.cos(2 * ori)
    s3 = tra * dop * np.cos(ret)
    return s0, s1, s2, s3


def s012_CPL_after_AR(ret, ori, tra):
    """
    Returns the first three Stokes parameters of circularly polarized light
    (CPL) that has passed through an attenuating retarder (AR) parametrized by
    its retardance (ret), slow-axis orientation (ori), and transmittance (tra).

    This model is used to model non-depolarizing samples, or situations where
    depolarization information is unavailable...e.g. when only a linear Stokes
    polarimeter is available.

    Parameters
    ----------
    ret, ori, tra: NDArray, identical shapes
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1

    Returns
    -------
    s0, s1, s2: NDArray
        First three Stokes parameters

    """
    s0 = tra
    s1 = tra * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * -np.sin(ret) * np.cos(2 * ori)
    return s0, s1, s2


# Inverse function group


def _s12_to_ori(s1, s2):
    """
    Converts s1 and s1 into a slow-axis orientation.

    This functions matches the sign convention used in s0123_CPL_after_ADR and
    s012_CPL_after_AR (see tests for examples), and matches the orientation
    range convention used in comp-micro: 0 <= ori < pi.

    Parameters
    ----------
    s1, s2: NDArray, identical shapes
        Stokes parameters

    Returns
    -------
    NDArray
        Slow-axis orientation with 0 <= ori < pi.
    """
    return (np.arctan2(s1, -s2) % (2 * np.pi)) / 2


def inverse_s0123_CPL_after_ADR(s0, s1, s2, s3):
    """
    Inverse of s0123_CPL_after_ADR.

    Given the Stokes parameters of circularly polarized light (CPL) that has
    passed through an attenuating depolarizing retarder (ADR), this function
    returns the parameters of the ADR, specifically its retardance (ret),
    slow-axis orientation (ori), transmittance (tra), and depolarization (dop).

    Note: this function is commonly used in QLIPP-type reconstructions. After
    converting raw intensities into Stokes parameters by applying A_inv, the
    Stokes parameters are uses to estimate the parameters of the sample ADR in
    a single step with this function.

    Parameters
    ----------
    s0, s1, s2, s3: NDArray, identical shapes
        Stokes parameters

    Returns
    ----------
    ret, ori, tra, dop: NDArray
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1
        dop: depolarization of ADR, 0 <= dop <= 1
    """
    len_pol = (s1**2 + s2**2 + s3**2) ** 0.5
    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / len_pol)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    dop = len_pol / s0
    return ret, ori, tra, dop


def inverse_s012_CPL_after_AR(s0, s1, s2):
    """
    Inverse of s012_CPL_after_ADR.

    Given the Stokes parameters of circularly polarized light (CPL) that has
    passed through an attenuating retarder (AR), this function returns the
    parameters of the ADR, specifically its retardance (ret), slow-axis
    orientation (ori), and transmittance (tra).

    Parameters
    ----------
    s0, s1, s2: NDArray, identical shapes
        First three Stokes parameters

    Returns
    ----------
    ret, ori, tra: NDArray, identical shapes
        ret: retardance of ADR, 2*pi periodic
        ori: slow-axis orientation of ADR, 2*pi periodic
        tra: transmittance of ADR, 0 <= tra <= 1
    """
    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / s0)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    return ret, ori, tra


def AR_mueller_from_CPL_projection(s0, s1, s2, s3):
    """
    Given the Stokes parameters of circularly polarized light (CPL) that has
    passed through an attenuating retarder (AR), this function returns the
    complete Mueller matrix of the AR.

    Parameters
    ----------
    s0, s1, s2, s3 : NDArray, identical shapes
        Stokes parameters

    Returns
    -------
    NDArray, float, M.shape = (4, 4,) + s0.shape
        Mueller matrix
    """
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


# Convenience function group


def inv_AR_mueller_from_CPL_projection(s0, s1, s2, s3):
    """
    Given the Stokes parameters of circularly polarized light (CPL) that has
    passed through an attenuating retarder (AR), this function returns the
    complete *INVERSE* Mueller matrix of the AR.

    Parameters
    ----------
    s0, s1, s2, s3 : NDArray, identical shapes
        Stokes parameters

    Returns
    -------
    NDArray with shape = (4, 4,) + s0.shape
        inverse Mueller matrix

    NOTE: this implementation calculates the full matrix then inverts it.
    TODO: (medium) Calculate the inverse matrix directly here and exploit the
    block-diagonal structure.
    TODO: (harder) Instead of calculating the entire inverse matrix and using
    matrix multiplication to apply the correction, only calculate the unique
    terms (exploit the fact that the lower block is a scaled rotation matrix)
    and write a function that applies the correction directly.

    """
    M = AR_mueller_from_CPL_projection(s0, s1, s2, s3)
    M_flip = np.moveaxis(M, (0, 1), (-2, -1))
    M_inv_flip = np.linalg.inv(M_flip)  # applied over the last two axes
    M_inv = np.moveaxis(M_inv_flip, (-2, -1), (0, 1))
    return M_inv


def mmul(matrix, vector):
    """Convenient matrix-multiply used for
        - applying Mueller matrices to Stokes vectors
        - applying A_inv to intensities

    Parameters
    ----------
    matrix : NDArray, shape = (N, M, ...)
    vector : NDArray, shape = (M, ...)

    where (...) shapes must be identical.

    Returns
    -------
    NDArray, shape = (N, ...)
    """
    if matrix.shape[1] != vector.shape[0]:
        ValueError("matrix.shape[1] is not equal to vector.shape[0]")

    if np.not_equal(matrix.shape[2:], vector.shape[1:]):
        ValueError("matrix.shape[2:] is not equal to vector.shape[1:]")

    return np.einsum("NM...,M...->N...", matrix, vector)

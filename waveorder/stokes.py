import numpy as np

# Forward function group


def A_matrix(swing, scheme="5-State"):
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
        raise KeyError(f"{scheme} is not implemented, use 4-State or 5-State")
    return A


def s0123_CPL_after_ADR(ret, ori, tra, dop):
    s0 = tra
    s1 = tra * dop * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * dop * -np.sin(ret) * np.cos(2 * ori)
    s3 = tra * dop * np.cos(ret)
    return s0, s1, s2, s3


def s012_CPL_after_AR(ret, ori, tra):
    s0 = tra
    s1 = tra * np.sin(ret) * np.sin(2 * ori)
    s2 = tra * -np.sin(ret) * np.cos(2 * ori)
    return s0, s1, s2


# Inverse function group


def _s12_to_ori(s1, s2):
    return (np.arctan2(s1, -s2) % (2 * np.pi)) / 2


def inverse_s0123_CPL_after_ADR(s0, s1, s2, s3):
    len_pol = (s1**2 + s2**2 + s3**2) ** 0.5
    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / len_pol)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    dop = len_pol / s0
    return ret, ori, tra, dop


def inverse_s012_CPL_after_AR(s0, s1, s2):
    ret = np.arcsin(((s1**2 + s2**2) ** 0.5) / s0)
    ori = _s12_to_ori(s1, s2)
    tra = s0
    return ret, ori, tra


def AR_mueller_from_CPL_projection(s0, s1, s2, s3):
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

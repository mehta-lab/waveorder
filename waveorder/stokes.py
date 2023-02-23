import numpy as np

# Forward function group


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

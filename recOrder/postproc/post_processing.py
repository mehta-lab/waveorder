import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import hsv_to_rgb


def generic_hsv_overlay(H, S, V, H_scale=None, S_scale=None, V_scale=None, mode='2D'):
    """
    Generates a generic HSV overlay in either 2D or 3D

    Parameters
    ----------
    H:          (nd-array) data to use in the Hue channel
    S:          (nd-array) data to use in the Saturation channel
    V:          (nd-array) data to use in the Value channel
    H_scale:    (tuple) values at which to clip the hue data for display
    S_scale:    (tuple) values at which to clip the saturation data for display
    V_scale:    (tuple) values at which to clip the value data for display
    mode:       (str) '3D' or '2D'

    Returns
    -------
    overlay:    (nd-array) RGB overlay array of shape (Z, Y, X, 3) or (Y, X, 3)

    """

    if H.shape != S.shape or H.shape != S.shape or S.shape != V.shape:
        raise ValueError(
            f'Channel shapes do not match: {H.shape} vs. {S.shape} vs. {V.shape}')

    if mode == '3D':
        overlay_final = np.zeros((H.shape[0], H.shape[1], H.shape[2], 3))
        slices = H.shape[0]
    else:
        overlay_final = np.zeros((1, H.shape[-2], H.shape[-1], 3))
        H = np.expand_dims(H, axis=0)
        S = np.expand_dims(S, axis=0)
        V = np.expand_dims(V, axis=0)
        slices = 1

    for i in range(slices):
        H_ = np.interp(H[i], H_scale, (0, 1))
        S_ = np.interp(S[i], S_scale, (0, 1))
        V_ = np.interp(V[i], V_scale, (0, 1))

        hsv = np.transpose(np.stack([H_, S_, V_]), (1, 2, 0))
        overlay_final[i] = hsv_to_rgb(hsv)

    return overlay_final[0] if mode == '2D' else overlay_final


def ret_ori_overlay(retardance, orientation, scale, mode='2D', cmap='JCh'):
    """
    This function will create an overlay of retardance and orientation with two different colormap options.
    HSV is the standard Hue, Saturation, Value colormap while JCh is a similar colormap but is perceptually uniform.

    Parameters
    ----------
    retardance:             (nd-array) retardance array of shape (N, Y, X) or (Y, X) in nanometers
    orientation:            (nd-array) orientation array of shape (N, Y, X) or (Y, X) in radian [0, pi]
    scale:                  (tuple) interpolation scale to use for retardance.  Typically use adjusted contrast limits
    mode:                   (str) '2D' or '3D'
    cmap:                   (str) 'JCh' or 'HSV'

    Returns
    -------
    overlay                 (nd-array) overlaid image of shape (N, Y, X, 3) or (Y, X, 3) RGB image

    """

    orientation = orientation * 180 / np.pi
    levels = 8
    ret_scale = (0, 10)
    noise_level = 1

    if retardance.shape != orientation.shape:
        raise ValueError(
            f'Retardance and Orientation shapes do not match: {retardance.shape} vs. {orientation.shape}')

    if mode == '3D':
        overlay_final = np.zeros((retardance.shape[0], retardance.shape[1], retardance.shape[2], 3))
        slices = retardance.shape[0]
    else:
        overlay_final = np.zeros((1, retardance.shape[-2], retardance.shape[-1], 3))
        orientation = np.expand_dims(orientation, axis=0)
        retardance = np.expand_dims(retardance, axis=0)
        slices = 1

    for i in range(slices):
        ret_ = np.interp(retardance[i], ret_scale, scale)
        ori_binned = np.round(orientation[i] / 180 * levels + 0.5) / levels - 1 / levels
        ori_ = np.interp(ori_binned, (0, 1), (0, 360))

        if cmap == 'JCh':
            J = ret_
            C = np.ones_like(J) * 60
            C[retardance[i] < noise_level] = 0
            h = ori_

            JCh = np.stack((J, C, h), axis=-1)
            JCh_rgb = cspace_convert(JCh, "JCh", "sRGB1")

            JCh_rgb[JCh_rgb < 0] = 0
            JCh_rgb[JCh_rgb > 1] = 1

            overlay_final[i] = JCh_rgb
        elif cmap == 'HSV':
            I_hsv = np.transpose(np.stack([ori_binned, np.ones_like(ori_binned),
                                           np.minimum(1, ret_ / np.max(ret_))]), (1, 2, 0))
            overlay_final[i] = hsv_to_rgb(I_hsv)

        else:
            raise ValueError(f'Colormap {cmap} not understood')

    return overlay_final[0] if mode == '2D' else overlay_final
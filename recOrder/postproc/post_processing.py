from scipy.ndimage import affine_transform
import numpy as np
from waveorder.util import wavelet_softThreshold
from colorspacious import cspace_convert
from matplotlib.colors import hsv_to_rgb


def generic_overlay(chan1, chan2, chan3, mode='2D', cmap='JCh'):
    levels = 8
    ret_scale = (0, 10)
    noise_level = 1

    if chan1.shape != chan2.shape or chan1.shape != chan3.shape or chan2.shape != chan3.shape:
        raise ValueError(
            f'Channel shapes do not match: {chan1.shape} vs. {chan2.shape} vs. {chan3.shape}')

    if mode == '3D':
        overlay_final = np.zeros((chan1.shape[0], chan1.shape[1], chan1.shape[2], 3))
        slices = chan1.shape[0]
    else:
        overlay_final = np.zeros((1, chan1.shape[-2], chan1.shape[-1], 3))
        chan1 = np.expand_dims(chan1, axis=0)
        chan2 = np.expand_dims(chan2, axis=0)
        chan3 = np.expand_dims(chan3, axis=0)
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
            I_hsv = np.transpose(np.stack([ori_, np.ones_like(ori_),
                                           np.minimum(1, ret_ / np.max(ret_))]), (1, 2, 0))
            overlay_final[i] = hsv_to_rgb(I_hsv)

        else:
            raise ValueError(f'Colormap {cmap} not understood')

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
            I_hsv = np.transpose(np.stack([ori_, np.ones_like(ori_),
                                           np.minimum(1, ret_ / np.max(ret_))]), (1, 2, 0))
            overlay_final[i] = hsv_to_rgb(I_hsv)

        else:
            raise ValueError(f'Colormap {cmap} not understood')

    return overlay_final[0] if mode == '2D' else overlay_final

def post_proc_denoise(data_volume, params):

    data_volume_denoised = np.copy(data_volume)

    if len(data_volume) == 1:
        data_volume_denoised = wavelet_softThreshold(data_volume[0], 'db8', params[1], params[2])
    else:
        for z in range(len(data_volume)):
            data_volume_denoised[z, :, :] = wavelet_softThreshold(data_volume[z], 'db8', params[1], params[2])

    return data_volume_denoised


def translate_3D(image_stack, shift, binning=1, size_z_param=0, size_z_um=0):
    """
    Parameters
    ----------
    image_stack: img stack of shape (Z, Y, X)
        list of images to translate

    shift: shift in terms of translation in [z, y, x]

        Shift directions: If you want to shift the image up and to the right shift = [0, +y, -x]

    Returns
    -------
    registered_image_stack:
    """

    registered_images = []

    matrix = [[1, 0, shift[1]], [0, 1, shift[2]]]
    for img in image_stack:
        #         if size_z_um == 0:  # 2D translation
        #             pass
        # #             shift[0] = 0
        #         else:
        #             # 3D translation. Scale z-shift according to the z-step size.
        #             shift[0] = shift[0] * size_z_param / size_z_um
        #         if not binning == 1:
        #             shift[1] = shift[1] / binning
        #             shift[2] = shift[2] / binning

        image = affine_transform(img, matrix, order=1)
        registered_images.append(image)
    return np.asarray(registered_images)

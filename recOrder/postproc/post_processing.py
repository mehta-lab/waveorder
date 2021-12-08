from scipy.ndimage import affine_transform
import numpy as np
from waveorder.util import wavelet_softThreshold
from colorspacious import cspace_convert


def ret_ori_overlay(retardance, orientation, scale, mode='2D'):
    # orientation is in radian

    orientation = orientation * 180 / np.pi
    levels = 8

    ret_scale = (0, 10)
    noise_level = 1

    if mode == '3D':
        if retardance.shape != orientation.shape:
            raise ValueError(
                f'Retardance and Orientation shapes do not match: {retardance.shape} vs. {orientation.shape}')

        overlay_final = np.zeros((retardance.shape[0], retardance.shape[1], retardance.shape[2], 3))
        for i in range(retardance.shape[0]):
            ret_ = np.interp(retardance[i], ret_scale, scale)
            ori_binned = np.round(orientation[i] / 180 * levels + 0.5) / levels - 1 / levels
            ori_ = np.interp(ori_binned, (0, 1), (0, 360))

            J = ret_
            C = np.ones_like(J) * 60
            C[retardance[i] < noise_level] = 0
            h = ori_

            JCh = np.stack((J, C, h), axis=-1)
            JCh_rgb = cspace_convert(JCh, "JCh", "sRGB1")

            JCh_rgb[JCh_rgb < 0] = 0
            JCh_rgb[JCh_rgb > 1] = 1

            overlay_final[i] = JCh_rgb

        return overlay_final

    else:
        ret_ = np.interp(retardance, ret_scale, scale)
        ori_binned = np.round(orientation / 180 * levels + 0.5) / levels - 1 / levels
        ori_ = np.interp(ori_binned, (0, 1), (0, 360))

        J = ret_
        C = np.ones_like(J) * 60
        C[retardance < noise_level] = 0
        h = ori_

        JCh = np.stack((J, C, h), axis=-1)
        JCh_rgb = cspace_convert(JCh, "JCh", "sRGB1")

        JCh_rgb[JCh_rgb < 0] = 0
        JCh_rgb[JCh_rgb > 1] = 1

        return JCh_rgb

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

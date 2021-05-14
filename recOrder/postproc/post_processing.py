from scipy.ndimage import affine_transform, sobel
from skimage.feature import register_translation
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import zarr
import tifffile as tiff

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
    for img in image_stack:
        if size_z_um == 0:  # 2D translation
            shift[0] = 0
        else:
            # 3D translation. Scale z-shift according to the z-step size.
            shift[0] = shift[0] * size_z_param / size_z_um
        if not binning == 1:
            shift[1] = shift[1] / binning
            shift[2] = shift[2] / binning

        image = affine_transform(img, np.ones(3), shift, order=1)
        registered_images.append(image)
    return np.asarray(registered_images)

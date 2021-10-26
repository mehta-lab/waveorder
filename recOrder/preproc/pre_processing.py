import numpy as np
from waveorder.util import wavelet_softThreshold
import cv2
import warnings

def preproc_denoise(stokes, params):

    stokes_denoised = np.copy(stokes)
    for chan in range(len(params)):

        if 'S0' in params[chan][0]:
            stokes_denoised[0] = wavelet_softThreshold(stokes[0], 'db8', params[chan][1], params[chan][2])

        elif 'S1' in params[chan][0]:
            stokes_denoised[1] = wavelet_softThreshold(stokes[1], 'db8', params[chan][1], params[chan][2])

        if 'S2' in params[chan][0]:
            stokes_denoised[2] = wavelet_softThreshold(stokes[2], 'db8', params[chan][1], params[chan][2])

        if 'S3' in params[chan][0]:
            stokes_denoised[3] = wavelet_softThreshold(stokes[3], 'db8', params[chan][1], params[chan][2])

    return stokes_denoised

def find_focus(stack):
    """
    Parameters
    ----------
    stack:       (nd-array) Image stack of dimension (Z, ...) to find focus

    Returns
    -------
    focus_idx:  (int) Index corresponding to the focal plane of the stack

    """
    def brenner_gradient(im):
        assert len(im.shape) == 2, 'Input image must be 2D'
        return np.mean((im[:-2, :] - im[2:, :]) ** 2)

    focus_scores = []
    for img in stack:
        focus_score = brenner_gradient(img)
        focus_scores.append(focus_score)

    focus_idx_min = np.argmin(focus_scores)
    focus_idx_max = np.argmax(focus_scores)

    return focus_idx_max, focus_idx_min

def get_autocontrast_limits(img, clip = .01):

    #TODO: Figure out how to do this for float images (ret, phase, etc.)
    # current only works for uint8, uint16, etc.

    data_type = img.dtype

    # Will raise an error if dtype is float, in that case cap the n_bins at 65536
    try:
        n_bins = int(np.iinfo(data_type).max)
    except:
        n_bins = 65536

    hist = cv2.calcHist([img.astype(data_type.name)], [0], None, [n_bins], [0, n_bins])
    cdf = np.cumsum(hist)

    maximum = cdf[-1]
    pix_clip = maximum * clip

    try:
        # Locate min cut
        min_val = 0
        while cdf[min_val] < pix_clip:
            min_val += 1

        # Locate max cut
        max_val = n_bins - 1
        while cdf[max_val] >= (maximum - pix_clip):
            max_val -= 1

        return min_val, max_val
    except IndexError as ex:
        warnings.warn(UserWarning(f'Pixel Data overexposed, please check the image, Warning Message: {ex}'))
        return 0, 65535

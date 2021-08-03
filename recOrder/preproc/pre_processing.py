import numpy as np
from waveorder.util import wavelet_softThreshold


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

    focus_idx_min = np.where(focus_scores == np.min(focus_scores))[0][0]
    focus_idx_max = np.where(focus_scores == np.max(focus_scores))[0][0]

    return focus_idx_max, focus_idx_min

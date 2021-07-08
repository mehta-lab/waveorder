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

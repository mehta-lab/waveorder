import numpy as np
from waveorder.util import wavelet_softThreshold


def preproc_denoise(stokes, params):

    stokes_denoised = np.copy(stokes)
    for chan in range(len(params)):

        if 'S0' in params[chan][0]:
            for z in range(len(stokes)):
                stokes_denoised[z, 0, :, :] = wavelet_softThreshold(stokes[z, 0, :, :], 'db8',
                                                                    params[chan][1], params[chan][2])
        elif 'S1' in params[chan][0]:
            for z in range(len(stokes)):
                stokes_denoised[z, 1, :, :] = wavelet_softThreshold(stokes[z, 1, :, :], 'db8',
                                                                    params[chan][1], params[chan][2])
        if 'S2' in params[chan][0]:
            for z in range(len(stokes)):
                stokes_denoised[z, 2, :, :] = wavelet_softThreshold(stokes[z, 2, :, :], 'db8',
                                                                    params[chan][1], params[chan][2])
        if 'S3' in params[chan][0]:
            for z in range(len(stokes)):
                stokes_denoised[z, 3, :, :] = wavelet_softThreshold(stokes[z, 3, :, :], 'db8',
                                                                    params[chan][1], params[chan][2])

    return stokes_denoised

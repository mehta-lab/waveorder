import numpy as np
import cupy as cp
from waveorder import optics, util


# OTF precomputations
def phase_2D_to_3D_OTF(
    YX_shape,
    YX_ps,
    Z_samples,
    lambda_illu,
    n_media,
    NA_illu,
    NA_obj,
    use_gpu=False,
    gpu_id=0,
):
    _, _, fxx, fyy = util.gen_coordinate(YX_shape, YX_ps)

    ill_pupil = optics.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)
    det_pupil = optics.gen_Pupil(fxx, fyy, NA_obj, lambda_illu)
    Hz_stack = optics.gen_Hz_stack(
        fxx, fyy, det_pupil, lambda_illu / n_media, Z_samples
    )

    ZYX_shape = YX_shape + (len(Z_samples),)
    Hu = np.zeros(ZYX_shape)
    Hp = np.zeros(ZYX_shape)
    for z in range(len(Z_samples)):
        Hu[z], Hp[z] = optics.WOTF_2D_compute(
            ill_pupil, det_pupil * Hz_stack[z], use_gpu, gpu_id
        )

    return Hu, Hp


def phase_3D_to_3D_OTF(
    ZYX_shape,
    YX_ps,
    Z_ps,
    Z_pad,
    lambda_illu,
    n_media,
    NA_illu,
    NA_obj,
    use_gpu=False,
    gpu_id=0,
):
    _, _, fxx, fyy = util.gen_coordinate(ZYX_shape[1:], YX_ps)
    Z_total = ZYX_shape[0] + 2 * Z_pad
    Z_samples = -np.fft.ifftshift((np.r_[0:Z_total] - Z_total // 2) * Z_ps)

    ill_pupil = optics.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)
    det_pupil = optics.gen_Pupil(fxx, fyy, NA_obj, lambda_illu)
    Hz_stack = optics.gen_Hz_stack(
        fxx, fyy, det_pupil, lambda_illu / n_media, Z_samples
    )
    G_fun_z_3D = optics.gen_Greens_function_z(
        fxx, fyy, lambda_illu / n_media, Z_samples
    )

    H_re, H_im = optics.WOTF_3D_compute(
        ill_pupil.astype("float32"),
        ill_pupil.astype("float32"),
        det_pupil.astype("complex64"),
        Hz_stack.astype("complex64"),
        G_fun_z_3D.astype("complex64"),
        Z_ps,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )

    return H_re, H_im

## Reconstructions
def phase_2D_to_3D_recon(
    ZYX_data,
    Hu,
    Hp,
    method="Tikhonov",
    reg_u=1e-6,
    reg_p=1e-6,
    bg_filter=True,
    use_gpu=False,
    gpu_id=0,
    **kwargs
):
    """

    2D phase reconstruction from defocused set of intensity images

    Parameters
    ----------
        ZYX_data  : numpy.ndarray
                    defocused set of intensity images with size (Z, Y, X)

        method    : str
                    denoiser for 2D phase reconstruction
                    'Tikhonov' for Tikhonov denoiser
                    'TV'       for TV denoiser

        reg_u     : float
                    Tikhonov regularization parameter for 2D absorption

        reg_p     : float
                    Tikhonov regularization parameter for 2D phase

        bg_filter : bool
                    option for slow-varying 2D background normalization with uniform filter

        use_gpu :   False

        gpu_id :    0

        **kwargs

    Returns
    -------
        mu_sample  : numpy.ndarray
                        2D absorption reconstruction with the size of (Y, X)

        phi_sample : numpy.ndarray
                        2D phase reconstruction (in the unit of rad) with the size of (Y, X)


    """

    S0_stack = util.inten_normalization(ZYX_data, bg_filter=bg_filter)

    if use_gpu:
        Hu = cp.array(Hu)
        Hp = cp.array(Hp)

        S0_stack_f = cp.fft.fft2(S0_stack, axes=(1, 2))

        AHA = [
            cp.sum(cp.abs(Hu) ** 2, axis=0) + reg_u,
            cp.sum(cp.conj(Hu) * Hp, axis=0),
            cp.sum(cp.conj(Hp) * Hu, axis=0),
            cp.sum(cp.abs(Hp) ** 2, axis=0) + reg_p,
        ]

        b_vec = [
            cp.sum(cp.conj(Hu) * S0_stack_f, axis=0),
            cp.sum(cp.conj(Hp) * S0_stack_f, axis=0),
        ]
    else:
        S0_stack_f = np.fft.fft2(S0_stack, axes=(1, 2))

        AHA = [
            np.sum(np.abs(Hu) ** 2, axis=0) + reg_u,
            np.sum(np.conj(Hu) * Hp, axis=0),
            np.sum(np.conj(Hp) * Hu, axis=0),
            np.sum(np.abs(Hp) ** 2, axis=0) + reg_p,
        ]

        b_vec = [
            np.sum(np.conj(Hu) * S0_stack_f, axis=0),
            np.sum(np.conj(Hp) * S0_stack_f, axis=0),
        ]

    # Deconvolution with Tikhonov regularization
    if method == "Tikhonov":
        mu_sample, phi_sample = util.Dual_variable_Tikhonov_deconv_2D(
            AHA, b_vec, use_gpu=use_gpu, gpu_id=gpu_id
        )

    # ADMM deconvolution with anisotropic TV regularization
    elif method == "TV":
        mu_sample, phi_sample = util.Dual_variable_ADMM_TV_deconv_2D(
            AHA, b_vec, use_gpu=use_gpu, gpu_id=gpu_id, **kwargs
        )

    phi_sample -= phi_sample.mean()

    return mu_sample, phi_sample

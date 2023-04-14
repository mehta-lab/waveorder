import torch
from waveorder import optics, util


def calc_TF(YX_shape, YX_ps, Z_pos_list, lamb_ill, n_media, NA_ill, NA_obj):
    frr = util.gen_radial_freq(YX_shape, YX_ps)

    ill_pupil = optics.gen_pupil(frr, NA_ill, lamb_ill)
    det_pupil = optics.gen_pupil(frr, NA_obj, lamb_ill)
    Hz_stack = optics.gen_Hz_stack(
        frr, det_pupil, lamb_ill / n_media, torch.tensor(Z_pos_list)
    )

    ZYX_shape = (len(Z_pos_list),) + YX_shape

    Hu = torch.zeros(ZYX_shape, dtype=torch.complex64)
    Hp = torch.zeros(ZYX_shape, dtype=torch.complex64)
    for z in range(len(Z_pos_list)):
        Hu[z], Hp[z] = optics.WOTF_2D_compute(
            ill_pupil, det_pupil * Hz_stack[z]
        )

    return Hu, Hp

def visualize_TF(viewer, Hu, Hp, ZYX_scale):
    raise NotImplementedError 

def apply_inv_TF(
    ZYX_data,
    Hu,
    Hp,
    method="Tikhonov",
    reg_u=1e-6,
    reg_p=1e-6,
    bg_filter=True,
    **kwargs
):
    S0_stack = util.inten_normalization(ZYX_data, bg_filter=bg_filter)

    S0_stack_f = torch.fft.fft2(S0_stack, axes=(1, 2))

    AHA = [
        torch.sum(torch.abs(Hu) ** 2, axis=0) + reg_u,
        torch.sum(torch.conj(Hu) * Hp, axis=0),
        torch.sum(torch.conj(Hp) * Hu, axis=0),
        torch.sum(torch.abs(Hp) ** 2, axis=0) + reg_p,
    ]

    b_vec = [
        torch.sum(torch.conj(Hu) * S0_stack_f, axis=0),
        torch.sum(torch.conj(Hp) * S0_stack_f, axis=0),
    ]

    # Deconvolution with Tikhonov regularization
    if method == "Tikhonov":
        mu_sample, phi_sample = util.Dual_variable_Tikhonov_deconv_2D(
            AHA, b_vec
        )

    # ADMM deconvolution with anisotropic TV regularization
    elif method == "TV":
        mu_sample, phi_sample = util.Dual_variable_ADMM_TV_deconv_2D(
            AHA, b_vec, **kwargs
        )

    phi_sample -= phi_sample.mean()

    return mu_sample, phi_sample

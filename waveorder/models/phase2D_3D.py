import torch
import numpy as np
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
    # TODO: consider generalizing w/ phase3Dto3D.visualize_TF
    arrays = [
        (torch.imag(Hu).numpy(), "Im(H_u)"),
        (torch.real(Hu).numpy(), "Re(H_u)"),
        (torch.imag(Hp).numpy(), "Im(H_p)"),
        (torch.real(Hp).numpy(), "Re(H_p)"),
    ]

    for array in arrays:
        lim = 0.5 * np.max(np.abs(array[0]))
        viewer.add_image(
            np.fft.ifftshift(array[0], axes=(1,2)),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1,1,1),
        )
    viewer.dims.order = (2, 0, 1)

def apply_TF(ZYX_obj, Hp):
    # Very simple simulation, consider adding noise and bkg knobs

    # TODO: extend to absorption, or restrict to just phase
    ZYX_obj_hat = torch.fft.fftn(ZYX_obj, dim=(1,2))
    ZYX_data = ZYX_obj_hat * torch.real(Hp)
    data = torch.real(torch.fft.ifftn(ZYX_data, dim=(1,2)))

    data = torch.tensor(data + 10)  # Add a direct background
    return data


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

    S0_stack_f = torch.fft.fft2(S0_stack, dim=(1, 2))

    AHA = [
        torch.sum(torch.abs(Hu) ** 2, dim=0) + reg_u,
        torch.sum(torch.conj(Hu) * Hp, dim=0),
        torch.sum(torch.conj(Hp) * Hu, dim=0),
        torch.sum(torch.abs(Hp) ** 2, dim=0) + reg_p,
    ]

    b_vec = [
        torch.sum(torch.conj(Hu) * S0_stack_f, dim=0),
        torch.sum(torch.conj(Hp) * S0_stack_f, dim=0),
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

    phi_sample -= torch.mean(phi_sample)

    return phi_sample

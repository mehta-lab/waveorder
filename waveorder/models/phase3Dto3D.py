import numpy as np
import torch
from waveorder import optics, util


def calc_TF(
    ZYX_shape,
    YX_ps,
    Z_ps,
    Z_pad,
    lamb_ill,
    n_media,
    NA_ill,
    NA_obj,
):
    frr = util.gen_radial_freq(ZYX_shape[1:], YX_ps)
    Z_total = ZYX_shape[0] + 2 * Z_pad
    Z_pos_list = torch.fft.ifftshift(
        (torch.arange(Z_total) - Z_total // 2) * Z_ps
    )

    ill_pupil = optics.gen_pupil(frr, NA_ill, lamb_ill)
    det_pupil = optics.gen_pupil(frr, NA_obj, lamb_ill)
    Hz_stack = optics.gen_Hz_stack(
        frr, det_pupil, lamb_ill / n_media, Z_pos_list
    )
    G_fun_z_3D = optics.gen_Greens_function_z(
        frr, det_pupil, lamb_ill / n_media, Z_pos_list
    )

    H_re, H_im = optics.WOTF_3D_compute(
        ill_pupil, ill_pupil, det_pupil, Hz_stack, G_fun_z_3D, Z_ps
    )

    return H_re, H_im


def visualize_TF(viewer, H_re, H_im, ZYX_scale):
    # TODO: consider generalizing w/ phase2Dto3D.visualize_TF
    arrays = [
        (torch.real(H_im).numpy(), "Re(H_im)"),
        (torch.imag(H_im).numpy(), "Im(H_im)"),
        (torch.real(H_re).numpy(), "Re(H_re)"),
        (torch.imag(H_re).numpy(), "Im(H_re)"),
    ]

    for array in arrays:
        lim = 0.5 * np.max(np.abs(array[0]))
        viewer.add_image(
            np.fft.ifftshift(array[0]),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / ZYX_scale,
        )
    viewer.dims.order = (2, 0, 1)


def apply_TF(ZYX_obj, H_re):
    # Very simple simulation, consider adding noise and bkg knobs

    # TODO: extend to absorption, or restrict to just phase
    ZYX_obj_hat = torch.fft.fftn(ZYX_obj)
    ZYX_data = ZYX_obj_hat * H_re
    data = torch.real(torch.fft.ifftn(ZYX_data))

    data = torch.tensor(data + 10)  # Add a direct background
    return data


# TODO CONSIDER MAKING THIS A PHASE-ONLY RECONSTRUCTION
def apply_inv_TF(
    ZYX_data,
    H_re,
    H_im,
    Z_ps,  # TODO: MOVE THIS PARAM TO OTF? (leaky param)
    lamb_ill,  # TOOD: MOVE THIS PARAM TO OTF? (leaky param)
    absorption_ratio=0.0,
    method="Tikhonov",
    **kwargs
):
    # TODO HANDLE PADDING
    # pad_z = H_re.shape[0] - ZYX_data.shape[0]

    # if pad_z < 0:
    #     raise ("")
    # elif pad_z == 0:
    #     S0_stack = util.inten_normalization_3D(ZYX_data)
    # elif pad_z >= 0:  ## FINISH THIS
    #     S0_pad = np.pad(
    #         ZYX_data,
    #         ((pad_z, pad_z), (0, 0), (0, 0)),
    #         mode="constant",
    #         constant_values=0,
    #     )
    #     if pad_z < ZYX_data.shape[0]:
    #         S0_pad[:pad_z, :, :] = (S0_stack[:pad_z, :, :])[::-1, :, :]
    #         S0_pad[-pad_z:, :] = (S0_stack[-pad_z:, :])[::-1, :, :]
    #     else:
    #         print(
    #             "pad_z is larger than number of z-slices, use zero padding (not effective) instead of reflection padding"
    #         )

    ZYX_data = util.inten_normalization_3D(ZYX_data)

    H_eff = H_re + absorption_ratio * H_im

    if method == "Tikhonov":
        f_real = util.Single_variable_Tikhonov_deconv_3D(
            ZYX_data, H_eff, **kwargs
        )

    elif method == "TV":
        f_real = util.Single_variable_ADMM_TV_deconv_3D(
            ZYX_data, H_eff, **kwargs
        )

    # TODO HANDLE UNPADDING
    # if Z_pad != 0:
    #    f_real = f_real[pad_z:-pad_z, ...]

    return f_real * Z_ps / 4 / np.pi * lamb_ill

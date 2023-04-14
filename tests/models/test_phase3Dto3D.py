from waveorder.models import phase3Dto3D


def test_calc_TF():
    Z_pad = 5
    H_re, H_im = phase3Dto3D.calc_TF(
        ZYX_shape=(20, 100, 101),
        YX_ps=6.5 / 40,
        Z_ps=2,
        Z_pad=Z_pad,
        lamb_ill=0.5,
        n_media=1.0,
        NA_ill=0.45,
        NA_obj=0.55,
    )

    assert H_re.shape == (20 + 2 * Z_pad, 100, 101)
    assert H_im.shape == (20 + 2 * Z_pad, 100, 101)

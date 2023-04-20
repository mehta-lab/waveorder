from waveorder.models import phase2D_3D


def test_calc_TF():
    Hu, Hp = phase2D_3D.calc_TF(
        YX_shape=(100, 101),
        YX_ps=6.5 / 40,
        Z_pos_list=[-1, 0, 1],
        lamb_ill=0.5,
        n_media=1.0,
        NA_ill=0.4,
        NA_obj=0.55,
    )

    assert Hu.shape == (3, 100, 101)
    assert Hp.shape == (3, 100, 101)

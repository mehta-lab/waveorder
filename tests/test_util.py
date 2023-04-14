from waveorder import util


def test_gen_coordinate():
    YX_shape = (5, 6)
    frr = util.gen_radial_freq(YX_shape, 1)

    assert frr.shape == YX_shape
    assert frr[0, 0] == 0

from waveorder import util
import torch


def test_gen_coordinate():
    YX_shape = (5, 6)
    frr = util.generate_radial_frequencies(YX_shape, 1)

    assert frr.shape == YX_shape
    assert frr[0, 0] == 0

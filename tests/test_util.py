from waveorder import util
import torch
import pytest


def test_gen_coordinate():
    YX_shape = (5, 6)
    frr = util.generate_radial_frequencies(YX_shape, 1)

    assert frr.shape == YX_shape
    assert frr[0, 0] == 0


# test util.pad_zyx function
@pytest.fixture
def zyx_data():
    return torch.ones((3, 4, 5))  # Example input data


def test_pad_zyx_negative_padding():
    zyx_data = torch.zeros((3, 4, 5))
    z_padding = -1
    with pytest.raises(Exception):
        util.pad_zyx_along_z(zyx_data, z_padding)


def test_pad_zyx_no_padding(zyx_data):
    z_padding = 0
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert torch.all(result == zyx_data)


def test_pad_zyx_small_padding(zyx_data):
    z_padding = 2
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert result.shape == (7, 4, 5)
    assert torch.all(result[:2] == torch.flip(zyx_data[:2], dims=[0]))
    assert torch.all(result[-2:] == torch.flip(zyx_data[-2:], dims=[0]))


def test_pad_zyx_large_padding(zyx_data):
    z_padding = 5
    result = util.pad_zyx_along_z(zyx_data, z_padding)
    assert result.shape == (13, 4, 5)
    assert torch.all(result[:5] == 0)
    assert torch.all(result[-5:] == 0)

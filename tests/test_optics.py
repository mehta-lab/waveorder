from waveorder import optics, util
import torch


def test_gen_pupil():
    frr = util.gen_radial_freq((10, 10), 0.5)
    pupil = optics.gen_pupil(frr, 0.5, 0.5)

    # Corners are in the pupil
    assert pupil[0, 0] == 1
    assert pupil[-1, -1] == 1

    # Center is outside the pupil
    assert pupil[5, 5] == 0


def test_gen_Hz_stack():
    lambda_in = 0.5
    z_positions = torch.tensor([-1, 0, 1])
    frr = util.gen_radial_freq((10, 10), 0.5)
    pupil = optics.gen_pupil(frr, 0.5, 0.5)

    Hz_stack = optics.gen_Hz_stack(frr, pupil, lambda_in, z_positions)

    assert Hz_stack.shape == (3, 10, 10)
    assert Hz_stack[1, 0, 0] == 1
    assert Hz_stack[1, 5, 5] == 0

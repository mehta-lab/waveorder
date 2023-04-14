from waveorder import optics, util


def test_gen_pupil():
    frr = util.gen_radial_freq((10, 10), 0.5)
    pupil = optics.gen_pupil(frr, 0.5, 0.5)

    # Corners are in the pupil
    assert pupil[0, 0] == 1
    assert pupil[-1, -1] == 1

    # Center is outside
    assert pupil[5, 5] == 0

import torch

from waveorder import optics, util


def test_generate_pupil():
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    # Corners are in the pupil
    assert pupil[0, 0] > 0.99
    assert pupil[-1, -1] > 0.99

    # Center is outside the pupil
    assert pupil[5, 5] < 0.01


def test_generate_pupil_gradient():
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    NA = torch.tensor(0.5, requires_grad=True)
    pupil = optics.generate_pupil(radial_frequencies, NA, 0.5)
    loss = pupil.sum()
    loss.backward()
    assert NA.grad is not None
    assert NA.grad != 0


def test_generate_tilted_pupil():
    fyy, fxx = util.generate_frequencies((64, 64), 0.1)
    pupil = optics.generate_tilted_pupil(fxx, fyy, 0.5, 0.532, 1.0, 0.0, 0.0)
    # Zero tilt: pupil should be centered and roughly circular
    shifted = torch.fft.fftshift(pupil)
    center = shifted[32, 32]
    corner = shifted[0, 0]
    assert center > 0.9  # center of pupil is bright
    assert corner < 0.1  # corner is outside pupil


def test_generate_tilted_pupil_gradient():
    fyy, fxx = util.generate_frequencies((32, 32), 0.1)
    tilt_z = torch.tensor(0.1, requires_grad=True)
    tilt_a = torch.tensor(0.0, requires_grad=True)
    pupil = optics.generate_tilted_pupil(fxx, fyy, 0.5, 0.532, 1.3, tilt_z, tilt_a)
    loss = pupil.sum()
    loss.backward()
    assert tilt_z.grad is not None


def test_generate_propagation_kernel():
    lambda_in = 0.5
    z_positions = torch.tensor([-1, 0, 1])
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    propagation_kernel = optics.generate_propagation_kernel(radial_frequencies, pupil, lambda_in, z_positions)

    assert propagation_kernel.shape == (3, 10, 10)
    assert abs(propagation_kernel[1, 0, 0] - 1) < 0.01
    assert abs(propagation_kernel[1, 5, 5]) < 0.01


def test_gen_Greens_function_z():
    wavelength = 0.5
    z_position_list = torch.tensor([0, 1, -1])  # note fftfreq coords
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    G = optics.generate_greens_function_z(radial_frequencies, pupil, wavelength, z_position_list)

    assert G.shape == (3, 10, 10)
    assert abs(G[1, 5, 5]) < 0.01


def test_WOTF_2D():
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    illumination_pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)
    detection_pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    (
        absorption_transfer_function,
        phase_transfer_function,
    ) = optics.compute_weak_object_transfer_function_2d(illumination_pupil, detection_pupil)

    # Absorption DC term
    assert abs(absorption_transfer_function[0, 0] - 2) < 0.01

    # No phase contrast for an in-focus slice
    assert torch.all(torch.abs(torch.real(phase_transfer_function)) < 0.01)
    assert torch.all(torch.abs(torch.imag(phase_transfer_function)) < 0.01)


def test_WOTF_3D():
    wavelength = 0.5
    index_of_refraction_media = 1.0
    yx_pixel_size = 0.25
    z_pixel_size = 0.2
    numerical_aperture_detection = 0.5
    z_position_list = torch.fft.fftfreq(11, 1 / z_pixel_size)

    frr = util.generate_radial_frequencies((10, 10), yx_pixel_size)
    source = optics.generate_pupil(frr, numerical_aperture_detection, wavelength)
    pupil = optics.generate_pupil(frr, numerical_aperture_detection, wavelength)

    Hz = optics.generate_propagation_kernel(frr, pupil, wavelength / index_of_refraction_media, z_position_list)
    G = optics.generate_greens_function_z(frr, pupil, wavelength / index_of_refraction_media, z_position_list)

    H_re, H_im = optics.compute_weak_object_transfer_function_3D(source, source, pupil, Hz, G, z_pixel_size)

    assert H_re.shape == (11, 10, 10)
    assert H_im.shape == (11, 10, 10)

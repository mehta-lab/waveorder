import torch

from waveorder import optics, util


def test_generate_pupil():
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    # Corners are in the pupil
    assert torch.isclose(pupil[0, 0], torch.tensor(1.0), rtol=1e-3)
    assert torch.isclose(pupil[-1, -1], torch.tensor(1.0), rtol=1e-3)

    # Center is outside the pupil
    assert pupil[5, 5] < 1e-3


def test_generate_pupil_cutoff():
    """
    Test generate_pupil at the cutoff frequency.
    """
    frr = torch.tensor([[0.5, 1.0, 1.5]])
    NA = 1.0
    lamb_in = 1.0
    pupil = optics.generate_pupil(frr, NA, lamb_in)
    # At cutoff, sigmoid should be ~0.5
    assert torch.isclose(pupil[0, 1], torch.tensor(0.5), atol=1e-3)


def test_generate_propagation_kernel():
    lambda_in = 0.5
    z_positions = torch.tensor([-1, 0, 1])
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies, pupil, lambda_in, z_positions
    )

    assert propagation_kernel.shape == (3, 10, 10)
    assert propagation_kernel[1, 0, 0] == 1
    assert torch.abs(propagation_kernel[1, 5, 5]) < 1e-3


def test_gen_Greens_function_z():
    wavelength = 0.5
    z_position_list = torch.tensor([0, 1, -1])  # note fftfreq coords
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    G = optics.generate_greens_function_z(
        radial_frequencies, pupil, wavelength, z_position_list
    )

    assert G.shape == (3, 10, 10)
    assert torch.abs(G[1, 5, 5]) < 1e-3


def test_WOTF_2D():
    radial_frequencies = util.generate_radial_frequencies((10, 10), 0.5)
    illumination_pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)
    detection_pupil = optics.generate_pupil(radial_frequencies, 0.5, 0.5)

    (
        absorption_transfer_function,
        phase_transfer_function,
    ) = optics.compute_weak_object_transfer_function_2d(
        illumination_pupil, detection_pupil
    )

    # Absorption DC term
    assert torch.isclose(
        torch.real(absorption_transfer_function[0, 0]),
        torch.tensor(2.0),
        rtol=1e-3,
    )
    assert torch.abs(torch.imag(absorption_transfer_function[0, 0])) < 1e-3

    # No phase contrast for an in-focus slice
    assert torch.all(torch.real(phase_transfer_function) == 0)
    assert torch.all(torch.imag(phase_transfer_function) == 0)


def test_WOTF_3D():
    wavelength = 0.5
    index_of_refraction_media = 1.0
    yx_pixel_size = 0.25
    z_pixel_size = 0.2
    numerical_aperture_detection = 0.5
    z_position_list = torch.fft.fftfreq(11, 1 / z_pixel_size)

    frr = util.generate_radial_frequencies((10, 10), yx_pixel_size)
    source = optics.generate_pupil(
        frr, numerical_aperture_detection, wavelength
    )
    pupil = optics.generate_pupil(
        frr, numerical_aperture_detection, wavelength
    )

    Hz = optics.generate_propagation_kernel(
        frr, pupil, wavelength / index_of_refraction_media, z_position_list
    )
    G = optics.generate_greens_function_z(
        frr, pupil, wavelength / index_of_refraction_media, z_position_list
    )

    H_re, H_im = optics.compute_weak_object_transfer_function_3D(
        source, source, pupil, Hz, G, z_pixel_size
    )

    assert H_re.shape == (11, 10, 10)
    assert H_im.shape == (11, 10, 10)

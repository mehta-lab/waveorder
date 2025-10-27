import numpy as np
import pytest

from waveorder.models import phase_thick_3d


@pytest.mark.parametrize("invert_phase_contrast", (True, False))
def test_calculate_transfer_function(invert_phase_contrast):
    z_padding = 5
    H_re, H_im = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(20, 100, 101),
        yx_pixel_size=6.5 / 40,
        z_pixel_size=2,
        z_padding=z_padding,
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.45,
        numerical_aperture_detection=0.55,
        invert_phase_contrast=invert_phase_contrast,
    )

    assert H_re.shape == (20 + 2 * z_padding, 100, 101)
    assert H_im.shape == (20 + 2 * z_padding, 100, 101)


# Helper function for testing reconstruction invariances
def simulate_phase_recon(
    z_pixel_size_um=0.1,
    yx_pixel_size_um=6.5 / 63,
):
    z_fov_um = 25
    yx_fov_um = 20

    n_z = np.int32(z_fov_um / z_pixel_size_um)
    n_yx = np.int32(yx_fov_um / yx_pixel_size_um)

    # Parameters
    # all lengths must use consistent units e.g. um
    simulation_arguments = {
        "zyx_shape": (n_z, n_yx, n_yx),
        "yx_pixel_size": yx_pixel_size_um,
        "z_pixel_size": z_pixel_size_um,
        "index_of_refraction_media": 1.3,
    }
    phantom_arguments = {
        "index_of_refraction_sample": 1.40,
        "sphere_radius": 5,
    }
    transfer_function_arguments = {
        "z_padding": 0,
        "wavelength_illumination": 0.532,
        "numerical_aperture_illumination": 0.9,
        "numerical_aperture_detection": 1.3,
    }

    # Create a phantom
    zyx_phase = phase_thick_3d.generate_test_phantom(
        **simulation_arguments, **phantom_arguments
    )

    # Calculate transfer function
    (
        real_potential_transfer_function,
        imag_potential_transfer_function,
    ) = phase_thick_3d.calculate_transfer_function(
        **simulation_arguments, **transfer_function_arguments
    )

    # Simulate
    zyx_data = phase_thick_3d.apply_transfer_function(
        zyx_phase,
        real_potential_transfer_function,
        transfer_function_arguments["z_padding"],
        brightness=1000,
    )

    # Reconstruct
    zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
        zyx_data,
        real_potential_transfer_function,
        imag_potential_transfer_function,
        transfer_function_arguments["z_padding"],
        regularization_strength=1e-3,
    )

    Z, Y, X = zyx_phase.shape
    recon_center = zyx_recon[Z // 2, Y // 2, X // 2].numpy()

    return recon_center


@pytest.mark.parametrize(
    "z_pixel_size_um, yx_pixel_size_um, tolerance",
    [
        (0.1, 6.5 / 63, 0.02),  # baseline
        (0.15, 6.5 / 63, 0.02),  # test z pixel size invariance
        (0.1, 0.8 * 6.5 / 63, 0.02),  # test yx pixel size invariance
    ],
)
def test_phase_invariance(z_pixel_size_um, yx_pixel_size_um, tolerance):
    baseline = simulate_phase_recon()
    recon = simulate_phase_recon(
        z_pixel_size_um=z_pixel_size_um, yx_pixel_size_um=yx_pixel_size_um
    )
    assert np.abs((recon - baseline) / baseline) < tolerance

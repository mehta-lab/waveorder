import numpy as np
import pytest
import torch

from waveorder._pixel_size import YXPixelSize
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
        "wavelength_illumination": 0.532,
        "index_of_refraction_media": 1.3,
    }
    phantom_arguments = {
        "index_of_refraction_sample": 1.40,
        "sphere_radius": 5,
    }
    transfer_function_arguments = {
        "z_padding": 0,
        "numerical_aperture_illumination": 0.9,
        "numerical_aperture_detection": 1.3,
    }

    # Create a phantom
    zyx_phase = phase_thick_3d.generate_test_phantom(**simulation_arguments, **phantom_arguments)

    # Calculate transfer function
    (
        real_potential_transfer_function,
        imag_potential_transfer_function,
    ) = phase_thick_3d.calculate_transfer_function(**simulation_arguments, **transfer_function_arguments)

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
    """Test that the reconstructed physical property (Δn) is invariant to voxel size.

    Reconstruction returns phase in cycles per voxel, which correctly scales with
    voxel size. This test converts back to Δn (the material property) to verify
    that the physical property is recovered invariant to discretization.
    """
    # Baseline with default parameters
    baseline_z_pixel_size_um = 0.1
    baseline = simulate_phase_recon(z_pixel_size_um=baseline_z_pixel_size_um)
    recon = simulate_phase_recon(z_pixel_size_um=z_pixel_size_um, yx_pixel_size_um=yx_pixel_size_um)

    # Convert from cycles per voxel to Δn (refractive index difference)
    # Δn = (cycles/voxel) × λ_medium / z_pixel_size
    wavelength_medium = 0.532 / 1.3  # λ_vacuum / n_media
    baseline_delta_n = baseline * wavelength_medium / baseline_z_pixel_size_um
    recon_delta_n = recon * wavelength_medium / z_pixel_size_um

    # The physical property Δn should be invariant to voxel size
    assert np.abs((recon_delta_n - baseline_delta_n) / baseline_delta_n) < tolerance


def test_calculate_transfer_function_isotropic_yx_pixel_size_equivalence():
    """Anisotropic YXPixelSize with y == x reproduces the legacy scalar output."""
    common = dict(
        zyx_shape=(10, 32, 32),
        z_pixel_size=0.5,
        z_padding=0,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    H_re_scalar, H_im_scalar = phase_thick_3d.calculate_transfer_function(yx_pixel_size=0.2, **common)
    H_re_model, H_im_model = phase_thick_3d.calculate_transfer_function(
        yx_pixel_size=YXPixelSize.isotropic(0.2), **common
    )
    assert torch.allclose(H_re_scalar, H_re_model)
    assert torch.allclose(H_im_scalar, H_im_model)


def test_calculate_transfer_function_anisotropic_runs():
    """y != x produces finite transfer functions of the expected shape."""
    H_re, H_im = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(10, 32, 32),
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_pixel_size=0.5,
        z_padding=0,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert H_re.shape == (10, 32, 32)
    assert H_im.shape == (10, 32, 32)
    assert torch.isfinite(H_re).all()
    assert torch.isfinite(H_im).all()


def test_reconstruct_anisotropic_smoke():
    """End-to-end reconstruct on random data with anisotropic yx pixel size runs."""
    zyx_shape = (10, 32, 32)
    zyx_data = torch.rand(zyx_shape)
    result = phase_thick_3d.reconstruct(
        zyx_data,
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert result.shape == zyx_shape
    assert np.all(np.isfinite(result.numpy()))


def test_reconstruct():
    zyx_shape = (10, 32, 32)
    zyx_data = torch.rand(zyx_shape)

    result = phase_thick_3d.reconstruct(
        zyx_data,
        yx_pixel_size=6.5 / 40,
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )

    assert result.shape == zyx_shape
    assert np.all(np.isfinite(result.numpy()))


def test_reconstruct_apodization_rolloff_smoke():
    """Inverse-filter apodization runs and zeroes the transverse Nyquist content."""
    zyx_shape = (8, 32, 32)
    zyx_data = torch.rand(zyx_shape)

    kwargs = dict(
        yx_pixel_size=0.325,
        z_pixel_size=2.0,
        wavelength_illumination=0.45,
        z_padding=0,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
    )
    phase_hard = phase_thick_3d.reconstruct(zyx_data, **kwargs)
    phase_apod = phase_thick_3d.reconstruct(zyx_data, apodization_rolloff=0.25, **kwargs)

    assert phase_apod.shape == zyx_shape
    assert np.all(np.isfinite(phase_apod.numpy()))

    nyquist_row_apod = np.abs(np.fft.fftn(phase_apod.numpy())[:, 16, :])
    nyquist_row_hard = np.abs(np.fft.fftn(phase_hard.numpy())[:, 16, :])
    assert nyquist_row_apod.max() < 1e-3 * nyquist_row_hard.max()

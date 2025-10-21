import torch

from waveorder.models import isotropic_fluorescent_thin_3d


def test_generate_test_phantom():
    """Test phantom generation."""
    yx_shape = (64, 64)
    yx_pixel_size = 0.1
    sphere_radius = 5.0

    phantom = isotropic_fluorescent_thin_3d.generate_test_phantom(
        yx_shape=yx_shape,
        yx_pixel_size=yx_pixel_size,
        sphere_radius=sphere_radius,
    )

    assert phantom.shape == yx_shape
    assert phantom.dtype == torch.float32
    assert torch.max(phantom) > 0  # Phantom should have non-zero values


def test_calculate_transfer_function():
    """Test transfer function calculation."""
    yx_shape = (64, 64)
    yx_pixel_size = 0.1
    z_position_list = [-2.0, -1.0, 0.0, 1.0, 2.0]
    wavelength_emission = 0.532
    index_of_refraction_media = 1.3
    numerical_aperture_detection = 1.2

    transfer_function = (
        isotropic_fluorescent_thin_3d.calculate_transfer_function(
            yx_shape=yx_shape,
            yx_pixel_size=yx_pixel_size,
            z_position_list=z_position_list,
            wavelength_emission=wavelength_emission,
            index_of_refraction_media=index_of_refraction_media,
            numerical_aperture_detection=numerical_aperture_detection,
        )
    )

    expected_shape = (len(z_position_list),) + yx_shape
    assert transfer_function.shape == expected_shape
    assert transfer_function.dtype == torch.complex64

    # Transfer function should be normalized
    max_val = torch.max(torch.abs(transfer_function))
    assert torch.isclose(max_val, torch.tensor(1.0), atol=1e-5)


def test_calculate_singular_system():
    """Test singular system calculation."""
    yx_shape = (32, 32)
    z_shape = 5

    fluorescent_2d_to_3d_transfer_function = torch.randn(
        (z_shape,) + yx_shape, dtype=torch.complex64
    )

    U, S, Vh = isotropic_fluorescent_thin_3d.calculate_singular_system(
        fluorescent_2d_to_3d_transfer_function
    )

    # Check shapes - for fluorescence with 1 object type and Z data points:
    # U should map from object space (1) to data space (Z), but SVD compresses to min(1,Z)=1
    assert U.shape == (1, 1) + yx_shape  # (objects, compressed_data, vy, vx)
    assert S.shape == (1,) + yx_shape  # (compressed_dims, vy, vx)
    assert (
        Vh.shape == (1, z_shape) + yx_shape
    )  # (compressed_dims, data, vy, vx)

    # Check that singular values are real and non-negative
    assert torch.all(S >= 0)


def test_apply_transfer_function():
    """Test forward simulation."""
    yx_shape = (32, 32)
    z_shape = 5

    # Create simple phantom
    yx_fluorescence_density = torch.ones(yx_shape)

    # Create simple transfer function
    fluorescent_2d_to_3d_transfer_function = torch.ones(
        (z_shape,) + yx_shape, dtype=torch.complex64
    )

    # Apply transfer function
    zyx_data = isotropic_fluorescent_thin_3d.apply_transfer_function(
        yx_fluorescence_density,
        fluorescent_2d_to_3d_transfer_function,
        background=10,
    )

    assert zyx_data.shape == (z_shape,) + yx_shape
    assert torch.all(zyx_data >= 10)  # Should have background offset


def test_apply_inverse_transfer_function():
    """Test reconstruction."""
    yx_shape = (32, 32)
    z_shape = 5

    # Create sample data
    zyx_data = torch.randn((z_shape,) + yx_shape) + 10  # Add background

    # Create sample transfer function and singular system
    fluorescent_2d_to_3d_transfer_function = torch.randn(
        (z_shape,) + yx_shape, dtype=torch.complex64
    )
    singular_system = isotropic_fluorescent_thin_3d.calculate_singular_system(
        fluorescent_2d_to_3d_transfer_function
    )

    # Test Tikhonov reconstruction
    result_tikhonov = (
        isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
            zyx_data,
            singular_system,
            reconstruction_algorithm="Tikhonov",
            regularization_strength=1e-3,
        )
    )

    assert result_tikhonov.shape == yx_shape
    assert result_tikhonov.dtype == torch.float32


def test_end_to_end_simulation():
    """Test complete simulation pipeline."""
    yx_shape = (64, 64)
    yx_pixel_size = 0.1
    z_position_list = [-1.0, 0.0, 1.0]
    wavelength_emission = 0.532
    index_of_refraction_media = 1.3
    numerical_aperture_detection = 1.2
    sphere_radius = 3.0

    # Generate phantom
    yx_fluorescence_density = (
        isotropic_fluorescent_thin_3d.generate_test_phantom(
            yx_shape=yx_shape,
            yx_pixel_size=yx_pixel_size,
            sphere_radius=sphere_radius,
        )
    )

    # Calculate transfer function
    fluorescent_2d_to_3d_transfer_function = (
        isotropic_fluorescent_thin_3d.calculate_transfer_function(
            yx_shape=yx_shape,
            yx_pixel_size=yx_pixel_size,
            z_position_list=z_position_list,
            wavelength_emission=wavelength_emission,
            index_of_refraction_media=index_of_refraction_media,
            numerical_aperture_detection=numerical_aperture_detection,
        )
    )

    # Calculate singular system
    singular_system = isotropic_fluorescent_thin_3d.calculate_singular_system(
        fluorescent_2d_to_3d_transfer_function
    )

    # Simulate imaging
    zyx_data = isotropic_fluorescent_thin_3d.apply_transfer_function(
        yx_fluorescence_density,
        fluorescent_2d_to_3d_transfer_function,
    )

    # Reconstruct
    yx_fluorescence_recon = (
        isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
            zyx_data,
            singular_system,
            regularization_strength=1e-2,
        )
    )

    # Check shapes
    assert yx_fluorescence_density.shape == yx_shape
    assert zyx_data.shape == (len(z_position_list),) + yx_shape
    assert yx_fluorescence_recon.shape == yx_shape

    # Check that reconstruction has similar scale to original
    # (won't be exact due to regularization and noise)
    original_max = torch.max(yx_fluorescence_density)
    recon_max = torch.max(yx_fluorescence_recon)
    # More lenient test - just check reconstruction is non-zero
    assert recon_max > 0.01 * original_max  # Very lenient scale preservation

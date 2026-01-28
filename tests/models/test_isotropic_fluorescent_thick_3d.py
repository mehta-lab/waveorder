import torch

from waveorder import util
from waveorder.models import isotropic_fluorescent_thick_3d


def test_pinhole_aperture_otf_small_diameter():
    """Test that pinhole OTF is broader for smaller diameter."""
    yx_shape = (128, 128)
    yx_pixel_size = 0.1
    radial_frequencies = util.generate_radial_frequencies(
        yx_shape, yx_pixel_size
    )

    # Small pinhole should give broad OTF (higher mean value)
    small_pinhole_otf = (
        isotropic_fluorescent_thick_3d._calculate_pinhole_aperture_otf(
            radial_frequencies, pinhole_diameter=0.1
        )
    )

    # Large pinhole should give narrow OTF (lower mean, more concentrated)
    large_pinhole_otf = (
        isotropic_fluorescent_thick_3d._calculate_pinhole_aperture_otf(
            radial_frequencies, pinhole_diameter=5.0
        )
    )

    # Small pinhole OTF should have higher mean (broader)
    assert small_pinhole_otf.mean() > large_pinhole_otf.mean()


def test_calculate_transfer_function():
    z_padding = 5
    transfer_function = (
        isotropic_fluorescent_thick_3d.calculate_transfer_function(
            zyx_shape=(20, 100, 101),
            yx_pixel_size=6.5 / 40,
            z_pixel_size=2,
            wavelength_emission=0.5,
            z_padding=z_padding,
            index_of_refraction_media=1.0,
            numerical_aperture_detection=0.55,
        )
    )

    assert transfer_function.shape == (20 + 2 * z_padding, 100, 101)


def test_confocal_otf_extended_support():
    """Test that confocal OTF has extended support compared to widefield."""
    z_padding = 0
    zyx_shape = (20, 64, 64)
    params = {
        "zyx_shape": zyx_shape,
        "yx_pixel_size": 0.2,
        "z_pixel_size": 0.5,
        "wavelength_emission": 0.5,
        "z_padding": z_padding,
        "index_of_refraction_media": 1.0,
        "numerical_aperture_detection": 0.55,
    }

    # Widefield OTF
    otf_widefield = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        **params, confocal_pinhole_diameter=None
    )

    # Confocal OTF with small pinhole
    otf_confocal = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        **params, confocal_pinhole_diameter=0.5
    )

    # Confocal OTF should have more high-frequency content (extended support)
    threshold = 0.01
    widefield_support = (torch.abs(otf_widefield) > threshold).sum()
    confocal_support = (torch.abs(otf_confocal) > threshold).sum()
    assert confocal_support > widefield_support


def test_apply_inverse_transfer_function():
    # Create sample data
    zyx_data = torch.randn(10, 5, 5)
    z_padding = 2
    optical_transfer_function = torch.randn(10 + 2 * z_padding, 5, 5)

    # Test Tikhonov method
    result_tikhonov = (
        isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
            zyx_data,
            optical_transfer_function,
            z_padding,
            reconstruction_algorithm="Tikhonov",
            regularization_strength=1e-3,
        )
    )
    assert result_tikhonov.shape == (10, 5, 5)

    # TODO: Fix TV method
    # result_tv = isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
    #    zyx_data,
    #    optical_transfer_function,
    #    z_padding,
    #    method="TV",
    #    reg_re=1e-3,
    #    rho=1e-3,
    #    itr=10,
    # )
    # assert result_tv.shape == (10, 5, 5)

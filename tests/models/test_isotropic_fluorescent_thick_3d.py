import pytest
import torch

from waveorder.models import isotropic_fluorescent_thick_3d


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

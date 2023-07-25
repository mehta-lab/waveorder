import pytest
import torch
from waveorder.models import isotropic_thin_3d


@pytest.mark.parametrize("axial_flip", (True, False))
def test_calculate_transfer_function(axial_flip):
    Hu, Hp = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(100, 101),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1, 0, 1],
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
        axial_flip=axial_flip,
    )

    assert Hu.shape == (3, 100, 101)
    assert Hp.shape == (3, 100, 101)

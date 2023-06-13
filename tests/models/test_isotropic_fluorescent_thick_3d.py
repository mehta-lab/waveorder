import pytest
from waveorder.models import isotropic_fluorescent_thick_3d


@pytest.mark.parametrize("axial_flip", (True, False))
def test_calculate_transfer_function(axial_flip):
    z_padding = 5
    transfer_function = (
        isotropic_fluorescent_thick_3d.calculate_transfer_function(
            zyx_shape=(20, 100, 101),
            yx_pixel_size=6.5 / 40,
            z_pixel_size=2,
            wavelength_illumination=0.5,
            z_padding=z_padding,
            index_of_refraction_media=1.0,
            numerical_aperture_detection=0.55,
            axial_flip=axial_flip,
        )
    )

    assert transfer_function.shape == (20 + 2 * z_padding, 100, 101)

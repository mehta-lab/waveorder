from waveorder.models import phase3D_3D


def test_calc_TF():
    Z_pad = 5
    H_re, H_im = phase3D_3D.calculate_transfer_function(
        zyx_shape=(20, 100, 101),
        yx_pixel_size=6.5 / 40,
        z_pixel_size=2,
        z_padding=Z_pad,
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.45,
        numerical_aperture_detection=0.55,
    )

    assert H_re.shape == (20 + 2 * Z_pad, 100, 101)
    assert H_im.shape == (20 + 2 * Z_pad, 100, 101)

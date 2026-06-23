"""Anisotropic yx_pixel_size guards on the birefringence vector model."""

import pytest

from waveorder._pixel_size import YXPixelSize
from waveorder.models import inplane_oriented_thick_pol3d_vector


def _common_kwargs():
    return dict(
        swing=0.1,
        scheme="5-State",
        zyx_shape=(4, 16, 16),
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )


def test_isotropic_yx_pixel_size_runs():
    """An isotropic YXPixelSize equal in y and x is accepted."""
    result = inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
        yx_pixel_size=YXPixelSize.isotropic(0.2),
        **_common_kwargs(),
    )
    # Result is a tuple (sfZYX_tf, intensity_to_stokes, *extras); just check non-empty.
    assert result is not None


def test_anisotropic_yx_pixel_size_raises_not_implemented():
    """Anisotropic yx_pixel_size on the birefringence vector model is unsupported."""
    with pytest.raises(NotImplementedError, match="Anisotropic"):
        inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
            yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
            **_common_kwargs(),
        )


def test_scalar_yx_pixel_size_still_accepted():
    """Legacy scalar yx_pixel_size keeps working."""
    result = inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
        yx_pixel_size=0.2,
        **_common_kwargs(),
    )
    assert result is not None

import numpy as np
import pytest
import torch

from waveorder.models import isotropic_thin_3d


@pytest.mark.parametrize("invert_phase_contrast", (True, False))
def test_calculate_transfer_function(invert_phase_contrast):
    Hu, Hp = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(100, 101),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1, 0, 1],
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
        invert_phase_contrast=invert_phase_contrast,
    )

    assert Hu.shape == (3, 100, 101)
    assert Hp.shape == (3, 100, 101)


def test_reconstruct():
    yx_shape = (32, 32)
    z_position_list = [-1.0, 0.0, 1.0]
    zyx_data = torch.rand((len(z_position_list),) + yx_shape)

    absorption, phase = isotropic_thin_3d.reconstruct(
        zyx_data,
        yx_pixel_size=6.5 / 40,
        z_position_list=z_position_list,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )

    assert absorption.shape == yx_shape
    assert phase.shape == yx_shape
    assert np.all(np.isfinite(absorption.numpy()))
    assert np.all(np.isfinite(phase.numpy()))

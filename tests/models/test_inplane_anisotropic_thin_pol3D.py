import pytest
import torch
from waveorder import stokes
from waveorder.models import inplane_anisotropic_thin_pol3d


def test_calculate_transfer_function():
    (
        intensity_to_stokes_matrix,
        inverse_background_mueller,
    ) = inplane_anisotropic_thin_pol3d.calculate_transfer_function(
        swing=0.1,
        polarized_illumination_scheme="5-State",
        no_sample_intensities=torch.rand((5, 128, 128)),
        with_sample_intensities=torch.rand((5, 10, 128, 128)),
    )

    assert intensity_to_stokes_matrix.shape == (4, 5)
    assert inverse_background_mueller.shape == (4, 4, 128, 128)

    # Test incompatible shape
    with pytest.raises(ValueError):
        inplane_anisotropic_thin_pol3d.calculate_transfer_function(
            swing=0.1,
            polarized_illumination_scheme="5-State",
            no_sample_intensities=torch.rand((5, 128, 128)),
            with_sample_intensities=torch.rand((5, 10, 129, 128)),
        )

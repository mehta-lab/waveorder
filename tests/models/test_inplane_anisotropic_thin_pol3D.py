import pytest
import torch
from waveorder import stokes
from waveorder.models import inplane_anisotropic_thin_pol3d


def test_calculate_transfer_function():
    intensity_to_stokes_matrix = (
        inplane_anisotropic_thin_pol3d.calculate_transfer_function(
            swing=0.1,
            scheme="5-State",
        )
    )

    assert intensity_to_stokes_matrix.shape == (4, 5)

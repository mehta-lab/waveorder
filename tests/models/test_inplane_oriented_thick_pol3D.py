import pytest
import torch

from waveorder import stokes
from waveorder.models import inplane_oriented_thick_pol3d


def test_calculate_transfer_function():
    intensity_to_stokes_matrix = (
        inplane_oriented_thick_pol3d.calculate_transfer_function(
            swing=0.1,
            scheme="5-State",
        )
    )

    assert intensity_to_stokes_matrix.shape == (4, 5)


def test_apply_inverse_transfer_function():
    input_shape = (5, 10, 5, 5)
    czyx_data = torch.rand(input_shape)

    intensity_to_stokes_matrix = (
        inplane_oriented_thick_pol3d.calculate_transfer_function(
            swing=0.1,
            scheme="5-State",
        )
    )

    results = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
        czyx_data=czyx_data,
        intensity_to_stokes_matrix=intensity_to_stokes_matrix,
    )

    assert len(results) == 4

    for result in results:
        assert result.shape == input_shape[1:]

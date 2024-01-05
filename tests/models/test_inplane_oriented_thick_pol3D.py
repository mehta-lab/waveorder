import pytest
import torch

from tests.conftest import _DEVICE
from waveorder.models import inplane_oriented_thick_pol3d


def test_calculate_transfer_function():
    intensity_to_stokes_matrix = (
        inplane_oriented_thick_pol3d.calculate_transfer_function(
            swing=0.1,
            scheme="5-State",
        )
    )

    assert intensity_to_stokes_matrix.shape == (4, 5)


@pytest.mark.parametrize(*_DEVICE)
@pytest.mark.parametrize("estimate_bg", [True, False])
def test_apply_inverse_transfer_function(device, estimate_bg):
    input_shape = (5, 10, 100, 100)
    czyx_data = torch.rand(input_shape, device=device)

    intensity_to_stokes_matrix = (
        inplane_oriented_thick_pol3d.calculate_transfer_function(
            swing=0.1,
            scheme="5-State",
        ).to(device)
    )

    results = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
        czyx_data=czyx_data,
        intensity_to_stokes_matrix=intensity_to_stokes_matrix,
        remove_estimated_background=estimate_bg,
    )

    assert len(results) == 4

    for result in results:
        assert result.shape == input_shape[1:]
        assert result.device.type == device

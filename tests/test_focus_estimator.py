import numpy as np
import pytest
import torch

from waveorder import focus


def test_focus_estimator(tmp_path):
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4, 5)), NA_det, lambda_ill, ps
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)),
            NA_det,
            lambda_ill,
            ps,
            midband_fractions=(-1, 0.5),
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)),
            NA_det,
            lambda_ill,
            ps,
            midband_fractions=(0.75, 0.5),
        )

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(
            np.zeros((2, 3, 4)), NA_det, lambda_ill, ps, mode="maxx"
        )

    plot_path = tmp_path.joinpath("test.pdf")
    data3D = np.random.random((11, 256, 256))
    slice = focus.focus_from_transverse_band(
        data3D, NA_det, lambda_ill, ps, plot_path=str(plot_path)
    )
    assert slice >= 0
    assert slice <= data3D.shape[0]
    assert plot_path.exists()

    # Check single slice
    slice = focus.focus_from_transverse_band(
        np.random.random((1, 10, 10)),
        NA_det,
        lambda_ill,
        ps,
    )
    assert slice == 0


def test_focus_estimator_snr(tmp_path):
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    z = np.linspace(-1, 1, 21)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    phantom = (np.sqrt(xx**2 + yy**2 + zz**2) < 0.5).astype(np.uint16)

    for snr in [1000, 100, 10, 1, 0.1]:
        np.random.seed(1)
        data = np.random.poisson(
            phantom * np.sqrt(snr), size=phantom.shape
        ) + np.random.normal(loc=0, scale=3, size=phantom.shape)

        plot_path = tmp_path / f"test-{snr}.pdf"
        slice = focus.focus_from_transverse_band(
            data,
            NA_det,
            lambda_ill,
            ps,
            plot_path=plot_path,
            threshold_FWHM=5,
        )
        assert plot_path.exists()
        if slice is not None:
            assert np.abs(slice - 10) <= 2


def test_compute_midband_power():
    """Test the compute_midband_power function with torch tensors."""
    # Test parameters
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4
    midband_fractions = (0.125, 0.25)

    # Create test data
    np.random.seed(42)
    test_2d_np = np.random.random((64, 64)).astype(np.float32)
    test_2d_torch = torch.from_numpy(test_2d_np)

    # Test the compute_midband_power function
    result = focus.compute_midband_power(
        test_2d_torch, NA_det, lambda_ill, ps, midband_fractions
    )

    # Check result properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])  # scalar tensor
    assert result.item() > 0  # should be positive

    # Test with different midband fractions
    result2 = focus.compute_midband_power(
        test_2d_torch, NA_det, lambda_ill, ps, (0.1, 0.3)
    )
    assert isinstance(result2, torch.Tensor)
    assert result2.item() > 0

    # Results should be different for different bands
    assert abs(result.item() - result2.item()) > 1e-6


def test_compute_midband_power_consistency():
    """Test that compute_midband_power is consistent with focus_from_transverse_band."""
    # Test parameters
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4
    midband_fractions = (0.125, 0.25)

    # Create 3D test data
    np.random.seed(42)
    test_3d = np.random.random((3, 32, 32)).astype(np.float32)

    # Test focus_from_transverse_band still works
    focus_slice = focus.focus_from_transverse_band(
        test_3d, NA_det, lambda_ill, ps, midband_fractions
    )

    assert isinstance(focus_slice, (int, np.integer))
    assert 0 <= focus_slice < test_3d.shape[0]

    # Manually compute midband power for each slice
    manual_powers = []
    for z in range(test_3d.shape[0]):
        power = focus.compute_midband_power(
            torch.from_numpy(test_3d[z]),
            NA_det,
            lambda_ill,
            ps,
            midband_fractions,
        )
        manual_powers.append(power.item())

    expected_focus_slice = np.argmax(manual_powers)
    assert focus_slice == expected_focus_slice

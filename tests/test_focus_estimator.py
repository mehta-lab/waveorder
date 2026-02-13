import numpy as np
import pytest
import torch

from waveorder import focus


def test_focus_estimator(tmp_path):
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    with pytest.raises(ValueError):
        focus.focus_from_transverse_band(np.zeros((2, 3, 4, 5)), NA_det, lambda_ill, ps)

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
        focus.focus_from_transverse_band(np.zeros((2, 3, 4)), NA_det, lambda_ill, ps, mode="maxx")

    plot_path = tmp_path.joinpath("test.pdf")
    data3D = np.random.random((11, 256, 256))
    slice = focus.focus_from_transverse_band(data3D, NA_det, lambda_ill, ps, plot_path=str(plot_path))
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
        data = np.random.poisson(phantom * np.sqrt(snr), size=phantom.shape) + np.random.normal(
            loc=0, scale=3, size=phantom.shape
        )

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
    result = focus.compute_midband_power(test_2d_torch, NA_det, lambda_ill, ps, midband_fractions)

    # Check result properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])  # scalar tensor
    assert result.item() > 0  # should be positive

    # Test with different midband fractions
    result2 = focus.compute_midband_power(test_2d_torch, NA_det, lambda_ill, ps, (0.1, 0.3))
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
    focus_slice = focus.focus_from_transverse_band(test_3d, NA_det, lambda_ill, ps, midband_fractions)

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


def test_focus_from_transverse_band_with_statistics():
    """Test focus_from_transverse_band with return_statistics=True."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    # Create test data
    np.random.seed(42)
    test_3d = np.random.random((5, 32, 32)).astype(np.float32)

    # Test without statistics (backward compatibility)
    focus_slice = focus.focus_from_transverse_band(test_3d, NA_det, lambda_ill, ps)
    assert isinstance(focus_slice, (int, np.integer, type(None)))

    # Test with statistics
    focus_slice_stats, stats = focus.focus_from_transverse_band(test_3d, NA_det, lambda_ill, ps, return_statistics=True)

    # Check that both return the same index
    assert focus_slice == focus_slice_stats

    # Check statistics structure
    assert isinstance(stats, dict)
    assert "peak_index" in stats
    assert "peak_FWHM" in stats
    assert isinstance(stats["peak_index"], int)
    assert isinstance(stats["peak_FWHM"], float)

    # Test with threshold
    idx_thresh, stats_thresh = focus.focus_from_transverse_band(
        test_3d,
        NA_det,
        lambda_ill,
        ps,
        threshold_FWHM=100.0,
        return_statistics=True,
    )
    # High threshold should result in None
    assert idx_thresh is None

    # Test single slice case with statistics
    single_slice = np.random.random((1, 32, 32)).astype(np.float32)
    idx_single, stats_single = focus.focus_from_transverse_band(
        single_slice, NA_det, lambda_ill, ps, return_statistics=True
    )
    assert idx_single == 0
    assert stats_single["peak_index"] is None
    assert stats_single["peak_FWHM"] is None


def test_subpixel_precision():
    """Test that sub-pixel precision returns float values when enabled."""
    # Test parameters
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    # Create synthetic test data with a clear peak between slices
    z_size, y_size, x_size = 11, 64, 64
    x = np.linspace(-1, 1, x_size)
    y = np.linspace(-1, 1, y_size)
    z = np.linspace(-5, 5, z_size)

    # Create a 3D Gaussian that peaks between slice indices
    test_data = np.zeros((z_size, y_size, x_size))
    true_peak_z = 5.3  # Peak between slices 5 and 6

    for i, z_val in enumerate(z):
        # Create Gaussian centered at true_peak_z position in physical space
        gaussian_2d = np.exp(-((x[None, :] ** 2 + y[:, None] ** 2) + (z_val - (true_peak_z - 5)) ** 2))
        test_data[i] = gaussian_2d

    # Test without sub-pixel precision (should return integer)
    focus_slice_int = focus.focus_from_transverse_band(
        test_data,
        NA_det,
        lambda_ill,
        ps,
        polynomial_fit_order=4,
        enable_subpixel_precision=False,
    )
    assert isinstance(focus_slice_int, (int, np.integer))

    # Test with sub-pixel precision (should return float)
    focus_slice_float = focus.focus_from_transverse_band(
        test_data,
        NA_det,
        lambda_ill,
        ps,
        polynomial_fit_order=4,
        enable_subpixel_precision=True,
    )

    # Should return a float
    assert isinstance(focus_slice_float, float)

    # Should be close to the true peak position
    assert abs(focus_slice_float - true_peak_z) < 1.0  # Within 1 slice

    # Sub-pixel result should be different from integer result
    assert focus_slice_float != focus_slice_int


def test_subpixel_precision_backward_compatibility():
    """Test that default behavior (integer results) is preserved."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    # Create simple test data
    test_data = np.random.random((5, 32, 32)).astype(np.float32)

    # Test default behavior (should return integer)
    focus_slice = focus.focus_from_transverse_band(
        test_data,
        NA_det,
        lambda_ill,
        ps,
        polynomial_fit_order=4,
    )

    assert isinstance(focus_slice, (int, np.integer))


def test_subpixel_precision_with_plotting(tmp_path):
    """Test that sub-pixel precision works with plotting."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    # Create test data
    test_data = np.random.random((7, 32, 32)).astype(np.float32)
    plot_path = tmp_path / "subpixel_test.pdf"

    # Should work without errors
    focus_slice = focus.focus_from_transverse_band(
        test_data,
        NA_det,
        lambda_ill,
        ps,
        polynomial_fit_order=4,
        enable_subpixel_precision=True,
        plot_path=str(plot_path),
    )

    assert isinstance(focus_slice, float)
    assert plot_path.exists()


def test_compute_midband_power_3d():
    """Test that compute_midband_power handles (Z, Y, X) input."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    np.random.seed(42)
    data_3d = torch.from_numpy(np.random.random((7, 64, 64)).astype(np.float32))

    result = focus.compute_midband_power(data_3d, NA_det, lambda_ill, ps)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)
    assert (result > 0).all()


def test_focus_from_transverse_band_tensor():
    """Test that focus_from_transverse_band accepts a torch Tensor."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    np.random.seed(42)
    data_np = np.random.random((11, 64, 64)).astype(np.float32)
    data_tensor = torch.from_numpy(data_np)

    result_np = focus.focus_from_transverse_band(data_np, NA_det, lambda_ill, ps)
    result_tensor = focus.focus_from_transverse_band(data_tensor, NA_det, lambda_ill, ps)
    assert result_np == result_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_focus_from_transverse_band_gpu():
    """Test focus_from_transverse_band with a GPU tensor."""
    ps = 6.5 / 100
    lambda_ill = 0.532
    NA_det = 1.4

    np.random.seed(42)
    data_np = np.random.random((7, 64, 64)).astype(np.float32)

    cpu_result = focus.focus_from_transverse_band(data_np, NA_det, lambda_ill, ps)
    gpu_result = focus.focus_from_transverse_band(torch.from_numpy(data_np).cuda(), NA_det, lambda_ill, ps)
    assert cpu_result == gpu_result


def test_z_focus_offset_float_type():
    """Test that z_focus_offset can accept float values in settings."""
    from waveorder.cli.settings import FourierTransferFunctionSettings

    # Test that float values are accepted
    settings = FourierTransferFunctionSettings(z_focus_offset=1.5)
    assert settings.z_focus_offset == 1.5
    assert isinstance(settings.z_focus_offset, float)

    # Test that "auto" still works
    settings_auto = FourierTransferFunctionSettings(z_focus_offset="auto")
    assert settings_auto.z_focus_offset == "auto"

    # Test that integers are converted to float
    settings_int = FourierTransferFunctionSettings(z_focus_offset=2)
    assert settings_int.z_focus_offset == 2
    assert isinstance(settings_int.z_focus_offset, (int, float))


def test_position_list_with_float_offset():
    """Test that _position_list_from_shape_scale_offset works correctly with float offsets."""
    from waveorder.api._utils import _position_list_from_shape_scale_offset

    # Test integer offset
    pos_int = _position_list_from_shape_scale_offset(5, 1.0, 0)
    expected_int = [2.0, 1.0, 0.0, -1.0, -2.0]
    assert pos_int == expected_int

    # Test float offset
    pos_float = _position_list_from_shape_scale_offset(5, 1.0, 0.5)
    expected_float = [2.5, 1.5, 0.5, -0.5, -1.5]
    assert pos_float == expected_float

    # Verify the difference is exactly the offset
    import numpy as np

    diff = np.array(pos_float) - np.array(pos_int)
    assert np.allclose(diff, 0.5)

    # Test with different scale and offset
    pos_scaled = _position_list_from_shape_scale_offset(4, 2.0, 0.3)
    # shape=4, shape//2=2, so indices are [0,1,2,3],
    # positions are [(-0+2+0.3)*2, (-1+2+0.3)*2, (-2+2+0.3)*2, (-3+2+0.3)*2] = [4.6, 2.6, 0.6, -1.4]
    expected_scaled = [4.6, 2.6, 0.6, -1.4]
    assert np.allclose(pos_scaled, expected_scaled)

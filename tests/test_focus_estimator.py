import pytest
import numpy as np
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
        data3D, ps, lambda_ill, NA_det, plot_path=str(plot_path)
    )
    assert slice >= 0
    assert slice <= data3D.shape[0]
    assert plot_path.exists()


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
            ps,
            lambda_ill,
            NA_det,
            plot_path=plot_path,
            threshold_FWHM=5,
        )
        assert plot_path.exists()
        if slice is not None:
            assert np.abs(slice - 10) <= 2

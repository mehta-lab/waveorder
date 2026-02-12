import numpy as np
import xarray as xr

from waveorder.api import birefringence, fluorescence, phase
from waveorder.api._utils import _output_channel_names

ZYX_SHAPE = (3, 4, 5)


# --- _output_channel_names ---


class TestOutputChannelNames:
    def test_biref_only(self):
        names = _output_channel_names(recon_biref=True)
        assert names == [
            "Retardance",
            "Orientation",
            "Transmittance",
            "Depolarization",
        ]

    def test_phase_2d(self):
        names = _output_channel_names(recon_phase=True, recon_dim=2)
        assert names == ["Phase2D"]

    def test_phase_3d(self):
        names = _output_channel_names(recon_phase=True, recon_dim=3)
        assert names == ["Phase3D"]

    def test_biref_and_phase_2d(self):
        names = _output_channel_names(
            recon_biref=True, recon_phase=True, recon_dim=2
        )
        assert names == [
            "Retardance",
            "Orientation",
            "Transmittance",
            "Depolarization",
            "Phase2D",
        ]
        # No joint decon channels for 2D
        assert "Retardance_Joint_Decon" not in names

    def test_biref_and_phase_3d(self):
        names = _output_channel_names(
            recon_biref=True, recon_phase=True, recon_dim=3
        )
        assert names == [
            "Retardance",
            "Orientation",
            "Transmittance",
            "Depolarization",
            "Phase3D",
            "Retardance_Joint_Decon",
            "Orientation_Joint_Decon",
            "Phase_Joint_Decon",
        ]

    def test_fluorescence_2d(self):
        names = _output_channel_names(
            recon_fluo=True, recon_dim=2, fluor_channel_name="GFP"
        )
        assert names == ["GFP_Density2D"]

    def test_fluorescence_3d(self):
        names = _output_channel_names(
            recon_fluo=True, recon_dim=3, fluor_channel_name="GFP"
        )
        assert names == ["GFP_Density3D"]

    def test_no_recon(self):
        assert _output_channel_names() == []


# --- birefringence reconstruction ---


def test_birefringence_returns_xarray(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    settings = birefringence.Settings()

    tf_ds = birefringence.compute_transfer_function(
        czyx, settings, [f"ch{i}" for i in range(4)]
    )

    result = birefringence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=3,
        settings=settings,
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == [
        "Retardance",
        "Orientation",
        "Transmittance",
        "Depolarization",
    ]
    assert result.shape[1:] == ZYX_SHAPE


def test_birefringence_2d_singleton_z(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    settings = birefringence.Settings()

    tf_ds = birefringence.compute_transfer_function(
        czyx, settings, [f"ch{i}" for i in range(4)]
    )

    result = birefringence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=2,
        settings=settings,
    )

    assert result.sizes["z"] == 1


def test_birefringence_inherits_yx_coords(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    settings = birefringence.Settings()

    tf_ds = birefringence.compute_transfer_function(
        czyx, settings, [f"ch{i}" for i in range(4)]
    )

    result = birefringence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=3,
        settings=settings,
    )

    np.testing.assert_array_equal(
        result.coords["y"].values, czyx.coords["y"].values
    )
    np.testing.assert_array_equal(
        result.coords["x"].values, czyx.coords["x"].values
    )


# --- phase reconstruction ---


def test_phase_3d_returns_xarray(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = phase.Settings()

    tf_ds = phase.compute_transfer_function(czyx, 3, settings)

    result = phase.apply_inverse_transfer_function(
        czyx, tf_ds, recon_dim=3, settings=settings
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == ["Phase3D"]
    assert result.shape == (1, *ZYX_SHAPE)


def test_phase_2d_returns_xarray(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = phase.Settings()

    tf_ds = phase.compute_transfer_function(czyx, 2, settings)

    result = phase.apply_inverse_transfer_function(
        czyx, tf_ds, recon_dim=2, settings=settings
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == ["Phase2D"]
    assert result.sizes["z"] == 1


# --- fluorescence reconstruction ---


def test_fluorescence_3d_returns_xarray(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = fluorescence.Settings()

    tf_ds = fluorescence.compute_transfer_function(czyx, 3, settings)

    result = fluorescence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=3,
        settings=settings,
        fluor_channel_name="GFP",
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == ["GFP_Density3D"]
    assert result.shape == (1, *ZYX_SHAPE)


def test_fluorescence_2d_returns_xarray(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = fluorescence.Settings()

    tf_ds = fluorescence.compute_transfer_function(czyx, 2, settings)

    result = fluorescence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=2,
        settings=settings,
        fluor_channel_name="GFP",
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == ["GFP_Density2D"]
    assert result.sizes["z"] == 1


# --- roundtrip: compute TF -> apply inverse ---


def test_phase_3d_roundtrip_finite(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = phase.Settings()

    tf_ds = phase.compute_transfer_function(czyx, 3, settings)
    result = phase.apply_inverse_transfer_function(
        czyx, tf_ds, recon_dim=3, settings=settings
    )

    assert np.all(np.isfinite(result.values))


def test_fluorescence_3d_roundtrip_finite(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    settings = fluorescence.Settings()

    tf_ds = fluorescence.compute_transfer_function(czyx, 3, settings)
    result = fluorescence.apply_inverse_transfer_function(
        czyx,
        tf_ds,
        recon_dim=3,
        settings=settings,
        fluor_channel_name="GFP",
    )

    assert np.all(np.isfinite(result.values))

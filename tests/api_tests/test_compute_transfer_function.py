import numpy as np
import pytest
import xarray as xr

from waveorder.api import (
    birefringence,
    birefringence_and_phase,
    fluorescence,
    phase,
)
from waveorder.api._utils import _position_list_from_shape_scale_offset

ZYX_SHAPE = (3, 4, 5)


# --- _position_list_from_shape_scale_offset ---


@pytest.mark.parametrize(
    "shape, scale, offset, expected",
    [
        (5, 1.0, 0.0, [2.0, 1.0, 0.0, -1.0, -2.0]),
        (4, 0.5, 1.0, [1.5, 1.0, 0.5, 0.0]),
        (5, 1.0, 0.5, [2.5, 1.5, 0.5, -0.5, -1.5]),
        (4, 2.0, 0.3, [4.6, 2.6, 0.6, -1.4]),
    ],
)
def test_position_list_from_shape_scale_offset(shape, scale, offset, expected):
    result = _position_list_from_shape_scale_offset(shape, scale, offset)
    np.testing.assert_allclose(result, expected)


# --- birefringence ---


def test_birefringence_returns_dataset(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = birefringence.compute_transfer_function(
        czyx,
        birefringence.Settings(),
        [f"State{i}" for i in range(4)],
    )
    assert isinstance(tf_ds, xr.Dataset)
    assert "intensity_to_stokes_matrix" in tf_ds


def test_birefringence_stokes_matrix_shape(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = birefringence.compute_transfer_function(
        czyx,
        birefringence.Settings(),
        [f"State{i}" for i in range(4)],
    )
    mat = tf_ds["intensity_to_stokes_matrix"].values
    assert mat.ndim == 2
    assert mat.shape[0] == 4  # Stokes parameters
    assert mat.shape[1] == 4  # input channels


# --- phase 3D ---


def test_phase_3d_returns_dataset(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 3, phase.Settings())
    assert isinstance(tf_ds, xr.Dataset)
    assert "real_potential_transfer_function" in tf_ds
    assert "imaginary_potential_transfer_function" in tf_ds


def test_phase_3d_shapes(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 3, phase.Settings())
    real_tf = tf_ds["real_potential_transfer_function"].values
    imag_tf = tf_ds["imaginary_potential_transfer_function"].values
    assert real_tf.shape == ZYX_SHAPE
    assert imag_tf.shape == ZYX_SHAPE


def test_phase_3d_no_singular_system(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 3, phase.Settings())
    assert "singular_system_U" not in tf_ds
    assert "singular_system_S" not in tf_ds
    assert "singular_system_Vh" not in tf_ds


# --- phase 2D ---


def test_phase_2d_returns_singular_system(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 2, phase.Settings())
    assert isinstance(tf_ds, xr.Dataset)
    assert "singular_system_U" in tf_ds
    assert "singular_system_S" in tf_ds
    assert "singular_system_Vh" in tf_ds


def test_phase_2d_no_real_imag(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 2, phase.Settings())
    assert "real_potential_transfer_function" not in tf_ds
    assert "imaginary_potential_transfer_function" not in tf_ds


# --- fluorescence 3D ---


def test_fluorescence_3d_returns_dataset(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 3, fluorescence.Settings())
    assert isinstance(tf_ds, xr.Dataset)
    assert "optical_transfer_function" in tf_ds


def test_fluorescence_3d_shape(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 3, fluorescence.Settings())
    otf = tf_ds["optical_transfer_function"].values
    assert otf.shape == ZYX_SHAPE


def test_fluorescence_3d_no_singular_system(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 3, fluorescence.Settings())
    assert "singular_system_U" not in tf_ds


# --- fluorescence 2D ---


def test_fluorescence_2d_returns_singular_system(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 2, fluorescence.Settings())
    assert isinstance(tf_ds, xr.Dataset)
    assert "singular_system_U" in tf_ds
    assert "singular_system_S" in tf_ds
    assert "singular_system_Vh" in tf_ds


def test_fluorescence_2d_no_otf(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 2, fluorescence.Settings())
    assert "optical_transfer_function" not in tf_ds


# --- birefringence_and_phase ---


def test_birefringence_and_phase_3d(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = birefringence_and_phase.compute_transfer_function(
        czyx,
        birefringence.Settings(),
        phase.Settings(),
        [f"State{i}" for i in range(4)],
        recon_dim=3,
    )
    assert isinstance(tf_ds, xr.Dataset)
    assert "intensity_to_stokes_matrix" in tf_ds
    assert "vector_transfer_function" in tf_ds
    assert "vector_singular_system_U" in tf_ds
    assert "vector_singular_system_S" in tf_ds
    assert "vector_singular_system_Vh" in tf_ds
    assert "real_potential_transfer_function" in tf_ds
    assert "imaginary_potential_transfer_function" in tf_ds


def test_birefringence_and_phase_2d(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = birefringence_and_phase.compute_transfer_function(
        czyx,
        birefringence.Settings(),
        phase.Settings(),
        [f"State{i}" for i in range(4)],
        recon_dim=2,
    )
    assert isinstance(tf_ds, xr.Dataset)
    assert "intensity_to_stokes_matrix" in tf_ds
    assert "vector_transfer_function" in tf_ds
    assert "vector_singular_system_U" in tf_ds
    # 2D should NOT have real/imag potential TFs
    assert "real_potential_transfer_function" not in tf_ds
    assert "imaginary_potential_transfer_function" not in tf_ds


# --- roundtrip: compute TF then extract tensors ---


def test_phase_3d_tensors_are_finite(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = phase.compute_transfer_function(czyx, 3, phase.Settings())
    assert np.all(np.isfinite(tf_ds["real_potential_transfer_function"].values))
    assert np.all(np.isfinite(tf_ds["imaginary_potential_transfer_function"].values))


def test_fluorescence_3d_otf_is_finite(make_czyx):
    czyx = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx, 3, fluorescence.Settings())
    assert np.all(np.isfinite(tf_ds["optical_transfer_function"].values))

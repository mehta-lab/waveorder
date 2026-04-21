"""Tests for the --save-all cleanup behavior in benchmarks.runner."""

from benchmarks.runner import SIZE_LIMIT_BYTES, _cleanup_large_outputs, _dir_size_bytes


def _make_zarr_dir(path, size_bytes):
    """Write a fake zarr directory with a file of approximately size_bytes."""
    path.mkdir(parents=True)
    (path / "0").write_bytes(b"x" * size_bytes)


class TestCleanupLargeOutputs:
    def test_save_all_keeps_everything(self, tmp_path):
        _make_zarr_dir(tmp_path / "transfer_function_cfg.zarr", 100)
        _make_zarr_dir(tmp_path / "reconstruction.zarr", SIZE_LIMIT_BYTES * 2)
        _make_zarr_dir(tmp_path / "simulated.zarr", SIZE_LIMIT_BYTES * 2)
        _make_zarr_dir(tmp_path / "cropped_input.zarr", 100)
        _cleanup_large_outputs(tmp_path, save_all=True)
        assert (tmp_path / "transfer_function_cfg.zarr").exists()
        assert (tmp_path / "reconstruction.zarr").exists()
        assert (tmp_path / "simulated.zarr").exists()
        assert (tmp_path / "cropped_input.zarr").exists()

    def test_default_deletes_tf(self, tmp_path):
        _make_zarr_dir(tmp_path / "transfer_function_cfg.zarr", 100)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert not (tmp_path / "transfer_function_cfg.zarr").exists()

    def test_default_deletes_cropped_input(self, tmp_path):
        _make_zarr_dir(tmp_path / "cropped_input.zarr", 100)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert not (tmp_path / "cropped_input.zarr").exists()

    def test_default_keeps_small_recon(self, tmp_path):
        _make_zarr_dir(tmp_path / "reconstruction.zarr", 100)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert (tmp_path / "reconstruction.zarr").exists()

    def test_default_deletes_large_recon(self, tmp_path):
        _make_zarr_dir(tmp_path / "reconstruction.zarr", SIZE_LIMIT_BYTES + 1)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert not (tmp_path / "reconstruction.zarr").exists()

    def test_default_deletes_large_simulated(self, tmp_path):
        _make_zarr_dir(tmp_path / "simulated.zarr", SIZE_LIMIT_BYTES + 1)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert not (tmp_path / "simulated.zarr").exists()

    def test_default_keeps_small_simulated(self, tmp_path):
        _make_zarr_dir(tmp_path / "simulated.zarr", 100)
        _cleanup_large_outputs(tmp_path, save_all=False)
        assert (tmp_path / "simulated.zarr").exists()


class TestDirSizeBytes:
    def test_missing(self, tmp_path):
        assert _dir_size_bytes(tmp_path / "missing") == 0

    def test_empty(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert _dir_size_bytes(tmp_path / "empty") == 0

    def test_sums_recursive(self, tmp_path):
        d = tmp_path / "data"
        (d / "sub").mkdir(parents=True)
        (d / "a").write_bytes(b"x" * 100)
        (d / "sub" / "b").write_bytes(b"x" * 50)
        assert _dir_size_bytes(d) == 150

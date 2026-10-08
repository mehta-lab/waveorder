import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from iohub.ngff import open_ome_zarr
from platformdirs import user_data_dir
from wget import download

from waveorder.cli import settings

_MM_OME_TIFF_ZIP_URL = "https://zenodo.org/record/6983916/files/waveOrder_test_data.zip"


@pytest.fixture
def make_czyx():
    """Factory fixture for creating synthetic CZYX test DataArrays."""

    def _make(zyx_shape=(5, 32, 32), n_channels=1):
        rng = np.random.default_rng(42)
        data = rng.random((n_channels,) + zyx_shape, dtype=np.float32)
        z_pixel_size = 2.0
        yx_pixel_size = 6.5 / 20

        return xr.DataArray(
            data,
            dims=("c", "z", "y", "x"),
            coords={
                "c": [f"ch{i}" for i in range(n_channels)],
                "z": np.arange(zyx_shape[0]) * z_pixel_size,
                "y": np.arange(zyx_shape[1]) * yx_pixel_size,
                "x": np.arange(zyx_shape[2]) * yx_pixel_size,
            },
        )

    return _make


def device_params():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available() and os.getenv("GITHUB_ACTIONS") == "false":
        devices.append("mps")
    return "device", devices


_DEVICE = device_params()


@pytest.fixture(scope="function")
def example_plate(tmp_path):
    plate_path = tmp_path / "input.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w-",
        channel_names=[f"State{i}" for i in range(4)] + ["BF"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position.create_zeros("0", (2, 5, 4, 5, 6), dtype=np.uint16)

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def birefringence_phase_recon_settings_function(tmp_path):
    recon_settings = settings.ReconstructionSettings(
        birefringence=settings.BirefringenceSettings(),
        phase=settings.PhaseSettings(),
    )
    dataset = open_ome_zarr(
        tmp_path / "input.zarr",
        layout="fov",
        mode="w-",
        channel_names=[f"State{i}" for i in range(4)],
    )
    yield recon_settings, dataset


@pytest.fixture(scope="session")
def mm_ome_tiff_dir():
    """Return path to a small Micro-Manager OME-TIFF dataset for CLI/GUI tests.

    Downloads ``waveOrder_test_data.zip`` (~26 MB) from Zenodo on first use
    and caches it under ``platformdirs.user_data_dir``. Returns the path to
    a 1t x 3c x 5z x 128 x 128 MMStack OME-TIFF folder (Cy5, DAPI, FITC).
    """
    cache_dir = Path(user_data_dir("waveorder-test-data-v1"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / "waveOrder_test_data.zip"
    extracted = cache_dir / "MM20_ome-tiffs"
    if not extracted.exists():
        if not archive.exists():
            print(
                f"Downloading MM OME-TIFF test data to {cache_dir} (~26 MB)...",
                file=sys.stderr,
            )
            download(_MM_OME_TIFF_ZIP_URL, out=str(archive))
        shutil.unpack_archive(archive, extract_dir=cache_dir)

    dataset_dir = extracted / "mm2.0-20201209_1t_5z_3c_512k_1"
    if not dataset_dir.is_dir():
        pytest.skip(f"Expected MM ome-tiff dataset not found at {dataset_dir}")
    return dataset_dir


@pytest.fixture(scope="function")
def fluorescence_recon_settings_function(tmp_path):
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        fluorescence=settings.FluorescenceSettings(),
    )
    dataset = open_ome_zarr(
        tmp_path / "input.zarr",
        layout="fov",
        mode="w-",
        channel_names=[f"State{i}" for i in range(4)],
    )
    yield recon_settings, dataset

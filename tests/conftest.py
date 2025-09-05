import os

import numpy as np
import pytest
import torch
import sys

from iohub.ngff import open_ome_zarr

from waveorder.cli import settings

# conftest.py
# Workaround for napari bug on macOS GitHub Actions:
# napari._qt.qt_event_loop.get_qapp() assumes sys.argv[0] is a string,
# but in CI it can be a pathlib.PosixPath. Coerce everything to str.
sys.argv = [str(a) for a in sys.argv]

# Additional macOS CI workarounds
if os.getenv("GITHUB_ACTIONS") == "true" and sys.platform == "darwin":
    # Set Qt backend to avoid GUI issues on macOS CI
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Disable napari viewer if needed
    os.environ.setdefault("NAPARI_DISABLE_FULLSCREEN", "1")


def device_params():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if (
        torch.backends.mps.is_available()
        and os.getenv("GITHUB_ACTIONS") == "false"
    ):
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

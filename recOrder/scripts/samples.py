import shutil
from pathlib import Path

from typing import Literal
from iohub import open_ome_zarr
from iohub.ngff import Plate
from napari.utils.notifications import show_warning
from platformdirs import user_data_dir
from wget import download


def _build_layer_list(dataset: Plate, layer_names: list[str]):
    layer_list = []
    for channel_name in layer_names:
        channel_index = dataset.channel_names.index(channel_name)
        position = dataset["0/0/0"]
        data = (position["0"][:, channel_index],)
        layer_dict = {"name": channel_name, "scale": position.scale[3:]}
        layer_list.append((data, layer_dict))

    return layer_list


def download_and_unzip(data_type: Literal["target", "embryo"]) -> tuple[Path]:
    """Downloads sample data .zip from zenodo, unzips, and returns Paths to the .zarr datasets.

    Skips the download if the files already exist.

    Uses platformdirs.user_data_dir to store data.
    """

    # Delete old data
    old_data_dirs = ["recOrder-sample-v1.4"]
    for old_data_dir in old_data_dirs:
        old_data_path = Path(user_data_dir(old_data_dir))
        if old_data_path.exists():
            shutil.rmtree(str(old_data_path))

    temp_dirpath = Path(user_data_dir("recOrder-sample-v1.5"))
    temp_dirpath.mkdir(exist_ok=True, parents=True)

    if data_type == "target":
        data_dirpath = temp_dirpath / "sample_contribution"
        data_size = "10 MB"
        data_url = "https://zenodo.org/record/8386856/files/sample_contribution.zip?download=1"
    elif data_type == "embryo":
        data_dirpath = temp_dirpath / "sample_contribution_embryo"
        data_size = "92 MB"
        data_url = "https://zenodo.org/record/8386856/files/sample_contribution_embryo.zip?download=1"

    if not data_dirpath.with_suffix(".zip").exists():
        show_warning(
            f"Downloading {data_size} sample contribution. This might take a moment..."
        )
        download(data_url, out=str(temp_dirpath))

    if not data_dirpath.exists():
        shutil.unpack_archive(
            data_dirpath.with_suffix(".zip"), extract_dir=temp_dirpath
        )

    data_path = data_dirpath / "raw_data.zarr"
    recon_path = data_dirpath / "reconstruction.zarr"
    return data_path, recon_path


def read_polarization_target_data():
    """Returns the polarization data sample contribution"""
    data_path, _ = download_and_unzip("target")
    dataset = open_ome_zarr(data_path)
    return _build_layer_list(dataset, dataset.channel_names)


def read_polarization_target_reconstruction():
    """Returns the polarization target reconstruction sample contribution"""
    _, recon_path = download_and_unzip("target")
    dataset = open_ome_zarr(recon_path)
    return _build_layer_list(dataset, ["Phase3D", "Retardance", "Orientation"])


def read_zebrafish_embryo_reconstruction():
    """Returns the embryo reconstruction sample contribution"""
    _, recon_path = download_and_unzip("embryo")
    dataset = open_ome_zarr(recon_path)
    return _build_layer_list(dataset, ["Retardance", "Orientation"])

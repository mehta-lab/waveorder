import shutil
from pathlib import Path

from iohub import open_ome_zarr
from napari.utils.notifications import show_warning
from platformdirs import user_data_dir
from wget import download


def download_and_unzip() -> tuple[Path]:
    """Downloads sample data .zip from zenodo, unzips, and returns Paths to the .zarr datasets.

    Skips the download if the files already exist.

    Uses platformdirs.user_data_dir to store data.
    """
    temp_dirpath = Path(user_data_dir("recOrder-sample-v1.4"))
    temp_dirpath.mkdir(exist_ok=True, parents=True)
    data_dirpath = temp_dirpath / "sample_contribution"

    if not data_dirpath.with_suffix(".zip").exists():
        show_warning(
            "Downloading 10 MB sample contribution. This might take a moment..."
        )
        data_url = "https://zenodo.org/record/8280720/files/sample_contribution.zip?download=1"
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
    data_path, _ = download_and_unzip()

    dataset = open_ome_zarr(data_path)
    layer_list = []
    for channel_index, channel_name in enumerate(dataset.channel_names):
        position = dataset["0/0/0"]
        data = (position["0"][0, channel_index],)
        layer_dict = {"name": channel_name, "scale": position.scale[3:]}
        layer_list.append((data, layer_dict))

    return layer_list


def read_polarization_target_reconstruction():
    """Returns the polarization target reconstruction sample contribution"""

    _, recon_path = download_and_unzip()
    dataset = open_ome_zarr(recon_path)

    layer_list = []
    for channel_index, channel_name in enumerate(
        ["Retardance", "Orientation"]
    ):
        position = dataset["0/0/0"]
        data = (position["0"][0, channel_index],)
        layer_dict = {"name": channel_name, "scale": position.scale[3:]}
        layer_list.append((data, layer_dict))

    return layer_list

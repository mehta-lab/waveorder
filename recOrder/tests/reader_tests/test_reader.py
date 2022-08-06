from recOrder.io._reader import napari_get_reader, ome_zarr_reader, ome_tif_reader
import os
from os.path import dirname, abspath
import yaml

def test_napari_get_reader(get_ometiff_data_dir, get_zarr_data_dir):

    _, ometiff_data = get_ometiff_data_dir
    _, zarr_data = get_zarr_data_dir
    assert(napari_get_reader(ometiff_data).__name__ == 'ome_tif_reader')
    assert(napari_get_reader(zarr_data).__name__ == 'ome_zarr_reader')

def test_readers(get_ometiff_data_dir, get_zarr_data_dir):

    _, ometiff_data = get_ometiff_data_dir
    _, zarr_data = get_zarr_data_dir

    data_from_ome = ome_tif_reader(ometiff_data)
    data_from_zarr = ome_zarr_reader(zarr_data)

    for data in [data_from_ome, data_from_zarr]:
        assert(len(data) == 3)
        assert(data[0][0].shape == (2, 4, 16, 128, 256))

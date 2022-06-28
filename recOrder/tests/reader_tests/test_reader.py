from recOrder.io._reader import napari_get_reader, ome_zarr_reader, ome_tif_reader
import os
from os.path import dirname, abspath
import yaml

def test_napari_get_reader(setup_test_data):
    folder, ometiff_data, zarr_data, bf_data = setup_test_data
    assert(napari_get_reader(ometiff_data).__name__ == 'ome_tif_reader')
    assert(napari_get_reader(zarr_data).__name__ == 'ome_zarr_reader')

def test_readers(setup_test_data):
    folder, ometiff_data, zarr_data, bf_data = setup_test_data

    data_from_ome = ome_tif_reader(ometiff_data)
    data_from_zarr = ome_zarr_reader(zarr_data)

    for data in [data_from_ome, data_from_zarr]:
        assert(len(data) == 3)
        assert(data[0][0].shape == (2, 4, 81, 231, 498))
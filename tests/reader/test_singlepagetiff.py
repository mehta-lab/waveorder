import zarr
import numpy as np
from waveorder.io.singlepagetiff import MicromanagerSequenceReader


def test_constructor_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    """
    test that constructor parses metadata properly
        no data extraction in this test
    """

    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerSequenceReader(one_folder, extract_data=False)

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)


def test_output_dims_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    """
    test that output dimensions are always (t, c, z, y, x)
    """

    # choose a random folder
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=False)

    assert(mmr.get_zarr(0).shape[0] == mmr.frames)
    assert(mmr.get_zarr(0).shape[1] == mmr.channels)
    assert(mmr.get_zarr(0).shape[2] == mmr.slices)
    assert(mmr.get_zarr(0).shape[3] == mmr.height)
    assert(mmr.get_zarr(0).shape[4] == mmr.width)


def test_output_dims_mm2gamma_incomplete(setup_mm2gamma_singlepage_tiffs_incomplete):
    """
    test that output dimensions are correct for interrupted data
    """

    # choose a random folder
    folder = setup_mm2gamma_singlepage_tiffs_incomplete
    mmr = MicromanagerSequenceReader(folder, extract_data=True)

    assert(mmr.get_zarr(0).shape[0] == mmr.frames)
    assert(mmr.get_zarr(0).shape[1] == mmr.channels)
    assert(mmr.get_zarr(0).shape[2] == mmr.slices)
    assert(mmr.get_zarr(0).shape[3] == mmr.height)
    assert(mmr.get_zarr(0).shape[4] == mmr.width)
    assert(mmr.get_zarr(0).shape[0] == 11)


def test_get_zarr_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_get_array_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


def test_get_num_positions_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    assert(mmr.get_num_positions() >= 1)


# repeat of above but using mm1.4.22 data

def test_constructor_mm1422(setup_mm1422_singlepage_tiffs):
    """
    test that constructor parses metadata properly
        no data extraction in this test
    """

    # choose a specific folder
    _, one_folder, _ = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerSequenceReader(one_folder, extract_data=False)

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)


def test_output_dims_mm1422(setup_mm1422_singlepage_tiffs):
    """
    test that output dimensions are always (t, c, z, y, x)
    """

    # choose a random folder
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=False)

    assert(mmr.get_zarr(0).shape[0] == mmr.frames)
    assert(mmr.get_zarr(0).shape[1] == mmr.channels)
    assert(mmr.get_zarr(0).shape[2] == mmr.slices)
    assert(mmr.get_zarr(0).shape[3] == mmr.height)
    assert(mmr.get_zarr(0).shape[4] == mmr.width)


def test_get_zarr_mm1422(setup_mm1422_singlepage_tiffs):
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_get_array_mm1422(setup_mm1422_singlepage_tiffs):
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


def test_get_num_positions_mm1422(setup_mm1422_singlepage_tiffs):
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerSequenceReader(rand_folder, extract_data=True)
    assert(mmr.get_num_positions() >= 1)


# uncertain whether the bottom tests are useful
# def test_read_tiff_series_mm2gamma(setup_mm2gamma_singlepage_tiffs):
#     pass
#
#
# def test_extract_coord_mm2gamma(setup_mm2gamma_singlepage_tiffs):
#     pass
#
#
# def test_shape_mm2gamma(setup_mm2gamma_singlepage_tiffs):
#     pass

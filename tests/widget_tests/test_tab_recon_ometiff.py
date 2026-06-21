"""Smoke test for Micro-Manager OME-TIFF support in the reconstruction tab.

Manual reproduction (open the plugin in napari and point it at the MM ome-tif
folder; channel names should auto-populate without any prior zarr conversion)::

    napari -w waveorder
"""

import pytest


@pytest.fixture
def recon_tab(make_napari_viewer):
    from waveorder.plugin import tab_recon
    from waveorder.plugin.main_widget import MainWidget

    tab_recon.HAS_INSTANCE = {"val": False, "instance": None}
    viewer = make_napari_viewer()
    widget = MainWidget(viewer)
    yield widget.ui.tab_reconstruction
    tab_recon.HAS_INSTANCE = {"val": False, "instance": None}


def test_validate_input_data_autopopulates_from_ometiff(recon_tab, mm_ome_tiff_dir):
    """Pointing the GUI at an MM ome-tif populates input_channel_names from
    MM metadata, with no zarr conversion needed."""
    ok, _msg = recon_tab.validate_input_data(str(mm_ome_tiff_dir), BG=True)
    assert ok is True
    assert recon_tab.input_channel_names == ["Cy5", "DAPI", "FITC"]
    assert "Micro-Manager OME-TIFF" in recon_tab.data_input_Label.tooltip
    # GUI must not trigger a conversion just from path selection.
    assert not (mm_ome_tiff_dir.parent / (mm_ome_tiff_dir.name + "_converted.zarr")).exists()

from napari.viewer import ViewerModel

from waveorder.plugin.main_widget import MainWidget


def test_dock_widget(make_napari_viewer):
    viewer: ViewerModel = make_napari_viewer()
    viewer.window.add_dock_widget(MainWidget(viewer))
    assert "waveorder" in list(viewer._window._dock_widgets.keys())[0]

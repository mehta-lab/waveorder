import napari
from recOrder.plugin.widget.calibration_plugin_widget import Calibration


def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(Calibration(viewer))
    napari.run()


if __name__ == "__main__":
    main()
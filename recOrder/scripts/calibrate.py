# This script can be modified to debug and test calibrations

import napari
from recOrder.plugin.widget.main_widget import MainWidget

def main():
    viewer = napari.Viewer()
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)
    recorder.change_gui_mode()
    recorder.run_calibration()

if __name__ == "__main__":
    main()
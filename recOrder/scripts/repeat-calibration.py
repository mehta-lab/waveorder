# This script can be modified to debug and test calibrations

import napari
import time
from recOrder.plugin.widget.main_widget import MainWidget

SAVE_DIR = "./"
SWINGS = [0.1, 0.03, 0.01, 0.005]
REPEATS = 5

def main():
    viewer = napari.Viewer()
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)
    recorder.ui.qbutton_gui_mode.click()
    recorder.calib_scheme = '5-State'

    for repeat in range(REPEATS):
        for swing in SWINGS:
            print("Calibrating with swing = " + str(swing))
            recorder.swing = swing
            recorder.directory = SAVE_DIR
            recorder.run_calibration()
            time.sleep(100)
            
if __name__ == "__main__":
    main()
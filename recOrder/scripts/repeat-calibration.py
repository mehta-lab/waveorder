# This script can be modified to debug and test calibrations

import napari
import time
import matplotlib.pyplot as plt
from recOrder.plugin.widget.main_widget import MainWidget

save_dir = "./"
swings = [0.1, 0.03, 0.01, 0.005]
repeats = 5

def main():
    viewer = napari.Viewer()
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)
    recorder.ui.qbutton_gui_mode.click()
    recorder.calib_scheme = '5-State'

    for repeat in range(repeats):
        for swing in swings:
            print("Calibrating with swing = " + str(swing))
            recorder.swing = swing
            recorder.directory = save_dir  
            recorder.run_calibration()
            time.sleep(100)
            
if __name__ == "__main__":
    main()
# This script can be modified to debug and test calibrations

import time

import napari

from waveorder.plugin.main_widget import MainWidget

SAVE_DIR = "./"
SWINGS = [0.1, 0.03, 0.01, 0.005]
REPEATS = 5


def main():
    viewer = napari.Viewer()
    waveorder = MainWidget(viewer)
    viewer.window.add_dock_widget(waveorder)
    waveorder.ui.qbutton_connect_to_mm.click()
    waveorder.calib_scheme = "5-State"

    for repeat in range(REPEATS):
        for swing in SWINGS:
            print("Calibrating with swing = " + str(swing))
            waveorder.swing = swing
            waveorder.directory = SAVE_DIR
            waveorder.run_calibration()
            time.sleep(100)


if __name__ == "__main__":
    main()

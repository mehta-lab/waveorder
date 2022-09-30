# This script can be modified to debug and test calibrations

import napari
import time
import matplotlib.pyplot as plt
from recOrder.plugin.widget.main_widget import MainWidget
from recOrder.calib.Calibration import QLIPP_Calibration
from recOrder.plugin.workers.calibration_workers import CalibrationWorker

save_dir = "Q:\\Talon\\2022_09_30_calibration\\"
swings = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
repeats = 10

def main():
    viewer = napari.Viewer()
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)

    for repeat in range(repeats):
        for swing in swings:
            print("Calibrating with swing = " + str(swing))
            recorder.ui.qbutton_gui_mode.click()
            recorder.swing = swing
            recorder.directory = save_dir
            
            #recorder.calib = QLIPP_Calibration(recorder.mmc, recorder.mm, group=recorder.config_group, lc_control_mode=recorder.calib_mode,
            #                           interp_method=recorder.interp_method, wavelength=recorder.wavelength)

            #recorder.worker = CalibrationWorker(recorder, recorder.calib)

            #recorder.worker.start()
            recorder.run_calibration()
            time.sleep(100)
            plt.savefig(save_dir+str(repeat)+"-"+str(swing)+".png", dpi=300)

if __name__ == "__main__":
    main()
# This script can be modified to debug and test calibrations

import napari
import time
import matplotlib.pyplot as plt
from recOrder.plugin.widget.main_widget import MainWidget
from recOrder.calib.Calibration import QLIPP_Calibration
from recOrder.plugin.workers.calibration_workers import CalibrationWorker

save_dir = "Q:\\Talon\\2022_10_01_repeats_0.05\\"
swing = 0.05
cal_repeats = 3
bkg_repeats = 3

def main():
    viewer = napari.Viewer()
    app = MainWidget(viewer)
    viewer.window.add_dock_widget(app)
    app.ui.qbutton_gui_mode.click()
    app.calib_scheme = '5-State'
    app.directory = save_dir
    app.save_directory = save_dir

    input("Move to background and press <Enter>")
    for cal_repeat in range(cal_repeats):
        print(f"Calibration repeat # {cal_repeat}")
        app.swing = swing
    
        print(f"Calibrating with swing = {swing}")
        app.run_calibration()
        time.sleep(90)
 
        for bkg_repeat in range(bkg_repeats):
            print(f"Background repeat # {bkg_repeat}")
            app.last_calib_meta_file = app.calib.meta_file
            app.capture_bg()
            time.sleep(20)
        
            input("Move to target and press <Enter>")
            app.ui.cb_bg_method.setCurrentIndex(1) # Set to "Measured" bg correction
            app.enter_bg_correction()
            app.save_name = f"cal-{cal_repeat}-bkg-{bkg_repeat}"
            app.acq_birefringence()
            time.sleep(15)

            input("Move to background and press <Enter>")

if __name__ == "__main__":
    main()
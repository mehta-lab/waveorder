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
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)
    recorder.ui.qbutton_gui_mode.click()
    recorder.calib_scheme = '5-State'
    recorder.directory = save_dir
    recorder.save_directory = save_dir

    input("Move to background and press <Enter>")
    for cal_repeat in range(cal_repeats):
        print(f"Calibration repeat # {cal_repeat}")
        recorder.swing = swing
    
        print(f"Calibrating with swing = {swing}")
        recorder.run_calibration()
        time.sleep(90)
 
        for bkg_repeat in range(bkg_repeats):
            print(f"Background repeat # {bkg_repeat}")
            recorder.last_calib_meta_file = recorder.calib.meta_file
            recorder.capture_bg()
            time.sleep(20)
        
            input("Move to target and press <Enter>")
            recorder.ui.cb_bg_method.setCurrentIndex(1) # Set to "Measured" bg correction
            recorder.enter_bg_correction()
            recorder.save_name = f"cal-{cal_repeat}-bkg-{bkg_repeat}"
            recorder.acq_birefringence()
            time.sleep(15)

            input("Move to background and press <Enter>")

if __name__ == "__main__":
    main()
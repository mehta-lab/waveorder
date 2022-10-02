# This script can be modified to debug and test calibrations

import napari
import time
import matplotlib.pyplot as plt
from recOrder.plugin.widget.main_widget import MainWidget
from recOrder.calib.Calibration import QLIPP_Calibration
from recOrder.plugin.workers.calibration_workers import CalibrationWorker

save_dir = "Q:\\Talon\\2022_10_01_repeat_acq\\"
swing = 0.03
repeats = 100
states = ['4-State']
modes = ['MM-Voltage']

def main():
    viewer = napari.Viewer()
    recorder = MainWidget(viewer)
    viewer.window.add_dock_widget(recorder)
    recorder.ui.qbutton_gui_mode.click()
    
    recorder.directory = save_dir
    recorder.save_directory = save_dir
    recorder.swing = swing

    input("Move to background and press <Enter>")
        
    for state in states:
        recorder.calib_scheme = state

        for mode in modes:
            recorder.calib_mode = mode

            print(f"Calibrating with swing = {swing}")
            recorder.run_calibration()
            time.sleep(90)

            recorder.last_calib_meta_file = recorder.calib.meta_file
            recorder.capture_bg()
            time.sleep(20)
    
            for repeat in range(repeats):
                print(f"Calibration repeat # {repeat}")
    
                recorder.ui.cb_bg_method.setCurrentIndex(1) # Set to "Measured" bg correction
                recorder.enter_bg_correction()
                recorder.save_name = f"mode-{mode}-{state}-repeat-{repeat}"
                recorder.acq_birefringence()
                time.sleep(15)

if __name__ == "__main__":
    main()
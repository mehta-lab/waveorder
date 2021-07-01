from recOrder.calib.Calibration import QLIPP_Calibration
from pycromanager import Bridge
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from qtpy.QtWidgets import QWidget, QFileDialog
from recOrder.plugin.qtdesigner import recOrder_calibration_v4
from pathlib import Path
import os
import numpy as np


class recOrder_Calibration(QWidget, QtCore.QObject):

    mm_status_changed = pyqtSignal(bool)
    progress_changed = pyqtSignal(int)
    intensity_changed = pyqtSignal(float)
    log_changed = pyqtSignal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI Elements
        self.ui = recOrder_calibration_v4.Ui_Form()
        self.ui.setupUi(self)

        # Setup Connections between elements

        # Recievers
        # =================================
        # Connect to Micromanager
        self.ui.qbutton_mm_connect.clicked[bool].connect(self.connect_to_mm)

        # Calibration Parameters
        self.ui.qbutton_browse.clicked[bool].connect(self.browse_dir_path)
        self.ui.le_directory.editingFinished.connect(self.enter_dir_path)
        self.ui.le_swing.editingFinished.connect(self.enter_swing)
        self.ui.le_wavelength.editingFinished.connect(self.enter_wavelength)
        self.ui.cb_calib_scheme.currentIndexChanged[int].connect(self.enter_calib_scheme)
        self.ui.chb_use_roi.stateChanged[int].connect(self.enter_use_cropped_roi)
        # self.ui.run_calib.clicked[bool].connect(self.run_calibration)

        # Capture Background
        self.ui.le_bg_folder.editingFinished.connect(self.enter_bg_folder_name)
        self.ui.le_n_avg.editingFinished.connect(self.enter_n_avg)
        self.ui.qbutton_capture_bg.clicked[bool].connect(self.plot)

        # Emitters
        # =================================#
        self.mm_status_changed.connect(self.handle_mm_status_update)
        self.progress_changed.connect(self.handle_progress_update)
        # self.extinction_changed.connect(self.handle_extinction_update)

        #Other Properties:
        self.mm = None
        self.mmc = None
        self.home_path = str(Path.home())
        self.directory = None
        self.swing = 0.1
        self.wavelength = 532
        self.calib_scheme = '4-State'
        self.use_cropped_roi = False

        # Init Plot
        plot_item = self.ui.plot_widget.getPlotItem()
        plot_item.enableAutoRange()
        plot_item.setLabel('left', 'Intensity - Reference')
        self.ui.plot_widget.setBackground((32, 34, 40))

        # Init Logger
        self.ui.te_log.setStyleSheet('background-color: rgb(32,34,40);')

    @pyqtSlot(bool)
    def connect_to_mm(self):
        try:
            bridge = Bridge(convert_camel_case=False)
            self.mmc = bridge.get_core()
            self.mm = bridge.get_studio()

            self.mm_status_changed.emit(True)
        except:
            self.mm_status_changed.emit(False)

    @pyqtSlot(bool)
    def handle_mm_status_update(self, value):
        if value:
            self.ui.le_mm_status.setText('Sucess!')
            # self.ui.le_mm_status.setStyleSheet("border: 1px solid green;")
            self.ui.le_mm_status.setStyleSheet("background-color: green;")
        else:
            self.ui.le_mm_status.setText('Failed.')
            self.ui.le_mm_status.setStyleSheet("background-color: red;")
            # self.ui.le_mm_status.setStyleSheet("border: 1px solid red;")

    @pyqtSlot(bool)
    def browse_dir_path(self):
        # self.ui.le_directory.setFocus()
        result = self._open_file_dialog(self.home_path)
        self.directory = result
        self.ui.le_directory.setText(result)

    @pyqtSlot()
    def enter_dir_path(self):
        path = self.ui.le_directory.text()
        if os.path.exists(path):
            self.directory = path
        else:
            self.ui.le_directory.setText('Path Does Not Exist')

    @pyqtSlot()
    def enter_swing(self):
        self.swing = self.ui.le_swing.text()

    @pyqtSlot()
    def enter_wavelength(self):
        self.wavelength = self.ui.le_directory.text()

    @pyqtSlot()
    def enter_calib_scheme(self):
        self.calib_scheme = self.ui.cb_calib_scheme.currentIndex()

    @pyqtSlot()
    def enter_use_cropped_roi(self):
        state = self.ui.chb_use_roi.checkState()
        if state == 2:
            self.use_cropped_roi = True
        elif state == 0:
            self.use_cropped_roi = False
        print(self.use_cropped_roi)

    @pyqtSlot()
    def run_calibration(self):
        self.calib = QLIPP_Calibration(self.mmc, self.mm)
        self.calib.swing = self.swing
        self.calib.wavelength = self.wavelength

        # Get Image Parameters
        self.mmc.snapImage()
        self.mmc.getImage()
        self.height, self.width = self.mmc.getImageHeight(), self.mmc.getImageWidth()
        self.ROI = (0, 0, self.width, self.height)

        # Check if change of ROI is needed
        if use_full_FOV is False:
            rect = self.check_and_get_roi()
            cont = self.display_and_check_ROI(rect)

            if not cont:
                print('\n---------Stopping Calibration---------\n')
                return
            else:
                self.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
                self.ROI = (rect.x, rect.y, rect.width, rect.height)

        # Calculate Blacklevel
        print('Calculating Blacklevel ...')
        self.I_Black = self.calc_blacklevel()
        print(f'Blacklevel: {self.I_Black}\n')

        # Set LC Wavelength:
        self.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.wavelength)

        self.opt_Iext()
        self.opt_I0()
        self.opt_I60(0.05, 0.05)
        self.opt_I120(0.05, 0.05)

        # Calculate Extinction
        self.extinction_ratio = self.calculate_extinction()

        # Write Metadata
        self.write_metadata(4)

        # Return ROI to full FOV
        if use_full_FOV is False:
            self.mmc.clearROI()

        print("\n=======Finished Calibration=======\n")
        print(f"EXTINCTION = {self.extinction_ratio}")

    @pyqtSlot(int)
    def handle_progress_update(self, value):
        self.ui.progress_bar.setValue(value)

    @pyqtSlot()
    def enter_bg_folder_name(self):
        self.bg_folder_name = self.ui.le_bg_folder.text()

    @pyqtSlot()
    def enter_bg_folder_name(self):
        self.n_avg = self.ui.le_n_avg.text()

    def handle_extinction_update(self, value):
        self.ui.le_extinction.setText(value)

    def _open_file_dialog(self, default_path):
        return self._open_dialog("select a directory",
                                 default_path)

    def _open_dialog(self, title, ref):
        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,
                                                title,
                                                ref,
                                                options=options)
        return path

    @pyqtSlot(bool)
    def plot(self):
        print('here')
        x = range(0, 10)
        y = range(0, 20, 2)

        self.ui.plot_widget.plot(x,y)
        self.ui.plot_widget.getPlotItem().autoRange()
        self.ui.te_log.appendPlainText('')

    def _init_plot(self):
        pass

    def _update_plot(self, value):
        pass


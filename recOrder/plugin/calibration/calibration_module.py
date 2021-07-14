from recOrder.calib.Calibration import QLIPP_Calibration
from pycromanager import Bridge
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
from qtpy.QtWidgets import QWidget, QFileDialog
from recOrder.plugin.qtdesigner import recOrder_calibration_v4, recOrder_calibration_v5
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes
from pathlib import Path
from napari import Viewer
from recOrder.calib.CoreFunctions import snap_image, set_lc_state
import os
import numpy as np
import logging


class recOrder_Calibration(QWidget, QtCore.QObject):

    mm_status_changed = pyqtSignal(bool)
    intensity_changed = pyqtSignal(float)
    log_changed = pyqtSignal(str)

    def __init__(self, napari_viewer: Viewer):
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
        self.ui.qbutton_calibrate.clicked[bool].connect(self.run_calibration)
        self.ui.qbutton_stop_calibrate.clicked[bool].connect(self.stop_calibration)
        self.ui.qbutton_calc_extinction.clicked[bool].connect(self.calc_extinction)

        # Capture Background
        self.ui.le_bg_folder.editingFinished.connect(self.enter_bg_folder_name)
        self.ui.le_n_avg.editingFinished.connect(self.enter_n_avg)
        self.ui.qbutton_capture_bg.clicked[bool].connect(self.capture_bg)

        # Logging
        log_box = QtLogger(self.ui.te_log)
        log_box.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_box)
        logging.getLogger().setLevel(logging.INFO)

        # Emitters
        # =================================#
        self.mm_status_changed.connect(self.handle_mm_status_update)

        #Other Properties:
        self.mm = None
        self.mmc = None
        self.home_path = str(Path.home())
        self.directory = None
        self.swing = 0.1
        self.wavelength = 532
        self.calib_scheme = '4-State'
        self.use_cropped_roi = False
        self.bg_folder_name = 'BG'
        self.n_avg = 20
        self.intensity_monitor = []

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

    @pyqtSlot(int)
    def handle_progress_update(self, value):
        self.ui.progress_bar.setValue(value)

    @pyqtSlot(str)
    def handle_extinction_update(self, value):
        self.ui.le_extinction.setText(value)

    @pyqtSlot(object)
    def handle_plot_update(self, value):
        self.intensity_monitor.append(value)
        self.ui.plot_widget.plot(self.intensity_monitor)
        self.ui.plot_widget.getPlotItem().autoRange()

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
        self.wavelength = self.ui.le_wavelength.text()

    @pyqtSlot()
    def enter_calib_scheme(self):
        index = self.ui.cb_calib_scheme.currentIndex()
        if index == 0:
            self.calib_scheme = '4-State'
        else:
            self.calib_scheme = '5-State'

    @pyqtSlot()
    def enter_use_cropped_roi(self):
        state = self.ui.chb_use_roi.checkState()
        if state == 2:
            self.use_cropped_roi = True
        elif state == 0:
            self.use_cropped_roi = False

    @pyqtSlot()
    def enter_n_avg(self):
        self.n_avg = self.ui.le_n_avg.text()

    @pyqtSlot(bool)
    def run_calibration(self):
        print('Starting Calibration')
        self.ui.progress_bar.setValue(0)
        self.calib = QLIPP_Calibration(self.mmc, self.mm)
        self.calib.swing = self.swing
        self.calib.wavelength = self.wavelength
        self.calib.meta_file = self.directory

        self.thread = QThread()
        self.worker = Worker(self, self.calib)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_calibration)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress_update.connect(self.handle_progress_update)
        self.worker.extinction_update.connect(self.handle_extinction_update)
        self.worker.intensity_update.connect(self.handle_plot_update)
        self.thread.start()
        self.ui.qbutton_calibrate.setEnabled(False)
        self.thread.finished.connect(
            lambda: self.ui.qbutton_calibrate.setEnabled(True)
        )

    @pyqtSlot(bool)
    def stop_calibration(self):
        #todo: add try, except
        self.worker.stop()
        self.worker.killthread()
        self.worker.finished.emit()

    @pyqtSlot(bool)
    def calc_extinction(self):
        set_lc_state('State0')
        extinction = np.mean(snap_image(self.mmc))
        set_lc_state('State1')
        state1 = np.mean(snap_image(self.mmc))
        extinction = self.calib.calculate_extinction(self.swing, self.calib.I_Black, extinction, state1)
        self.ui.le_extinction.setText(str(extinction))

    @pyqtSlot(bool)
    def capture_bg(self):
        bg_path = os.path.join(self.directory, self.ui.le_bg_folder.text())
        if not os.path.exists(bg_path):
            os.mkdir(bg_path)
        imgs = self.calib.capture_bg(self.n_avg, bg_path)
        img_dim = (imgs.shape[-2], imgs.shape[-1])
        N_channel = 4 if self.calib_scheme == '4-State' else 5

        recon = initialize_reconstructor(img_dim, self.wavelength, self.swing, N_channel, True,
                                         None, None, None, None, None, None, None, bg_option='None')

        stokes = reconstruct_qlipp_stokes(imgs, recon, None)
        birefringence = reconstruct_qlipp_birefringence(stokes, recon)
        retardance = birefringence[0] / (2 * np.pi) * self.wavelength

        if self.viewer.layers['Background Images']:
            self.viewer.layers['Background Images'].data = imgs
        elif self.viewer.layers['Background Retardance']:
            self.viewer.layers['Background Retardance'].data = retardance
        elif self.viewer.layers['Background Orientation']:
            self.viewer.layers['Background Orientation'].data = birefringence[1]
        else:
            self.viewer.add_image(imgs, name='Background Images', colormap='gray')
            self.viewer.add_image(retardance, name='Background Retardance', colormap='gray')
            self.viewer.add_image(birefringence[1], name='Background Orientation', colormap='gray')


    @pyqtSlot()
    def enter_bg_folder_name(self):
        self.bg_folder_name = self.ui.le_bg_folder.text()

    @pyqtSlot()
    def enter_bg_folder_name(self):
        self.n_avg = self.ui.le_n_avg.text()

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

        self.ui.plot_widget.plot(x, y)
        self.ui.plot_widget.getPlotItem().autoRange()
        self.ui.te_log.appendPlainText('')


class Worker(QtCore.QObject):

    progress_update = pyqtSignal(int)
    extinction_update = pyqtSignal(str)
    intensity_update = pyqtSignal(object)
    finished = pyqtSignal()


    def __init__(self, calib_window, calib):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib

    def stop(self):
        self.threadactive = False
        self.wait()

    def killthread(self):
        self.thread.stop()

    def run_calibration(self):

        self.calib.intensity_emitter = self.intensity_update
        self.calib.get_full_roi()
        self.progress_update.emit(1)

        # TODO: Decide if displaying ROI is useful feature,
        # include in a pop-up window or napari window?  How to prompt to continue?

        # Check if change of ROI is needed
        if self.calib_window.use_cropped_roi:
            rect = self.calib.check_and_get_roi()
            cont = self.calib.display_and_check_ROI(rect)

            if not cont:
                print('\n---------Stopping Calibration---------\n')
                return
            else:
                self.calib_window.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
                self.calib.ROI = (rect.x, rect.y, rect.width, rect.height)

        # Calculate Blacklevel
        print('Calculating Blacklevel ...')
        self.calib.calc_blacklevel()
        print(f'Blacklevel: {self.calib.I_Black}\n')
        self.progress_update.emit(10)

        # Set LC Wavelength:
        self.calib_window.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.calib_window.wavelength)

        # Optimize States
        self._calibrate_4state() if self.calib_window.calib_scheme == '4-State' else self._calibrate_5state()

        # Write Metadata
        self.calib.write_metadata()

        # Return ROI to full FOV
        if self.calib_window.use_cropped_roi:
            self.calib_window.mmc.clearROI()

        # Calculate Extinction
        extinction_ratio = self.calculate_extinction()
        self.extinction_update.emit(str(extinction_ratio))
        self.progress_update.emit(100)

        print("\n=======Finished Calibration=======\n")
        print(f"EXTINCTION = {extinction_ratio}")
        self.finish.emit()

    def _calibrate_4state(self):

        self.calib.opt_Iext()
        self.progress_update.emit(60)
        self.calib.opt_I0()
        self.progress_update.emit(65)
        self.calib.opt_I60(0.05, 0.05)
        self.progress_update.emit(75)
        self.calib.opt_I120(0.05, 0.05)
        self.progress_update.emit(85)

    def _calibrate_5state(self):

        self.calib.opt_Iext()
        self.progress_update.emit(50)
        self.calib.opt_I0()
        self.progress_update.emit(55)
        self.calib.opt_I45(0.05, 0.05)
        self.progress_update.emit(65)
        self.calib.opt_I90(0.05, 0.05)
        self.progress_update.emit(75)
        self.calib.opt_I135(0.05, 0.05)
        self.progress_update.emit(85)


class QtLogger(logging.Handler):

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

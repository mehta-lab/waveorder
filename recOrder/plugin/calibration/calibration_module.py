from recOrder.calib.Calibration import QLIPP_Calibration
from pycromanager import Bridge
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
from qtpy.QtWidgets import QWidget, QFileDialog
from recOrder.plugin.qtdesigner import recOrder_calibration_v4, recOrder_calibration_v5
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes
from recOrder.acq.acq_single_stack import acquire_2D, acquire_3D
from pathlib import Path
from napari import Viewer
from recOrder.calib.CoreFunctions import set_lc_state, snap_and_average
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
        self.ui = recOrder_calibration_v5.Ui_Form()
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
        self.ui.qbutton_calc_extinction.clicked[bool].connect(self.calc_extinction)

        # Capture Background
        self.ui.le_bg_folder.editingFinished.connect(self.enter_bg_folder_name)
        self.ui.le_n_avg.editingFinished.connect(self.enter_n_avg)
        self.ui.qbutton_capture_bg.clicked[bool].connect(self.capture_bg)

        # Advanced
        self.ui.cb_loglevel.currentIndexChanged[int].connect(self.enter_log_level)

        ######### Acquisition Tab #########
        self.ui.qbutton_browse_save_path.clicked[bool].connect(self.browse_save_path)
        self.ui.chb_save_imgs.stateChanged[int].connect(self.enter_save_imgs)
        self.ui.le_zstart.editingFinished.connect(self.enter_zstart)
        self.ui.le_zend.editingFinished.connect(self.enter_zend)
        self.ui.le_zstep.editingFinished.connect(self.enter_zstep)
        self.ui.cb_birefringence.currentIndexChanged[int].connect(self.enter_birefringence_dim)
        self.ui.cb_phase.currentIndexChanged[int].connect(self.enter_phase_dim)
        self.ui.qbutton_acq_birefringence.clicked[bool].connect(self.acq_birefringence)
        self.ui.qbutton_acq_phase.clicked[bool].connect(self.acq_phase)

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
        self.save_imgs = False
        self.birefringence_dim = '2D'
        self.phase_dim = '2D'
        self.z_start = None
        self.z_end = None
        self.z_step = None

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
            self.calib = QLIPP_Calibration(self.mmc, self.mm)

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

    @pyqtSlot(object)
    def handle_bg_image_update(self, value):
        print(self.viewer.layers)
        if 'Background Images' in self.viewer.layers:
            self.viewer.layers['Background Images'].data = value
        else:
            self.viewer.add_image(value, name='Background Images', colormap='gray')

    @pyqtSlot(object)
    def handle_bire_image_update(self, value):

        if 'Background Retardance' in self.viewer.layers:
            self.viewer.layers['Background Retardance'].data = value[0]
        else:
            self.viewer.add_image(value[0], name='Background Retardance', colormap='gray')

        if 'Background Orientation' in self.viewer.layers:
            self.viewer.layers['Background Orientation'].data = value[1]
        else:
            self.viewer.add_image(value[1], name='Background Orientation', colormap='gray')

    @pyqtSlot(bool)
    def browse_dir_path(self):
        # self.ui.le_directory.setFocus()
        result = self._open_file_dialog(self.home_path)
        self.directory = result
        self.ui.le_directory.setText(result)

    @pyqtSlot(bool)
    def browse_save_path(self):
        # self.ui.le_directory.setFocus()
        result = self._open_file_dialog(self.home_path)
        self.directory = result
        self.ui.le_save_path.setText(result)

    @pyqtSlot()
    def enter_dir_path(self):
        path = self.ui.le_directory.text()
        if os.path.exists(path):
            self.directory = path
        else:
            self.ui.le_directory.setText('Path Does Not Exist')

    @pyqtSlot()
    def enter_swing(self):
        self.swing = float(self.ui.le_swing.text())

    @pyqtSlot()
    def enter_wavelength(self):
        self.wavelength = int(self.ui.le_wavelength.text())

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
    def enter_bg_folder_name(self):
        self.bg_folder_name = self.ui.le_bg_folder.text()

    @pyqtSlot()
    def enter_n_avg(self):
        self.n_avg = int(self.ui.le_n_avg.text())

    def enter_log_level(self):
        index = self.ui.cb_loglevel.currentIndex()
        if index == 0:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.DEBUG)

    @pyqtSlot()
    def enter_save_imgs(self):
        state = self.ui.chb_save_imgs.checkState()
        if state == 2:
            self.save_imgs = True
        elif state == 0:
            self.save_imgs = False

    @pyqtSlot()
    def enter_zstart(self):
        self.z_start = int(self.ui.le_zstart.text())

    @pyqtSlot()
    def enter_zend(self):
        self.z_end = int(self.ui.le_zend.text())

    @pyqtSlot()
    def enter_zstep(self):
        self.z_step = int(self.ui.le_zstep.text())

    @pyqtSlot()
    def enter_birefringence_dim(self):
        state = self.ui.cb_birefringence.checkState()
        if state == 2:
            self.birefringence_dim = '2D'
        elif state == 0:
            self.birefringence_dim = '3D'

    @pyqtSlot()
    def enter_phase_dim(self):
        state = self.ui.cb_phase.checkState()
        if state == 2:
            self.phase_dim = '2D'
        elif state == 0:
            self.phase_dim = '3D'

    @pyqtSlot(bool)
    def run_calibration(self):
        logging.info('Starting Calibration')
        self.ui.progress_bar.setValue(0)
        self.calib.swing = self.swing
        self.calib.wavelength = self.wavelength
        self.calib.meta_file = os.path.join(self.directory, 'calibration_metadata.txt')

        self.calibration_thread = QThread()
        self.calib_worker = CalibrationWorker(self, self.calib)
        self.calib_worker.moveToThread(self.calibration_thread)
        self.calibration_thread.started.connect(self.calib_worker.run)
        self.calib_worker.finished.connect(self.calibration_thread.quit)
        self.calib_worker.finished.connect(self.calib_worker.deleteLater)
        self.calibration_thread.finished.connect(self.calibration_thread.deleteLater)
        self.calib_worker.progress_update.connect(self.handle_progress_update)
        self.calib_worker.extinction_update.connect(self.handle_extinction_update)
        self.calib_worker.intensity_update.connect(self.handle_plot_update)
        self.calibration_thread.setTerminationEnabled(True)
        self.calibration_thread.start()

        self._disable_buttons()
        self.calibration_thread.finished.connect(self._enable_buttons)

    @pyqtSlot(bool)
    def calc_extinction(self):
        set_lc_state(self.mmc, 'State0')
        extinction = snap_and_average(self.calib.snap_manager)
        set_lc_state(self.mmc, 'State1')
        state1 = snap_and_average(self.calib.snap_manager)
        extinction = self.calib.calculate_extinction(self.swing, self.calib.I_Black, extinction, state1)
        self.ui.le_extinction.setText(str(extinction))

    @pyqtSlot(bool)
    def capture_bg(self):
        self.capture_bg_thread = QThread()
        self.capture_bg_worker = BackgroundCaptureWorker(self, self.calib)
        self.capture_bg_worker.moveToThread(self.capture_bg_thread)
        self.capture_bg_thread.started.connect(self.capture_bg_worker.run)
        self.capture_bg_worker.bg_image_emitter.connect(self.handle_bg_image_update)
        self.capture_bg_worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.capture_bg_worker.finished.connect(self.capture_bg_thread.quit)
        self.capture_bg_worker.finished.connect(self.capture_bg_worker.deleteLater)
        self.capture_bg_thread.finished.connect(self.capture_bg_thread.deleteLater)
        self.capture_bg_thread.start()

        self._disable_buttons()
        self.capture_bg_thread.finished.connect(self._enable_buttons)

    @pyqtSlot(bool)
    def acq_birefringence(self):

        self.acq_thread = QThread()
        self.acq_worker = AcquisitionWorker(self, self.calib)
        self.capture_bg_worker.moveToThread(self.capture_bg_thread)
        self.capture_bg_thread.started.connect(self.capture_bg_worker.run)
        self.capture_bg_worker.bg_image_emitter.connect(self.handle_bg_image_update)
        self.capture_bg_worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.capture_bg_worker.finished.connect(self.capture_bg_thread.quit)
        self.capture_bg_worker.finished.connect(self.capture_bg_worker.deleteLater)
        self.capture_bg_thread.finished.connect(self.capture_bg_thread.deleteLater)
        self.capture_bg_thread.start()

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

    def _disable_buttons(self):
        self.ui.qbutton_calibrate.setEnabled(False)
        self.ui.qbutton_capture_bg.setEnabled(False)
        self.ui.qbutton_calc_extinction.setEnabled(False)

    def _enable_buttons(self):
        self.ui.qbutton_calibrate.setEnabled(True)
        self.ui.qbutton_capture_bg.setEnabled(True)
        self.ui.qbutton_calc_extinction.setEnabled(True)

    def _stop_thread(self):
        self.calib_worker._running = False
        self.calibration_thread.quit()
        self.calibration_thread.wait()

class CalibrationWorker(QtCore.QObject):

    progress_update = pyqtSignal(int)
    extinction_update = pyqtSignal(str)
    intensity_update = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib
        self._running = True

    def stop(self):
        self._running = False
        self.finished.emit()

    def run(self):

        self.calib.intensity_emitter = self.intensity_update
        self.calib.get_full_roi()
        self.progress_update.emit(1)

        # TODO: Decide if displaying ROI is useful feature,
        # include in a pop-up window or napari window?  How to prompt to continue?

        # Check if change of ROI is needed
        if self.calib_window.use_cropped_roi:
            rect = self.calib.check_and_get_roi()
            # cont = self.calib.display_and_check_ROI(rect)
            self.calib_window.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
            self.calib.ROI = (rect.x, rect.y, rect.width, rect.height)

        # Calculate Blacklevel
        logging.info('Calculating Blacklevel ...')
        logging.debug('Calculating Blacklevel ...')
        self.calib.calc_blacklevel()
        logging.info(f'Blacklevel: {self.calib.I_Black}\n')
        logging.debug(f'Blacklevel: {self.calib.I_Black}\n')

        self.progress_update.emit(10)

        # Set LC Wavelength:
        self.calib_window.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.calib_window.wavelength)

        # Optimize States
        self._calibrate_4state() if self.calib_window.calib_scheme == '4-State' else self._calibrate_5state()

        # Return ROI to full FOV
        if self.calib_window.use_cropped_roi:
            self.calib_window.mmc.clearROI()

        # Calculate Extinction
        extinction_ratio = self.calib.calculate_extinction(self.calib.swing, self.calib.I_Black, self.calib.I_Ext,
                                                           self.calib.I_Elliptical)
        self.calib.extinction_ratio = extinction_ratio
        self.extinction_update.emit(str(extinction_ratio))

        # Write Metadata
        self.calib.write_metadata()
        self.progress_update.emit(100)

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {extinction_ratio}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {extinction_ratio}")
        self.finished.emit()
        self._running = False

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

class BackgroundCaptureWorker(QtCore.QObject):

    bg_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib

    def run(self):

        bg_path = os.path.join(self.calib_window.directory, self.calib_window.ui.le_bg_folder.text())
        if not os.path.exists(bg_path):
            os.mkdir(bg_path)
        imgs = self.calib.capture_bg(self.calib_window.n_avg, bg_path)
        img_dim = (imgs.shape[-2], imgs.shape[-1])
        N_channel = 4 if self.calib_window.calib_scheme == '4-State' else 5

        recon = initialize_reconstructor(img_dim, self.calib_window.wavelength, self.calib_window.swing, N_channel,
                                         True, 1, 1, 1, 1, 1, 0, 1, bg_option='None', mode='2D')

        stokes = reconstruct_qlipp_stokes(imgs, recon, None)
        birefringence = reconstruct_qlipp_birefringence(stokes, recon)
        retardance = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength

        self.bg_image_emitter.emit(imgs)
        self.bire_image_emitter.emit([retardance, birefringence[1]])
        self.finished.emit()

#todo: potentially switch to thread pooling now, want to reuse thread so phase reconstructor can be
# computed once
class AcquisitionWorker(QtCore.QObject):

    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib, mode):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib
        self.mode = mode

        if self.mode == 'birefringence' and self.calib_window.birefringence_dim == '2D':
            self.dim = '2D'
        else:
            self.dim = '3D'

    def run(self):

        if self.dim == '2D':
            stack = acquire_2D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib.snap_manager)

        elif self.dim == '3D':
            stack = acquire_3D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib_window.z_start, self.calib_window.z_end, self.calib_window.z_step)

    def _reconstruct(self, stack):
        if self.mode == 'phase' or 'all':
            recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                             self.calib_window.swing, stack.shape[0], False,
                                             1, 1, 1, 1, 1, 0, 1, bg_option='None', mode='2D')

        else:
            recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                             self.calib_window.swing, stack.shape[0],
                                             True, 1, 1, 1, 1, 1, 0, 1, bg_option='None', mode='2D')

    def _save_imgs(self, birefringence, phase):
        pass


class QtLogger(logging.Handler):

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

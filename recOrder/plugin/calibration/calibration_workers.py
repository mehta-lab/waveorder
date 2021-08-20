from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes
import os
import numpy as np
import logging


class CalibrationWorker(QtCore.QObject):
    """
    Class to execute calibration
    """

    # Initialize signals that emit to widget handlers
    progress_update = pyqtSignal(int)
    extinction_update = pyqtSignal(str)
    intensity_update = pyqtSignal(object)
    calib_assessment = pyqtSignal(str)
    calib_assessment_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib

    def run(self):
        """
        Runs the full calibration algorithm and emits necessary signals.

        Returns
        -------

        """

        self.calib.intensity_emitter = self.intensity_update
        self.calib.get_full_roi()
        self.progress_update.emit(1)

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
        self._assess_calibration()

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {extinction_ratio}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {extinction_ratio}")

        # Let thread know that it can finish + deconstruct
        self.finished.emit()

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

    def _assess_calibration(self):

        if 0.2 < self.calib.lca_ext < 0.4:
            if 0.4 < self.calib.lcb_ext < 0.75:
                if self.calib.extinction_ratio >= 100:
                    self.calib_assessment.emit('good')
                    self.calib_assessment_msg.emit('Sucessful Calibration')
                elif 80 <= self.calib.extinction_ratio < 100:
                    self.calib_assessment.emit('okay')
                    self.calib_assessment_msg.emit('Sucessful Calibration, Okay Extinction Ratio')
                else:
                    self.calib_assessment.emit('bad')
                    self.calib_assessment_msg.emit('Poor Extinction, try tuning the linear polarizer to be '
                                                   'perpendicular to the long edge of the LC housing')
            else:
                self.calib_assessment.emit('bad')
                self.calib_assessment_msg.emit('Wrong analyzer handedness or linear polarizer 90 degrees off')
        else:
            self.calib_assessment.emit('bad')
            self.calib_assessment_msg.emit('Calibration Failed, unknown origin of issue')


class BackgroundCaptureWorker(QtCore.QObject):
    """
    Class to execute background capture.
    """

    # Initialize signals to emit to widget handlers
    bg_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib

    def run(self):

        # Make the background folder
        bg_path = os.path.join(self.calib_window.directory, self.calib_window.ui.le_bg_folder.text())
        if not os.path.exists(bg_path):
            os.mkdir(bg_path)

        # capture and return background images
        imgs = self.calib.capture_bg(self.calib_window.n_avg, bg_path)
        img_dim = (imgs.shape[-2], imgs.shape[-1])

        # Prep parameters + initialize reconstructor
        N_channel = 4 if self.calib_window.calib_scheme == '4-State' else 5
        recon = initialize_reconstructor(img_dim, self.calib_window.wavelength, self.calib_window.swing, N_channel,
                                         True, 1, 1, 1, 1, 1, 0, 1, bg_option='None', mode='2D')

        # Reconstruct birefringence from BG images
        stokes = reconstruct_qlipp_stokes(imgs, recon, None)
        birefringence = reconstruct_qlipp_birefringence(stokes, recon)
        retardance = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength

        # Save metadata file and emit imgs
        self.calib.meta_file = os.path.join(bg_path, 'calibration_metadata.txt')
        self.calib.write_metadata()
        self.bg_image_emitter.emit(imgs)
        self.bire_image_emitter.emit([retardance, birefringence[1]])

        # Let thread know that it can finish + deconstruct
        self.finished.emit()

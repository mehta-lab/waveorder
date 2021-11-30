from PyQt5.QtCore import pyqtSignal
from napari.qt.threading import WorkerBaseSignals, WorkerBase
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes
import os
import numpy as np
import logging


class CalibrationSignals(WorkerBaseSignals):
    progress_update = pyqtSignal(int)
    extinction_update = pyqtSignal(str)
    intensity_update = pyqtSignal(object)
    calib_assessment = pyqtSignal(str)
    calib_assessment_msg = pyqtSignal(str)

class BackgroundSignals(WorkerBaseSignals):
    bg_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)


class CalibrationWorker(WorkerBase):
    """
    Class to execute calibration
    """

    def __init__(self, calib_window, calib):
        super().__init__(SignalsClass=CalibrationSignals)

        self.calib_window = calib_window
        self.calib = calib

    # def work(self):
    #
    #     raise ValueError('Test')

    def work(self):
        """
        Runs the full calibration algorithm and emits necessary signals.

        Returns
        -------

        """

        self.calib.intensity_emitter = self.intensity_update
        self.calib.get_full_roi()
        self.progress_update.emit(1)
        self.progress_update.emit(1)
        self._check_abort()

        # Check if change of ROI is needed
        if self.calib_window.use_cropped_roi:
            rect = self.calib.check_and_get_roi()
            # cont = self.calib.display_and_check_ROI(rect)
            self.calib_window.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
            self.calib.ROI = (rect.x, rect.y, rect.width, rect.height)
        self._check_abort()
        # Calculate Blacklevel
        logging.info('Calculating Blacklevel ...')
        logging.debug('Calculating Blacklevel ...')
        self.calib.calc_blacklevel()
        logging.info(f'Blacklevel: {self.calib.I_Black}\n')
        logging.debug(f'Blacklevel: {self.calib.I_Black}\n')
        self._check_abort()

        self.progress_update.emit(10)

        # Set LC Wavelength:
        self.calib_window.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.calib_window.wavelength)
        self._check_abort()
        # Optimize States
        self._calibrate_4state() if self.calib_window.calib_scheme == '4-State' else self._calibrate_5state()

        # Return ROI to full FOV
        if self.calib_window.use_cropped_roi:
            self.calib_window.mmc.clearROI()
        self._check_abort()

        # Calculate Extinction
        extinction_ratio = self.calib.calculate_extinction(self.calib.swing, self.calib.I_Black, self.calib.I_Ext,
                                                           self.calib.I_Elliptical)
        self._check_abort()

        self.calib.extinction_ratio = extinction_ratio
        self.extinction_update.emit(str(extinction_ratio))

        # Write Metadata
        self.calib.write_metadata()
        self.progress_update.emit(100)
        self._check_abort()
        self._assess_calibration()
        self._check_abort()

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {extinction_ratio}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {extinction_ratio}")
        self._check_abort()

        # Let thread know that it can finish + deconstruct
        # self.finished.emit()

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()

    def _calibrate_4state(self):

        self._check_abort()
        self.calib.opt_Iext()
        self._check_abort()
        self.progress_update.emit(60)

        self.calib.opt_I0()
        self.progress_update.emit(65)
        self._check_abort()

        self.calib.opt_I60(0.05, 0.05)
        self.progress_update.emit(75)
        self._check_abort()

        self.calib.opt_I120(0.05, 0.05)
        self.progress_update.emit(85)
        self._check_abort()


    def _calibrate_5state(self):

        self.calib.opt_Iext()
        self.progress_update.emit(50)
        self._check_abort()

        self.calib.opt_I0()
        self.progress_update.emit(55)
        self._check_abort()

        self.calib.opt_I45(0.05, 0.05)
        self.progress_update.emit(65)
        self._check_abort()

        self.calib.opt_I90(0.05, 0.05)
        self.progress_update.emit(75)
        self._check_abort()

        self.calib.opt_I135(0.05, 0.05)
        self.progress_update.emit(85)
        self._check_abort()

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


class BackgroundCaptureWorker(WorkerBase):
    """
    Class to execute background capture.
    """

    def __init__(self, calib_window, calib):
        super().__init__(SignalsClass=BackgroundSignals)
        self.calib_window = calib_window
        self.calib = calib

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()

    # def work(self):
    #
    #     raise ValueError('Test')


    def work(self):

        # Make the background folder
        bg_path = os.path.join(self.calib_window.directory, self.calib_window.ui.le_bg_folder.text())
        if not os.path.exists(bg_path):
            os.mkdir(bg_path)

        self._check_abort()

        # capture and return background images
        imgs = self.calib.capture_bg(self.calib_window.n_avg, bg_path)
        img_dim = (imgs.shape[-2], imgs.shape[-1])

        # initialize reconstructor
        recon = initialize_reconstructor('birefringence',
                                         image_dim=img_dim,
                                         calibration_scheme=self.calib_window.calib_scheme,
                                         wavelength_nm=self.calib_window.wavelength,
                                         swing=self.calib_window.swing,
                                         bg_correction='None')

        self._check_abort()

        # Reconstruct birefringence from BG images
        stokes = reconstruct_qlipp_stokes(imgs, recon, None)
        self._check_abort()

        birefringence = reconstruct_qlipp_birefringence(stokes, recon)
        self._check_abort()

        retardance = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength

        # Save metadata file and emit imgs
        self.calib.meta_file = os.path.join(bg_path, 'calibration_metadata.txt')
        self.calib.write_metadata()
        self._check_abort()
        self.bg_image_emitter.emit(imgs)
        self.bire_image_emitter.emit([retardance, birefringence[1]])

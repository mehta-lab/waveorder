from PyQt5.QtCore import pyqtSignal
from napari.qt.threading import WorkerBaseSignals, WorkerBase, thread_worker
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes
from recOrder.io.core_functions import set_lc_state, snap_and_average
from recOrder.io.utils import MockEmitter
from recOrder.calib.Calibration import LC_DEVICE_NAME
from recOrder.io.metadata_reader import MetadataReader, get_last_metadata_file
import os
import numpy as np
import glob
import logging
import json


class CalibrationSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    progress_update = pyqtSignal(tuple)
    extinction_update = pyqtSignal(str)
    intensity_update = pyqtSignal(object)
    calib_assessment = pyqtSignal(str)
    calib_assessment_msg = pyqtSignal(str)
    calib_file_emit = pyqtSignal(str)
    plot_sequence_emit = pyqtSignal(str)
    aborted = pyqtSignal()


class BackgroundSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    bg_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    aborted = pyqtSignal()


class CalibrationWorker(WorkerBase):
    """
    Class to execute calibration
    """

    def __init__(self, calib_window, calib):
        super().__init__(SignalsClass=CalibrationSignals)

        # initialize current state of GUI + Calibration class
        self.calib_window = calib_window
        self.calib = calib

    def work(self):
        """
        Runs the full calibration algorithm and emits necessary signals.
        """

        self.plot_sequence_emit.emit('Coarse')
        self.calib.intensity_emitter = self.intensity_update
        self.calib.plot_sequence_emitter = self.plot_sequence_emit
        self.calib.get_full_roi()
        self.progress_update.emit((1, 'Calculating Blacklevel...'))
        self._check_abort()

        # Check if change of ROI is needed
        if self.calib_window.use_cropped_roi:
            rect = self.calib.check_and_get_roi()
            self.calib_window.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
            self.calib.ROI = (rect.x, rect.y, rect.width, rect.height)

        self._check_abort()

        logging.info('Calculating Black Level ...')
        logging.debug('Calculating Black Level ...')
        self.calib.close_shutter_and_calc_blacklevel()

        # Calculate Blacklevel
        logging.info(f'Black Level: {self.calib.I_Black:.0f}\n')
        logging.debug(f'Black Level: {self.calib.I_Black:.0f}\n')

        self._check_abort()
        self.progress_update.emit((10, 'Calibrating Extinction State...'))

        # Open shutter
        self.calib.open_shutter()

        # Set LC Wavelength:
        self.calib.set_wavelength(int(self.calib_window.wavelength))
        if self.calib_window.calib_mode == 'MM-Retardance':
            self.calib_window.mmc.setProperty(LC_DEVICE_NAME, 'Wavelength', self.calib_window.wavelength)

        self._check_abort()

        # Optimize States
        self._calibrate_4state() if self.calib_window.calib_scheme == '4-State' else self._calibrate_5state()

        # Reset shutter autoshutter
        self.calib.reset_shutter()

        # Return ROI to full FOV
        if self.calib_window.use_cropped_roi:
            self.calib_window.mmc.clearROI()

        self._check_abort()

        # Calculate Extinction
        extinction_ratio = self.calib.calculate_extinction(self.calib.swing, self.calib.I_Black, self.calib.I_Ext,
                                                           self.calib.I_Elliptical)
        self._check_abort()

        # Update main GUI with extinction ratio
        self.calib.extinction_ratio = extinction_ratio
        self.extinction_update.emit(str(extinction_ratio))

        # Write Metadata
        self.calib.meta_file = os.path.join(self.calib_window.directory, 'calibration_metadata.txt')
        idx = 1
        while os.path.exists(self.calib.meta_file):
            if self.calib.meta_file == os.path.join(self.calib_window.directory, 'calibration_metadata.txt'):
                self.calib.meta_file = os.path.join(self.calib_window.directory, 'calibration_metadata_1.txt')
            else:
                idx += 1
                self.calib.meta_file = os.path.join(self.calib_window.directory, f'calibration_metadata_{idx}.txt')


        self.calib.write_metadata(notes=self.calib_window.ui.le_notes_field.text())
        self.calib_file_emit.emit(self.calib.meta_file)
        self.progress_update.emit((100, 'Finished'))

        self._check_abort()

        # Perform calibration assessment based on retardance values
        self._assess_calibration()

        self._check_abort()

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {extinction_ratio:.2f}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {extinction_ratio:.2f}")

    def _check_abort(self):
        """
        Called if the user presses the STOP button.
        Needed to be checked after every major step to stop the process
        """

        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def _calibrate_4state(self):
        """
        Run through the 4-state calibration algorithm
        """

        self.calib.calib_scheme = '4-State'

        self._check_abort()

        # Optimize Extinction State
        self.calib.opt_Iext()

        self._check_abort()
        self.progress_update.emit((60, 'Calibrating State 1...'))

        # Optimize first elliptical (reference) state
        self.calib.opt_I0()
        self.progress_update.emit((65, 'Calibrating State 2...'))

        self._check_abort()

        # Optimize 60 deg state
        self.calib.opt_I60(0.05, 0.05)
        self.progress_update.emit((75, 'Calibrating State 3...'))

        self._check_abort()

        # Optimize 120 deg state
        self.calib.opt_I120(0.05, 0.05)
        self.progress_update.emit((85, 'Writing Metadata...'))

        self._check_abort()

    def _calibrate_5state(self):

        self.calib.calib_scheme = '5-State'

        # Optimize Extinction State
        self.calib.opt_Iext()
        self.progress_update.emit((50, 'Calibrating State 1...'))

        self._check_abort()

        # Optimize First elliptical state
        self.calib.opt_I0()
        self.progress_update.emit((55, 'Calibrating State 2...'))

        self._check_abort()

        # Optimize 45 deg state
        self.calib.opt_I45(0.05, 0.05)
        self.progress_update.emit((65, 'Calibrating State 3...'))

        self._check_abort()

        # Optimize 90 deg state
        self.calib.opt_I90(0.05, 0.05)
        self.progress_update.emit((75, 'Calibrating State 4...'))

        self._check_abort()

        # Optimize 135 deg state
        self.calib.opt_I135(0.05, 0.05)
        self.progress_update.emit((85, 'Writing Metadata...'))

        self._check_abort()

    def _assess_calibration(self):
        """
        Assesses the quality of calibration based off retardance values.
        Attempts to determine whether certain optical components are out of place.
        """

        if self.calib.extinction_ratio >= 100:
            self.calib_assessment.emit('good')
            self.calib_assessment_msg.emit('Successful Calibration')
        elif 80 <= self.calib.extinction_ratio < 100:
            self.calib_assessment.emit('okay')
            self.calib_assessment_msg.emit('Successful Calibration, Okay Extinction Ratio')
        else:
            self.calib_assessment.emit('bad')
            message = ("Possibilities are: a) linear polarizer and LC are not oriented properly, "
                       "b) circular analyzer has wrong handedness, "
                       "c) the condenser is not setup for Kohler illumination, "
                       "d) a component, such as autofocus dichroic or sample chamber, distorts the polarization state")

            self.calib_assessment_msg.emit('Poor Extinction. '+message)


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
            return True

    def work(self):

        # Make the background folder
        bg_path = os.path.join(self.calib_window.directory, self.calib_window.ui.le_bg_folder.text())
        if not os.path.exists(bg_path):
            os.mkdir(bg_path)
        else:

            # increment background paths
            idx = 1
            while os.path.exists(bg_path+f'_{idx}'):
                idx += 1

            bg_path = bg_path+f'_{idx}'
            for state_file in glob.glob(os.path.join(bg_path, 'State*')):
                os.remove(state_file)

            if os.path.exists(os.path.join(bg_path, 'calibration_metadata.txt')):
                os.remove(os.path.join(bg_path, 'calibration_metadata.txt'))

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

        # Convert retardance to nm
        retardance = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength

        # Save metadata file and emit imgs
        self.calib.meta_file = os.path.join(bg_path, 'calibration_metadata.txt')

        microscope_params = {
             'n_objective_media': float(self.calib_window.ui.le_n_media.text()) if self.calib_window.ui.le_n_media.text() != '' else None,
             'objective_NA': float(self.calib_window.ui.le_obj_na.text()) if self.calib_window.ui.le_obj_na.text() != '' else None,
             'condenser_NA': float(self.calib_window.ui.le_cond_na.text()) if self.calib_window.ui.le_cond_na.text() != '' else None,
             'magnification': float(self.calib_window.ui.le_mag.text()) if self.calib_window.ui.le_mag.text() != '' else None,
             'pixel_size': float(self.calib_window.ui.le_ps.text()) if self.calib_window.ui.le_ps.text() != '' else None
        }

        self.calib.write_metadata(notes=self.calib_window.ui.le_notes_field.text(), microscope_params=microscope_params)

        # Update last calibration file
        note = self.calib_window.ui.le_notes_field.text()

        with open(self.calib_window.last_calib_meta_file, 'r') as file:
            current_json = json.load(file)

        old_note = current_json['Notes']
        if old_note is None or old_note == '' or old_note == note:
            current_json['Notes'] = note
        else:
            current_json['Notes'] = old_note + ', ' + note

        current_json['Microscope Parameters'] = microscope_params

        with open(self.calib_window.last_calib_meta_file, 'w') as file:
            json.dump(current_json, file, indent=1)

        self._check_abort()

        # Emit background images + background birefringence
        self.bg_image_emitter.emit(imgs)
        self.bire_image_emitter.emit([retardance, birefringence[1]])


@thread_worker
def load_calibration(calib, metadata: MetadataReader):
    """
    Sets MM properties based upon calibration metadata file


    Parameters
    ----------
    calib:          (object) recOrder Calibration Class
    metadata:       (object) MetadataReader instance

    Returns
    -------
    calib           (object) updated recOrder Calibration Class
    """

    for state, lca, lcb in zip([f'State{i}' for i in range(5)], metadata.LCA_retardance, metadata.LCB_retardance):
        calib.define_lc_state(state, lca, lcb)

    # Calculate black level after loading these properties
    calib.intensity_emitter = MockEmitter()
    calib.close_shutter_and_calc_blacklevel()
    calib.open_shutter()
    set_lc_state(calib.mmc, calib.group, 'State0')
    calib.I_Ext = snap_and_average(calib.snap_manager)
    set_lc_state(calib.mmc, calib.group, 'State1')
    calib.I_Elliptical = snap_and_average(calib.snap_manager)
    calib.reset_shutter()

    yield str(calib.calculate_extinction(calib.swing, calib.I_Black, calib.I_Ext, calib.I_Elliptical))

    return calib

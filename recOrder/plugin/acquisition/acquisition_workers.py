from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_qlipp_phase2D, reconstruct_qlipp_phase3D
from recOrder.acq.acq_single_stack import acquire_2D, acquire_3D
from recOrder.io.utils import load_bg
import logging
from waveorder.io.writer import WaveorderWriter
import json
import numpy as np
import os

class AcquisitionWorker(QtCore.QObject):

    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, calib_window, calib, mode):
        super().__init__()
        self.calib_window = calib_window
        self.calib = calib
        self.mode = mode
        self.n_slices = None

        if self.mode == 'birefringence' and self.calib_window.birefringence_dim == '2D':
            self.dim = '2D'
        else:
            self.dim = '3D'

    def run(self):

        logging.info('Running Acquisition...')
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')
            stack = acquire_2D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib.snap_manager)
            self.n_slices = 1

        else:
            logging.debug('Acquiring 3D stack')
            stack = acquire_3D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib_window.z_start, self.calib_window.z_end, self.calib_window.z_step)

            self.n_slices = len(range(self.calib_window.z_start, self.calib_window.z_end+self.calib_window.z_step,
                                      self.calib_window.z_step))

        birefringence, phase = self._reconstruct(stack)
        if self.calib_window.save_imgs:
            logging.debug('Saving Images')
            self._save_imgs(birefringence, phase)

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')
        self.bire_image_emitter.emit(birefringence)
        self.phase_image_emitter.emit(phase)
        self.finished.emit()

    def _reconstruct(self, stack):
        if self.mode == 'phase' or self.mode == 'all':
            if not self.calib_window.phase_reconstructor:
                logging.debug('Using previous reconstruction settings')
                recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                 self.calib_window.swing, stack.shape[0], False,
                                                 self.calib_window.obj_na, self.calib_window.cond_na,
                                                 self.calib_window.mag, self.n_slices, self.calib_window.z_step,
                                                 self.calib_window.pad_z, self.calib_window.ps,
                                                 self.calib_window.bg_option, mode=self.dim)
                self.phase_reconstructor_emitter.emit(recon)
            else:
                if self._reconstructor_changed():
                    logging.debug('Reconstruction settings changed, updating reconstructor')
                    recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                     self.calib_window.swing, stack.shape[0], False,
                                                     self.calib_window.obj_na, self.calib_window.cond_na,
                                                     self.calib_window.mag, self.n_slices, self.calib_window.z_step,
                                                     self.calib_window.pad_z, self.calib_window.ps,
                                                     self.calib_window.bg_option, mode=self.dim)
                else:
                    recon = self.calib_window.phase_reconstructor

        else:
            logging.debug('Creating birefringence only reconstructor')
            recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                             self.calib_window.swing, stack.shape[0],
                                             True, 1, 1, 1, 1, 1, 0, 1, bg_option=self.calib_window.bg_option, mode='2D')

        if self.calib_window.bg_option != 'None':
            logging.debug('Loading BG Data')
            bg_data = self._load_bg(self.calib_window.acq_bg_directory, stack.shape[-2], stack.shape[-1])
            bg_stokes = recon.Stokes_recon(bg_data)
            bg_stokes = recon.Stokes_transform(bg_stokes)
        else:
            logging.debug('No Background Correction method chosen')
            bg_stokes = None

        logging.debug('Reconstructing...')
        stokes = reconstruct_qlipp_stokes(stack, recon, bg_stokes)
        birefringence = None
        phase = None

        if self.mode == 'all':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            phase = reconstruct_qlipp_phase2D(stokes[0], recon) if self.dim == '2D' \
                else reconstruct_qlipp_phase3D(stokes[0], recon)

        elif self.mode == 'phase':
            phase = reconstruct_qlipp_phase2D(stokes[0], recon) if self.dim == '2D' \
                else reconstruct_qlipp_phase3D(stokes[0], recon)
        elif self.mode == 'birefringence':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)

        else:
            raise ValueError('Reconstruction Mode Not Understood')

        birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength
        return birefringence, phase

    def _save_imgs(self, birefringence, phase):
        writer = WaveorderWriter(self.calib_window.save_directory, 'physical')

        if birefringence is not None:
            i = 0
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Birefringence_Snap_{i}.zarr')):
                i += 1
            writer.create_zarr_root(f'Birefringence_Snap_{i}.zarr')
            writer.create_position(0)
            writer.init_array()
            writer.write(birefringence)

        if phase is not None:
            i = 0
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Phase_Snap_{i}.zarr')):
                i += 1
            writer.create_zarr_root(f'Phase_Snap_{i}.zarr')
            writer.create_position(0)
            writer.write(birefringence)

    def _load_bg(self, path, height, width):

        try:
            meta_path = open(os.path.join(path,'calibration_metadata.txt'))
            roi = json.load(meta_path)['Summary']['ROI Used (x, y, width, height)']
        except:
            roi = None

        bg_data = load_bg(path, height, width, roi)

        return bg_data

    def _reconstructor_changed(self):
        changed = None

        attr_list = {'phase_dim': 'mode',
                     'n_slices': 'N_defocus',
                     'mag': 'mag',
                     'pad_z': 'pad_z',
                     'n_media': 'n_media',
                     'ps': 'ps',
                     'swing': 'chi',
                     'bg_option': 'bg_option'
                     }
        attr_modified_list = {'obj_na': 'NA_obj',
                              'cond_na': 'NA_illu',
                              'wavelength': 'lambda_illu'
                              }

        for key, value in attr_list.items():
            if getattr(self.calib_window, key) != getattr(self.calib_window.phase_reconstructor, value):
                changed = True
            else:
                changed = False
        for key, value in attr_modified_list.items():
            if key == 'wavelength':
                if self.calib_window.wavelength * 1e-3 / self.calib_window.n_media != \
                        self.calib_window.phase_reconstructor.lambda_illu:
                    changed = True
                else:
                    changed = False
            elif getattr(self.calib_window, key)/self.calib_window.n_media != \
                    getattr(self.calib_window.phase_reconstructor, value):
                changed = True
            else:
                changed = False

        return changed
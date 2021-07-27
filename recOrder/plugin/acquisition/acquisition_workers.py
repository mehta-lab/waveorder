from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_qlipp_phase2D, reconstruct_qlipp_phase3D
from recOrder.acq.acq_single_stack import acquire_2D, acquire_3D
from recOrder.io.utils import load_bg
import logging
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

        if self.dim == '2D':
            stack = acquire_2D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib.snap_manager)
            self.n_slices = 1

        elif self.dim == '3D':
            stack = acquire_3D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib_window.z_start, self.calib_window.z_end, self.calib_window.z_step)

            self.n_slices = len(range(self.calib_window.z_start, self.calib_window.z_end+self.calib_window.z_step,
                                      self.calib_window.z_step))

        birefringence, phase = self._reconstruct(stack)
        self.finished.emit()

    def _reconstruct(self, stack):
        if self.mode == 'phase' or 'all':
            if not self.calib_window.phase_reconstructor:
                recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                 self.calib_window.swing, stack.shape[0], False,
                                                 self.calib_window.obj_na, self.calib_window.cond_na, self.calib_window.mag,
                                                 self.n_slices, self.calib_window.z_step, self.calib_window.pad_z,
                                                 self.calib_window.ps, self.calib_window.bg_option, mode=self.dim)
                self.phase_reconstructor_emitter.emit(recon)
            else:
                if self._reconstructor_changed():
                    recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                     self.calib_window.swing, stack.shape[0], False,
                                                     self.calib_window.obj_na, self.calib_window.cond_na,
                                                     self.calib_window.mag,
                                                     self.n_slices, self.calib_window.z_step, self.calib_window.pad_z,
                                                     self.calib_window.ps, self.calib_window.bg_option, mode=self.dim)
                else:
                    recon = self.calib_window.phase_reconstructor

        else:
            recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                             self.calib_window.swing, stack.shape[0],
                                             True, 1, 1, 1, 1, 1, 0, 1, bg_option='None', mode='2D')

        if self.bg_option != 'None':
            bg_data = self._load_bg(self.calib_window.acq_bg_directory)
            bg_stokes = recon.Stokes_recon(bg_data)
            bg_stokes = recon.Stokes_transform(bg_stokes)
        else:
            bg_stokes = None

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
        pass

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
                            }

        for key, value in attr_list.items():
            if getattr(self.calib_window, key) != getattr(self.calib_window.phase_reconstructor, value):
                changed = True
            else:
                changed = False
        for key, value in attr_modified_list.items():
            if getattr(self.calib_window, key)/self.calib_window.n_media != getattr(self.calib_window.phase_reconstructor, value):
                changed = True
            else:
                changed = False

        if self.calib_window.wavelength * 1e-3 / self.calib_window.n_media != self.calib_window.phase_reconstructor.lambda_illu:
            changed = True
        else:
            changed = False

        return changed
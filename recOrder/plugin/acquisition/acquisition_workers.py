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

# TODO: Change Acquisition to Using MDA?  This would allow for easy sequencing...
# TODO: Cache common OTF's on local computers and use those for reconstruction
# TODO: Fix bug in dimensionality, 2D/3D doesn't make a difference
class AcquisitionWorker(QtCore.QObject):
    """
    Class to execute a birefringence/phase acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    # Initialize signals to emit to widget handlers
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

        # Acquire 2D stack
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')
            stack = acquire_2D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib.snap_manager)
            self.n_slices = 1

        # Acquire 3D stack
        else:
            logging.debug('Acquiring 3D stack')
            stack = acquire_3D(self.calib_window.mm, self.calib_window.mmc, self.calib_window.calib_scheme,
                               self.calib_window.z_start, self.calib_window.z_end, self.calib_window.z_step)

            self.n_slices = int(len(np.arange(self.calib_window.z_start, self.calib_window.z_end+self.calib_window.z_step,
                                      self.calib_window.z_step)))

        # Reconstruct snapped images
        birefringence, phase = self._reconstruct(stack)

        # Save images if specified
        if self.calib_window.save_imgs:
            logging.debug('Saving Images')
            self._save_imgs(birefringence, phase)

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')

        # Emit the images and let thread know function is finished
        self.bire_image_emitter.emit(birefringence)
        self.phase_image_emitter.emit(phase)
        self.finished.emit()

    def _reconstruct(self, stack):
        """
        Method to reconstruct, given a 2D or 3D stack.  First need to initialize the reconstructor given
        what type of acquisition it is (birefringence only skips a lot of heavy compute needed for phase).
        This function also checks to see if the reconstructor needs to be updated from previous acquisitions

        Parameters
        ----------
        stack:          (nd-array) Dimensions are either (Z, C, Y, X) or (C, Y, X)

        Returns
        -------

        """

        # Initialize the heavy reconstuctor
        if self.mode == 'phase' or self.mode == 'all':

            # if no reconstructor has been initialized before
            if not self.calib_window.phase_reconstructor:
                logging.debug('Computing new reconstructor')
                recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                 self.calib_window.swing, stack.shape[0], False,
                                                 self.calib_window.obj_na, self.calib_window.cond_na,
                                                 self.calib_window.mag, self.n_slices, self.calib_window.z_step,
                                                 self.calib_window.pad_z, self.calib_window.ps,
                                                 self.calib_window.bg_option, mode=self.dim)
                self.phase_reconstructor_emitter.emit(recon)

            # if previous reconstructor exists
            else:

                # compute new reconstructor
                if self._reconstructor_changed():
                    logging.debug('Reconstruction settings changed, updating reconstructor')
                    recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                                     self.calib_window.swing, stack.shape[0], False,
                                                     self.calib_window.obj_na, self.calib_window.cond_na,
                                                     self.calib_window.mag, self.n_slices, self.calib_window.z_step,
                                                     self.calib_window.pad_z, self.calib_window.ps,
                                                     self.calib_window.bg_option, mode=self.dim)
                # use previous reconstructor
                else:
                    logging.debug('Using previous reconstruction settings')
                    recon = self.calib_window.phase_reconstructor

        # if phase isn't desired, initialize the lighter birefringence only reconstructor
        else:
            logging.debug('Creating birefringence only reconstructor')
            recon = initialize_reconstructor((stack.shape[-2], stack.shape[-1]), self.calib_window.wavelength,
                                             self.calib_window.swing, stack.shape[0],
                                             True, 1, 1, 1, 1, 1, 0, 1, bg_option=self.calib_window.bg_option, mode='2D')

        # Check to see if background correction is desired and compute BG stokes
        if self.calib_window.bg_option != 'None':
            logging.debug('Loading BG Data')
            bg_data = self._load_bg(self.calib_window.acq_bg_directory, stack.shape[-2], stack.shape[-1])
            bg_stokes = recon.Stokes_recon(bg_data)
            bg_stokes = recon.Stokes_transform(bg_stokes)
        else:
            logging.debug('No Background Correction method chosen')
            bg_stokes = None

        # Begin reconstruction with stokes (needed for birefringence or phase)
        logging.debug('Reconstructing...')
        stokes = reconstruct_qlipp_stokes(stack, recon, bg_stokes)

        # initialize empty variables to pass along
        birefringence = None
        phase = None

        # reconstruct both phase and birefringence
        if self.mode == 'all':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            phase = reconstruct_qlipp_phase2D(stokes[0], recon) if self.dim == '2D' \
                else reconstruct_qlipp_phase3D(stokes[0], recon)

        # reconstruct phase only
        elif self.mode == 'phase':
            phase = reconstruct_qlipp_phase2D(stokes[0], recon) if self.dim == '2D' \
                else reconstruct_qlipp_phase3D(stokes[0], recon)

        # reconstruct birefringence only
        elif self.mode == 'birefringence':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength

        else:
            raise ValueError('Reconstruction Mode Not Understood')

        # return both variables, could contain images or could be null
        return birefringence, phase

    def _save_imgs(self, birefringence, phase):
        """
        function to save images.  Seperates out both birefringence and phase into separate zarr stores.
        Makes sure file names do not overlap, i.e. nothing is overwritten.

        Parameters
        ----------
        birefringence:      (nd-array or None) birefringence image(s)
        phase:              (nd-array or None) phase image(s)

        Returns
        -------

        """
        writer = WaveorderWriter(self.calib_window.save_directory, 'physical')

        if birefringence is not None:

            # initialize
            chunk_size = (1,1,1,birefringence.shape[-2],birefringence.shape[-1])
            i = 0

            # increment filename one more than last found saved snap
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Birefringence_Snap_{i}.zarr')):
                i += 1

            # create zarr root and position group
            writer.create_zarr_root(f'Birefringence_Snap_{i}.zarr')
            writer.create_position(0)

            # Check if 2D
            if len(birefringence.shape) == 3:
                writer.init_array((1,4,1,birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = 0

            # Check if 3D
            else:
                writer.init_array((1, 4, birefringence.shape[-3], birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = [0, birefringence.shape[-3]]

            # Write the data to disk
            writer.write(birefringence, t=0, c=[0,4], z=z)

        if phase is not None:

            # initialize
            chunk_size = (1,1,1,phase.shape[-2],phase.shape[-1])

            # increment filename one more than last found saved snap
            i = 0
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Phase_Snap_{i}.zarr')):
                i += 1

            # create zarr root and position group
            writer.create_zarr_root(f'Phase_Snap_{i}.zarr')
            writer.create_position(0)

            # Check if 2D
            if len(phase.shape) == 2:
                writer.init_array((1,1,1,phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase2D'])
                z = 0

            # Check if 3D
            else:
                writer.init_array((1, 1, phase.shape[-3], phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase3D'])
                print(np.shape(phase))
                z = [0, phase.shape[-3]]

            # Write data to disk
            writer.write(phase, t=0, c=0, z=z)

    def _load_bg(self, path, height, width):
        """
        Load background and calibration metadata.

        Parameters
        ----------
        path:           (str) path to the BG folder
        height:         (int) height of BG image
        width:          (int) widht of BG image

        Returns
        -------

        """

        #TODO: Change to just accept ROI
        try:
            meta_path = open(os.path.join(path,'calibration_metadata.txt'))
            roi = json.load(meta_path)['Summary']['ROI Used (x, y, width, height)']
        except:
            roi = None

        bg_data = load_bg(path, height, width, roi)

        return bg_data

    def _reconstructor_changed(self):
        """
        Function to check if the reconstructor has changed from the previous one in memory.
        Serves to check if the worker attributes and reconstructor attributes have diverged.

        Returns
        -------

        """


        changed = None

        # Attributes that are directly equivalent to worker attributes
        attr_list = {'phase_dim': 'phase_deconv',
                     'n_slices': 'N_defocus',
                     'mag': 'mag',
                     'pad_z': 'pad_z',
                     'n_media': 'n_media',
                     'ps': 'ps',
                     'swing': 'chi',
                     'bg_option': 'bg_option'
                     }

        # attributes that are modified upon passing them to reconstructor
        attr_modified_list = {'obj_na': 'NA_obj',
                              'cond_na': 'NA_illu',
                              'wavelength': 'lambda_illu'
                              }

        # check if equivalent attributes have diverged
        for key, value in attr_list.items():
            if getattr(self.calib_window, key) != getattr(self.calib_window.phase_reconstructor, value):
                changed = True
            else:
                changed = False

        # modify attributes to be equivalent and check for divergence
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
from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_phase2D, reconstruct_phase3D
from recOrder.acq.acq_functions import generate_acq_settings, acquire_from_settings
from recOrder.io.utils import load_bg
from napari.qt.threading import WorkerBaseSignals, WorkerBase
import logging
from waveorder.io.writer import WaveorderWriter
import json
import numpy as np
import os
import shutil
import time


class AcquisitionSignals(WorkerBaseSignals):
    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    aborted = pyqtSignal()


# TODO: Cache common OTF's on local computers and use those for reconstruction
# TODO: Fix bug in dimensionality, 2D/3D doesn't make a difference?
class AcquisitionWorker(WorkerBase):
    """
    Class to execute a birefringence/phase acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window, calib, mode):
        super().__init__(SignalsClass=AcquisitionSignals)
        self.calib_window = calib_window
        self.calib = calib
        self.mode = mode
        self.n_slices = None
        self.prefix = 'recOrderPluginSnap'
        self.dm = self.calib_window.mm.displays()

        if self.mode == 'birefringence' and self.calib_window.birefringence_dim == '2D':
            self.dim = '2D'
        else:
            self.dim = '3D'

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):

        logging.info('Running Acquisition...')
        save_dir = self.calib_window.save_directory if self.calib_window.save_directory else self.calib_window.directory

        channels = ['State0', 'State1', 'State2', 'State3']
        if self.calib_window.calib_scheme == '5-State':
            channels.append('State4')

        self._check_abort()

        # Acquire 2D stack
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')

            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group='Channel',
                                             channels=channels,
                                             save_dir=save_dir,
                                             prefix=self.prefix)
            self._check_abort()
            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True) # (1, 4, 1, Y, X) array
            time.sleep(1)
            self._cleanup_acq()

        # Acquire 3D stack
        else:
            logging.debug('Acquiring 3D stack')
            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group='Channel',
                                             channels=channels,
                                             zstart=self.calib_window.z_start,
                                             zend=self.calib_window.z_end,
                                             zstep=self.calib_window.z_step,
                                             save_dir=save_dir,
                                             prefix=self.prefix)

            self._check_abort()

            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)  # (1, 4, Z, Y, X) array
            time.sleep(1)
            self._cleanup_acq()
            self._check_abort()

        # Reconstruct snapped images
        self._check_abort()
        self.n_slices = stack.shape[2]
        birefringence, phase = self._reconstruct(stack[0])
        self._check_abort()

        # Save images if specified
        if self.calib_window.save_imgs:
            logging.debug('Saving Images')
            self._save_imgs(birefringence, phase)

        self._check_abort()

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')

        # Emit the images and let thread know function is finished
        self.bire_image_emitter.emit(birefringence)
        self.phase_image_emitter.emit(phase)

    def _reconstruct(self, stack):
        """
        Method to reconstruct, given a 2D or 3D stack.  First need to initialize the reconstructor given
        what type of acquisition it is (birefringence only skips a lot of heavy compute needed for phase).
        This function also checks to see if the reconstructor needs to be updated from previous acquisitions

        Parameters
        ----------
        stack:          (nd-array) Dimensions are either (C, Z Y, X)

        Returns
        -------

        """

        # get rid of z-dimension if 2D acquisition
        stack = stack[:, 0] if self.n_slices == 1 else stack

        self._check_abort()

        # Initialize the heavy reconstuctor
        if self.mode == 'phase' or self.mode == 'all':

            self._check_abort()
            # if no reconstructor has been initialized before
            if not self.calib_window.phase_reconstructor:
                logging.debug('Computing new reconstructor')


                recon = initialize_reconstructor('QLIPP',
                                                 image_dim=(stack.shape[-2], stack.shape[-1]),
                                                 wavelength_nm=self.calib_window.wavelength,
                                                 swing=self.calib_window.swing,
                                                 calibration_scheme=self.calib_window.calib_scheme,
                                                 NA_obj=self.calib_window.obj_na,
                                                 NA_illu=self.calib_window.cond_na,
                                                 mag=self.calib_window.mag,
                                                 n_slices=self.n_slices,
                                                 z_step_um=self.calib_window.z_step,
                                                 pad_z=self.calib_window.pad_z,
                                                 pixel_size_um=self.calib_window.ps,
                                                 bg_correction=self.calib_window.bg_option,
                                                 n_obj_media=self.calib_window.n_media,
                                                 mode=self.calib_window.phase_dim,
                                                 use_gpu=False, gpu_id=0)

                self.phase_reconstructor_emitter.emit(recon)

            # if previous reconstructor exists
            else:
                self._check_abort()
                # compute new reconstructor
                if self._reconstructor_changed():
                    logging.debug('Reconstruction settings changed, updating reconstructor')
                    recon = initialize_reconstructor('QLIPP',
                                                     image_dim=(stack.shape[-2], stack.shape[-1]),
                                                     wavelength_nm=self.calib_window.wavelength,
                                                     swing=self.calib_window.swing,
                                                     calibration_scheme=self.calib_window.calib_scheme,
                                                     NA_obj=self.calib_window.obj_na,
                                                     NA_illu=self.calib_window.cond_na,
                                                     mag=self.calib_window.mag,
                                                     n_slices=self.n_slices,
                                                     z_step_um=self.calib_window.z_step,
                                                     pad_z=self.calib_window.pad_z,
                                                     pixel_size_um=self.calib_window.ps,
                                                     bg_correction=self.calib_window.bg_option,
                                                     n_obj_media=self.calib_window.n_media,
                                                     mode=self.calib_window.phase_dim,
                                                     use_gpu=False, gpu_id=0)
                # use previous reconstructor
                else:

                    logging.debug('Using previous reconstruction settings')
                    recon = self.calib_window.phase_reconstructor

        # if phase isn't desired, initialize the lighter birefringence only reconstructor
        else:
            self._check_abort()
            logging.debug('Creating birefringence only reconstructor')
            recon = initialize_reconstructor('birefringence',
                                             image_dim=(stack.shape[-2], stack.shape[-1]),
                                             calibration_scheme=self.calib_window.calib_scheme,
                                             wavelength_nm=self.calib_window.wavelength,
                                             swing=self.calib_window.swing,
                                             bg_correction=self.calib_window.bg_option,
                                             n_slices=self.n_slices)


        # Check to see if background correction is desired and compute BG stokes
        if self.calib_window.bg_option != 'None':
            logging.debug('Loading BG Data')
            self._check_abort()
            bg_data = self._load_bg(self.calib_window.acq_bg_directory, stack.shape[-2], stack.shape[-1])
            self._check_abort()
            bg_stokes = recon.Stokes_recon(bg_data)
            self._check_abort()
            bg_stokes = recon.Stokes_transform(bg_stokes)
            self._check_abort()
        else:
            logging.debug('No Background Correction method chosen')
            bg_stokes = None

        # Begin reconstruction with stokes (needed for birefringence or phase)
        logging.debug('Reconstructing...')
        self._check_abort()
        stokes = reconstruct_qlipp_stokes(stack, recon, bg_stokes)
        self._check_abort()

        # initialize empty variables to pass along
        birefringence = None
        phase = None

        # reconstruct both phase and birefringence
        if self.mode == 'all':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength
            self._check_abort()
            phase = reconstruct_phase2D(stokes[0], recon) if self.calib_window.phase_dim == '2D' \
                else reconstruct_phase3D(stokes[0], recon)
            self._check_abort()

        # reconstruct phase only
        elif self.mode == 'phase':
            phase = reconstruct_phase2D(stokes[0], recon) if self.calib_window.phase_dim == '2D' \
                else reconstruct_phase3D(stokes[0], recon)
            self._check_abort()

        # reconstruct birefringence only
        elif self.mode == 'birefringence':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength
            self._check_abort()

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
        writer = WaveorderWriter(self.calib_window.save_directory)

        if birefringence is not None:

            # initialize
            chunk_size = (1,1,1,birefringence.shape[-2],birefringence.shape[-1])
            i = 0

            # increment filename one more than last found saved snap
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Birefringence_Snap_{i}.zarr')):
                i += 1

            # create zarr root and position group
            writer.create_zarr_root(f'Birefringence_Snap_{i}.zarr')
            # Check if 2D
            if len(birefringence.shape) == 3:
                writer.init_array(0, (1, 4, 1, birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = 0

            # Check if 3D
            else:
                writer.init_array(0, (1, 4, birefringence.shape[-3], birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = [0, birefringence.shape[-3]]

            # Write the data to disk
            writer.write(birefringence, p=0, t=0, c=[0, 4], z=z)

        if phase is not None:

            # initialize
            chunk_size = (1,1,1,phase.shape[-2],phase.shape[-1])

            # increment filename one more than last found saved snap
            i = 0
            while os.path.exists(os.path.join(self.calib_window.save_directory, f'Phase_Snap_{i}.zarr')):
                i += 1

            # create zarr root and position group
            writer.create_zarr_root(f'Phase_Snap_{i}.zarr')

            # Check if 2D
            if len(phase.shape) == 2:
                writer.init_array(0, (1, 1, 1, phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase2D'])
                z = 0

            # Check if 3D
            else:
                writer.init_array(0, (1, 1, phase.shape[-3], phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase3D'])
                z = [0, phase.shape[-3]]

            # Write data to disk
            writer.write(phase, p=0, t=0, c=0, z=z)

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
            meta_path.close()
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
                     'pad_z': 'pad_z',
                     'n_media': 'n_media',
                     'ps': 'ps',
                     'swing': 'chi',
                     'bg_option': 'bg_option'
                     }

        # attributes that are modified upon passing them to reconstructor
        attr_modified_list = {'obj_na': 'NA_obj',
                              'cond_na': 'NA_illu',
                              'wavelength': 'lambda_illu',
                              'n_slices': 'N_defocus',
                              'ps': 'ps'
                              }

        self._check_abort()
        # check if equivalent attributes have diverged
        for key, value in attr_list.items():
            if getattr(self.calib_window, key) != getattr(self.calib_window.phase_reconstructor, value):
                changed = True
                break
            else:
                changed = False

        if not changed:
            # modify attributes to be equivalent and check for divergence
            for key, value in attr_modified_list.items():
                if key == 'wavelength':
                    if self.calib_window.wavelength * 1e-3 / self.calib_window.n_media != \
                            self.calib_window.phase_reconstructor.lambda_illu:
                        changed = True
                        break
                    else:
                        changed = False
                elif key == 'n_slices':
                    if getattr(self, key) != getattr(self.calib_window.phase_reconstructor, value):
                        changed = True
                        break
                    else:
                        changed = False

                elif key == 'ps':
                    if getattr(self, key) / float(self.calib_window.mag) != getattr(self.calib_window.phase_reconstructor, value):
                        changed = True
                        break
                    else:
                        changed = False
                else:
                    if getattr(self.calib_window, key)/self.calib_window.n_media != \
                            getattr(self.calib_window.phase_reconstructor, value):
                        changed = True
                    else:
                        changed = False

        return changed

    def _cleanup_acq(self):
        disps = self.dm.getAllDataViewers()

        for i in range(disps.size()):
            disp = disps.get(i)

            if self.prefix in disp.getName():
                dp = disp.getDataProvider()
                dir_ = dp.getSummaryMetadata().getDirectory()
                prefix = dp.getSummaryMetadata().getPrefix()
                closed = False
                disp.close()
                while not closed:
                    closed = disp.isClosed()
                dp.close()

                try:
                    shutil.rmtree(os.path.join(dir_, prefix))
                except PermissionError as ex:
                    dp.close()
                break
            else:
                continue





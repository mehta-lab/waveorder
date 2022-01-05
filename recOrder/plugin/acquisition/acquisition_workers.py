from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_phase2D, reconstruct_phase3D
from recOrder.acq.acq_functions import generate_acq_settings, acquire_from_settings
from recOrder.io.utils import load_bg
from recOrder.compute import QLIPPBirefringenceCompute
from napari.qt.threading import WorkerBaseSignals, WorkerBase
import logging
from waveorder.io.writer import WaveorderWriter
import tifffile as tiff
import json
import numpy as np
import os
import zarr
import shutil
import time
import glob


class AcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    aborted = pyqtSignal()

class ListeningSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    store_emitter = pyqtSignal(object)
    dim_emitter = pyqtSignal(tuple)
    aborted = pyqtSignal()

# TODO: Cache common OTF's on local computers and use those for reconstruction
class AcquisitionWorker(WorkerBase):
    """
    Class to execute a birefringence/phase acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window, calib, mode):
        super().__init__(SignalsClass=AcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init properties
        self.calib = calib
        self.mode = mode
        self.n_slices = None
        self.prefix = 'recOrderPluginSnap'
        self.dm = self.calib_window.mm.displays()

        # Determine whether 2D or 3D acquisition is needed
        if self.mode == 'birefringence' and self.calib_window.birefringence_dim == '2D':
            self.dim = '2D'
        else:
            self.dim = '3D'

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """

        logging.info('Running Acquisition...')
        save_dir = self.calib_window.save_directory if self.calib_window.save_directory else self.calib_window.directory

        # List the Channels to acquire, if 5-state then append 5th channel
        channels = ['State0', 'State1', 'State2', 'State3']
        if self.calib_window.calib_scheme == '5-State':
            channels.append('State4')

        if save_dir is None:
            raise ValueError('save directory is empty, please specify a directory in the plugin')

        self._check_abort()

        # Acquire 2D stack
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')

            # Generate MDA Settings
            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group='Channel',
                                             channels=channels,
                                             save_dir=save_dir,
                                             prefix=self.prefix)
            self._check_abort()

            # Acquire from MDA settings uses MM MDA GUI
            # Returns (1, 4/5, 1, Y, X) array
            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)

            # Sleep to make sure resources get unblocked before attempting cleanup
            time.sleep(1)

            # Cleanup acquisition by closing window + deleting temp directory
            self._cleanup_acq()

        # Acquire 3D stack
        else:
            logging.debug('Acquiring 3D stack')

            # Generate MDA Settings
            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group='Channel',
                                             channels=channels,
                                             zstart=self.calib_window.z_start,
                                             zend=self.calib_window.z_end,
                                             zstep=self.calib_window.z_step,
                                             save_dir=save_dir,
                                             prefix=self.prefix)

            self._check_abort()

            # Acquire from MDA settings uses MM MDA GUI
            # Returns (1, 4/5, Z, Y, X) array
            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)

            # Sleep to make sure resources get unblocked before attempting cleanup
            time.sleep(1)

            # Cleanup acquisition by closing window + deleting temp directory
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
        stack:          (nd-array) Dimensions are (C, Z, Y, X)

        Returns
        -------

        """

        # get rid of z-dimension if 2D acquisition
        stack = stack[:, 0] if self.n_slices == 1 else stack

        self._check_abort()

        # Initialize the heavy reconstuctor
        if self.mode == 'phase' or self.mode == 'all':

            self._check_abort()

            # if no reconstructor has been initialized before, create new reconstructor
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

                # Emit reconstructor to be saved for later reconstructions
                self.phase_reconstructor_emitter.emit(recon)

            # if previous reconstructor exists
            else:
                self._check_abort()

                # compute new reconstructor if the old reconstructor properties have been modified
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
        # no need to save this reconstructor for later as it is pretty quick to compute
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
            if self.calib_window.birefringence_dim == '2D':
                birefringence = reconstruct_qlipp_birefringence(stokes[:, :, :, stokes.shape[-1]//2], recon)
            else:
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
            chunk_size = (1, 1, 1, birefringence.shape[-2],birefringence.shape[-1])
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
                     'bg_option': 'bg_option'
                     }

        # attributes that are modified upon passing them to reconstructor
        attr_modified_list = {'obj_na': 'NA_obj',
                              'cond_na': 'NA_illu',
                              'wavelength': 'lambda_illu',
                              'n_slices': 'N_defocus',
                              'swing': 'chi',
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
                if key == 'swing':
                    if self.calib_window.calib_scheme == '5-State':
                        if self.calib_window.swing * 2 * np.pi != self.calib_window.phase_reconstructor.chi:
                            changed = True
                        else:
                            changed = False
                    else:
                        if self.calib_window.swing != self.calib_window.phase_reconstructor.chi:
                            changed = True
                        else:
                            changed = False

                elif key == 'wavelength':
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
                    if getattr(self.calib_window, key) / float(self.calib_window.mag) != getattr(self.calib_window.phase_reconstructor, value):
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

        # Get display windows
        disps = self.dm.getAllDataViewers()

        # loop through display window and find one with matching prefix
        for i in range(disps.size()):
            disp = disps.get(i)

            # close the datastore and grab the path to where the data is saved
            if self.prefix in disp.getName():
                dp = disp.getDataProvider()
                dir_ = dp.getSummaryMetadata().getDirectory()
                prefix = dp.getSummaryMetadata().getPrefix()
                closed = False
                disp.close()
                while not closed:
                    closed = disp.isClosed()
                dp.close()

                # Try to delete the data, sometime it isn't cleaned up quickly enough and will
                # return an error.  In this case, catch the error and then try to close again (seems to work).
                try:
                    shutil.rmtree(os.path.join(dir_, prefix))
                except PermissionError as ex:
                    dp.close()
                break
            else:
                continue


class ListeningWorker(WorkerBase):
    """
    Class to execute a birefringence/phase acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window, bg_data):
        super().__init__(SignalsClass=ListeningSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init properties
        self.n_slices = None
        self.n_channels = None
        self.n_frames = None
        self.n_pos = None
        self.shape = None
        self.dtype = None
        self.root = None
        self.prefix = None
        self.store = None
        self.save_directory = None
        self.bg_data = bg_data
        self.reconstructor = None

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def get_byte_offset(self, offsets, page):
        """
        Gets the byte offset from the tiff tag metadata

        Parameters
        ----------
        offsets:             (dict) Offset dictionary list
        page:               (int) Page to look at for the offset

        Returns
        -------
        byte offset:        (int) byte offset for the image array

        """

        if page == 0:
            array_offset = offsets[page] + 210
        else:
            array_offset = offsets[page] + 162

        return array_offset

    def listen_for_images(self, array, file, offsets, interval, dim3, dim2, dim1, dim0, dim_order):
        """

        Parameters
        ----------
        array:          (nd-array) numpy array of size (C, Z)
        file:           (string) filepath corresponding to the desired tiff image
        offsets:        (dict) dictionary of offsets corresponding to byte offsets of pixel data in tiff image
        interval:       (int) time interval between timepoints in seconds
        dim3:           (int) outermost dimension to value begin at
        dim2:           (int) first-inner dimension value to begin at
        dim1:           (int) second-inner dimension value to begin at
        dim0:           (int) innermost dimension value to begin at
        dim_order:      (int) 1, 2, 3, or 4 corresponding to the dimensionality ordering of the acquisition (MM-provided)

        Returns
        -------
        array:                      (nd-array) partially filled array of size (C, Z) to continue filling in next iteration
        index:                      (int) current page number at end of function
        dim3:                       (int) dimension values corresponding to where the next iteration should begin
        dim2:                       (int) dimension values corresponding to where the next iteration should begin
        dim1:                       (int) dimension values corresponding to where the next iteration should begin
        dim0:                       (int) dimension values corresponding to where the next iteration should begin

        """

        # Order dimensions that we will loop through in order to match the acquisition
        if dim_order == 0:
            dims = [[dim3, self.n_frames], [dim2, self.n_pos], [dim1, self.n_slices], [dim0, self.n_channels]]
            channel_dim = 0
        elif dim_order == 1:
            dims = [[dim3, self.n_frames], [dim2, self.n_pos], [dim1, self.n_channels], [dim0, self.n_slices]]
            channel_dim = 1
        elif dim_order == 2:
            dims = [[dim3, self.n_pos], [dim2, self.n_frames], [dim1, self.n_slices], [dim0, self.n_channels]]
            channel_dim = 0
        else:
            dims = [[dim3, self.n_pos], [dim2, self.n_frames], [dim1, self.n_channels], [dim0, self.n_slices]]
            channel_dim = 1

        print(dims)
        idx = 0
        for dim3 in range(dims[0][0], dims[0][1]):
            for dim2 in range(dims[1][0], dims[1][1]):
                for dim1 in range(dims[2][0], dims[2][1]):
                    for dim0 in range(dims[3][0], dims[3][1]):

                        print('iter', dim3, dim2, dim1, dim0)
                        # GET OFFSET AND WAIT
                        if idx > 0:
                            try:
                                offset = self.get_byte_offset(offsets, idx)
                            except IndexError:
                                # NEED TO STOP BECAUSE RAN OUT OF PAGES OR REACHED END OF ACQ
                                return array, idx, dim3, dim2, dim1, dim0
                            while offset == 162:
                                self._check_abort()
                                tf = tiff.TiffFile(file)
                                time.sleep(interval)
                                offsets = tf.micromanager_metadata['IndexMap']['Offset']
                                tf.close()
                                offset = self.get_byte_offset(offsets, idx)
                        else:
                            offset = self.get_byte_offset(offsets, idx)

                        if idx < (self.n_slices * self.n_channels * self.n_frames * self.n_pos):

                            # Assign dimensions based off acquisition order to correctly add image to array
                            if dim_order == 0:
                                t, p, c, z = dim3, dim2, dim0, dim1
                            elif dim_order == 1:
                                t, p, c, z = dim3, dim2, dim1, dim0
                            elif dim_order == 2:
                                t, p, c, z = dim2, dim3, dim0, dim1
                            else:
                                t, p, c, z = dim2, dim3, dim1, dim0

                            # If Channel first, compute birefringence here
                            if channel_dim == 0 and dim0 == self.n_channels - 1:
                                print('computing', dim3, dim2, dim1, dim0)
                                self._check_abort()

                                # Need to add last channel image before compute
                                img = np.memmap(file, dtype=self.dtype, mode='r', offset=offset, shape=self.shape)
                                array[c, z] = img

                                # Compute birefringence
                                self.compute_and_save(array[:, z], p, t, z)
                                idx += 1

                            # If Z First or channels not finished, add slice to the array
                            else:
                                self._check_abort()
                                print('adding', c, z)
                                img = np.memmap(file, dtype=self.dtype, mode='r', offset=offset, shape=self.shape)
                                array[c, z] = img
                                idx += 1

                    # Reset Range to 0 to account for starting this function in middle of a dimension
                    if idx < (self.n_slices * self.n_channels * self.n_frames * self.n_pos):
                        dims[2][0] = 0
                        dims[3][0] = 0

                    # If z-first, compute the birefringence here
                    if channel_dim == 1 and dim1 == self.n_channels - 1:
                        print('computing', dim3, dim2, dim1, dim0)
                        self._check_abort()
                        self.compute_and_save(array, p, t, dim0)
                        # idx += 1
                    else:
                        continue

                # Reset range to 0 to account for starting this function in the middle of a dimension
                if idx < (self.n_slices * self.n_channels * self.n_frames * self.n_pos):
                    dims[1][0] = 0

        # Return at the end of the acquisition
        return array, idx, dim3, dim2, dim1, dim0

    def compute_and_save(self, array, p, t, z):

        if self.n_slices == 1:
            array = array[:, 0]

        birefringence = self.reconstructor.reconstruct(array)

        if self.prefix not in self.calib_window.viewer.layers:
            self.store = zarr.open(os.path.join(self.root, self.prefix+'.zarr'))
            self.store.zeros(name='Birefringence',
                             shape=(self.n_pos,
                                    self.n_frames,
                                    2,
                                    self.n_slices,
                                    self.shape[0],
                                    self.shape[1]),
                             chunks=(1, 1, 1, 1, self.shape[0], self.shape[1]),
                             overwrite=True)

            if len(birefringence.shape) == 4:
                self.store['Birefringence'][p, t] = birefringence
            else:
                self.store['Birefringence'][p, t, :, z] = birefringence

            self.store_emitter.emit(self.store)

        else:
            if len(birefringence.shape) == 4:
                self.store['Birefringence'][p, t] = birefringence
            else:
                self.store['Birefringence'][p, t, :, z] = birefringence

        self.dim_emitter.emit((p, t, z))

    def work(self):
        acq_man = self.calib_window.mm.acquisitions()

        while not acq_man.isAcquisitionRunning():
            pass

        time.sleep(1)

        # Get all of the dataset dimensions, settings
        s = acq_man.getAcquisitionSettings()
        self.root = s.root()
        self.prefix = s.prefix()
        self.n_frames, self.interval = (s.numFrames(), s.intervalMs() / 1000) if s.useFrames() else (1, 1)
        self.n_channels = s.channels().size() if s.useChannels() else 1
        self.n_slices = s.slices().size() if s.useChannels() else 1
        self.n_pos = 1 if not s.usePositionList() else \
            self.calib_window.mm.getPositionListManager().getPositionList().getNumberOfPositions()
        dim_order = s.acqOrderMode()

        # Get File Path corresponding to current dataset
        path = os.path.join(self.root, self.prefix)
        files = [fn for fn in glob.glob(path + '*') if '.zarr' not in fn]
        index = max([int(x.strip(path + '_')) for x in files])

        self.prefix = self.prefix + f'_{index}'
        full_path = os.path.join(self.root, self.prefix)
        first_file_path = os.path.join(full_path, self.prefix + '_MMStack.ome.tif')

        file = tiff.TiffFile(first_file_path)
        file_path = first_file_path
        self.shape = (file.micromanager_metadata['Summary']['Height'], file.micromanager_metadata['Summary']['Width'])
        self.dtype = file.pages[0].dtype
        offsets = file.micromanager_metadata['IndexMap']['Offset']
        file.close()

        self._check_abort()

        # Init Reconstruction Class
        self.reconstructor = QLIPPBirefringenceCompute(self.shape,
                                                       self.calib_window.calib_scheme,
                                                       self.calib_window.wavelength,
                                                       self.calib_window.swing,
                                                       self.n_slices,
                                                       self.calib_window.bg_option,
                                                       self.bg_data)

        self._check_abort()

        # initialize dimensions / array for the loop
        idx, dim3, dim2, dim1, dim0 = 0, 0, 0, 0, 0
        array = np.zeros((self.n_channels, self.n_slices, self.shape[0], self.shape[1]))
        file_count = 0
        total_idx = 0

        # Run until the function has collected the totality of the data
        while total_idx < (self.n_slices * self.n_channels * self.n_frames * self.n_pos):

            self._check_abort()

            # this will loop through reading images in a single file as it's being written
            # when it has successfully loaded all of the images from the file, it'll move on to the next
            array, idx, dim3, dim2, dim1, dim0 = self.listen_for_images(array,
                                                            file_path,
                                                            offsets,
                                                            self.interval,
                                                            dim3, dim2, dim1, dim0, dim_order)

            total_idx += idx
            print(self.n_slices, self.n_channels, self.n_frames, self.n_pos)
            print(total_idx, idx, dim3, dim2, dim1, dim0)

            # If acquisition is not finished, grab the next file and listen for images
            if total_idx != self.n_slices * self.n_channels * self.n_frames * self.n_pos:
                time.sleep(1)
                file_count += 1
                file_path = os.path.join(full_path, self.prefix + f'_MMStack_{file_count}.ome.tif')
                file = tiff.TiffFile(file_path)
                offsets = file.micromanager_metadata['IndexMap']['Offset']
                file.close()




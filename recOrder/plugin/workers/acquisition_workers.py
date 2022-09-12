from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_phase2D, reconstruct_phase3D
from recOrder.acq.acq_functions import generate_acq_settings, acquire_from_settings
from recOrder.io.utils import load_bg, extract_reconstruction_parameters
from recOrder.compute import QLIPPBirefringenceCompute
from recOrder.compute.fluorescence_compute import initialize_fluorescence_reconstructor, deconvolve_fluorescence_3D, \
    deconvolve_fluorescence_2D, calculate_background
from recOrder.io.zarr_converter import ZarrConverter
from recOrder.io.metadata_reader import MetadataReader, get_last_metadata_file
from recOrder.io.utils import ram_message, rec_bkg_to_wo_bkg
from napari.qt.threading import WorkerBaseSignals, WorkerBase
import logging
from waveorder.io.writer import WaveorderWriter
import tifffile as tiff
import numpy as np
import os
import zarr
import shutil
import time
import glob


class PolarizationAcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    meta_emitter = pyqtSignal(dict)
    aborted = pyqtSignal()


class BFAcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    meta_emitter = pyqtSignal(dict)
    aborted = pyqtSignal()

class FluorescenceAcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    fluor_image_emitter = pyqtSignal(object)
    fluor_reconstructor_emitter = pyqtSignal(object)
    meta_emitter = pyqtSignal(dict)
    aborted = pyqtSignal()

class ListeningSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    store_emitter = pyqtSignal(object)
    dim_emitter = pyqtSignal(tuple)
    aborted = pyqtSignal()


class BFAcquisitionWorker(WorkerBase):
    """
    Class to execute a fluorescence deconvolution acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window):
        super().__init__(SignalsClass=BFAcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init Properties
        self.prefix = 'recOrderPluginSnap'
        self.dm = self.calib_window.mm.displays()
        self.dim = '2D' if self.calib_window.ui.cb_phase.currentIndex() == 0 else '3D'
        self.img_dim = None

        save_dir = self.calib_window.save_directory if self.calib_window.save_directory else self.calib_window.directory

        if save_dir is None:
            raise ValueError('save directory is empty, please specify a directory in the plugin')

        # increment filename one more than last found saved snap
        i = 0
        prefix = self.calib_window.save_name
        snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'
        while os.path.exists(os.path.join(save_dir, snap_dir)):
            i += 1
            snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'

        self.snap_dir = os.path.join(save_dir, snap_dir)
        os.mkdir(self.snap_dir)

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """
        logging.info(ram_message())
        logging.info('Running Acquisition...')
        self._check_abort()

        channel_idx = self.calib_window.ui.cb_acq_channel.currentIndex()
        channel = self.calib_window.ui.cb_acq_channel.itemText(channel_idx)
        channel_group = None

        groups = self.calib_window.mmc.getAvailableConfigGroups()
        group_list = []
        for i in range(groups.size()):
            group_list.append(groups.get(i))

        for group in group_list:
            config = self.calib_window.mmc.getAvailableConfigs(group)
            for idx in range(config.size()):
                if channel in config.get(idx):
                    channel_group = group
                    break

        # Acquire 3D stack
        logging.debug('Acquiring 3D stack')

        # Generate MDA Settings
        settings = generate_acq_settings(self.calib_window.mm,
                                         channel_group=channel_group,
                                         channels=[channel],
                                         zstart=self.calib_window.z_start,
                                         zend=self.calib_window.z_end,
                                         zstep=self.calib_window.z_step,
                                         save_dir=self.snap_dir,
                                         prefix=self.prefix)

        self._check_abort()

        # Acquire from MDA settings uses MM MDA GUI
        # Returns (1, 4/5, Z, Y, X) array
        stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)
        self._check_abort()

        # Cleanup acquisition by closing window, converting to zarr, and deleting temp directory
        self._cleanup_acq()

        # Reconstruct snapped images
        self.n_slices = stack.shape[2]

        phase, meta = self._reconstruct(stack[0])
        self._check_abort()

        # Save images
        logging.debug('Saving Images')
        self._save_imgs(phase, meta)

        self._check_abort()

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')

        # Emit the images and let thread know function is finished
        self.phase_image_emitter.emit(phase)
        self.meta_emitter.emit(meta)

    def _reconstruct(self, stack):
        """
        Method to reconstruct, given a 2D or 3D stack.
        This function also checks to see if the reconstructor needs to be updated from previous acquisitions

        Parameters
        ----------
        stack:          (nd-array) Dimensions are (C, Z, Y, X)

        Returns
        -------

        """

        self.img_dim = (stack.shape[-2], stack.shape[-1], stack.shape[-3])
        self._check_abort()

        # Initialize the reconstuctor

        # if no reconstructor has been initialized before, create new reconstructor
        if not self.calib_window.phase_reconstructor:
            logging.debug('Computing new reconstructor')

            recon = initialize_reconstructor('PhaseFromBF',
                                             image_dim=(stack.shape[-2], stack.shape[-1]),
                                             wavelength_nm=int(self.calib_window.ui.le_recon_wavelength.text()),
                                             NA_obj=self.calib_window.obj_na,
                                             NA_illu=self.calib_window.cond_na,
                                             mag=self.calib_window.mag,
                                             n_slices=self.img_dim[-1],
                                             z_step_um=self.calib_window.z_step,
                                             pad_z=self.calib_window.pad_z,
                                             pixel_size_um=self.calib_window.ps,
                                             n_obj_media=self.calib_window.n_media,
                                             mode=self.calib_window.phase_dim,
                                             use_gpu=self.calib_window.use_gpu,
                                             gpu_id=self.calib_window.gpu_id)

            # Emit reconstructor to be saved for later reconstructions
            self.phase_reconstructor_emitter.emit(recon)

        # if previous reconstructor exists
        else:
            self._check_abort()

            # compute new reconstructor if the old reconstructor properties have been modified
            if self._reconstructor_changed():
                logging.debug('Reconstruction settings changed, updating reconstructor')

                recon = initialize_reconstructor('PhaseFromBF',
                                                 image_dim=(stack.shape[-2], stack.shape[-1]),
                                                 wavelength_nm=int(self.calib_window.ui.le_recon_wavelength.text()),
                                                 NA_obj=self.calib_window.obj_na,
                                                 NA_illu=self.calib_window.cond_na,
                                                 mag=self.calib_window.mag,
                                                 n_slices=self.n_slices,
                                                 z_step_um=self.calib_window.z_step,
                                                 pad_z=self.calib_window.pad_z,
                                                 pixel_size_um=self.calib_window.ps,
                                                 n_obj_media=self.calib_window.n_media,
                                                 mode=self.calib_window.phase_dim,
                                                 use_gpu=self.calib_window.use_gpu,
                                                 gpu_id=self.calib_window.gpu_id)

                # Emit reconstructor to be saved for later reconstructions
                self.phase_reconstructor_emitter.emit(recon)

            # use previous reconstructor
            else:
                logging.debug('Using previous reconstruction settings')
                recon = self.calib_window.phase_reconstructor

        # Begin reconstruction with stokes (needed for birefringence or phase)
        logging.debug('Reconstructing...')
        self._check_abort()

        regularizer = 'Tikhonov' if self.calib_window.ui.cb_phase_denoiser.currentIndex() == 0 else 'TV'
        reg = float(self.calib_window.ui.le_phase_strength.text())

        # Perform deconvolution
        if self.dim == '2D':

            phase = reconstruct_phase2D(stack[0],
                                        recon,
                                        method=regularizer,
                                        reg_p=reg,
                                        itr=int(self.calib_window.ui.le_itr.text()),
                                        rho=float(self.calib_window.ui.le_rho.text()))
        else:

            phase = reconstruct_phase3D(stack[0],
                                        recon,
                                        method=regularizer,
                                        reg_re=reg,
                                        itr=int(self.calib_window.ui.le_itr.text()),
                                        rho=float(self.calib_window.ui.le_rho.text()))

        self._check_abort()

        # Update metadata in zarr attributes with reconstruction parameters
        meta = extract_reconstruction_parameters(recon, magnification=self.calib_window.mag)
        meta['regularization_method'] = regularizer
        meta['regularization_strength'] = reg
        if regularizer == 'TV':
            meta['rho'] = float(self.calib_window.ui.le_rho.text())
            meta['itr'] = int(self.calib_window.ui.le_itr.text())

        # return both variables, could contain images or could be null
        return phase, meta

    def _save_imgs(self, phase, meta=None):
        """
        function to save images.

        Parameters
        ----------
        fluor:      (nd-array or None) deconvolved fluorescence image or stack

        Returns
        -------

        """

        writer = WaveorderWriter(self.snap_dir)

        # initialize
        chunk_size = (1, 1, 1, phase.shape[-2], phase.shape[-1])
        prefix = self.calib_window.save_name
        name = f'PhaseSnap.zarr' if not prefix else f'{prefix}_PhaseSnap.zarr'

        # create zarr root and position group
        writer.create_zarr_root(name)

        # Check if 2D
        if len(phase.shape) == 2:
            writer.init_array(0, (1, 1, 1, phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase2D'])
            z = 0

        # Check if 3D
        else:
            writer.init_array(0, (1, 1, phase.shape[-3], phase.shape[-2], phase.shape[-1]), chunk_size,
                              ['Phase3D'])

            z = slice(0, phase.shape[-3])


        # Write data to disk
        writer.write(phase, p=0, t=0, c=0, z=z)

        current_meta = writer.store.attrs.asdict()
        current_meta['recOrder'] = meta
        writer.store.attrs.put(current_meta)

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
                     'use_gpu': 'use_gpu',
                     'gpu_id': 'gpu_id'
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
                    save_prefix = self.calib_window.save_name if self.calib_window.save_name else None
                    name = f'RawBFDataSnap.zarr' if not save_prefix else f'{save_prefix}_RawBFDataSnap.zarr'
                    out_path = os.path.join(self.snap_dir, name)
                    converter = ZarrConverter(os.path.join(dir_, prefix), out_path, 'ometiff', False, False)
                    converter.run_conversion()
                    shutil.rmtree(os.path.join(dir_, prefix))
                except PermissionError as ex:
                    dp.close()
                break
            else:
                continue


class FluorescenceAcquisitionWorker(WorkerBase):
    """
    Class to execute a fluorescence deconvolution acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window):
        super().__init__(SignalsClass=FluorescenceAcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init Properties
        self.prefix = 'recOrderPluginSnap'
        self.dm = self.calib_window.mm.displays()
        self.dim = '2D' if self.calib_window.ui.cb_fluor_dim.currentIndex() == 0 else '3D'

        self.img_dim = None

        save_dir = self.calib_window.save_directory if self.calib_window.save_directory else self.calib_window.directory

        if save_dir is None:
            raise ValueError('save directory is empty, please specify a directory in the plugin')

        # increment filename one more than last found saved snap
        i = 0
        prefix = self.calib_window.save_name
        snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'
        while os.path.exists(os.path.join(save_dir, snap_dir)):
            i += 1
            snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'

        self.snap_dir = os.path.join(save_dir, snap_dir)
        os.mkdir(self.snap_dir)

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """
        logging.info(ram_message())
        logging.info('Running Acquisition...')

        self._check_abort()

        channel_idx = self.calib_window.ui.cb_acq_channel.currentIndex()
        channel = self.calib_window.ui.cb_acq_channel.itemText(channel_idx)
        channel_group = None

        groups = self.calib_window.mmc.getAvailableConfigGroups()
        group_list = []
        for i in range(groups.size()):
            group_list.append(groups.get(i))

        for group in group_list:
            config = self.calib_window.mmc.getAvailableConfigs(group)
            for idx in range(config.size()):
                if channel in config.get(idx):
                    channel_group = group
                    break

        # Acquire 2D stack
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')

            # Generate MDA Settings
            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group=channel_group,
                                             channels=[channel],
                                             save_dir=self.snap_dir,
                                             prefix=self.prefix)
            self._check_abort()

            # Acquire from MDA settings uses MM MDA GUI
            # Returns (1, 4/5, 1, Y, X) array
            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)

        # Acquire 3D stack
        else:
            logging.debug('Acquiring 3D stack')

            # Generate MDA Settings
            settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group=channel_group,
                                             channels=[channel],
                                             zstart=self.calib_window.z_start,
                                             zend=self.calib_window.z_end,
                                             zstep=self.calib_window.z_step,
                                             save_dir=self.snap_dir,
                                             prefix=self.prefix)

            self._check_abort()

            # Acquire from MDA settings uses MM MDA GUI
            # Returns (1, 4/5, Z, Y, X) array
            stack = acquire_from_settings(self.calib_window.mm, settings, grab_images=True)
            self._check_abort()

        # Cleanup acquisition by closing window, converting to zarr, and deleting temp directory
        self._cleanup_acq()

        # Reconstruct snapped images
        self._check_abort()
        self.n_slices = stack.shape[2]
        fluor, meta = self._reconstruct(stack[0])
        self._check_abort()

        # Save images
        logging.debug('Saving Images')
        self._save_imgs(fluor, meta)

        self._check_abort()

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')

        # Emit the images and let thread know function is finished
        self.fluor_image_emitter.emit(fluor)
        self.meta_emitter.emit(meta)

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
        self.img_dim = (stack.shape[-2], stack.shape[-1], stack.shape[-3]) if self.n_slices != 1 else (stack.shape[-2], stack.shape[-1], 1)

        self._check_abort()

        # Initialize the reconstuctor

        # if no reconstructor has been initialized before, create new reconstructor
        if not self.calib_window.fluor_reconstructor:
            logging.debug('Computing new reconstructor')

            recon = initialize_fluorescence_reconstructor(img_dim=self.img_dim,
                                                          wavelength_nm=int(
                                                              self.calib_window.ui.le_recon_wavelength.text()),
                                                          pixel_size_um=self.calib_window.ps,
                                                          z_step_um=self.calib_window.z_step,
                                                          NA_obj=self.calib_window.obj_na,
                                                          magnification=self.calib_window.mag,
                                                          mode=self.dim,
                                                          n_obj_media=self.calib_window.n_media,
                                                          pad_z=self.calib_window.pad_z,
                                                          use_gpu=self.calib_window.use_gpu,
                                                          gpu_id=self.calib_window.gpu_id)

            # Emit reconstructor to be saved for later reconstructions
            self.fluor_reconstructor_emitter.emit(recon)

        # if previous reconstructor exists
        else:
            self._check_abort()

            # compute new reconstructor if the old reconstructor properties have been modified
            if self._reconstructor_changed():
                logging.debug('Reconstruction settings changed, updating reconstructor')

                recon = initialize_fluorescence_reconstructor(img_dim=self.img_dim,
                                                              wavelength_nm=int(
                                                                  self.calib_window.ui.le_recon_wavelength.text()),
                                                              pixel_size_um=self.calib_window.ps,
                                                              z_step_um=self.calib_window.z_step,
                                                              NA_obj=self.calib_window.obj_na,
                                                              magnification=self.calib_window.mag,
                                                              mode=self.dim,
                                                              n_obj_media=self.calib_window.n_media,
                                                              pad_z=self.calib_window.pad_z,
                                                              use_gpu=self.calib_window.use_gpu,
                                                              gpu_id=self.calib_window.gpu_id)

                # Emit reconstructor to be saved for later reconstructions
                self.fluor_reconstructor_emitter.emit(recon)

            # use previous reconstructor
            else:
                logging.debug('Using previous reconstruction settings')
                recon = self.calib_window.fluor_reconstructor

        # Get BG information
        if self.calib_window.ui.chb_autocalc_bg.checkState() == 2:
            bg_level = calculate_background(stack)

        else:
            try:
                bg_level = [int(self.calib_window.ui.le_fluor_bg.text())]
            except ValueError:
                bg_level = [0]

        # Perform deconvolution
        if self.dim == '2D':

            fluor_deconvolved = deconvolve_fluorescence_2D(data=stack,
                                                           reconstructor=recon,
                                                           bg_level=bg_level,
                                                           reg=[float(self.calib_window.ui.le_fluor_strength.text())])
        else:

            fluor_deconvolved = deconvolve_fluorescence_3D(data=stack,
                                                           reconstructor=recon,
                                                           bg_level=bg_level,
                                                           reg=[float(self.calib_window.ui.le_fluor_strength.text())])



        # Update metadata in zarr attributes with reconstruction parameters
        meta = extract_reconstruction_parameters(recon, magnification=self.calib_window.mag)
        meta['regularization_strength'] = float(self.calib_window.ui.le_fluor_strength.text())
        meta['bg_level'] = bg_level

        # return both variables, could contain images or could be null
        return fluor_deconvolved, meta

    def _save_imgs(self, fluor, meta=None):
        """
        function to save images.

        Parameters
        ----------
        fluor:      (nd-array or None) deconvolved fluorescence image or stack

        Returns
        -------

        """

        writer = WaveorderWriter(self.snap_dir)

        # initialize
        prefix = self.calib_window.save_name
        chunk_size = (1, 1, 1, fluor.shape[-2], fluor.shape[-1])
        name = f'FluorescenceDeconvolvedSnap.zarr' if not prefix else f'{prefix}_FluorescenceDeconvolvedSnap.zarr'

        # create zarr root and position group
        writer.create_zarr_root(name)

        # Check if 2D
        if len(fluor.shape) == 2:
            writer.init_array(0, (1, 1, 1, fluor.shape[-2], fluor.shape[-1]), chunk_size, ['FluorDeconvolved2D'])
            z = 0

        # Check if 3D
        else:
            writer.init_array(0, (1, 1, fluor.shape[-3], fluor.shape[-2], fluor.shape[-1]), chunk_size,
                              ['FluorDeconvolved3D'])

            z = slice(0, fluor.shape[-3])


        # Write data to disk
        writer.write(fluor, p=0, t=0, c=0, z=z)

        current_meta = writer.store.attrs.asdict()
        current_meta['recOrder'] = meta
        writer.store.attrs.put(current_meta)

    def _reconstructor_changed(self):
        """
        Function to check if the reconstructor has changed from the previous one in memory.
        Serves to check if the worker attributes and reconstructor attributes have diverged.

        Returns
        -------

        """

        recon = self.calib_window.fluor_reconstructor

        if (recon.N, recon.M, recon.N_defocus) != self.img_dim:
            changed = True

        elif recon.n_media != self.calib_window.n_media:
            changed = True

        elif recon.NA_obj * recon.n_media != self.calib_window.obj_na:
            changed = True

        elif recon.lambda_emiss * recon.n_media != int(self.calib_window.ui.le_recon_wavelength.text()) / 1000:
            changed = True

        elif recon.ps * self.calib_window.mag != self.calib_window.ps:
            changed = True

        elif recon.psz != self.calib_window.z_step:
            changed = True

        elif recon.pad_z != self.calib_window.pad_z:
            changed = True

        elif recon.use_gpu != self.calib_window.use_gpu:
            changed = True

        elif recon.gpu_id != self.calib_window.gpu_id:
            changed = True

        elif recon.deconv_mode != self.dim+'-WF':
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
                    save_prefix = self.calib_window.save_name if self.calib_window.save_name else None
                    name = f'RawFluorDataSnap.zarr' if not save_prefix else f'{save_prefix}_RawFluorDataSnap.zarr'
                    out_path = os.path.join(self.snap_dir, name)
                    converter = ZarrConverter(os.path.join(dir_, prefix), out_path, 'ometiff', False, False)
                    converter.run_conversion()
                    shutil.rmtree(os.path.join(dir_, prefix))
                except PermissionError as ex:
                    dp.close()
                break
            else:
                continue


# TODO: Cache common OTF's on local computers and use those for reconstruction
class PolarizationAcquisitionWorker(WorkerBase):
    """
    Class to execute a birefringence/phase acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window, calib, mode):
        super().__init__(SignalsClass=PolarizationAcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init properties
        self.calib = calib
        self.mode = mode
        self.n_slices = None
        self.prefix = 'recOrderPluginSnap'
        self.dm = self.calib_window.mm.displays()
        self.channel_group = self.calib_window.config_group

        # Determine whether 2D or 3D acquisition is needed
        if self.mode == 'birefringence' and self.calib_window.birefringence_dim == '2D':
            self.dim = '2D'
        else:
            self.dim = '3D'

        save_dir = self.calib_window.save_directory if self.calib_window.save_directory else self.calib_window.directory

        if save_dir is None:
            raise ValueError('save directory is empty, please specify a directory in the plugin')

        # increment filename one more than last found saved snap
        i = 0
        prefix = self.calib_window.save_name
        snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'
        while os.path.exists(os.path.join(save_dir, snap_dir)):
            i += 1
            snap_dir = f'recOrderPluginSnap_{i}' if not prefix else f'{prefix}_recOrderPluginSnap_{i}'

        self.snap_dir = os.path.join(save_dir, snap_dir)
        os.mkdir(self.snap_dir)

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """
        logging.info(ram_message())
        logging.info('Running Acquisition...')

        # List the Channels to acquire, if 5-state then append 5th channel
        channels = ['State0', 'State1', 'State2', 'State3']
        if self.calib_window.calib_scheme == '5-State':
            channels.append('State4')

        self._check_abort()

        # Acquire 2D stack
        if self.dim == '2D':
            logging.debug('Acquiring 2D stack')

            # Generate MDA Settings
            self.settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group=self.channel_group,
                                             channels=channels,
                                             save_dir=self.snap_dir,
                                             prefix=self.prefix,
                                             keep_shutter_open_channels=True)
            self._check_abort()
            # acquire images
            stack = self._acquire()

        # Acquire 3D stack
        else:
            logging.debug('Acquiring 3D stack')

            # Generate MDA Settings
            self.settings = generate_acq_settings(self.calib_window.mm,
                                             channel_group=self.channel_group,
                                             channels=channels,
                                             zstart=self.calib_window.z_start,
                                             zend=self.calib_window.z_end,
                                             zstep=self.calib_window.z_step,
                                             save_dir=self.snap_dir,
                                             prefix=self.prefix,
                                             keep_shutter_open_channels=True,
                                             keep_shutter_open_slices=True)

            self._check_abort()

            # set acquisition order to channel-first
            self.settings['slicesFirst'] = False
            self.settings['acqOrderMode'] = 0  # TIME_POS_SLICE_CHANNEL

            # acquire images
            stack = self._acquire()

        # Cleanup acquisition by closing window, converting to zarr, and deleting temp directory
        self._cleanup_acq()

        # Reconstruct snapped images
        self._check_abort()
        self.n_slices = stack.shape[2]
        birefringence, phase, meta = self._reconstruct(stack[0])
        self._check_abort()

        # Save images
        logging.debug('Saving Images')
        self._save_imgs(birefringence, phase, meta)

        self._check_abort()

        logging.info('Finished Acquisition')
        logging.debug('Finished Acquisition')

        # Emit the images and let thread know function is finished
        self.bire_image_emitter.emit(birefringence)
        self.phase_image_emitter.emit(phase)
        self.meta_emitter.emit(meta)

    def _check_exposure(self) -> None:
        """
        Check that all LF channels have the same exposure settings. If not, abort Acquisition.
        """
        logging.debug('Verifying exposure times...')
        # parse exposure times
        channel_exposures = []
        for channel in self.settings['channels']:
            channel_exposures.append(channel['exposure'])

        channel_exposures = np.array(channel_exposures)
        # check if exposure times are equal
        if not np.all(channel_exposures == channel_exposures[0]):
            error_exposure_msg = f'The MDA exposure times are not equal! Aborting Acquisition.\n' \
                                 f'Please manually set the exposure times to the same value from the MDA menu.'

            raise ValueError(error_exposure_msg)        

        self._check_abort()

    def _acquire(self) -> np.ndarray:
        """
        Acquire images.

        Returns
        -------
        stack:          (nd-array) Dimensions are (C, Z, Y, X). Z=1 for 2D acquisition.
        """
        # check if exposure times are the same
        self._check_exposure()

        # Acquire from MDA settings uses MM MDA GUI
        # Returns (1, 4/5, Z, Y, X) array
        stack = acquire_from_settings(self.calib_window.mm, self.settings, grab_images=True)
        self._check_abort()

        return stack

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

        wo_background_correction = rec_bkg_to_wo_bkg(self.calib_window.bg_option)

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
                                                 bg_correction=wo_background_correction,
                                                 n_obj_media=self.calib_window.n_media,
                                                 mode=self.calib_window.phase_dim,
                                                 use_gpu=self.calib_window.use_gpu,
                                                 gpu_id=self.calib_window.gpu_id)

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
                                                     bg_correction=wo_background_correction,
                                                     n_obj_media=self.calib_window.n_media,
                                                     mode=self.calib_window.phase_dim,
                                                     use_gpu=self.calib_window.use_gpu,
                                                     gpu_id=self.calib_window.gpu_id)

                    self.phase_reconstructor_emitter.emit(recon)

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
                                             bg_correction=wo_background_correction,
                                             n_slices=self.n_slices)

        # Prepare background corrections for waveorder
        # This block mimics qlipp_pipeline.py L110-119.
        if self.calib_window.bg_option in ['global', 'local_fit+']:
            logging.debug('Loading BG Data')
            self._check_abort()
            bg_data = self._load_bg(self.calib_window.acq_bg_directory, stack.shape[-2], stack.shape[-1])
            self._check_abort()
            bg_stokes = recon.Stokes_recon(bg_data)
            self._check_abort()
            bg_stokes = recon.Stokes_transform(bg_stokes)
            self._check_abort()
        elif self.calib_window.bg_option == 'local_fit':
            bg_stokes = np.zeros((5, stack.shape[-2], stack.shape[-1]))
            bg_stokes[0, ...] = 1  # Set background to "identity" Stokes parameters.
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

        regularizer = 'Tikhonov' if self.calib_window.ui.cb_phase_denoiser.currentIndex() == 0 else 'TV'
        reg = float(self.calib_window.ui.le_phase_strength.text())

        # reconstruct both phase and birefringence
        if self.mode == 'all':
            if self.calib_window.birefringence_dim == '2D':
                birefringence = reconstruct_qlipp_birefringence(stokes[:, stokes.shape[1]//2, :, :], recon)
            else:
                birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength
            self._check_abort()

            if self.calib_window.phase_dim == '2D':
                phase = reconstruct_phase2D(stokes[0],
                                            recon,
                                            method=regularizer,
                                            reg_p=reg,
                                            itr=int(self.calib_window.ui.le_itr.text()),
                                            rho=float(self.calib_window.ui.le_rho.text()))
            else:
                phase = reconstruct_phase3D(stokes[0],
                                            recon,
                                            method=regularizer,
                                            reg_re=reg,
                                            itr=int(self.calib_window.ui.le_itr.text()),
                                            rho=float(self.calib_window.ui.le_rho.text()))

            self._check_abort()

        # reconstruct phase only
        elif self.mode == 'phase':
            if self.calib_window.phase_dim == '2D':
                phase = reconstruct_phase2D(stokes[0],
                                            recon,
                                            method=regularizer,
                                            reg_p=reg,
                                            itr=int(self.calib_window.ui.le_itr.text()),
                                            rho=float(self.calib_window.ui.le_rho.text()))
            else:
                phase = reconstruct_phase3D(stokes[0],
                                            recon,
                                            method=regularizer,
                                            reg_re=reg,
                                            itr=int(self.calib_window.ui.le_itr.text()),
                                            rho=float(self.calib_window.ui.le_rho.text()))
            self._check_abort()

        # reconstruct birefringence only
        elif self.mode == 'birefringence':
            birefringence = reconstruct_qlipp_birefringence(stokes, recon)
            birefringence[0] = birefringence[0] / (2 * np.pi) * self.calib_window.wavelength
            self._check_abort()

        else:
            raise ValueError('Reconstruction Mode Not Understood')

        meta = extract_reconstruction_parameters(recon, self.calib_window.mag)
        meta['regularization_method'] = regularizer
        meta['regularization_strength'] = reg
        if regularizer == 'TV':
            meta['rho'] = float(self.calib_window.ui.le_rho.text())
            meta['itr'] = int(self.calib_window.ui.le_itr.text())

        # return both variables, could contain images or could be null
        return birefringence, phase, meta

    def _save_imgs(self, birefringence, phase, meta=None):
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
        writer = WaveorderWriter(self.snap_dir)

        if birefringence is not None:

            # initialize
            chunk_size = (1, 1, 1, birefringence.shape[-2],birefringence.shape[-1])

            # increment filename one more than last found saved snap
            prefix = self.calib_window.save_name
            name = f'BirefringenceSnap.zarr' if not prefix else f'{prefix}_BirefringenceSnap.zarr'

            # create zarr root and position group
            writer.create_zarr_root(name)

            # Check if 2D
            if len(birefringence.shape) == 3:
                writer.init_array(0, (1, 4, 1, birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = 0

            # Check if 3D
            else:
                writer.init_array(0, (1, 4, birefringence.shape[-3], birefringence.shape[-2], birefringence.shape[-1]),
                                  chunk_size, ['Retardance', 'Orientation', 'BF', 'Pol'])
                z = slice(0, birefringence.shape[-3])

            # Write the data to disk
            writer.write(birefringence, p=0, t=0, c=slice(0, 4), z=z)

            current_meta = writer.store.attrs.asdict()
            current_meta['recOrder'] = meta
            writer.store.attrs.put(current_meta)

        if phase is not None:

            # initialize
            chunk_size = (1, 1, 1, phase.shape[-2], phase.shape[-1])

            # increment filename one more than last found saved snap
            prefix = self.calib_window.save_name
            name = f'PhaseSnap.zarr' if not prefix else f'{prefix}_PhaseSnap.zarr'

            # create zarr root and position group
            writer.create_zarr_root(name)

            # Check if 2D
            if len(phase.shape) == 2:
                writer.init_array(0, (1, 1, 1, phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase2D'])
                z = 0

            # Check if 3D
            else:
                writer.init_array(0, (1, 1, phase.shape[-3], phase.shape[-2], phase.shape[-1]), chunk_size, ['Phase3D'])
                z = slice(0, phase.shape[-3])

            # Write data to disk
            writer.write(phase, p=0, t=0, c=0, z=z)

            current_meta = writer.store.attrs.asdict()
            current_meta['recOrder'] = meta
            writer.store.attrs.put(current_meta)

    def _load_bg(self, path, height, width):
        """
        # TODO: remove ROI for 1.0.0

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
            metadata_path = get_last_metadata_file(path)
            metadata = MetadataReader(metadata_path)
            roi = metadata.ROI
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
                     'bg_option': 'bg_option',
                     'use_gpu': 'use_gpu',
                     'gpu_id': 'gpu_id'
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
                    save_prefix = self.calib_window.save_name if self.calib_window.save_name else None
                    name = f'RawPolDataSnap.zarr' if not save_prefix else f'{save_prefix}_RawPolDataSnap.zarr'
                    out_path = os.path.join(self.snap_dir, name)
                    converter = ZarrConverter(os.path.join(dir_, prefix), out_path, 'ometiff', False, False)
                    converter.run_conversion()
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
        Gets the byte offset from the tiff tag metadata.

        210 accounts for header data + page header data.
        162 accounts page header data

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

        # Loop through dimensions in the order they are acquired
        idx = 0
        for dim3 in range(dims[0][0], dims[0][1]):
            for dim2 in range(dims[1][0], dims[1][1]):
                for dim1 in range(dims[2][0], dims[2][1]):
                    for dim0 in range(dims[3][0], dims[3][1]):

                        # GET OFFSET AND WAIT
                        if idx > 0:
                            try:
                                offset = self.get_byte_offset(offsets, idx)
                            except IndexError:
                                # NEED TO STOP BECAUSE RAN OUT OF PAGES OR REACHED END OF ACQ
                                return array, idx, dim3, dim2, dim1, dim0

                            # Checks if the offset in metadata is 0, but technically self.get_byte_offset() adds
                            # 162 to every offset to account for tiff file header bytes
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

                            self._check_abort()

                            # Add Image to array
                            img = np.memmap(file, dtype=self.dtype, mode='r', offset=offset, shape=self.shape)
                            array[c, z] = img

                            # If Channel first, compute birefringence here
                            if channel_dim == 0 and dim0 == self.n_channels - 1:
                                self._check_abort()

                                # Compute birefringence
                                self.compute_and_save(array[:, z], p, t, z)

                            idx += 1

                    # Reset Range to 0 to account for starting this function in middle of a dimension
                    if idx < (self.n_slices * self.n_channels * self.n_frames * self.n_pos):
                        dims[2][0] = 0
                        dims[3][0] = 0

                    # If z-first, compute the birefringence here
                    if channel_dim == 1 and dim1 == self.n_channels - 1:

                        # Assign dimensions based off acquisition order to correctly add image to array
                        if dim_order == 0:
                            t, p, c, z = dim3, dim2, dim0, dim1
                        elif dim_order == 1:
                            t, p, c, z = dim3, dim2, dim1, dim0
                        elif dim_order == 2:
                            t, p, c, z = dim2, dim3, dim0, dim1
                        else:
                            t, p, c, z = dim2, dim3, dim1, dim0

                        self._check_abort()
                        self.compute_and_save(array, p, t, dim0)
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

        # If the napari layer doesn't exist yet, create the zarr store to emit to napari
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

            # Add data and send out the store.  Once the store has been emitted, we can add data to the store
            # without needing to emit the store again (thanks to napari's handling of zarr datasets)
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

        # Emit the current dimensions so that we can update the napari dimension slider
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

            # If acquisition is not finished, grab the next file and listen for images
            if total_idx != self.n_slices * self.n_channels * self.n_frames * self.n_pos:
                time.sleep(1)
                file_count += 1
                file_path = os.path.join(full_path, self.prefix + f'_MMStack_{file_count}.ome.tif')
                file = tiff.TiffFile(file_path)
                offsets = file.micromanager_metadata['IndexMap']['Offset']
                file.close()

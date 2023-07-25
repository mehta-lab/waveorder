from __future__ import annotations

from qtpy.QtCore import Signal
from iohub import open_ome_zarr
from iohub.convert import TIFFConverter
from recOrder.cli import settings
from recOrder.cli.compute_transfer_function import (
    compute_transfer_function_cli,
)
from recOrder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_cli,
)
from recOrder.acq.acq_functions import (
    generate_acq_settings,
    acquire_from_settings,
)
from recOrder.io.utils import model_to_yaml
from recOrder.io.utils import ram_message
from napari.qt.threading import WorkerBaseSignals, WorkerBase
from napari.utils.notifications import show_warning
import logging
import numpy as np
import os
import shutil

# type hint/check
from typing import TYPE_CHECKING

# avoid runtime import error
if TYPE_CHECKING:
    from recOrder.plugin.main_widget import MainWidget
    from recOrder.calib.Calibration import QLIPP_Calibration


def _generate_reconstruction_config_from_gui(
    reconstruction_config_path,
    mode,
    calib_window,
    input_channel_names,
):
    if mode == "birefringence" or mode == "all":
        if calib_window.bg_option == "None":
            background_path = ""
            remove_estimated_background = False
        elif calib_window.bg_option == "Measured":
            background_path = calib_window.acq_bg_directory
            remove_estimated_background = False
        elif calib_window.bg_option == "Estimated":
            background_path = ""
            remove_estimated_background = True
        elif calib_window.bg_option == "Measured + Estimated":
            background_path = calib_window.acq_bg_directory
            remove_estimated_background = True

        birefringence_transfer_function_settings = (
            settings.BirefringenceTransferFunctionSettings(
                swing=calib_window.swing,
            )
        )
        birefringence_apply_inverse_settings = settings.BirefringenceApplyInverseSettings(
            wavelength_illumination=calib_window.recon_wavelength / 1000,
            background_path=background_path,
            remove_estimated_background=remove_estimated_background,
            orientation_flip=False,  # TODO connect this to a recOrder button
            orientation_rotate=False,  # TODO connect this to a recOrder button
        )
        birefringence_settings = settings.BirefringenceSettings(
            transfer_function=birefringence_transfer_function_settings,
            apply_inverse=birefringence_apply_inverse_settings,
        )
    else:
        birefringence_settings = None

    if mode == "phase" or mode == "all":
        phase_transfer_function_settings = (
            settings.PhaseTransferFunctionSettings(
                wavelength_illumination=calib_window.recon_wavelength / 1000,
                yx_pixel_size=calib_window.ps / calib_window.mag,
                z_pixel_size=calib_window.z_step,
                z_padding=calib_window.pad_z,
                index_of_refraction_media=calib_window.n_media,
                numerical_aperture_detection=calib_window.obj_na,
                numerical_aperture_illumination=calib_window.cond_na,
                axial_flip=False,  # TODO: connect this to a recOrder button
            )
        )
        phase_apply_inverse_settings = settings.FourierApplyInverseSettings(
            reconstruction_algorithm=calib_window.phase_regularizer,
            regularization_strength=calib_window.ui.le_phase_strength.text(),
            TV_rho_strength=calib_window.ui.le_rho.text(),
            TV_iterations=calib_window.ui.le_itr.text(),
        )
        phase_settings = settings.PhaseSettings(
            transfer_function=phase_transfer_function_settings,
            apply_inverse=phase_apply_inverse_settings,
        )
    else:
        phase_settings = None

    reconstruction_settings = settings.ReconstructionSettings(
        input_channel_names=input_channel_names,
        reconstruction_dimension=int(calib_window.acq_mode[0]),
        birefringence=birefringence_settings,
        phase=phase_settings,
    )

    model_to_yaml(reconstruction_settings, reconstruction_config_path)


class PolarizationAcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = Signal(object)
    bire_image_emitter = Signal(object)
    phase_reconstructor_emitter = Signal(object)
    aborted = Signal()


class BFAcquisitionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = Signal(object)
    phase_reconstructor_emitter = Signal(object)
    aborted = Signal()


class BFAcquisitionWorker(WorkerBase):
    """
    Class to execute a brightfield acquisition.  First step is to snap the images follow by a second
    step of reconstructing those images.
    """

    def __init__(self, calib_window: MainWidget):
        super().__init__(SignalsClass=BFAcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init Properties
        self.prefix = "recOrderPluginSnap"
        self.dm = self.calib_window.mm.displays()
        self.dim = (
            "2D"
            if self.calib_window.ui.cb_acq_mode.currentIndex() == 0
            else "3D"
        )
        self.img_dim = None

        save_dir = (
            self.calib_window.save_directory
            if self.calib_window.save_directory
            else self.calib_window.directory
        )

        if save_dir is None:
            raise ValueError(
                "save directory is empty, please specify a directory in the plugin"
            )

        # increment filename one more than last found saved snap
        i = 0
        prefix = self.calib_window.save_name
        snap_dir = (
            f"recOrderPluginSnap_{i}"
            if not prefix
            else f"{prefix}_recOrderPluginSnap_{i}"
        )
        while os.path.exists(os.path.join(save_dir, snap_dir)):
            i += 1
            snap_dir = (
                f"recOrderPluginSnap_{i}"
                if not prefix
                else f"{prefix}_recOrderPluginSnap_{i}"
            )

        self.snap_dir = os.path.join(save_dir, snap_dir)
        os.mkdir(self.snap_dir)

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError("Stop Requested")

    def _check_ram(self):
        """
        Show a warning if RAM < 32 GB.
        """
        is_warning, msg = ram_message()
        if is_warning:
            show_warning(msg)
        else:
            logging.info(msg)

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """
        self._check_ram()
        logging.info("Running Acquisition...")
        self._check_abort()
        self.calib_window._dump_gui_state(self.snap_dir)

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

        # Create and validate reconstruction settings
        self.config_path = os.path.join(
            self.snap_dir, "reconstruction_settings.yml"
        )
        _generate_reconstruction_config_from_gui(
            self.config_path,
            "phase",
            self.calib_window,
            input_channel_names=["BF"],
        )

        # Acquire 3D stack
        logging.debug("Acquiring 3D stack")

        # Generate MDA Settings
        settings = generate_acq_settings(
            self.calib_window.mm,
            channel_group=channel_group,
            channels=[channel],
            zstart=self.calib_window.z_start,
            zend=self.calib_window.z_end,
            zstep=self.calib_window.z_step,
            save_dir=self.snap_dir,
            prefix=self.prefix,
            keep_shutter_open_slices=True,
        )

        self._check_abort()

        # Acquire from MDA settings uses MM MDA GUI
        # Returns (1, 4/5, Z, Y, X) array
        stack = acquire_from_settings(
            self.calib_window.mm,
            settings,
            grab_images=True,
            restore_settings=True,
        )
        self._check_abort()

        # Cleanup acquisition by closing window, converting to zarr, and deleting temp directory
        self._cleanup_acq()

        # Reconstruct snapped images
        self.n_slices = stack.shape[2]

        phase = self._reconstruct()
        self._check_abort()

        # # Save images
        # logging.debug("Saving Images")
        # self._save_imgs(phase, meta)

        # self._check_abort()

        logging.info("Finished Acquisition")
        logging.debug("Finished Acquisition")

        # Emit the images and let thread know function is finished
        self.phase_image_emitter.emit(phase)

    def _reconstruct(self):
        """
        Method to reconstruct
        """
        self._check_abort()

        # Create i/o paths
        transfer_function_path = os.path.join(
            self.snap_dir, "transfer_function.zarr"
        )
        reconstruction_path = os.path.join(
            self.snap_dir, "reconstruction.zarr"
        )

        input_data_path = os.path.join(self.latest_out_path, "0", "0", "0")

        # TODO: skip if config files match
        compute_transfer_function_cli(
            input_data_path,
            self.config_path,
            transfer_function_path,
        )

        apply_inverse_transfer_function_cli(
            input_data_path,
            transfer_function_path,
            self.config_path,
            reconstruction_path,
        )

        # Read reconstruction to pass to emitters
        with open_ome_zarr(reconstruction_path, mode="r") as dataset:
            phase = dataset["0"][0]

        return phase

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
                    save_prefix = (
                        self.calib_window.save_name
                        if self.calib_window.save_name
                        else None
                    )
                    name = (
                        f"RawBFDataSnap.zarr"
                        if not save_prefix
                        else f"{save_prefix}_RawBFDataSnap.zarr"
                    )
                    self.latest_out_path = os.path.join(self.snap_dir, name)
                    converter = TIFFConverter(
                        os.path.join(dir_, prefix),
                        self.latest_out_path,
                        data_type="ometiff",
                        grid_layout=False,
                        label_positions=False,
                    )
                    converter.run()
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

    def __init__(
        self, calib_window: MainWidget, calib: QLIPP_Calibration, mode: str
    ):
        super().__init__(SignalsClass=PolarizationAcquisitionSignals)

        # Save current state of GUI window
        self.calib_window = calib_window

        # Init properties
        self.calib = calib
        self.mode = mode
        self.n_slices = None
        self.prefix = "recOrderPluginSnap"
        self.dm = self.calib_window.mm.displays()
        self.channel_group = self.calib_window.config_group

        # Determine whether 2D or 3D acquisition is needed
        if self.mode == "birefringence" and self.calib_window.acq_mode == "2D":
            self.dim = "2D"
        else:
            self.dim = "3D"

        save_dir = (
            self.calib_window.save_directory
            if self.calib_window.save_directory
            else self.calib_window.directory
        )

        if save_dir is None:
            raise ValueError(
                "save directory is empty, please specify a directory in the plugin"
            )

        # increment filename one more than last found saved snap
        i = 0
        prefix = self.calib_window.save_name
        snap_dir = (
            f"recOrderPluginSnap_{i}"
            if not prefix
            else f"{prefix}_recOrderPluginSnap_{i}"
        )
        while os.path.exists(os.path.join(save_dir, snap_dir)):
            i += 1
            snap_dir = (
                f"recOrderPluginSnap_{i}"
                if not prefix
                else f"{prefix}_recOrderPluginSnap_{i}"
            )

        self.snap_dir = os.path.join(save_dir, snap_dir)
        os.mkdir(self.snap_dir)

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError("Stop Requested")

    def _check_ram(self):
        """
        Show a warning if RAM < 32 GB.
        """
        is_warning, msg = ram_message()
        if is_warning:
            show_warning(msg)
        else:
            logging.info(msg)

    def work(self):
        """
        Function that runs the 2D or 3D acquisition and reconstructs the data
        """
        self._check_ram()
        logging.info("Running Acquisition...")
        self.calib_window._dump_gui_state(self.snap_dir)

        # List the Channels to acquire, if 5-state then append 5th channel
        channels = ["State0", "State1", "State2", "State3"]
        if self.calib.calib_scheme == "5-State":
            channels.append("State4")

        self._check_abort()

        # Create and validate reconstruction settings
        self.config_path = os.path.join(
            self.snap_dir, "reconstruction_settings.yml"
        )
        _generate_reconstruction_config_from_gui(
            self.config_path,
            self.mode,
            self.calib_window,
            input_channel_names=channels,
        )

        # Acquire 2D stack
        if self.dim == "2D":
            logging.debug("Acquiring 2D stack")

            # Generate MDA Settings
            self.settings = generate_acq_settings(
                self.calib_window.mm,
                channel_group=self.channel_group,
                channels=channels,
                save_dir=self.snap_dir,
                prefix=self.prefix,
                keep_shutter_open_channels=True,
            )
            self._check_abort()
            # acquire images
            stack = self._acquire()

        # Acquire 3D stack
        else:
            logging.debug("Acquiring 3D stack")

            # Generate MDA Settings
            self.settings = generate_acq_settings(
                self.calib_window.mm,
                channel_group=self.channel_group,
                channels=channels,
                zstart=self.calib_window.z_start,
                zend=self.calib_window.z_end,
                zstep=self.calib_window.z_step,
                save_dir=self.snap_dir,
                prefix=self.prefix,
                keep_shutter_open_channels=True,
                keep_shutter_open_slices=True,
            )

            self._check_abort()

            # set acquisition order to channel-first
            self.settings["slicesFirst"] = False
            self.settings["acqOrderMode"] = 0  # TIME_POS_SLICE_CHANNEL

            # acquire images
            stack = self._acquire()

        # Cleanup acquisition by closing window, converting to zarr, and deleting temp directory
        self._cleanup_acq()

        # Reconstruct snapped images
        self._check_abort()
        self.n_slices = stack.shape[2]
        birefringence, phase = self._reconstruct()
        self._check_abort()

        if self.calib_window.orientation_offset:
            birefringence = self._orientation_offset(birefringence)

        # Save images
        # logging.debug("Saving Images")
        # self._save_imgs(birefringence, phase)

        # self._check_abort()

        logging.info("Finished Acquisition")
        logging.debug("Finished Acquisition")

        # Emit the images and let thread know function is finished
        self.bire_image_emitter.emit(birefringence)
        self.phase_image_emitter.emit(phase)

    def _orientation_offset(self, birefringence: np.ndarray):
        """Apply +90 degree orientation offset

        Parameters
        ----------
        birefringence : np.ndarray
            [2, Y, X] retardance and orientation

        Returns
        -------
        ndarray
            birefringence with offset
        """
        show_warning(
            (
                "Applying a +90 degree orientation offset to birefringence! "
                "This will affect both visualization and the saved reconstruction."
            )
        )

        # TODO: Move these to waveorder, connect to the `orientation_flip`
        # and `orientation_rotate` parameters, and connect to recOrder buttons.

        # 90 degree flip
        birefringence[1] = np.fmod(birefringence[1] + np.pi / 2, np.pi)

        # Reflection
        birefringence[1] = np.fmod(-birefringence[1], np.pi) + np.pi

        return birefringence

    def _check_exposure(self) -> None:
        """
        Check that all LF channels have the same exposure settings. If not, abort Acquisition.
        """
        # parse exposure times
        channel_exposures = []
        for channel in self.settings["channels"]:
            channel_exposures.append(channel["exposure"])
        logging.debug(f"Verifying exposure times: {channel_exposures}")
        channel_exposures = np.array(channel_exposures)
        # check if exposure times are equal
        if not np.all(channel_exposures == channel_exposures[0]):
            error_exposure_msg = (
                f"The MDA exposure times are not equal! Aborting Acquisition.\n"
                f"Please manually set the exposure times to the same value from the MDA menu."
            )

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
        stack = acquire_from_settings(
            self.calib_window.mm,
            self.settings,
            grab_images=True,
            restore_settings=True,
        )
        self._check_abort()

        return stack

    def _reconstruct(self):
        """
        Method to reconstruct.  First need to initialize the reconstructor given
        what type of acquisition it is (birefringence only skips a lot of heavy compute needed for phase).
        This function also checks to see if the reconstructor needs to be updated from previous acquisitions

        """
        self._check_abort()

        # Create config and i/o paths
        transfer_function_path = os.path.join(
            self.snap_dir, "transfer_function.zarr"
        )
        reconstruction_path = os.path.join(
            self.snap_dir, "reconstruction.zarr"
        )

        input_data_path = os.path.join(self.latest_out_path, "0", "0", "0")

        # TODO: skip if config files match
        compute_transfer_function_cli(
            input_data_path,
            self.config_path,
            transfer_function_path,
        )

        apply_inverse_transfer_function_cli(
            input_data_path,
            transfer_function_path,
            self.config_path,
            reconstruction_path,
        )

        # Read reconstruction to pass to emitters
        with open_ome_zarr(reconstruction_path, mode="r") as dataset:
            czyx_data = dataset["0"][0]
            birefringence = czyx_data[0:4]
            try:
                phase = czyx_data[4]
            except:
                phase = None

        return birefringence, phase

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
                    save_prefix = (
                        self.calib_window.save_name
                        if self.calib_window.save_name
                        else None
                    )
                    name = (
                        f"RawPolDataSnap.zarr"
                        if not save_prefix
                        else f"{save_prefix}_RawPolDataSnap.zarr"
                    )
                    self.latest_out_path = os.path.join(self.snap_dir, name)
                    converter = TIFFConverter(
                        os.path.join(dir_, prefix),
                        self.latest_out_path,
                        data_type="ometiff",
                        grid_layout=False,
                        label_positions=False,
                    )
                    converter.run()
                    shutil.rmtree(os.path.join(dir_, prefix))
                except PermissionError as ex:
                    dp.close()
                break
            else:
                continue

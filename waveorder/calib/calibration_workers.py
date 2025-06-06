from __future__ import annotations

import json
import logging
from pathlib import Path

# type hint/check
from typing import TYPE_CHECKING

import numpy as np
from iohub import open_ome_zarr
from napari.qt.threading import WorkerBase, WorkerBaseSignals, thread_worker
from qtpy.QtCore import Signal

from waveorder.calib.Calibration import LC_DEVICE_NAME
from waveorder.cli import settings
from waveorder.cli.apply_inverse_transfer_function import (
    _apply_inverse_transfer_function_cli,
)
from waveorder.cli.compute_transfer_function import (
    _compute_transfer_function_cli,
)
from waveorder.io.core_functions import set_lc_state, snap_and_average
from waveorder.io.metadata_reader import MetadataReader
from waveorder.io.utils import MockEmitter, add_index_to_path, model_to_yaml

# avoid runtime import error
if TYPE_CHECKING:
    pass

    from waveorder.calib.Calibration import QLIPP_Calibration
    from waveorder.plugin.main_widget import MainWidget


class CalibrationSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    progress_update = Signal(tuple)
    extinction_update = Signal(str)
    intensity_update = Signal(object)
    calib_assessment = Signal(str)
    calib_assessment_msg = Signal(str)
    calib_file_emit = Signal(Path)
    plot_sequence_emit = Signal(str)
    lc_states = Signal(tuple)
    aborted = Signal()


class BackgroundSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    bg_image_emitter = Signal(tuple)
    bire_image_emitter = Signal(tuple)
    bg_path_update_emitter = Signal(Path)
    aborted = Signal()


class CalibrationWorkerBase(WorkerBase):
    """
    Base class for creating calibration workers.
    """

    def __init_subclass__(cls, signals: WorkerBaseSignals):
        """Called when creating calibration worker classes.

        Parameters
        ----------
        signals : WorkerBaseSignals
            Qt Signals class for the created worker class to send data across threads.
        """
        super().__init_subclass__()
        cls.signals = signals

    def __init__(self, calib_window: MainWidget, calib: QLIPP_Calibration):
        """Initialize the worker object.

        Parameters
        ----------
        calib_window : MainWidget
            The waveorder plugin's main GUI widget object containing metadata input.
        calib : QLIPP_Calibration
            waveorder calibration backend object.
        """
        super().__init__(SignalsClass=self.signals)
        self.calib_window = calib_window
        self.calib = calib

    def _check_abort(self):
        """
        Called if the user presses the STOP button.
        Needs to be checked after every major step to stop the process
        """
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError("Stop Requested.")

    def _write_meta_file(self, meta_file: str):
        self.calib.meta_file = meta_file
        self.calib.write_metadata(
            notes=self.calib_window.ui.le_notes_field.text()
        )


class CalibrationWorker(CalibrationWorkerBase, signals=CalibrationSignals):
    """
    Class to execute calibration
    """

    def __init__(self, calib_window, calib):
        super().__init__(calib_window, calib)

    def work(self):
        """
        Runs the full calibration algorithm and emits necessary signals.
        """

        self.plot_sequence_emit.emit("Coarse")
        self.calib.intensity_emitter = self.intensity_update
        self.calib.plot_sequence_emitter = self.plot_sequence_emit
        self.progress_update.emit((1, "Calculating Blacklevel..."))
        self._check_abort()

        logging.info("Calculating Black Level ...")
        logging.debug("Calculating Black Level ...")
        self.calib.close_shutter_and_calc_blacklevel()

        # Calculate Blacklevel
        logging.info(f"Black Level: {self.calib.I_Black:.0f}\n")
        logging.debug(f"Black Level: {self.calib.I_Black:.0f}\n")

        self._check_abort()
        self.progress_update.emit((10, "Calibrating Extinction State..."))

        # Open shutter
        self.calib.open_shutter()

        # Set LC Wavelength:
        self.calib.set_wavelength(int(self.calib_window.wavelength))
        if self.calib_window.calib_mode == "MM-Retardance":
            self.calib_window.mmc.setProperty(
                LC_DEVICE_NAME, "Wavelength", self.calib_window.wavelength
            )

        self._check_abort()

        # Optimize States
        (
            self._calibrate_4state()
            if self.calib_window.calib_scheme == "4-State"
            else self._calibrate_5state()
        )

        # Reset shutter autoshutter
        self.calib.reset_shutter()

        # Calculate Extinction
        extinction_ratio = self.calib.calculate_extinction(
            self.calib.swing,
            self.calib.I_Black,
            self.calib.I_Ext,
            self.calib.I_Elliptical,
        )
        self._check_abort()

        # Update main GUI with extinction ratio
        self.calib.extinction_ratio = extinction_ratio
        self.extinction_update.emit(str(extinction_ratio))

        # determine metadata filename
        meta_file = (
            Path(self.calib_window.directory) / "calibration_metadata.txt"
        )
        meta_file = add_index_to_path(meta_file)

        # Write Metadata
        self._write_meta_file(meta_file)
        self.calib_file_emit.emit(self.calib.meta_file)
        self.progress_update.emit((100, "Finished"))

        self._check_abort()

        # Perform calibration assessment based on retardance values
        self._assess_calibration()

        self._check_abort()

        # Emit calibrated LC states for plotting
        self.lc_states.emit((self.calib.pol_states, self.calib.lc_states))

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {extinction_ratio:.2f}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {extinction_ratio:.2f}")

    def _calibrate_4state(self):
        """
        Run through the 4-state calibration algorithm
        """

        search_radius = np.min((self.calib.swing / self.calib.ratio, 0.05))

        self.calib.calib_scheme = "4-State"

        self._check_abort()

        # Optimize Extinction State
        self.calib.opt_Iext()

        self._check_abort()
        self.progress_update.emit((60, "Calibrating State 1..."))

        # Optimize first elliptical (reference) state
        self.calib.opt_I0()
        self.progress_update.emit((65, "Calibrating State 2..."))

        self._check_abort()

        # Optimize 60 deg state
        self.calib.opt_I60(search_radius, search_radius)
        self.progress_update.emit((75, "Calibrating State 3..."))

        self._check_abort()

        # Optimize 120 deg state
        self.calib.opt_I120(search_radius, search_radius)
        self.progress_update.emit((85, "Writing Metadata..."))

        self._check_abort()

    def _calibrate_5state(self):
        search_radius = np.min((self.calib.swing, 0.05))

        self.calib.calib_scheme = "5-State"

        # Optimize Extinction State
        self.calib.opt_Iext()
        self.progress_update.emit((50, "Calibrating State 1..."))

        self._check_abort()

        # Optimize First elliptical state
        self.calib.opt_I0()
        self.progress_update.emit((55, "Calibrating State 2..."))

        self._check_abort()

        # Optimize 45 deg state
        self.calib.opt_I45(search_radius, search_radius)
        self.progress_update.emit((65, "Calibrating State 3..."))

        self._check_abort()

        # Optimize 90 deg state
        self.calib.opt_I90(search_radius, search_radius)
        self.progress_update.emit((75, "Calibrating State 4..."))

        self._check_abort()

        # Optimize 135 deg state
        self.calib.opt_I135(search_radius, search_radius)
        self.progress_update.emit((85, "Writing Metadata..."))

        self._check_abort()

    def _assess_calibration(self):
        """
        Assesses the quality of calibration based off retardance values.
        Attempts to determine whether certain optical components are out of place.
        """

        if self.calib.extinction_ratio >= 100:
            self.calib_assessment.emit("good")
            self.calib_assessment_msg.emit("Successful Calibration")
        elif 80 <= self.calib.extinction_ratio < 100:
            self.calib_assessment.emit("okay")
            self.calib_assessment_msg.emit(
                "Successful Calibration, Okay Extinction Ratio"
            )
        else:
            self.calib_assessment.emit("bad")
            message = (
                "Possibilities are: a) linear polarizer and LC are not oriented properly, "
                "b) circular analyzer has wrong handedness, "
                "c) the condenser is not setup for Kohler illumination, "
                "d) a component, such as autofocus dichroic or sample chamber, distorts the polarization state"
            )

            self.calib_assessment_msg.emit("Poor Extinction. " + message)


class BackgroundCaptureWorker(
    CalibrationWorkerBase, signals=BackgroundSignals
):
    """
    Class to execute background capture.
    """

    def __init__(self, calib_window, calib):
        super().__init__(calib_window, calib)

    def work(self):
        # Make the background folder
        bg_path = (
            Path(self.calib_window.directory)
            / self.calib_window.ui.le_bg_folder.text()
        )
        bg_path = add_index_to_path(bg_path)
        bg_path.mkdir()

        self._check_abort()

        # capture and return background images
        imgs = self.calib.capture_bg(self.calib_window.n_avg, bg_path)

        # build background-specific reconstruction settings
        reconstruction_settings = settings.ReconstructionSettings(
            input_channel_names=[
                f"State{i}"
                for i in range(int(self.calib_window.calib_scheme[0]))
            ],
            reconstruction_dimension=2,
            birefringence=settings.BirefringenceSettings(
                transfer_function=settings.BirefringenceTransferFunctionSettings(
                    swing=self.calib_window.swing
                ),
                apply_inverse=settings.BirefringenceApplyInverseSettings(
                    wavelength_illumination=self.calib_window.recon_wavelength
                    / 1000,
                    background_path="",
                    remove_estimated_background=False,
                    flip_orientation=False,
                    rotate_orientation=False,
                ),
            ),
        )

        reconstruction_config_path = bg_path / "reconstruction_settings.yml"
        model_to_yaml(reconstruction_settings, reconstruction_config_path)

        input_data_path = bg_path / "background.zarr" / "0" / "0" / "0"
        transfer_function_path = bg_path / "transfer_function.zarr"
        reconstruction_path = bg_path / "reconstruction.zarr"

        _compute_transfer_function_cli(
            input_position_dirpath=input_data_path,
            config_filepath=reconstruction_config_path,
            output_dirpath=transfer_function_path,
        )

        _apply_inverse_transfer_function_cli(
            input_position_dirpaths=[input_data_path],
            transfer_function_dirpath=transfer_function_path,
            config_filepath=reconstruction_config_path,
            output_dirpath=reconstruction_path,
        )

        # Load reconstructions from file for layers
        with open_ome_zarr(reconstruction_path, mode="r") as dataset:
            self.retardance = dataset["0/0/0/0"][0, 0, 0]
            self.birefringence = dataset["0/0/0/0"][0, :, 0]
            scale = dataset["0/0/0"].scale

        # Save metadata file and emit imgs
        meta_file = bg_path / "polarization_calibration.txt"
        self._write_meta_file(meta_file)

        # Update last calibration file
        note = self.calib_window.ui.le_notes_field.text()

        with open(self.calib_window.last_calib_meta_file, "r") as file:
            current_json = json.load(file)

        old_note = current_json["Notes"]
        if old_note is None or old_note == "" or old_note == note:
            current_json["Notes"] = note
        else:
            current_json["Notes"] = old_note + ", " + note

        with open(self.calib_window.last_calib_meta_file, "w") as file:
            json.dump(current_json, file, indent=1)

        self._check_abort()

        # Emit background images + background birefringence
        self.bg_image_emitter.emit((imgs, scale))
        self.bire_image_emitter.emit(
            ((self.retardance, self.birefringence[1]), scale)
        )

        # Emit bg path
        self.bg_path_update_emitter.emit(bg_path)


@thread_worker
def load_calibration(calib, metadata: MetadataReader):
    """
    Sets MM properties based upon calibration metadata file


    Parameters
    ----------
    calib:          (object) waveorder Calibration Class
    metadata:       (object) MetadataReader instance

    Returns
    -------
    calib           (object) updated waveorder Calibration Class
    """
    calib.calib_scheme = metadata.Calibration_scheme

    def _set_calib_attrs(calib, metadata):
        """Set the retardance attributes in the waveorder Calibration object"""
        if calib.calib_scheme == "4-State":
            lc_states = ["ext", "0", "60", "120"]
        elif calib.calib_scheme == "5-State":
            lc_states = ["ext", "0", "45", "90", "135"]
        else:
            raise ValueError(
                "Invalid calibration scheme in metadata: {calib.calib_scheme}"
            )
        for side in ("A", "B"):
            retardance_values = metadata.__getattribute__(
                "LC" + side + "_retardance"
            )
            for i, state in enumerate(lc_states):
                # set the retardance value attribute (e.g. 'lca_0')
                retardance_name = "lc" + side.lower() + "_" + state
                setattr(calib, retardance_name, retardance_values[i])
                # set the swing value attribute (e.g. 'swing0')
                if state != "ext":
                    swing_name = "swing" + state
                    setattr(calib, swing_name, metadata.Swing_measured[i - 1])

    _set_calib_attrs(calib, metadata)

    for state, lca, lcb in zip(
        [f"State{i}" for i in range(5)],
        metadata.LCA_retardance,
        metadata.LCB_retardance,
    ):
        calib.define_lc_state(state, lca, lcb)

    # Calculate black level after loading these properties
    calib.intensity_emitter = MockEmitter()
    calib.close_shutter_and_calc_blacklevel()
    calib.open_shutter()
    set_lc_state(calib.mmc, calib.group, "State0")
    calib.I_Ext = snap_and_average(calib.snap_manager)
    set_lc_state(calib.mmc, calib.group, "State1")
    calib.I_Elliptical = snap_and_average(calib.snap_manager)
    calib.reset_shutter()

    yield str(
        calib.calculate_extinction(
            calib.swing, calib.I_Black, calib.I_Ext, calib.I_Elliptical
        )
    )

    return calib

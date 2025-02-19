import os, json, subprocess, time, datetime, uuid
import socket, threading
from pathlib import Path

from qtpy import QtCore
from qtpy.QtCore import Qt, QEvent, QThread, Signal
from qtpy.QtWidgets import *
from magicgui.widgets import *

from iohub.ngff import open_ome_zarr

from typing import List, Literal, Union, Final, Annotated
from magicgui import widgets
from magicgui.type_map import get_widget_class
import warnings

try:
    from napari import Viewer
    from napari.utils import notifications
except:pass

from recOrder.io import utils
from recOrder.cli import settings, jobs_mgmt

import concurrent.futures

import importlib.metadata

import pydantic.v1, pydantic
from pydantic.v1 import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    root_validator,
    validator,
)

try:
    # Use version specific pydantic import for ModelMetaclass
    # prefer to pin to 1.10.19
    version = importlib.metadata.version("pydantic")
    # print("Your Pydantic library ver:{v}.".format(v=version))
    if version >= "2.0.0":
        print(
            "Your Pydantic library ver:{v}. Recommended ver is: 1.10.19".format(
                v=version
            )
        )
        from pydantic.main import ValidationError
        from pydantic.main import BaseModel
        from pydantic.main import ModelMetaclass
    elif version >= "1.10.19":
        from pydantic.main import ValidationError
        from pydantic.main import BaseModel
        from pydantic.main import ModelMetaclass
    else:
        print(
            "Your Pydantic library ver:{v}. Recommended ver is: 1.10.19".format(
                v=version
            )
        )
        from pydantic.main import ValidationError
        from pydantic.main import BaseModel
        from pydantic.main import ModelMetaclass
except:
    print("Pydantic library was not found. Ver 1.10.19 is recommended.")

STATUS_submitted_pool = "Submitted_Pool"
STATUS_submitted_job = "Submitted_Job"
STATUS_running_pool = "Running_Pool"
STATUS_running_job = "Running_Job"
STATUS_finished_pool = "Finished_Pool"
STATUS_finished_job = "Finished_Job"
STATUS_errored_pool = "Errored_Pool"
STATUS_errored_job = "Errored_Job"
STATUS_user_cleared_job = "User_Cleared_Job"
STATUS_user_cancelled_job = "User_Cancelled_Job"

MSG_SUCCESS = {"msg": "success"}
JOB_COMPLETION_STR = "Job completed successfully"
JOB_RUNNING_STR = "Starting with JobEnvironment"
JOB_TRIGGERED_EXC = "Submitted job triggered an exception"
JOB_OOM_EVENT = "oom_kill event"

_validate_alert = "⚠"
_validate_ok = "✔️"
_green_dot = "🟢"
_red_dot = "🔴"
_info_icon = "ⓘ"

# For now replicate CLI processing modes - these could reside in the CLI settings file as well
# for consistency
OPTION_TO_MODEL_DICT = {
    "birefringence": {"enabled": False, "setting": None},
    "phase": {"enabled": False, "setting": None},
    "fluorescence": {"enabled": False, "setting": None},
}

CONTAINERS_INFO = {}

# This keeps an instance of the MyWorker server that is listening
# napari will not stop processes and the Hide event is not reliable
HAS_INSTANCE = {"val": False, "instance": None}

# Components Queue list for new Jobs spanned from single processing
NEW_WIDGETS_QUEUE = []
NEW_WIDGETS_QUEUE_THREADS = []
MULTI_JOBS_REFS = {}
ROW_POP_QUEUE = []

# Main class for the Reconstruction tab
# Not efficient since instantiated from GUI
# Does not have access to common functions in main_widget
# ToDo : From main_widget and pass self reference
class Ui_ReconTab_Form(QWidget):

    def __init__(self, parent=None, stand_alone=False):
        super().__init__(parent)
        self._ui = parent
        self.stand_alone = stand_alone
        self.viewer: Viewer = None
        if HAS_INSTANCE["val"]:
            self.current_dir_path = str(Path.cwd())
            self.directory = str(Path.cwd())
            self.input_directory = HAS_INSTANCE["input_directory"]
            self.output_directory = HAS_INSTANCE["output_directory"]
            self.model_directory = HAS_INSTANCE["model_directory"]
            self.yaml_model_file = HAS_INSTANCE["yaml_model_file"]
        else:
            self.directory = str(Path.cwd())
            self.current_dir_path = str(Path.cwd())
            self.input_directory = str(Path.cwd())
            self.output_directory = str(Path.cwd())
            self.model_directory = str(Path.cwd())
            self.yaml_model_file = str(Path.cwd())

        self.input_directory_dataset = None
        self.input_directory_datasetMeta = None
        self.input_channel_names = []

        # Parent (Widget) which holds the GUI ##############################
        self.recon_tab_mainScrollArea = QScrollArea()
        self.recon_tab_mainScrollArea.setWidgetResizable(True)

        self.recon_tab_widget = QWidget()
        self.recon_tab_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.recon_tab_layout = QVBoxLayout()
        self.recon_tab_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.recon_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.recon_tab_layout.setSpacing(0)
        self.recon_tab_widget.setLayout(self.recon_tab_layout)
        self.recon_tab_mainScrollArea.setWidget(self.recon_tab_widget)

        # Top Section Group - Data ##############################
        group_box_Data_groupBox_widget = QGroupBox("Data")
        group_box_Data_layout = QVBoxLayout()
        group_box_Data_layout.setContentsMargins(0, 5, 0, 0)
        group_box_Data_layout.setSpacing(0)
        group_box_Data_groupBox_widget.setLayout(group_box_Data_layout)

        # Input Data ##############################
        self.data_input_widget = QWidget()
        self.data_input_widget_layout = QHBoxLayout()
        self.data_input_widget_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.data_input_widget.setLayout(self.data_input_widget_layout)

        self.data_input_Label = widgets.Label(value="Input Store")
        # self.data_input_Label.native.setMinimumWidth(97)
        self.data_input_LineEdit = widgets.LineEdit(value=self.input_directory)
        self.data_input_PushButton = widgets.PushButton(label="Browse")
        # self.data_input_PushButton.native.setMinimumWidth(75)
        self.data_input_PushButton.clicked.connect(self.browse_dir_path_input)
        self.data_input_LineEdit.changed.connect(
            self.read_and_set_input_path_on_validation
        )

        self.data_input_widget_layout.addWidget(self.data_input_Label.native)
        self.data_input_widget_layout.addWidget(
            self.data_input_LineEdit.native
        )
        self.data_input_widget_layout.addWidget(
            self.data_input_PushButton.native
        )

        # Output Data ##############################
        self.data_output_widget = QWidget()
        self.data_output_widget_layout = QHBoxLayout()
        self.data_output_widget_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.data_output_widget.setLayout(self.data_output_widget_layout)

        self.data_output_Label = widgets.Label(value="Output Directory")
        self.data_output_LineEdit = widgets.LineEdit(
            value=self.output_directory
        )
        self.data_output_PushButton = widgets.PushButton(label="Browse")
        # self.data_output_PushButton.native.setMinimumWidth(75)
        self.data_output_PushButton.clicked.connect(
            self.browse_dir_path_output
        )
        self.data_output_LineEdit.changed.connect(
            self.read_and_set_out_path_on_validation
        )

        self.data_output_widget_layout.addWidget(self.data_output_Label.native)
        self.data_output_widget_layout.addWidget(
            self.data_output_LineEdit.native
        )
        self.data_output_widget_layout.addWidget(
            self.data_output_PushButton.native
        )

        self.data_input_Label.native.setMinimumWidth(115)
        self.data_output_Label.native.setMinimumWidth(115)

        group_box_Data_layout.addWidget(self.data_input_widget)
        group_box_Data_layout.addWidget(self.data_output_widget)
        self.recon_tab_layout.addWidget(group_box_Data_groupBox_widget)

        ##################################

        # Middle Section - Models ##############################
        # Selection modes, New, Load, Clear
        # Pydantic Models ScrollArea

        group_box_Models_groupBox_widget = QGroupBox("Models")
        group_box_Models_layout = QVBoxLayout()
        group_box_Models_layout.setContentsMargins(0, 5, 0, 0)
        group_box_Models_layout.setSpacing(0)
        group_box_Models_groupBox_widget.setLayout(group_box_Models_layout)

        self.models_widget = QWidget()
        self.models_widget_layout = QHBoxLayout()
        self.models_widget_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.models_widget.setLayout(self.models_widget_layout)

        self.modes_selected = OPTION_TO_MODEL_DICT.copy()

        # Make a copy of the Reconstruction settings mode, these will be used as template
        for mode in self.modes_selected.keys():
            self.modes_selected[mode]["setting"] = None

        # Checkboxes for the modes to select single or combination of modes
        for mode in self.modes_selected.keys():
            self.modes_selected[mode]["Checkbox"] = widgets.Checkbox(
                name=mode, label=mode
            )
            self.models_widget_layout.addWidget(
                self.modes_selected[mode]["Checkbox"].native
            )

        # PushButton to create a copy of the model - UI
        self.models_new_PushButton = widgets.PushButton(label="New")
        # self.models_new_PushButton.native.setMinimumWidth(100)
        self.models_new_PushButton.clicked.connect(self.build_acq_contols)

        self.models_load_PushButton = DropButton(text="Load", recon_tab=self)
        # self.models_load_PushButton.setMinimumWidth(90)

        # Passing model location label to model location selector
        self.models_load_PushButton.clicked.connect(
            lambda: self.browse_dir_path_model()
        )

        # PushButton to clear all copies of models that are create for UI
        self.models_clear_PushButton = widgets.PushButton(label="Clear")
        # self.models_clear_PushButton.native.setMinimumWidth(110)
        self.models_clear_PushButton.clicked.connect(self.clear_all_models)

        self.models_widget_layout.addWidget(self.models_new_PushButton.native)
        self.models_widget_layout.addWidget(self.models_load_PushButton)
        self.models_widget_layout.addWidget(
            self.models_clear_PushButton.native
        )

        # Middle scrollable component which will hold Editable/(vertical) Expanding UI
        self.models_scrollArea = QScrollArea()
        self.models_scrollArea.setWidgetResizable(True)
        self.models_container_widget = DropWidget(self)
        self.models_container_widget_layout = QVBoxLayout()
        self.models_container_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.models_container_widget_layout.setSpacing(2)
        self.models_container_widget_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.models_container_widget.setLayout(
            self.models_container_widget_layout
        )
        self.models_scrollArea.setWidget(self.models_container_widget)

        group_box_Models_layout.addWidget(self.models_widget)
        group_box_Models_layout.addWidget(self.models_scrollArea)

        ##################################

        # Create the splitter to resize Middle and Bottom Sections if required ##################################
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)
        splitter.setSizes([600, 200])

        self.recon_tab_layout.addWidget(splitter)

        # Reconstruction ##################################
        # Run, Processing, On-The-Fly
        group_box_Reconstruction_groupBox_widget = QGroupBox(
            "Reconstruction Queue"
        )
        group_box_Reconstruction_layout = QVBoxLayout()
        group_box_Reconstruction_layout.setContentsMargins(5, 10, 5, 5)
        group_box_Reconstruction_layout.setSpacing(2)
        group_box_Reconstruction_groupBox_widget.setLayout(
            group_box_Reconstruction_layout
        )

        splitter.addWidget(group_box_Models_groupBox_widget)
        splitter.addWidget(group_box_Reconstruction_groupBox_widget)

        my_splitter_handle = splitter.handle(1)
        my_splitter_handle.setStyleSheet("background: 1px rgb(128,128,128);")
        splitter.setStyleSheet(
            """QSplitter::handle:pressed {background-color: #ca5;}"""
        )

        # PushButton to validate and Run the yaml file(s) based on selection against the Input store
        self.reconstruction_run_PushButton = widgets.PushButton(
            name="RUN Model"
        )
        self.reconstruction_run_PushButton.native.setMinimumWidth(100)
        self.reconstruction_run_PushButton.clicked.connect(
            self.build_model_and_run
        )

        group_box_Reconstruction_layout.addWidget(
            self.reconstruction_run_PushButton.native
        )

        # Tabs - Processing & On-The-Fly
        tabs_Reconstruction = QTabWidget()
        group_box_Reconstruction_layout.addWidget(tabs_Reconstruction)

        # Table for Jobs processing entries
        tab1_processing_widget = QWidget()
        tab1_processing_widget_layout = QVBoxLayout()
        tab1_processing_widget_layout.setContentsMargins(5, 5, 5, 5)
        tab1_processing_widget_layout.setSpacing(2)
        tab1_processing_widget.setLayout(tab1_processing_widget_layout)
        self.proc_table_QFormLayout = QFormLayout()
        self.proc_table_QFormLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        tab1_processing_form_widget = QWidget()
        tab1_processing_form_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        tab1_processing_form_widget.setLayout(self.proc_table_QFormLayout)
        tab1_processing_widget_layout.addWidget(tab1_processing_form_widget)

        _clear_results_btn = widgets.PushButton(label="Clear Results")
        _clear_results_btn.clicked.connect(self.clear_results_table)
        tab1_processing_widget_layout.addWidget(_clear_results_btn.native)

        # Table for On-The-Fly processing entries
        tab2_processing_widget = QWidget()
        tab2_processing_widget_layout = QVBoxLayout()
        tab2_processing_widget_layout.setContentsMargins(0, 0, 0, 0)
        tab2_processing_widget_layout.setSpacing(0)
        tab2_processing_widget.setLayout(tab2_processing_widget_layout)
        self.proc_OTF_table_QFormLayout = QFormLayout()
        self.proc_OTF_table_QFormLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        _proc_OTF_table_widget = QWidget()
        _proc_OTF_table_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        _proc_OTF_table_widget.setLayout(self.proc_OTF_table_QFormLayout)
        tab2_processing_widget_layout.addWidget(_proc_OTF_table_widget)
        tab2_processing_widget.setMaximumHeight(100)

        tabs_Reconstruction.addTab(tab1_processing_widget, "Processing")
        tabs_Reconstruction.addTab(tab2_processing_widget, "On-The-Fly")

        # Editable List holding pydantic class(es) as per user selection
        self.pydantic_classes = list()
        self.prev_model_settings = {}
        self.index = 0
        self.pollData = False

        # Stores Model & Components values which cause validation failure - can be highlighted on the model field as Red
        self.modelHighlighterVals = {}

        # handle napari's close widget and avoid starting a second server
        if HAS_INSTANCE["val"]:
            self.worker: MyWorker = HAS_INSTANCE["MyWorker"]
            self.worker.set_new_instances(
                self.proc_table_QFormLayout, self, self._ui
            )
        else:
            self.worker = MyWorker(self.proc_table_QFormLayout, self, self._ui)
            HAS_INSTANCE["val"] = True
            HAS_INSTANCE["MyWorker"] = self.worker

        self.app = QApplication.instance()
        self.app.lastWindowClosed.connect(
            self.myCloseEvent
        )  # this line is connection to signal close

    ######################################################

    # our defined close event since napari doesnt do
    def myCloseEvent(self):
        event = QEvent(QEvent.Type.Close)
        self.closeEvent(event)
        # self.app.exit()

    # on napari close - cleanup
    def closeEvent(self, event):
        if event.type() == QEvent.Type.Close:
            self.worker.stop_server()

    def hideEvent(self, event):
        if event.type() == QEvent.Type.Hide and (
            self._ui is not None and self._ui.isVisible()
        ):
            pass

    def showEvent(self, event):
        if event.type() == QEvent.Type.Show:
            pass

    def set_viewer(self, viewer):
        self.viewer = viewer

    def show_dataset(self, data_path):
        # Show reconstruction data
        try:
            if self.viewer is not None:
                self.viewer.open(data_path, plugin="napari-ome-zarr")
        except Exception as exc:
            self.message_box(exc.args)

    def confirm_dialog(self, msg="Confirm your selection ?"):
        qm = QMessageBox
        ret = qm.question(
            self.recon_tab_widget,
            "Confirm",
            msg,
            qm.Yes | qm.No,
        )
        if ret == qm.Yes:
            return True
        else:
            return False

    # Copied from main_widget
    # ToDo: utilize common functions
    # Input data selector
    def browse_dir_path_input(self):
        if len(self.pydantic_classes) > 0 and not self.confirm_dialog(
            "Changing Input Data will reset your models. Continue ?"
        ):
            return
        else:
            self.clear_all_models(silent=True)
        try:
            result = self.open_file_dialog(
                self.input_directory, "dir", filter="ZARR Storage (*.zarr)"
            )
            # .zarr is a folder but we could implement a filter to scan for "ending with" and present those if required
        except Exception as exc:
            self.message_box(exc.args)
            return

        if result == "":
            return

        self.data_input_LineEdit.value = result

    def browse_dir_path_output(self):
        try:
            result = self.open_file_dialog(self.output_directory, "dir")
        except Exception as exc:
            self.message_box(exc.args)
            return

        if result == "":
            return

        if not Path(result).exists():
            self.message_box("Output Directory path must exist !")
            return

        self.data_output_LineEdit.value = result

    def browse_dir_path_inputBG(self, elem):
        result = self.open_file_dialog(self.directory, "dir")
        if result == "":
            return

        ret, ret_msg = self.validate_input_data(result, BG=True)
        if not ret:
            self.message_box(ret_msg)
            return

        elem.value = result

    def validate_input_data(
        self, input_data_folder: str, metadata=False, BG=False
    ) -> bool:
        try:
            self.input_channel_names = []
            self.data_input_Label.value = "Input Store"
            input_paths = Path(input_data_folder)
            with open_ome_zarr(input_paths, mode="r") as dataset:
                try:
                    self.input_channel_names = dataset.channel_names
                    self.data_input_Label.value = (
                        "Input Store" + " " + _info_icon
                    )
                    self.data_input_Label.tooltip = (
                        "Channel Names:\n- "
                        + "\n- ".join(self.input_channel_names)
                    )
                except Exception as exc:
                    print(exc.args)

                try:
                    string_pos = []
                    i = 0
                    for pos_paths, pos in dataset.positions():
                        string_pos.append(pos_paths)
                        if i == 0:
                            axes = pos.zgroup.attrs["multiscales"][0]["axes"]
                            string_array_n = [str(x["name"]) for x in axes]
                            string_array = [
                                str(x)
                                for x in pos.zgroup.attrs["multiscales"][0][
                                    "datasets"
                                ][0]["coordinateTransformations"][0]["scale"]
                            ]
                            string_scale = []
                            for i in range(len(string_array_n)):
                                string_scale.append(
                                    "{n}={d}".format(
                                        n=string_array_n[i], d=string_array[i]
                                    )
                                )
                            txt = "\n\nScale: " + ", ".join(string_scale)
                            self.data_input_Label.tooltip += txt
                        i += 1
                    txt = "\n\nFOV: " + ", ".join(string_pos)
                    self.data_input_Label.tooltip += txt
                except Exception as exc:
                    print(exc.args)

                if not BG and metadata:
                    self.input_directory_dataset = dataset

                if not BG:
                    self.pollData = False
                    zattrs = dataset.zattrs
                    if self.is_dataset_acq_running(zattrs):
                        if self.confirm_dialog(
                            msg="This seems like an in-process Acquisition. Would you like to process data on-the-fly ?"
                        ):
                            self.pollData = True

                return True, MSG_SUCCESS
            raise Exception(
                "Dataset does not appear to be a valid ome-zarr storage"
            )
        except Exception as exc:
            return False, exc.args

    # call back for input LineEdit path changed manually
    # include data validation
    def read_and_set_input_path_on_validation(self):
        if (
            self.data_input_LineEdit.value is None
            or len(self.data_input_LineEdit.value) == 0
        ):
            self.data_input_LineEdit.value = self.input_directory
            self.message_box("Input data path cannot be empty")
            return
        if not Path(self.data_input_LineEdit.value).exists():
            self.data_input_LineEdit.value = self.input_directory
            self.message_box("Input data path must point to a valid location")
            return

        result = self.data_input_LineEdit.value
        valid, ret_msg = self.validate_input_data(result)

        if valid:
            self.directory = Path(result).parent.absolute()
            self.current_dir_path = result
            self.input_directory = result

            self.prev_model_settings = {}

            self.save_last_paths()
        else:
            self.data_input_LineEdit.value = self.input_directory
            self.message_box(ret_msg)

        self.data_output_LineEdit.value = Path(
            self.input_directory
        ).parent.absolute()

    def read_and_set_out_path_on_validation(self):
        if (
            self.data_output_LineEdit.value is None
            or len(self.data_output_LineEdit.value) == 0
        ):
            self.data_output_LineEdit.value = self.output_directory
            self.message_box("Output data path cannot be empty")
            return
        if not Path(self.data_output_LineEdit.value).exists():
            self.data_output_LineEdit.value = self.output_directory
            self.message_box("Output data path must point to a valid location")
            return

        self.output_directory = self.data_output_LineEdit.value

        self.validate_model_output_paths()

    def validate_model_output_paths(self):
        if len(self.pydantic_classes) > 0:
            for model_item in self.pydantic_classes:
                output_LineEdit = model_item["output_LineEdit"]
                output_Button = model_item["output_Button"]
                model_item["output_parent_dir"] = self.output_directory

                full_out_path = os.path.join(
                    Path(self.output_directory).absolute(),
                    output_LineEdit.value,
                )
                model_item["output"] = full_out_path

                save_path_exists = (
                    True if Path(full_out_path).exists() else False
                )
                output_LineEdit.label = (
                    "" if not save_path_exists else (_validate_alert + " ")
                ) + "Output Data:"
                output_LineEdit.tooltip = (
                    ""
                    if not save_path_exists
                    else (_validate_alert + "Output file exists")
                )
                output_Button.text = (
                    "" if not save_path_exists else (_validate_alert + " ")
                ) + "Output Data:"
                output_Button.tooltip = (
                    ""
                    if not save_path_exists
                    else (_validate_alert + "Output file exists")
                )

    def is_dataset_acq_running(self, zattrs: dict) -> bool:
        """
        Checks the zattrs for CurrentDimensions & FinalDimensions key and tries to figure if
        data acquisition is running
        """

        required_order = ["time", "position", "z", "channel"]
        if "CurrentDimensions" in zattrs.keys():
            my_dict = zattrs["CurrentDimensions"]
            sorted_dict_acq = {
                k: my_dict[k]
                for k in sorted(my_dict, key=lambda x: required_order.index(x))
            }
        if "FinalDimensions" in zattrs.keys():
            my_dict = zattrs["FinalDimensions"]
            sorted_dict_final = {
                k: my_dict[k]
                for k in sorted(my_dict, key=lambda x: required_order.index(x))
            }
            if sorted_dict_acq != sorted_dict_final:
                return True
        return False

    # Output data selector
    def browse_model_dir_path_output(self, elem):
        result = self.open_file_dialog(self.output_directory, "save")
        if result == "":
            return

        save_path_exists = True if Path(result).exists() else False
        elem.label = "Output Data:" + (
            "" if not save_path_exists else (" " + _validate_alert)
        )
        elem.tooltip = "" if not save_path_exists else "Output file exists"

        elem.value = Path(result).name

        self.save_last_paths()

    # call back for output LineEdit path changed manually
    def read_and_set_output_path_on_validation(self, elem1, elem2, save_path):
        if elem1.value is None or len(elem1.value) == 0:
            elem1.value = Path(save_path).name

        save_path = os.path.join(
            Path(self.output_directory).absolute(), elem1.value
        )

        save_path_exists = True if Path(save_path).exists() else False
        elem1.label = (
            "" if not save_path_exists else (_validate_alert + " ")
        ) + "Output Data:"
        elem1.tooltip = (
            ""
            if not save_path_exists
            else (_validate_alert + "Output file exists")
        )
        elem2.text = (
            "" if not save_path_exists else (_validate_alert + " ")
        ) + "Output Data:"
        elem2.tooltip = (
            ""
            if not save_path_exists
            else (_validate_alert + "Output file exists")
        )

        self.save_last_paths()

    # Copied from main_widget
    # ToDo: utilize common functions
    # Output data selector
    def browse_dir_path_model(self):
        results = self.open_file_dialog(
            self.directory, "files", filter="YAML Files (*.yml)"
        )  # returns list
        if len(results) == 0 or results == "":
            return

        self.model_directory = str(Path(results[0]).parent.absolute())
        self.directory = self.model_directory
        self.current_dir_path = self.model_directory

        self.save_last_paths()
        self.open_model_files(results)

    def open_model_files(self, results: List):
        pydantic_models = list()
        for result in results:
            self.yaml_model_file = result

            with open(result, "r") as yaml_in:
                yaml_object = utils.yaml.safe_load(
                    yaml_in
                )  # yaml_object will be a list or a dict
            jsonString = json.dumps(self.convert(yaml_object))
            json_out = json.loads(jsonString)
            json_dict = dict(json_out)

            selected_modes = list(OPTION_TO_MODEL_DICT.copy().keys())
            exclude_modes = list(OPTION_TO_MODEL_DICT.copy().keys())

            for k in range(len(selected_modes) - 1, -1, -1):
                if selected_modes[k] in json_dict.keys():
                    exclude_modes.pop(k)
                else:
                    selected_modes.pop(k)

            pruned_pydantic_class, ret_msg = self.build_model(selected_modes)
            if pruned_pydantic_class is None:
                self.message_box(ret_msg)
                return

            pydantic_model, ret_msg = self.get_model_from_file(
                self.yaml_model_file
            )
            if pydantic_model is None:
                if (
                    isinstance(ret_msg, List)
                    and len(ret_msg) == 2
                    and len(ret_msg[0]["loc"]) == 3
                    and ret_msg[0]["loc"][2] == "background_path"
                ):
                    pydantic_model = pruned_pydantic_class  # if only background_path fails validation
                    json_dict["birefringence"]["apply_inverse"][
                        "background_path"
                    ] = ""
                    self.message_box(
                        "background_path:\nPath was invalid and will be reset"
                    )
                else:
                    self.message_box(ret_msg)
                    return
            else:
                # make sure "background_path" is valid
                bg_loc = json_dict["birefringence"]["apply_inverse"][
                    "background_path"
                ]
                if bg_loc != "":
                    extension = os.path.splitext(bg_loc)[1]
                    if len(extension) > 0:
                        bg_loc = Path(
                            os.path.join(
                                str(Path(bg_loc).parent.absolute()),
                                "background.zarr",
                            )
                        )
                    else:
                        bg_loc = Path(os.path.join(bg_loc, "background.zarr"))
                    if not bg_loc.exists() or not self.validate_input_data(
                        str(bg_loc)
                    ):
                        self.message_box(
                            "background_path:\nPwas invalid and will be reset"
                        )
                        json_dict["birefringence"]["apply_inverse"][
                            "background_path"
                        ] = ""
                    else:
                        json_dict["birefringence"]["apply_inverse"][
                            "background_path"
                        ] = str(bg_loc.parent.absolute())

            pydantic_model = self.create_acq_contols2(
                selected_modes, exclude_modes, pydantic_model, json_dict
            )
            if pydantic_model is None:
                self.message_box("Error - pydantic model returned None")
                return

            pydantic_models.append(pydantic_model)

        return pydantic_models

    # useful when using close widget and not napari close and we might need them again
    def save_last_paths(self):
        HAS_INSTANCE["current_dir_path"] = self.current_dir_path
        HAS_INSTANCE["input_directory"] = self.input_directory
        HAS_INSTANCE["output_directory"] = self.output_directory
        HAS_INSTANCE["model_directory"] = self.model_directory
        HAS_INSTANCE["yaml_model_file"] = self.yaml_model_file

    # clears the results table
    def clear_results_table(self):
        index = self.proc_table_QFormLayout.rowCount()
        if index < 1:
            self.message_box("There are no processing results to clear !")
            return
        if self.confirm_dialog():
            for i in range(self.proc_table_QFormLayout.rowCount()):
                self.proc_table_QFormLayout.removeRow(0)

    def remove_row(self, row, expID):
        try:
            if row < self.proc_table_QFormLayout.rowCount():
                widgetItem = self.proc_table_QFormLayout.itemAt(row)
                if widgetItem is not None:
                    name_widget = widgetItem.widget()
                    toolTip_string = str(name_widget.toolTip)
                    if expID in toolTip_string:
                        self.proc_table_QFormLayout.removeRow(
                            row
                        )  # removeRow vs takeRow for threads ?
        except Exception as exc:
            print(exc.args)

    # marks fields on the Model that cause a validation error
    def model_highlighter(self, errs):
        try:
            for uid in errs.keys():
                self.modelHighlighterVals[uid] = {}
                container = errs[uid]["cls"]
                self.modelHighlighterVals[uid]["errs"] = errs[uid]["errs"]
                self.modelHighlighterVals[uid]["items"] = []
                self.modelHighlighterVals[uid]["tooltip"] = []
                if len(errs[uid]["errs"]) > 0:
                    self.model_highlighter_setter(
                        errs[uid]["errs"], container, uid
                    )
        except Exception as exc:
            print(exc.args)
            # more of a test feature - no need to show up

    # format all model errors into a display format for napari error message box
    def format_string_for_error_display(self, errs):
        try:
            ret_str = ""
            for uid in errs.keys():
                if len(errs[uid]["errs"]) > 0:
                    ret_str += errs[uid]["collapsibleBox"] + "\n"
                    for idx in range(len(errs[uid]["errs"])):
                        ret_str += f"{'>'.join(errs[uid]['errs'][idx]['loc'])}:\n{errs[uid]['errs'][idx]['msg']} \n"
                    ret_str += "\n"
            return ret_str
        except Exception as exc:
            return ret_str

    # recursively fix the container for highlighting
    def model_highlighter_setter(
        self, errs, container: Container, containerID, lev=0
    ):
        try:
            layout = container.native.layout()
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    widget = layout.itemAt(i).widget()
                    if (
                        (
                            not isinstance(widget._magic_widget, CheckBox)
                            and not isinstance(
                                widget._magic_widget, PushButton
                            )
                        )
                        and not isinstance(widget._magic_widget, LineEdit)
                        and isinstance(
                            widget._magic_widget._inner_widget, Container
                        )
                        and not (widget._magic_widget._inner_widget is None)
                    ):
                        self.model_highlighter_setter(
                            errs,
                            widget._magic_widget._inner_widget,
                            containerID,
                            lev + 1,
                        )
                    else:
                        for idx in range(len(errs)):
                            if len(errs[idx]["loc"]) - 1 < lev:
                                pass
                            elif (
                                isinstance(widget._magic_widget, CheckBox)
                                or isinstance(widget._magic_widget, LineEdit)
                                or isinstance(widget._magic_widget, PushButton)
                            ):
                                if widget._magic_widget.label == errs[idx][
                                    "loc"
                                ][lev].replace("_", " "):
                                    if widget._magic_widget.tooltip is None:
                                        widget._magic_widget.tooltip = "-\n"
                                        self.modelHighlighterVals[containerID][
                                            "items"
                                        ].append(widget._magic_widget)
                                        self.modelHighlighterVals[containerID][
                                            "tooltip"
                                        ].append(widget._magic_widget.tooltip)
                                    widget._magic_widget.tooltip += (
                                        errs[idx]["msg"] + "\n"
                                    )
                                    widget._magic_widget.native.setStyleSheet(
                                        "border:1px solid rgb(255, 255, 0); border-width: 1px;"
                                    )
                            elif (
                                widget._magic_widget._label_widget.value
                                == errs[idx]["loc"][lev].replace("_", " ")
                            ):
                                if (
                                    widget._magic_widget._label_widget.tooltip
                                    is None
                                ):
                                    widget._magic_widget._label_widget.tooltip = (
                                        "-\n"
                                    )
                                    self.modelHighlighterVals[containerID][
                                        "items"
                                    ].append(
                                        widget._magic_widget._label_widget
                                    )
                                    self.modelHighlighterVals[containerID][
                                        "tooltip"
                                    ].append(
                                        widget._magic_widget._label_widget.tooltip
                                    )
                                widget._magic_widget._label_widget.tooltip += (
                                    errs[idx]["msg"] + "\n"
                                )
                                widget._magic_widget._label_widget.native.setStyleSheet(
                                    "border:1px solid rgb(255, 255, 0); border-width: 1px;"
                                )
                                if (
                                    widget._magic_widget._inner_widget.tooltip
                                    is None
                                ):
                                    widget._magic_widget._inner_widget.tooltip = (
                                        "-\n"
                                    )
                                    self.modelHighlighterVals[containerID][
                                        "items"
                                    ].append(
                                        widget._magic_widget._inner_widget
                                    )
                                    self.modelHighlighterVals[containerID][
                                        "tooltip"
                                    ].append(
                                        widget._magic_widget._inner_widget.tooltip
                                    )
                                widget._magic_widget._inner_widget.tooltip += (
                                    errs[idx]["msg"] + "\n"
                                )
                                widget._magic_widget._inner_widget.native.setStyleSheet(
                                    "border:1px solid rgb(255, 255, 0); border-width: 1px;"
                                )
        except Exception as exc:
            print(exc.args)

    # recursively fix the container for highlighting
    def model_reset_highlighter_setter(self):
        try:
            for containerID in self.modelHighlighterVals.keys():
                items = self.modelHighlighterVals[containerID]["items"]
                tooltip = self.modelHighlighterVals[containerID]["tooltip"]
                i = 0
                for widItem in items:
                    widItem.native.setStyleSheet(
                        "border:1px solid rgb(0, 0, 0); border-width: 0px;"
                    )
                    widItem.tooltip = tooltip[i]
                    i += 1

        except Exception as exc:
            print(exc.args)

        except Exception as exc:
            print(exc.args)

    # passes msg to napari notifications
    def message_box(self, msg, type="exc"):
        if len(msg) > 0:
            try:
                json_object = msg
                json_txt = ""
                for err in json_object:
                    json_txt = (
                        json_txt
                        + "Loc: {loc}\nMsg:{msg}\nType:{type}\n\n".format(
                            loc=err["loc"], msg=err["msg"], type=err["type"]
                        )
                    )
                json_txt = str(json_txt)
                # ToDo: format it better
                # formatted txt does not show up properly in msg-box ??
            except:
                json_txt = str(msg)

            # show is a message box
            if self.stand_alone:
                self.message_box_stand_alone(json_txt)
            else:
                if type == "exc":
                    notifications.show_error(json_txt)
                else:
                    notifications.show_info(json_txt)

    def message_box_stand_alone(self, msg):
        q = QMessageBox(
            QMessageBox.Warning,
            "Message",
            str(msg),
            parent=self.recon_tab_widget,
        )
        q.setStandardButtons(QMessageBox.StandardButton.Ok)
        q.setIcon(QMessageBox.Icon.Warning)
        q.exec_()

    def cancel_job(self, btn: PushButton):
        if self.confirm_dialog():
            btn.enabled = False
            btn.text = btn.text + " (cancel called)"

    def add_widget(
        self, parentLayout: QVBoxLayout, expID, jID, table_entry_ID="", pos=""
    ):

        jID = str(jID)
        _cancelJobBtntext = "Cancel Job {jID} ({posName})".format(
            jID=jID, posName=pos
        )
        _cancelJobButton = widgets.PushButton(
            name="JobID", label=_cancelJobBtntext, enabled=True, value=False
        )
        _cancelJobButton.clicked.connect(
            lambda: self.cancel_job(_cancelJobButton)
        )
        _txtForInfoBox = "Updating {id}-{pos}: Please wait... \nJobID assigned: {jID} ".format(
            id=table_entry_ID, jID=jID, pos=pos
        )
        _scrollAreaCollapsibleBoxDisplayWidget = ScrollableLabel(
            text=_txtForInfoBox
        )

        _scrollAreaCollapsibleBoxWidgetLayout = QVBoxLayout()
        _scrollAreaCollapsibleBoxWidgetLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(
            _cancelJobButton.native
        )
        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(
            _scrollAreaCollapsibleBoxDisplayWidget
        )

        _scrollAreaCollapsibleBoxWidget = QWidget()
        _scrollAreaCollapsibleBoxWidget.setLayout(
            _scrollAreaCollapsibleBoxWidgetLayout
        )
        _scrollAreaCollapsibleBox = QScrollArea()
        _scrollAreaCollapsibleBox.setWidgetResizable(True)
        _scrollAreaCollapsibleBox.setMinimumHeight(300)
        _scrollAreaCollapsibleBox.setWidget(_scrollAreaCollapsibleBoxWidget)

        _collapsibleBoxWidgetLayout = QVBoxLayout()
        _collapsibleBoxWidgetLayout.addWidget(_scrollAreaCollapsibleBox)

        _collapsibleBoxWidget = CollapsibleBox(
            table_entry_ID + " - " + pos
        )  # tableEntryID, tableEntryShortDesc - should update with processing status
        _collapsibleBoxWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _collapsibleBoxWidget.setContentLayout(_collapsibleBoxWidgetLayout)

        parentLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        parentLayout.addWidget(_collapsibleBoxWidget)

        MULTI_JOBS_REFS[expID + jID] = {}
        MULTI_JOBS_REFS[expID + jID]["cancelBtn"] = _cancelJobButton
        MULTI_JOBS_REFS[expID + jID][
            "infobox"
        ] = _scrollAreaCollapsibleBoxDisplayWidget
        NEW_WIDGETS_QUEUE.remove(expID + jID)

    def add_table_entry_job(self, proc_params):

        tableEntryID = proc_params["tableEntryID"]
        parentLayout: QVBoxLayout = proc_params["parent_layout"]

        _cancelJobButton = widgets.PushButton(
            name="JobID", label="Cancel Job", value=False, enabled=False
        )
        _cancelJobButton.clicked.connect(
            lambda: self.cancel_job(_cancelJobButton)
        )
        _txtForInfoBox = "Updating {id}: Please wait...".format(
            id=tableEntryID
        )
        _scrollAreaCollapsibleBoxDisplayWidget = ScrollableLabel(
            text=_txtForInfoBox
        )
        _scrollAreaCollapsibleBoxDisplayWidget.setFixedHeight(300)

        proc_params["table_entry_infoBox"] = (
            _scrollAreaCollapsibleBoxDisplayWidget
        )
        proc_params["cancelJobButton"] = _cancelJobButton
        parentLayout.addWidget(_cancelJobButton.native)
        parentLayout.addWidget(_scrollAreaCollapsibleBoxDisplayWidget)

        return proc_params

    def add_remove_check_OTF_table_entry(
        self, OTF_dir_path, bool_msg, do_check=False
    ):
        if do_check:
            try:
                for row in range(self.proc_OTF_table_QFormLayout.rowCount()):
                    widgetItem = self.proc_OTF_table_QFormLayout.itemAt(row)
                    if widgetItem is not None:
                        name_widget: QWidget = widgetItem.widget()
                        name_string = str(name_widget.objectName())
                        if OTF_dir_path in name_string:
                            for item in name_widget.findChildren(QPushButton):
                                _poll_Stop_PushButton: QPushButton = item
                                return _poll_Stop_PushButton.isChecked()
                return False
            except Exception as exc:
                print(exc.args)
                return False
        else:
            if bool_msg:
                _poll_otf_label = ScrollableLabel(
                    text=OTF_dir_path + " " + _green_dot
                )
                _poll_Stop_PushButton = QPushButton("Stop")
                _poll_Stop_PushButton.setCheckable(
                    True
                )  # Make the button checkable
                _poll_Stop_PushButton.clicked.connect(
                    lambda: self.stop_OTF_push_button_call(
                        _poll_otf_label, OTF_dir_path + " " + _red_dot
                    )
                )

                _poll_data_widget = QWidget()
                _poll_data_widget.setObjectName(OTF_dir_path)
                _poll_data_widget_layout = QHBoxLayout()
                _poll_data_widget.setLayout(_poll_data_widget_layout)
                _poll_data_widget_layout.addWidget(_poll_otf_label)
                _poll_data_widget_layout.addWidget(_poll_Stop_PushButton)

                self.proc_OTF_table_QFormLayout.insertRow(0, _poll_data_widget)
            else:
                try:
                    for row in range(
                        self.proc_OTF_table_QFormLayout.rowCount()
                    ):
                        widgetItem = self.proc_OTF_table_QFormLayout.itemAt(
                            row
                        )
                        if widgetItem is not None:
                            name_widget: QWidget = widgetItem.widget()
                            name_string = str(name_widget.objectName())
                            if OTF_dir_path in name_string:
                                self.proc_OTF_table_QFormLayout.removeRow(row)
                except Exception as exc:
                    print(exc.args)

    def stop_OTF_push_button_call(self, label, txt):
        _poll_otf_label: QLabel = label
        _poll_otf_label.setText(txt)
        self.setDisabled(True)

    # adds processing entry to _qwidgetTabEntry_layout as row item
    # row item will be purged from table as processing finishes
    # there could be 3 tabs for this processing table status
    # Running, Finished, Errored
    def addTableEntry(self, table_entry_ID, table_entry_short_desc, proc_params):
        _scrollAreaCollapsibleBoxWidgetLayout = QVBoxLayout()
        _scrollAreaCollapsibleBoxWidgetLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        _scrollAreaCollapsibleBoxWidget = QWidget()
        _scrollAreaCollapsibleBoxWidget.setLayout(
            _scrollAreaCollapsibleBoxWidgetLayout
        )
        _scrollAreaCollapsibleBoxWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        _scrollAreaCollapsibleBox = QScrollArea()
        _scrollAreaCollapsibleBox.setWidgetResizable(True)
        _scrollAreaCollapsibleBox.setWidget(_scrollAreaCollapsibleBoxWidget)
        _scrollAreaCollapsibleBox.setMinimumHeight(300)
        _scrollAreaCollapsibleBox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        _collapsibleBoxWidgetLayout = QVBoxLayout()
        _collapsibleBoxWidgetLayout.addWidget(_scrollAreaCollapsibleBox)

        _collapsibleBoxWidget = CollapsibleBox(table_entry_ID)
        _collapsibleBoxWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _collapsibleBoxWidget.setContentLayout(_collapsibleBoxWidgetLayout)

        _expandingTabEntryWidgetLayout = QVBoxLayout()
        _expandingTabEntryWidgetLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        _expandingTabEntryWidgetLayout.addWidget(_collapsibleBoxWidget)

        _expandingTabEntryWidget = QWidget()
        _expandingTabEntryWidget.toolTip = table_entry_short_desc
        _expandingTabEntryWidget.setLayout(_expandingTabEntryWidgetLayout)
        _expandingTabEntryWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        proc_params["tableEntryID"] = table_entry_ID
        proc_params["parent_layout"] = _scrollAreaCollapsibleBoxWidgetLayout
        proc_params = self.add_table_entry_job(proc_params)

        # instead of adding, insert at 0 to keep latest entry on top
        # self.proc_table_QFormLayout.addRow(_expandingTabEntryWidget)
        self.proc_table_QFormLayout.insertRow(0, _expandingTabEntryWidget)

        proc_params["table_layout"] = self.proc_table_QFormLayout
        proc_params["table_entry"] = _expandingTabEntryWidget

        self.worker.run_in_pool(proc_params)
        # result = self.worker.getResult(proc_params["exp_id"])
        # print(result)

    # Builds the model as required
    def build_model(self, selected_modes):
        try:
            birefringence = None
            phase = None
            fluorescence = None
            chNames = ["State0"]
            exclude_modes = ["birefringence", "phase", "fluorescence"]
            if "birefringence" in selected_modes and "phase" in selected_modes:
                birefringence = settings.BirefringenceSettings()
                phase = settings.PhaseSettings()
                chNames = ["State0", "State1", "State2", "State3"]
                exclude_modes = ["fluorescence"]
            elif "birefringence" in selected_modes:
                birefringence = settings.BirefringenceSettings()
                chNames = ["State0", "State1", "State2", "State3"]
                exclude_modes = ["fluorescence", "phase"]
            elif "phase" in selected_modes:
                phase = settings.PhaseSettings()
                chNames = ["BF"]
                exclude_modes = ["birefringence", "fluorescence"]
            elif "fluorescence" in selected_modes:
                fluorescence = settings.FluorescenceSettings()
                chNames = ["FL"]
                exclude_modes = ["birefringence", "phase"]

            model = None
            try:
                model = settings.ReconstructionSettings(
                    input_channel_names=chNames,
                    birefringence=birefringence,
                    phase=phase,
                    fluorescence=fluorescence,
                )
            except ValidationError as exc:
                # use v1 and v2 differ for ValidationError - newer one is not caught properly
                return None, exc.errors()

            model = self.fix_model(
                model, exclude_modes, "input_channel_names", chNames
            )
            return model, "+".join(selected_modes) + ": MSG_SUCCESS"

        except Exception as exc:
            return None, exc.args

    # ToDo: Temporary fix to over ride the 'input_channel_names' default value
    # Needs revisitation
    def fix_model(self, model, exclude_modes, attr_key, attr_val):
        try:
            for mode in exclude_modes:
                model = settings.ReconstructionSettings.copy(
                    model,
                    exclude={mode},
                    deep=True,
                    update={attr_key: attr_val},
                )
            settings.ReconstructionSettings.__setattr__(
                model, attr_key, attr_val
            )
            if hasattr(model, attr_key):
                model.__fields__[attr_key].default = attr_val
                model.__fields__[attr_key].field_info.default = attr_val
        except Exception as exc:
            return print(exc.args)
        return model

    # Creates UI controls from model based on selections
    def build_acq_contols(self):

        # Make a copy of selections and unsed for deletion
        selected_modes = []
        exclude_modes = []

        for mode in self.modes_selected.keys():
            enabled = self.modes_selected[mode]["Checkbox"].value
            if not enabled:
                exclude_modes.append(mode)
            else:
                selected_modes.append(mode)

        self.create_acq_contols2(selected_modes, exclude_modes)

    def create_acq_contols2(
        self, selected_modes, exclude_modes, my_loaded_model=None, json_dict=None
    ):
        # duplicate settings from the prev model on new model creation
        if json_dict is None and len(self.pydantic_classes) > 0:
            ret = self.build_model_and_run(
                validate_return_prev_model_json_txt=True
            )
            if ret is None:
                return
            key, json_txt = ret
            self.prev_model_settings[key] = json.loads(json_txt)
        if json_dict is None:
            key = "-".join(selected_modes)
            if key in self.prev_model_settings.keys():
                json_dict = self.prev_model_settings[key]

        # initialize the top container and specify what pydantic class to map from
        if my_loaded_model is not None:
            pydantic_class = my_loaded_model
        else:
            pydantic_class, ret_msg = self.build_model(selected_modes)
            if pydantic_class is None:
                self.message_box(ret_msg)
                return

        # Final constant UI val and identifier
        _idx: Final[int] = self.index
        _str: Final[str] = str(uuid.uuid4())

        # Container holding the pydantic UI components
        # Multiple instances/copies since more than 1 might be created
        recon_pydantic_container = widgets.Container(
            name=_str, scrollable=False
        )

        self.add_pydantic_to_container(
            pydantic_class, recon_pydantic_container, exclude_modes, json_dict
        )

        # Run a validation check to see if the selected options are permitted
        # before we create the GUI
        # get the kwargs from the container/class
        pydantic_kwargs = {}
        pydantic_kwargs, ret_msg = self.get_and_validate_pydantic_args(
            recon_pydantic_container,
            pydantic_class,
            pydantic_kwargs,
            exclude_modes,
        )
        if pydantic_kwargs is None:
            self.message_box(ret_msg)
            return

        # For list element, this needs to be cleaned and parsed back as an array
        input_channel_names, ret_msg = self.clean_string_for_list(
            "input_channel_names", pydantic_kwargs["input_channel_names"]
        )
        if input_channel_names is None:
            self.message_box(ret_msg)
            return
        pydantic_kwargs["input_channel_names"] = input_channel_names

        time_indices, ret_msg = self.clean_string_int_for_list(
            "time_indices", pydantic_kwargs["time_indices"]
        )
        if time_indices is None:
            self.message_box(ret_msg)
            return
        pydantic_kwargs["time_indices"] = time_indices

        if "birefringence" in pydantic_kwargs.keys():
            background_path, ret_msg = self.clean_path_string_when_empty(
                "background_path",
                pydantic_kwargs["birefringence"]["apply_inverse"][
                    "background_path"
                ],
            )
            if background_path is None:
                self.message_box(ret_msg)
                return
            pydantic_kwargs["birefringence"]["apply_inverse"][
                "background_path"
            ] = background_path

        # validate and return errors if None
        pydantic_model, ret_msg = self.validate_pydantic_model(
            pydantic_class, pydantic_kwargs
        )
        if pydantic_model is None:
            self.message_box(ret_msg)
            return

        # generate a json from the instantiated model, update the json_display
        # most of this will end up in a table as processing proceeds
        json_txt, ret_msg = self.validate_and_return_json(pydantic_model)
        if json_txt is None:
            self.message_box(ret_msg)
            return

        # PushButton to delete a UI container
        # Use case when a wrong selection of input modes get selected eg Bire+Fl
        # Preferably this root level validation should occur before values arevalidated
        # in order to display and avoid this to occur
        _del_button = widgets.PushButton(name="Delete Model")

        c_mode = "-and-".join(selected_modes)
        c_mode_short = "".join(
            item[:3].capitalize() for item in selected_modes
        )
        if c_mode in CONTAINERS_INFO.keys():
            CONTAINERS_INFO[c_mode] += 1
        else:
            CONTAINERS_INFO[c_mode] = 1
        num_str = "{:02d}".format(CONTAINERS_INFO[c_mode])
        c_mode_str = f"{c_mode} - {num_str}"

        # Output Data location
        # These could be multiple based on user selection for each model
        # Inherits from Input by default at creation time
        name_without_ext = os.path.splitext(Path(self.input_directory).name)[0]
        save_path = os.path.join(
            Path(self.output_directory).absolute(),
            (
                name_without_ext
                + ("_" + c_mode_short + "_" + num_str)
                + ".zarr"
            ),
        )
        save_path_exists = True if Path(save_path).exists() else False
        _output_data_loc = widgets.LineEdit(
            value=Path(save_path).name,
            tooltip=(
                ""
                if not save_path_exists
                else (_validate_alert + " Output file exists")
            ),
        )
        _output_data_btn = widgets.PushButton(
            text=("" if not save_path_exists else (_validate_alert + " "))
            + "Output Data:",
            tooltip=(
                ""
                if not save_path_exists
                else (_validate_alert + " Output file exists")
            ),
        )

        # Passing location label to output location selector
        _output_data_btn.clicked.connect(
            lambda: self.browse_model_dir_path_output(_output_data_loc)
        )
        _output_data_loc.changed.connect(
            lambda: self.read_and_set_output_path_on_validation(
                _output_data_loc, _output_data_btn, save_path
            )
        )

        _show_CheckBox = widgets.CheckBox(
            name="Show after Reconstruction", value=True
        )
        _show_CheckBox.max_width = 200
        _rx_Label = widgets.Label(value="rx")
        _rx_LineEdit = widgets.LineEdit(name="rx", value=1)
        _rx_LineEdit.max_width = 50
        _validate_button = widgets.PushButton(name="Validate")

        # Passing all UI components that would be deleted
        _expandingTabEntryWidget = QWidget()
        _del_button.clicked.connect(
            lambda: self.delete_model(
                _expandingTabEntryWidget,
                recon_pydantic_container.native,
                _output_data_loc.native,
                _output_data_btn.native,
                _show_CheckBox.native,
                _validate_button.native,
                _del_button.native,
                _str,
            )
        )

        # HBox for Output Data
        _hBox_widget = QWidget()
        _hBox_layout = QHBoxLayout()
        _hBox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        _hBox_widget.setLayout(_hBox_layout)
        _hBox_layout.addWidget(_output_data_btn.native)
        _hBox_layout.addWidget(_output_data_loc.native)

        # Add this container to the main scrollable widget
        _scrollAreaCollapsibleBoxWidgetLayout = QVBoxLayout()
        _scrollAreaCollapsibleBoxWidgetLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        _scrollAreaCollapsibleBoxWidget = MyWidget()
        _scrollAreaCollapsibleBoxWidget.setLayout(
            _scrollAreaCollapsibleBoxWidgetLayout
        )

        _scrollAreaCollapsibleBox = QScrollArea()
        _scrollAreaCollapsibleBox.setWidgetResizable(True)
        _scrollAreaCollapsibleBox.setWidget(_scrollAreaCollapsibleBoxWidget)

        _collapsibleBoxWidgetLayout = QVBoxLayout()
        _collapsibleBoxWidgetLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scrollbar = _scrollAreaCollapsibleBox.horizontalScrollBar()
        _scrollAreaCollapsibleBoxWidget.resized.connect(
            lambda: self.check_scrollbar_visibility(scrollbar)
        )

        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(
            scrollbar, alignment=Qt.AlignmentFlag.AlignTop
        )  # Place at the top

        _collapsibleBoxWidgetLayout.addWidget(_scrollAreaCollapsibleBox)

        _collapsibleBoxWidget = CollapsibleBox(
            c_mode_str
        )  # tableEntryID, tableEntryShortDesc - should update with processing status

        _validate_button.clicked.connect(
            lambda: self.validate_model(_str, _collapsibleBoxWidget)
        )

        _hBox_widget2 = QWidget()
        _hBox_layout2 = QHBoxLayout()
        _hBox_layout2.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        _hBox_widget2.setLayout(_hBox_layout2)
        _hBox_layout2.addWidget(_show_CheckBox.native)
        _hBox_layout2.addWidget(_validate_button.native)
        _hBox_layout2.addWidget(_del_button.native)
        _hBox_layout2.addWidget(_rx_Label.native)
        _hBox_layout2.addWidget(_rx_LineEdit.native)

        _expandingTabEntryWidgetLayout = QVBoxLayout()
        _expandingTabEntryWidgetLayout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        _expandingTabEntryWidgetLayout.addWidget(_collapsibleBoxWidget)

        _expandingTabEntryWidget.toolTip = c_mode_str
        _expandingTabEntryWidget.setLayout(_expandingTabEntryWidgetLayout)
        _expandingTabEntryWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _expandingTabEntryWidget.layout().setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(
            recon_pydantic_container.native
        )
        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(_hBox_widget)
        _scrollAreaCollapsibleBoxWidgetLayout.addWidget(_hBox_widget2)

        _scrollAreaCollapsibleBox.setMinimumHeight(
            _scrollAreaCollapsibleBoxWidgetLayout.sizeHint().height() + 20
        )
        _collapsibleBoxWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _collapsibleBoxWidget.setContentLayout(_collapsibleBoxWidgetLayout)

        self.models_container_widget_layout.addWidget(_expandingTabEntryWidget)

        # Store a copy of the pydantic container along with all its associated components and properties
        # We dont needs a copy of the class but storing for now
        # This will be used for making deletion edits and looping to create our final run output
        # uuid - used for identiying in editable list
        self.pydantic_classes.append(
            {
                "uuid": _str,
                "c_mode_str": c_mode_str,
                "collapsibleBoxWidget": _collapsibleBoxWidget,
                "class": pydantic_class,
                "input": self.data_input_LineEdit,
                "output": os.path.join(
                    Path(self.output_directory).absolute(),
                    _output_data_loc.value,
                ),
                "output_parent_dir": str(
                    Path(self.output_directory).absolute()
                ),
                "output_LineEdit": _output_data_loc,
                "output_Button": _output_data_btn,
                "container": recon_pydantic_container,
                "selected_modes": selected_modes.copy(),
                "exclude_modes": exclude_modes.copy(),
                "poll_data": self.pollData,
                "show": _show_CheckBox,
                "rx": _rx_LineEdit,
            }
        )
        self.index += 1

        if self.index > 1:
            self.reconstruction_run_PushButton.text = "RUN {n} Models".format(
                n=self.index
            )
        else:
            self.reconstruction_run_PushButton.text = "RUN Model"

        return pydantic_model

    def check_scrollbar_visibility(self, scrollbar):
        h_scrollbar = scrollbar

        # Hide scrollbar if not needed
        h_scrollbar.setVisible(h_scrollbar.maximum() > h_scrollbar.minimum())

    def validate_model(self, _str, _collapsibleBoxWidget):
        i = 0
        model_entry_item = None
        for item in self.pydantic_classes:
            if item["uuid"] == _str:
                model_entry_item = item
                break
            i += 1
        if model_entry_item is not None:
            cls = item["class"]
            cls_container = item["container"]
            exclude_modes = item["exclude_modes"]
            c_mode_str = item["c_mode_str"]

            # build up the arguments for the pydantic model given the current container
            if cls is None:
                self.message_box("No model defined !")
                return

            pydantic_kwargs = {}
            pydantic_kwargs, ret_msg = self.get_and_validate_pydantic_args(
                cls_container, cls, pydantic_kwargs, exclude_modes
            )
            if pydantic_kwargs is None:
                self.message_box(ret_msg)
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                return

            input_channel_names, ret_msg = self.clean_string_for_list(
                "input_channel_names", pydantic_kwargs["input_channel_names"]
            )
            if input_channel_names is None:
                self.message_box(ret_msg)
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                return
            pydantic_kwargs["input_channel_names"] = input_channel_names

            time_indices, ret_msg = self.clean_string_int_for_list(
                "time_indices", pydantic_kwargs["time_indices"]
            )
            if time_indices is None:
                self.message_box(ret_msg)
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                return
            pydantic_kwargs["time_indices"] = time_indices

            time_indices, ret_msg = self.clean_string_int_for_list(
                "time_indices", pydantic_kwargs["time_indices"]
            )
            if time_indices is None:
                self.message_box(ret_msg)
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                return
            pydantic_kwargs["time_indices"] = time_indices

            if "birefringence" in pydantic_kwargs.keys():
                background_path, ret_msg = self.clean_path_string_when_empty(
                    "background_path",
                    pydantic_kwargs["birefringence"]["apply_inverse"][
                        "background_path"
                    ],
                )
                if background_path is None:
                    self.message_box(ret_msg)
                    _collapsibleBoxWidget.setNewName(
                        f"{c_mode_str} {_validate_alert}"
                    )
                    return
                pydantic_kwargs["birefringence"]["apply_inverse"][
                    "background_path"
                ] = background_path

            # validate and return errors if None
            pydantic_model, ret_msg = self.validate_pydantic_model(
                cls, pydantic_kwargs
            )
            if pydantic_model is None:
                self.message_box(ret_msg)
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                return
            if ret_msg == MSG_SUCCESS:
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_ok}"
                )
            else:
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )

    # UI components deletion - maybe just needs the parent container instead of individual components
    def delete_model(self, wid0, wid1, wid2, wid3, wid4, wid5, wid6, _str):

        if not self.confirm_dialog():
            return False

        # if wid5 is not None:
        #     wid5.setParent(None)
        # if wid4 is not None:
        #     wid4.setParent(None)
        # if wid3 is not None:
        #     wid3.setParent(None)
        # if wid2 is not None:
        #     wid2.setParent(None)
        # if wid1 is not None:
        #     wid1.setParent(None)
        if wid0 is not None:
            wid0.setParent(None)

        # Find and remove the class from our pydantic model list using uuid
        i = 0
        for item in self.pydantic_classes:
            if item["uuid"] == _str:
                self.pydantic_classes.pop(i)
                break
            i += 1
        self.index = len(self.pydantic_classes)
        if self.index > 1:
            self.reconstruction_run_PushButton.text = "RUN {n} Models".format(
                n=self.index
            )
        else:
            self.reconstruction_run_PushButton.text = "RUN Model"

    # Clear all the generated pydantic models and clears the pydantic model list
    def clear_all_models(self, silent=False):

        if silent or self.confirm_dialog():
            index = self.models_container_widget_layout.count() - 1
            while index >= 0:
                myWidget = self.models_container_widget_layout.itemAt(
                    index
                ).widget()
                if myWidget is not None:
                    myWidget.setParent(None)
                index -= 1
            self.pydantic_classes.clear()
            CONTAINERS_INFO.clear()
            self.index = 0
            self.reconstruction_run_PushButton.text = "RUN Model"
            self.prev_model_settings = {}

    # Displays the json output from the pydantic model UI selections by user
    # Loops through all our stored pydantic classes
    def build_model_and_run(self, validate_return_prev_model_json_txt=False):
        # we dont want to have a partial run if there are N models
        # so we will validate them all first and then run in a second loop
        # first pass for validating
        # second pass for creating yaml and processing

        if len(self.pydantic_classes) == 0:
            self.message_box("Please create a processing model first !")
            return

        self.model_reset_highlighter_setter()  # reset the container elements that might be highlighted for errors
        _collectAllErrors = {}
        _collectAllErrorsBool = True
        for item in self.pydantic_classes:
            cls = item["class"]
            cls_container = item["container"]
            selected_modes = item["selected_modes"]
            exclude_modes = item["exclude_modes"]
            uuid_str = item["uuid"]
            _collapsibleBoxWidget = item["collapsibleBoxWidget"]
            c_mode_str = item["c_mode_str"]

            _collectAllErrors[uuid_str] = {}
            _collectAllErrors[uuid_str]["cls"] = cls_container
            _collectAllErrors[uuid_str]["errs"] = []
            _collectAllErrors[uuid_str]["collapsibleBox"] = c_mode_str

            # build up the arguments for the pydantic model given the current container
            if cls is None:
                self.message_box(ret_msg)
                return

            # get the kwargs from the container/class
            pydantic_kwargs = {}
            pydantic_kwargs, ret_msg = self.get_and_validate_pydantic_args(
                cls_container, cls, pydantic_kwargs, exclude_modes
            )
            if pydantic_kwargs is None and not _collectAllErrorsBool:
                self.message_box(ret_msg)
                return

            # For list element, this needs to be cleaned and parsed back as an array
            input_channel_names, ret_msg = self.clean_string_for_list(
                "input_channel_names", pydantic_kwargs["input_channel_names"]
            )
            if input_channel_names is None and not _collectAllErrorsBool:
                self.message_box(ret_msg)
                return
            pydantic_kwargs["input_channel_names"] = input_channel_names

            time_indices, ret_msg = self.clean_string_int_for_list(
                "time_indices", pydantic_kwargs["time_indices"]
            )
            if time_indices is None and not _collectAllErrorsBool:
                self.message_box(ret_msg)
                return
            pydantic_kwargs["time_indices"] = time_indices

            if "birefringence" in pydantic_kwargs.keys():
                background_path, ret_msg = self.clean_path_string_when_empty(
                    "background_path",
                    pydantic_kwargs["birefringence"]["apply_inverse"][
                        "background_path"
                    ],
                )
                if background_path is None and not _collectAllErrorsBool:
                    self.message_box(ret_msg)
                    return
                pydantic_kwargs["birefringence"]["apply_inverse"][
                    "background_path"
                ] = background_path

            # validate and return errors if None
            pydantic_model, ret_msg = self.validate_pydantic_model(
                cls, pydantic_kwargs
            )
            if ret_msg == MSG_SUCCESS:
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_ok}"
                )
            else:
                _collapsibleBoxWidget.setNewName(
                    f"{c_mode_str} {_validate_alert}"
                )
                _collectAllErrors[uuid_str]["errs"] = ret_msg
            if pydantic_model is None and not _collectAllErrorsBool:
                self.message_box(ret_msg)
                return

            # generate a json from the instantiated model, update the json_display
            # most of this will end up in a table as processing proceeds
            json_txt, ret_msg = self.validate_and_return_json(pydantic_model)
            if json_txt is None and not _collectAllErrorsBool:
                self.message_box(ret_msg)
                return

        # check if we collected any validation errors before continuing
        for uu_key in _collectAllErrors.keys():
            if len(_collectAllErrors[uu_key]["errs"]) > 0:
                self.model_highlighter(_collectAllErrors)
                fmt_str = self.format_string_for_error_display(_collectAllErrors)
                self.message_box(fmt_str)
                return

        if validate_return_prev_model_json_txt:
            return "-".join(selected_modes), json_txt

        # generate a time-stamp for our yaml files to avoid overwriting
        # files generated at the same time will have an index suffix
        now = datetime.datetime.now()
        ms = now.strftime("%f")[:3]
        unique_id = now.strftime("%Y_%m_%d_%H_%M_%S_") + ms

        if self.pollData:
            data = open_ome_zarr(self.input_directory, mode="r")
            if "CurrentDimensions" in data.zattrs.keys():
                my_dict_time_indices = data.zattrs["CurrentDimensions"]["time"]
                # get the prev time_index, since this is current acq
                if my_dict_time_indices - 1 > 1:
                    time_indices = list(range(0, my_dict_time_indices))
                else:
                    time_indices = 0

            pollDataThread = threading.Thread(
                target=self.add_poll_loop,
                args=(self.input_directory, my_dict_time_indices - 1),
            )
            pollDataThread.start()

        i = 0
        for item in self.pydantic_classes:
            i += 1
            cls = item["class"]
            cls_container = item["container"]
            selected_modes = item["selected_modes"]
            exclude_modes = item["exclude_modes"]
            c_mode_str = item["c_mode_str"]
            output_LineEdit = item["output_LineEdit"]
            output_parent_dir = item["output_parent_dir"]

            full_out_path = os.path.join(
                output_parent_dir, output_LineEdit.value
            )

            # gather input/out locations
            input_dir = f"{item['input'].value}"
            output_dir = full_out_path

            # build up the arguments for the pydantic model given the current container
            if cls is None:
                self.message_box("No model defined !")
                return

            pydantic_kwargs = {}
            pydantic_kwargs, ret_msg = self.get_and_validate_pydantic_args(
                cls_container, cls, pydantic_kwargs, exclude_modes
            )
            if pydantic_kwargs is None:
                self.message_box(ret_msg)
                return

            input_channel_names, ret_msg = self.clean_string_for_list(
                "input_channel_names", pydantic_kwargs["input_channel_names"]
            )
            if input_channel_names is None:
                self.message_box(ret_msg)
                return
            pydantic_kwargs["input_channel_names"] = input_channel_names

            if not self.pollData:
                time_indices, ret_msg = self.clean_string_int_for_list(
                    "time_indices", pydantic_kwargs["time_indices"]
                )
                if time_indices is None:
                    self.message_box(ret_msg)
                    return
                pydantic_kwargs["time_indices"] = time_indices

                time_indices, ret_msg = self.clean_string_int_for_list(
                    "time_indices", pydantic_kwargs["time_indices"]
                )
                if time_indices is None:
                    self.message_box(ret_msg)
                    return
            pydantic_kwargs["time_indices"] = time_indices

            if "birefringence" in pydantic_kwargs.keys():
                background_path, ret_msg = self.clean_path_string_when_empty(
                    "background_path",
                    pydantic_kwargs["birefringence"]["apply_inverse"][
                        "background_path"
                    ],
                )
                if background_path is None:
                    self.message_box(ret_msg)
                    return
                pydantic_kwargs["birefringence"]["apply_inverse"][
                    "background_path"
                ] = background_path

            # validate and return errors if None
            pydantic_model, ret_msg = self.validate_pydantic_model(
                cls, pydantic_kwargs
            )
            if pydantic_model is None:
                self.message_box(ret_msg)
                return

            # generate a json from the instantiated model, update the json_display
            # most of this will end up in a table as processing proceeds
            json_txt, ret_msg = self.validate_and_return_json(pydantic_model)
            if json_txt is None:
                self.message_box(ret_msg)
                return

            # save the yaml files
            # path is next to saved data location
            save_config_path = str(Path(output_dir).parent.absolute())
            yml_file_name = "-and-".join(selected_modes)
            yml_file = (
                yml_file_name + "-" + unique_id + "-{:02d}".format(i) + ".yml"
            )
            config_path = os.path.join(save_config_path, yml_file)
            utils.model_to_yaml(pydantic_model, config_path)

            # Input params for table entry
            # Once ALL entries are entered we can deleted ALL model containers
            # Table will need a low priority update thread to refresh status queried from CLI
            # Table entries will be purged on completion when Result is returned OK
            # Table entries will show an error msg when processing finishes but Result not OK
            # Table fields ID / DateTime, Reconstruction type, Input Location, Output Location, Progress indicator, Stop button

            # addl_txt = "ID:" + unique_id + "-"+ str(i) + "\nInput:" + input_dir + "\nOutput:" + output_dir
            # self.json_display.value = self.json_display.value + addl_txt + "\n" + json_txt+ "\n\n"
            expID = "{tID}-{idx}".format(tID=unique_id, idx=i)
            tableID = "{tName}: ({tID}-{idx})".format(
                tName=c_mode_str, tID=unique_id, idx=i
            )
            tableDescToolTip = "{tName}: ({tID}-{idx})".format(
                tName=yml_file_name, tID=unique_id, idx=i
            )

            proc_params = {}
            proc_params["exp_id"] = expID
            proc_params["desc"] = tableDescToolTip
            proc_params["config_path"] = str(Path(config_path).absolute())
            proc_params["input_path"] = str(Path(input_dir).absolute())
            proc_params["output_path"] = str(Path(output_dir).absolute())
            proc_params["output_path_parent"] = str(
                Path(output_dir).parent.absolute()
            )
            proc_params["show"] = item["show"].value
            proc_params["rx"] = item["rx"].value

            self.addTableEntry(tableID, tableDescToolTip, proc_params)

    def add_poll_loop(self, input_data_path, last_time_index):
        _pydantic_classes = self.pydantic_classes.copy()
        required_order = ["time", "position", "z", "channel"]
        _pollData = True

        tableEntryWorker = AddOTFTableEntryWorkerThread(
            input_data_path, True, False
        )
        tableEntryWorker.add_tableOTFentry_signal.connect(
            self.add_remove_check_OTF_table_entry
        )
        tableEntryWorker.start()
        _breakFlag = False
        while True:
            time.sleep(10)
            zattrs_data = None
            try:
                _stopCalled = self.add_remove_check_OTF_table_entry(
                    input_data_path, True, do_check=True
                )
                if _stopCalled:
                    tableEntryWorker2 = AddOTFTableEntryWorkerThread(
                        input_data_path, False, False
                    )
                    tableEntryWorker2.add_tableOTFentry_signal.connect(
                        self.add_remove_check_OTF_table_entry
                    )
                    tableEntryWorker2.start()

                    # let child threads finish their work before exiting the parent thread
                    while tableEntryWorker2.isRunning():
                        time.sleep(1)
                    time.sleep(5)
                    break
                try:
                    data = open_ome_zarr(input_data_path, mode="r")
                    zattrs_data = data.zattrs
                except PermissionError:
                    pass  # On-The-Fly dataset will throw Permission Denied when being written
                    # Maybe we can read the zaatrs directly in that case
                    # If this write/read is a constant issue then the zattrs 'CurrentDimensions' key
                    # should be updated less frequently, instead of current design of updating with
                    # each image

                if zattrs_data is None:
                    zattrs_data = self.load_zattrs_directly_as_dict(
                        input_data_path
                    )

                if zattrs_data is not None:
                    if "CurrentDimensions" in zattrs_data.keys():
                        my_dict1 = zattrs_data["CurrentDimensions"]
                        sorted_dict_acq = {
                            k: my_dict1[k]
                            for k in sorted(
                                my_dict1, key=lambda x: required_order.index(x)
                            )
                        }
                        my_dict_time_indices_curr = zattrs_data[
                            "CurrentDimensions"
                        ]["time"]
                        # print(sorted_dict_acq)

                    if "FinalDimensions" in zattrs_data.keys():
                        my_dict2 = zattrs_data["FinalDimensions"]
                        sorted_dict_final = {
                            k: my_dict2[k]
                            for k in sorted(
                                my_dict2, key=lambda x: required_order.index(x)
                            )
                        }
                        # print(sorted_dict_final)

                    # use the prev time_index, since this is current acq and we need for other dims to finish acq for this t
                    # or when all dims match - signifying acq finished
                    if (
                        my_dict_time_indices_curr - 2 > last_time_index
                        or json.dumps(sorted_dict_acq)
                        == json.dumps(sorted_dict_final)
                    ):

                        now = datetime.datetime.now()
                        ms = now.strftime("%f")[:3]
                        unique_id = now.strftime("%Y_%m_%d_%H_%M_%S_") + ms

                        i = 0
                        for item in _pydantic_classes:
                            i += 1
                            cls = item["class"]
                            cls_container = item["container"]
                            selected_modes = item["selected_modes"]
                            exclude_modes = item["exclude_modes"]
                            c_mode_str = item["c_mode_str"]
                            output_LineEdit = item["output_LineEdit"]
                            output_parent_dir = item["output_parent_dir"]

                            full_out_path = os.path.join(
                                output_parent_dir, output_LineEdit.value
                            )
                            # gather input/out locations
                            input_dir = f"{item['input'].value}"
                            output_dir = full_out_path

                            pydantic_kwargs = {}
                            pydantic_kwargs, ret_msg = (
                                self.get_and_validate_pydantic_args(
                                    cls_container,
                                    cls,
                                    pydantic_kwargs,
                                    exclude_modes,
                                )
                            )

                            input_channel_names, ret_msg = (
                                self.clean_string_for_list(
                                    "input_channel_names",
                                    pydantic_kwargs["input_channel_names"],
                                )
                            )
                            pydantic_kwargs["input_channel_names"] = (
                                input_channel_names
                            )

                            if _pollData:
                                if json.dumps(sorted_dict_acq) == json.dumps(
                                    sorted_dict_final
                                ):
                                    time_indices = list(
                                        range(
                                            last_time_index,
                                            my_dict_time_indices_curr,
                                        )
                                    )
                                    _breakFlag = True
                                else:
                                    time_indices = list(
                                        range(
                                            last_time_index,
                                            my_dict_time_indices_curr - 2,
                                        )
                                    )
                                pydantic_kwargs["time_indices"] = time_indices

                            if "birefringence" in pydantic_kwargs.keys():
                                background_path, ret_msg = (
                                    self.clean_path_string_when_empty(
                                        "background_path",
                                        pydantic_kwargs["birefringence"][
                                            "apply_inverse"
                                        ]["background_path"],
                                    )
                                )

                                pydantic_kwargs["birefringence"][
                                    "apply_inverse"
                                ]["background_path"] = background_path

                            # validate and return errors if None
                            pydantic_model, ret_msg = (
                                self.validate_pydantic_model(
                                    cls, pydantic_kwargs
                                )
                            )

                            # save the yaml files
                            # path is next to saved data location
                            save_config_path = str(
                                Path(output_dir).parent.absolute()
                            )
                            yml_file_name = "-and-".join(selected_modes)
                            yml_file = (
                                yml_file_name
                                + "-"
                                + unique_id
                                + "-{:02d}".format(i)
                                + ".yml"
                            )
                            config_path = os.path.join(
                                save_config_path, yml_file
                            )
                            utils.model_to_yaml(pydantic_model, config_path)

                            expID = "{tID}-{idx}".format(tID=unique_id, idx=i)
                            tableID = "{tName}: ({tID}-{idx})".format(
                                tName=c_mode_str, tID=unique_id, idx=i
                            )
                            tableDescToolTip = "{tName}: ({tID}-{idx})".format(
                                tName=yml_file_name, tID=unique_id, idx=i
                            )

                            proc_params = {}
                            proc_params["exp_id"] = expID
                            proc_params["desc"] = tableDescToolTip
                            proc_params["config_path"] = str(
                                Path(config_path).absolute()
                            )
                            proc_params["input_path"] = str(
                                Path(input_dir).absolute()
                            )
                            proc_params["output_path"] = str(
                                Path(output_dir).absolute()
                            )
                            proc_params["output_path_parent"] = str(
                                Path(output_dir).parent.absolute()
                            )
                            proc_params["show"] = False
                            proc_params["rx"] = 1

                            tableEntryWorker1 = AddTableEntryWorkerThread(
                                tableID, tableDescToolTip, proc_params
                            )
                            tableEntryWorker1.add_tableentry_signal.connect(
                                self.addTableEntry
                            )
                            tableEntryWorker1.start()

                        if (
                            json.dumps(sorted_dict_acq)
                            == json.dumps(sorted_dict_final)
                            and _breakFlag
                        ):

                            tableEntryWorker2 = AddOTFTableEntryWorkerThread(
                                input_data_path, False, False
                            )
                            tableEntryWorker2.add_tableOTFentry_signal.connect(
                                self.add_remove_check_OTF_table_entry
                            )
                            tableEntryWorker2.start()

                            # let child threads finish their work before exiting the parent thread
                            while (
                                tableEntryWorker1.isRunning()
                                or tableEntryWorker2.isRunning()
                            ):
                                time.sleep(1)
                            time.sleep(5)
                            break

                        last_time_index = my_dict_time_indices_curr - 2
            except Exception as exc:
                print(exc.args)
                print(
                    "Exiting polling for dataset: {data_path}".format(
                        data_path=input_data_path
                    )
                )
                break

    def load_zattrs_directly_as_dict(self, zattrsFilePathDir):
        try:
            file_path = os.path.join(zattrsFilePathDir, ".zattrs")
            f = open(file_path, "r")
            txt = f.read()
            f.close()
            return json.loads(txt)
        except Exception as exc:
            print(exc.args)
        return None

    # ======= These function do not implement validation
    # They simply make the data from GUI translate to input types
    # that the model expects: for eg. GUI txt field will output only str
    # when the model needs integers

    # util function to parse list elements displayed as string
    def remove_chars(self, string, chars_to_remove):
        for char in chars_to_remove:
            string = string.replace(char, "")
        return string

    # util function to parse list elements displayed as string
    def clean_string_for_list(self, field, string):
        chars_to_remove = ["[", "]", "'", '"', " "]
        if isinstance(string, str):
            string = self.remove_chars(string, chars_to_remove)
        if len(string) == 0:
            return None, {"msg": field + " is invalid"}
        if "," in string:
            string = string.split(",")
            return string, MSG_SUCCESS
        if isinstance(string, str):
            string = [string]
            return string, MSG_SUCCESS
        return string, MSG_SUCCESS

    # util function to parse list elements displayed as string, int, int as list of strings, int range
    # [1,2,3], 4,5,6 , 5-95
    def clean_string_int_for_list(self, field, string):
        chars_to_remove = ["[", "]", "'", '"', " "]
        if Literal[string] == Literal["all"]:
            return string, MSG_SUCCESS
        if Literal[string] == Literal[""]:
            return string, MSG_SUCCESS
        if isinstance(string, str):
            string = self.remove_chars(string, chars_to_remove)
        if len(string) == 0:
            return None, {"msg": field + " is invalid"}
        if "-" in string:
            string = string.split("-")
            if len(string) == 2:
                try:
                    x = int(string[0])
                    if not isinstance(x, int):
                        raise
                except Exception as exc:
                    return None, {
                        "msg": field + " first range element is not an integer"
                    }
                try:
                    y = int(string[1])
                    if not isinstance(y, int):
                        raise
                except Exception as exc:
                    return None, {
                        "msg": field
                        + " second range element is not an integer"
                    }
                if y > x:
                    return list(range(x, y + 1)), MSG_SUCCESS
                else:
                    return None, {
                        "msg": field
                        + " second integer cannot be smaller than first"
                    }
            else:
                return None, {"msg": field + " is invalid"}
        if "," in string:
            string = string.split(",")
            return string, MSG_SUCCESS
        return string, MSG_SUCCESS

    # util function to set path to empty - by default empty path has a "."
    def clean_path_string_when_empty(self, field, string):
        if isinstance(string, Path) and string == Path(""):
            string = ""
            return string, MSG_SUCCESS
        return string, MSG_SUCCESS

    # get the pydantic_kwargs and catches any errors in doing so
    def get_and_validate_pydantic_args(
        self, cls_container, cls, pydantic_kwargs, exclude_modes
    ):
        try:
            try:
                self.get_pydantic_kwargs(
                    cls_container, cls, pydantic_kwargs, exclude_modes
                )
                return pydantic_kwargs, MSG_SUCCESS
            except ValidationError as exc:
                return None, exc.errors()
        except Exception as exc:
            return None, exc.args

    # validate the model and return errors for user actioning
    def validate_pydantic_model(self, cls, pydantic_kwargs):
        # instantiate the pydantic model form the kwargs we just pulled
        try:
            try:
                pydantic_model = settings.ReconstructionSettings.parse_obj(
                    pydantic_kwargs
                )
                return pydantic_model, MSG_SUCCESS
            except ValidationError as exc:
                return None, exc.errors()
        except Exception as exc:
            return None, exc.args

    # test to make sure model coverts to json which should ensure compatibility with yaml export
    def validate_and_return_json(self, pydantic_model):
        try:
            json_format = pydantic_model.json(indent=4)
            return json_format, MSG_SUCCESS
        except Exception as exc:
            return None, exc.args

    # gets a copy of the model from a yaml file
    # will get all fields (even those that are optional and not in yaml) and default values
    # model needs further parsing against yaml file for fields
    def get_model_from_file(self, model_file_path):
        pydantic_model = None
        try:
            try:
                pydantic_model = utils.yaml_to_model(
                    model_file_path, settings.ReconstructionSettings
                )
            except ValidationError as exc:
                return pydantic_model, exc.errors()
            if pydantic_model is None:
                raise Exception("utils.yaml_to_model - returned a None model")
            return pydantic_model, MSG_SUCCESS
        except Exception as exc:
            return None, exc.args

    # handles json with boolean properly and converts to lowercase string
    # as required
    def convert(self, obj):
        if isinstance(obj, bool):
            return str(obj).lower()
        if isinstance(obj, (list, tuple)):
            return [self.convert(item) for item in obj]
        if isinstance(obj, dict):
            return {
                self.convert(key): self.convert(value)
                for key, value in obj.items()
            }
        return obj

    # Main function to add pydantic model to container
    # https://github.com/chrishavlin/miscellaneous_python/blob/main/src/pydantic_magicgui_roundtrip.py
    # Has limitation and can cause breakages for unhandled or incorrectly handled types
    # Cannot handle Union types/typing - for now being handled explicitly
    # Ignoring NoneType since those should be Optional but maybe needs displaying ??
    # ToDo: Needs revisitation, Union check
    # Displaying Union field "time_indices" as LineEdit component
    # excludes handles fields that are not supposed to show up from __fields__
    # json_dict adds ability to provide new set of default values at time of container creation

    def add_pydantic_to_container(
        self,
        py_model: Union[BaseModel, ModelMetaclass],
        container: widgets.Container,
        excludes=[],
        json_dict=None,
    ):
        # recursively traverse a pydantic model adding widgets to a container. When a nested
        # pydantic model is encountered, add a new nested container

        for field, field_def in py_model.__fields__.items():
            if field_def is not None and field not in excludes:
                def_val = field_def.default
                ftype = field_def.type_
                toolTip = ""
                try:
                    for f_val in field_def.class_validators.keys():
                        toolTip = f"{toolTip}{f_val} "
                except Exception as e:
                    pass
                if isinstance(ftype, BaseModel) or isinstance(
                    ftype, ModelMetaclass
                ):
                    json_val = None
                    if json_dict is not None:
                        json_val = json_dict[field]
                    # the field is a pydantic class, add a container for it and fill it
                    new_widget_cls = widgets.Container
                    new_widget = new_widget_cls(name=field_def.name)
                    new_widget.tooltip = toolTip
                    self.add_pydantic_to_container(
                        ftype, new_widget, excludes, json_val
                    )
                # ToDo: Implement Union check, tried:
                # pydantic.typing.is_union(ftype)
                # isinstance(ftype, types.UnionType)
                # https://stackoverflow.com/questions/45957615/how-to-check-a-variable-against-union-type-during-runtime
                elif isinstance(ftype, type(Union[NonNegativeInt, List, str])):
                    if (
                        field == "background_path"
                    ):  # field == "background_path":
                        new_widget_cls, ops = get_widget_class(
                            def_val,
                            Annotated[Path, {"mode": "d"}],
                            dict(name=field, value=def_val),
                        )
                        new_widget = new_widget_cls(**ops)
                        toolTip = (
                            "Select the folder containing background.zarr"
                        )
                    elif field == "time_indices":  # field == "time_indices":
                        new_widget_cls, ops = get_widget_class(
                            def_val, str, dict(name=field, value=def_val)
                        )
                        new_widget = new_widget_cls(**ops)
                    else:  # other Union cases
                        new_widget_cls, ops = get_widget_class(
                            def_val, str, dict(name=field, value=def_val)
                        )
                        new_widget = new_widget_cls(**ops)
                    new_widget.tooltip = toolTip
                    if isinstance(new_widget, widgets.EmptyWidget):
                        warnings.warn(
                            message=f"magicgui could not identify a widget for {py_model}.{field}, which has type {ftype}"
                        )
                elif isinstance(def_val, float):
                    # parse the field, add appropriate widget
                    def_step_size = 0.001
                    if field_def.name == "regularization_strength":
                        def_step_size = 0.00001
                    if def_val > -1 and def_val < 1:
                        new_widget_cls, ops = get_widget_class(
                            None,
                            ftype,
                            dict(
                                name=field_def.name,
                                value=def_val,
                                step=float(def_step_size),
                            ),
                        )
                        new_widget = new_widget_cls(**ops)
                        new_widget.tooltip = toolTip
                    else:
                        new_widget_cls, ops = get_widget_class(
                            None,
                            ftype,
                            dict(name=field_def.name, value=def_val),
                        )
                        new_widget = new_widget_cls(**ops)
                        new_widget.tooltip = toolTip
                    if isinstance(new_widget, widgets.EmptyWidget):
                        warnings.warn(
                            message=f"magicgui could not identify a widget for {py_model}.{field}, which has type {ftype}"
                        )
                else:
                    # parse the field, add appropriate widget
                    new_widget_cls, ops = get_widget_class(
                        None, ftype, dict(name=field_def.name, value=def_val)
                    )
                    new_widget = new_widget_cls(**ops)
                    if isinstance(new_widget, widgets.EmptyWidget):
                        warnings.warn(
                            message=f"magicgui could not identify a widget for {py_model}.{field}, which has type {ftype}"
                        )
                    else:
                        new_widget.tooltip = toolTip
                if json_dict is not None and (
                    not isinstance(new_widget, widgets.Container)
                    or (isinstance(new_widget, widgets.FileEdit))
                ):
                    if field in json_dict.keys():
                        if isinstance(new_widget, widgets.CheckBox):
                            new_widget.value = (
                                True if json_dict[field] == "true" else False
                            )
                        elif isinstance(new_widget, widgets.FileEdit):
                            if len(json_dict[field]) > 0:
                                extension = os.path.splitext(json_dict[field])[
                                    1
                                ]
                                if len(extension) > 0:
                                    new_widget.value = Path(
                                        json_dict[field]
                                    ).parent.absolute()  # CLI accepts BG folder not .zarr
                                else:
                                    new_widget.value = Path(json_dict[field])
                        else:
                            new_widget.value = json_dict[field]
                container.append(new_widget)

    # refer - add_pydantic_to_container() for comments
    def get_pydantic_kwargs(
        self,
        container: widgets.Container,
        pydantic_model,
        pydantic_kwargs: dict,
        excludes=[],
        json_dict=None,
    ):
        # given a container that was instantiated from a pydantic model, get the arguments
        # needed to instantiate that pydantic model from the container.

        # traverse model fields, pull out values from container
        for field, field_def in pydantic_model.__fields__.items():
            if field_def is not None and field not in excludes:
                ftype = field_def.type_
                if isinstance(ftype, BaseModel) or isinstance(
                    ftype, ModelMetaclass
                ):
                    # go deeper
                    pydantic_kwargs[field] = (
                        {}
                    )  # new dictionary for the new nest level
                    # any pydantic class will be a container, so pull that out to pass
                    # to the recursive call
                    sub_container = getattr(container, field_def.name)
                    self.get_pydantic_kwargs(
                        sub_container,
                        ftype,
                        pydantic_kwargs[field],
                        excludes,
                        json_dict,
                    )
                else:
                    # not a pydantic class, just pull the field value from the container
                    if hasattr(container, field_def.name):
                        value = getattr(container, field_def.name).value
                        pydantic_kwargs[field] = value

    # copied from main_widget
    # file open/select dialog
    def open_file_dialog(self, default_path, type, filter="All Files (*)"):
        if type == "dir":
            return self.open_dialog(
                "select a directory", str(default_path), type, filter
            )
        elif type == "file":
            return self.open_dialog(
                "select a file", str(default_path), type, filter
            )
        elif type == "files":
            return self.open_dialog(
                "select file(s)", str(default_path), type, filter
            )
        elif type == "save":
            return self.open_dialog(
                "save a file", str(default_path), type, filter
            )
        else:
            return self.open_dialog(
                "select a directory", str(default_path), type, filter
            )

    def open_dialog(self, title, ref, type, filter="All Files (*)"):
        """
        opens pop-up dialogue for the user to choose a specific file or directory.

        Parameters
        ----------
        title:          (str) message to display at the top of the pop up
        ref:            (str) reference path to start the search at
        type:           (str) type of file the user is choosing (dir, file, or save)

        Returns
        -------

        """

        options = QFileDialog.DontUseNativeDialog
        if type == "dir":
            path = QFileDialog.getExistingDirectory(
                None, title, ref, options=options
            )
        elif type == "file":
            path = QFileDialog.getOpenFileName(
                None, title, ref, filter=filter, options=options
            )[0]
        elif type == "files":
            path = QFileDialog.getOpenFileNames(
                None, title, ref, filter=filter, options=options
            )[0]
        elif type == "save":
            path = QFileDialog.getSaveFileName(
                None, "Choose a save name", ref, filter=filter, options=options
            )[0]
        else:
            raise ValueError("Did not understand file dialogue type")

        return path


class MyWorker:

    def __init__(self, formLayout, tab_recon: Ui_ReconTab_Form, parentForm):
        super().__init__()
        self.formLayout: QFormLayout = formLayout
        self.tab_recon: Ui_ReconTab_Form = tab_recon
        self.ui: QWidget = parentForm
        self.max_cores = os.cpu_count()
        # In the case of CLI, we just need to submit requests in a non-blocking way
        self.threadPool = int(self.max_cores / 2)
        self.results = {}
        self.pool = None
        self.futures = []
        # https://click.palletsprojects.com/en/stable/testing/
        # self.runner = CliRunner()
        # jobs_mgmt.shared_var_jobs = self.JobsManager.shared_var_jobs
        self.JobsMgmt = jobs_mgmt.JobsManagement()
        self.useServer = True
        self.serverRunning = True
        self.server_socket = None
        self.isInitialized = False

    def initialize(self):
        if not self.isInitialized:
            thread = threading.Thread(target=self.start_server)
            thread.start()
            self.workerThreadRowDeletion = RowDeletionWorkerThread(
                self.formLayout
            )
            self.workerThreadRowDeletion.removeRowSignal.connect(
                self.tab_recon.remove_row
            )
            self.workerThreadRowDeletion.start()
            self.isInitialized = True

    def set_new_instances(self, formLayout, tab_recon, parentForm):
        self.formLayout: QFormLayout = formLayout
        self.tab_recon: Ui_ReconTab_Form = tab_recon
        self.ui: QWidget = parentForm
        self.workerThreadRowDeletion.set_new_instances(formLayout)

    def find_widget_row_in_layout(self, strID):
        layout: QFormLayout = self.formLayout
        for idx in range(0, layout.rowCount()):
            widgetItem = layout.itemAt(idx)
            name_widget = widgetItem.widget()
            toolTip_string = str(name_widget.toolTip)
            if strID in toolTip_string:
                name_widget.setParent(None)
                return idx
        return -1

    def start_server(self):
        try:
            if not self.useServer:
                return

            self.server_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            self.server_socket.bind(("localhost", jobs_mgmt.SERVER_PORT))
            self.server_socket.listen(
                50
            )  # become a server socket, maximum 50 connections

            while self.serverRunning:
                client_socket, address = self.server_socket.accept()
                if self.ui is not None and not self.ui.isVisible():
                    break
                try:
                    # dont block the server thread
                    thread = threading.Thread(
                        target=self.decode_client_data,
                        args=("", "", "", "", client_socket),
                    )
                    thread.start()
                except Exception as exc:
                    print(exc.args)
                    time.sleep(1)

            self.server_socket.close()
        except Exception as exc:
            if not self.serverRunning:
                self.serverRunning = True
                return  # ignore - will cause an exception on napari close but that is fine and does the job
            print(exc.args)

    def stop_server(self):
        try:
            if self.server_socket is not None:
                self.serverRunning = False
                self.server_socket.close()
        except Exception as exc:
            print(exc.args)

    def get_max_CPU_cores(self):
        return self.max_cores

    def set_pool_threads(self, t):
        if t > 0 and t < self.max_cores:
            self.threadPool = t

    def start_pool(self):
        if self.pool is None:
            self.pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threadPool
            )

    def shut_down_pool(self):
        self.pool.shutdown(wait=True)

    # This method handles each client response thread. It parses the information received from the client
    # and is responsible for parsing each well/pos Job if the case may be and starting individual update threads 
    # using the tableUpdateAndCleaupThread() method
    # This is also handling an unused "CoNvErTeR" functioning that can be implemented on 3rd party apps
    def decode_client_data(self,
        expIdx="",
        jobIdx="",
        wellName="",
        logs_folder_path="",
        client_socket=None,):
        
        if client_socket is not None and expIdx == "" and jobIdx == "":
            try:
                buf = client_socket.recv(10240)
                if len(buf) > 0:
                    if b"\n" in buf:
                        dataList = buf.split(b"\n")
                    else:
                        dataList = [buf]
                    for data in dataList:
                        if len(data) > 0:
                            decoded_string = data.decode()
                            if (
                                "CoNvErTeR" in decoded_string
                            ):  # this request came from an agnostic route - requires processing
                                json_str = str(decoded_string)
                                json_obj = json.loads(json_str)
                                converter_params = json_obj["CoNvErTeR"]
                                input_data = converter_params["input"]
                                output_data = converter_params["output"]
                                recon_params = converter_params["params"]
                                expID = recon_params["expID"]
                                mode = recon_params["mode"]
                                if "config_path" in recon_params.keys():
                                    config_path = recon_params["config_path"]
                                else:
                                    config_path = ""

                                proc_params = {}
                                proc_params["exp_id"] = expID
                                proc_params["desc"] = expID
                                proc_params["input_path"] = str(input_data)
                                proc_params["output_path"] = str(output_data)
                                proc_params["output_path_parent"] = str(
                                    Path(output_data).parent.absolute()
                                )
                                proc_params["show"] = False
                                proc_params["rx"] = 1

                                if config_path == "":
                                    model = None
                                    if (
                                        len(self.tab_recon.pydantic_classes)
                                        > 0
                                    ):
                                        for (
                                            item
                                        ) in self.tab_recon.pydantic_classes:
                                            if mode == item["selected_modes"]:
                                                cls = item["class"]
                                                cls_container = item[
                                                    "container"
                                                ]
                                                exclude_modes = item[
                                                    "exclude_modes"
                                                ]
                                                output_LineEdit = item[
                                                    "output_LineEdit"
                                                ]
                                                output_parent_dir = item[
                                                    "output_parent_dir"
                                                ]
                                                full_out_path = os.path.join(
                                                    output_parent_dir,
                                                    output_LineEdit.value,
                                                )

                                                # gather input/out locations
                                                output_dir = full_out_path
                                                if output_data == "":
                                                    output_data = output_dir
                                                    proc_params[
                                                        "output_path"
                                                    ] = str(output_data)

                                                # build up the arguments for the pydantic model given the current container
                                                if cls is None:
                                                    self.tab_recon.message_box(
                                                        "No model defined !"
                                                    )
                                                    return

                                                pydantic_kwargs = {}
                                                pydantic_kwargs, ret_msg = (
                                                    self.tab_recon.get_and_validate_pydantic_args(
                                                        cls_container,
                                                        cls,
                                                        pydantic_kwargs,
                                                        exclude_modes,
                                                    )
                                                )
                                                if pydantic_kwargs is None:
                                                    self.tab_recon.message_box(
                                                        ret_msg
                                                    )
                                                    return

                                                (
                                                    input_channel_names,
                                                    ret_msg,
                                                ) = self.tab_recon.clean_string_for_list(
                                                    "input_channel_names",
                                                    pydantic_kwargs[
                                                        "input_channel_names"
                                                    ],
                                                )
                                                if input_channel_names is None:
                                                    self.tab_recon.message_box(
                                                        ret_msg
                                                    )
                                                    return
                                                pydantic_kwargs[
                                                    "input_channel_names"
                                                ] = input_channel_names

                                                time_indices, ret_msg = (
                                                    self.tab_recon.clean_string_int_for_list(
                                                        "time_indices",
                                                        pydantic_kwargs[
                                                            "time_indices"
                                                        ],
                                                    )
                                                )
                                                if time_indices is None:
                                                    self.tab_recon.message_box(
                                                        ret_msg
                                                    )
                                                    return
                                                pydantic_kwargs[
                                                    "time_indices"
                                                ] = time_indices

                                                time_indices, ret_msg = (
                                                    self.tab_recon.clean_string_int_for_list(
                                                        "time_indices",
                                                        pydantic_kwargs[
                                                            "time_indices"
                                                        ],
                                                    )
                                                )
                                                if time_indices is None:
                                                    self.tab_recon.message_box(
                                                        ret_msg
                                                    )
                                                    return
                                                pydantic_kwargs[
                                                    "time_indices"
                                                ] = time_indices

                                                if (
                                                    "birefringence"
                                                    in pydantic_kwargs.keys()
                                                ):
                                                    (
                                                        background_path,
                                                        ret_msg,
                                                    ) = self.tab_recon.clean_path_string_when_empty(
                                                        "background_path",
                                                        pydantic_kwargs[
                                                            "birefringence"
                                                        ]["apply_inverse"][
                                                            "background_path"
                                                        ],
                                                    )
                                                    if background_path is None:
                                                        self.tab_recon.message_box(
                                                            ret_msg
                                                        )
                                                        return
                                                    pydantic_kwargs[
                                                        "birefringence"
                                                    ]["apply_inverse"][
                                                        "background_path"
                                                    ] = background_path

                                                # validate and return errors if None
                                                pydantic_model, ret_msg = (
                                                    self.tab_recon.validate_pydantic_model(
                                                        cls, pydantic_kwargs
                                                    )
                                                )
                                                if pydantic_model is None:
                                                    self.tab_recon.message_box(
                                                        ret_msg
                                                    )
                                                    return
                                                model = pydantic_model
                                                break
                                    if model is None:
                                        model, msg = self.tab_recon.build_model(
                                            mode
                                        )
                                    yaml_path = os.path.join(
                                        str(
                                            Path(output_data).parent.absolute()
                                        ),
                                        expID + ".yml",
                                    )
                                    utils.model_to_yaml(model, yaml_path)
                                proc_params["config_path"] = str(yaml_path)

                                tableEntryWorker = AddTableEntryWorkerThread(
                                    expID, expID, proc_params
                                )
                                tableEntryWorker.add_tableentry_signal.connect(
                                    self.tab_recon.addTableEntry
                                )
                                tableEntryWorker.start()
                                time.sleep(10)
                                return
                            else:
                                json_str = str(decoded_string)
                                json_obj = json.loads(json_str)
                                for k in json_obj:
                                    expIdx = k
                                    jobIdx = json_obj[k]["jID"]
                                    wellName = json_obj[k]["pos"]
                                    logs_folder_path = json_obj[k]["log"]
                                if (
                                    expIdx not in self.results.keys()
                                ):  # this job came from agnostic CLI route - no processing
                                    now = datetime.datetime.now()
                                    ms = now.strftime("%f")[:3]
                                    unique_id = (
                                        now.strftime("%Y_%m_%d_%H_%M_%S_") + ms
                                    )
                                    expIdx = expIdx + "-" + unique_id
                                self.JobsMgmt.put_Job_in_list(
                                    None,
                                    expIdx,
                                    str(jobIdx),
                                    wellName,
                                    mode="server",
                                )
                                # print("Submitting Job: {job} expIdx: {expIdx}".format(job=jobIdx, expIdx=expIdx))
                                thread = threading.Thread(
                                    target=self.table_update_and_cleaup_thread,
                                    args=(
                                        expIdx,
                                        jobIdx,
                                        wellName,
                                        logs_folder_path,
                                        client_socket,
                                    ),
                                )
                                thread.start()
                return
            except Exception as exc:
                print(exc.args)

    # the table update thread can be called from multiple points/threads
    # on errors - table row item is updated but there is no row deletion
    # on successful processing - the row item is expected to be deleted
    # row is being deleted from a seperate thread for which we need to connect using signal

    # This is handling essentially each job thread. Points of entry are on a failed job submission 
    # which then calls this to update based on the expID (used for .yml naming). On successful job 
    # submissions jobID, the point of entry is via the socket connection the GUI is listening and 
    # then spawns a new thread to avoid blocking of other connections.
    # If a job submission spawns more jobs then this also calls other methods via signal to create 
    # the required GUI components in the main thread.
    # Once we have expID and jobID this thread periodically loops and updates each job status and/or
    # the job error by reading the log files. Using certain keywords 
    # eg JOB_COMPLETION_STR = "Job completed successfully" we determine the progress. We also create 
    # a map for expID which might have multiple jobs to determine when a reconstruction is 
    # finished vs a single job finishing.
    # The loop ends based on user, time-out, job(s) completion and errors and handles removal of 
    # processing GUI table items (on main thread).
    # Based on the conditions the loop will end calling clientRelease()
    def table_update_and_cleaup_thread(
        self,
        expIdx="",
        jobIdx="",
        wellName="",
        logs_folder_path="",
        client_socket=None,
    ):
        jobIdx = str(jobIdx)
        
        # ToDo: Another approach to this could be to implement a status thread on the client side
        # Since the client is already running till the job is completed, the client could ping status
        # at regular intervals and also provide results and exceptions we currently read from the file
        # Currently we only send JobID/UniqueID pair from Client to Server. This would reduce multiple threads
        # server side.

        if expIdx != "" and jobIdx != "":
            # this request came from server listening so we wait for the Job to finish and update progress
            if (
                expIdx not in self.results.keys()
            ):  
                proc_params = {}
                tableID = "{exp} - {job} ({pos})".format(
                    exp=expIdx, job=jobIdx, pos=wellName
                )
                proc_params["exp_id"] = expIdx
                proc_params["desc"] = tableID
                proc_params["config_path"] = ""
                proc_params["input_path"] = ""
                proc_params["output_path"] = ""
                proc_params["output_path_parent"] = ""
                proc_params["show"] = False
                proc_params["rx"] = 1

                tableEntryWorker = AddTableEntryWorkerThread(
                    tableID, tableID, proc_params
                )
                tableEntryWorker.add_tableentry_signal.connect(
                    self.tab_recon.addTableEntry
                )
                tableEntryWorker.start()

                while expIdx not in self.results.keys():
                    time.sleep(1)

                params = self.results[expIdx]["JobUNK"].copy()
                params["status"] = STATUS_running_job
            else:
                params = self.results[expIdx]["JobUNK"].copy()

            if (
                jobIdx not in self.results[expIdx].keys()
                and len(self.results[expIdx].keys()) == 1
            ):
                # this is the first job
                params["primary"] = True
                self.results[expIdx][jobIdx] = params
            elif (
                jobIdx not in self.results[expIdx].keys()
                and len(self.results[expIdx].keys()) > 1
            ):
                # this is a new job
                # we need to create cancel and job status windows and add to parent container
                params["primary"] = False
                NEW_WIDGETS_QUEUE.append(expIdx + jobIdx)
                parentLayout: QVBoxLayout = params["parent_layout"]
                worker_thread = AddWidgetWorkerThread(
                    parentLayout, expIdx, jobIdx, params["desc"], wellName
                )
                worker_thread.add_widget_signal.connect(
                    self.tab_recon.add_widget
                )
                NEW_WIDGETS_QUEUE_THREADS.append(worker_thread)

                while len(NEW_WIDGETS_QUEUE_THREADS) > 0:
                    s_worker_thread = NEW_WIDGETS_QUEUE_THREADS.pop(0)
                    s_worker_thread.start()
                    time.sleep(1)

                # wait for new components reference
                while expIdx + jobIdx in NEW_WIDGETS_QUEUE:
                    time.sleep(1)

                _cancelJobBtn = MULTI_JOBS_REFS[expIdx + jobIdx]["cancelBtn"]
                _infoBox = MULTI_JOBS_REFS[expIdx + jobIdx]["infobox"]
                params["table_entry_infoBox"] = _infoBox
                params["cancelJobButton"] = _cancelJobBtn

                self.results[expIdx][jobIdx] = params

            _infoBox: ScrollableLabel = params["table_entry_infoBox"]
            _cancelJobBtn: PushButton = params["cancelJobButton"]

            _txtForInfoBox = "Updating {id}-{pos}: Please wait... \nJobID assigned: {jID} ".format(
                id=params["desc"], pos=wellName, jID=jobIdx
            )
            try:
                _cancelJobBtn.text = "Cancel Job {jID} ({posName})".format(
                    jID=jobIdx, posName=wellName
                )
                _cancelJobBtn.enabled = True
                _infoBox.setText(_txtForInfoBox)
            except:
                # deleted by user - no longer needs updating
                params["status"] = STATUS_user_cleared_job
                return
            _tUpdateCount = 0
            _tUpdateCountTimeout = (
                jobs_mgmt.JOBS_TIMEOUT * 60
            )  # 5 mins - match executor time-out
            _lastUpdate_jobTXT = ""
            jobTXT = ""
            # print("Updating Job: {job} expIdx: {expIdx}".format(job=jobIdx, expIdx=expIdx))
            while True:
                time.sleep(1)  # update every sec and exit on break
                try:
                    if "cancel called" in _cancelJobBtn.text:
                        json_obj = {
                            "uID": expIdx,
                            "jID": jobIdx,
                            "command": "cancel",
                        }
                        json_str = json.dumps(json_obj) + "\n"
                        client_socket.send(json_str.encode())
                        params["status"] = STATUS_user_cancelled_job
                        _infoBox.setText(
                            "User called for Cancel Job Request\n"
                            + "Please check terminal output for Job status..\n\n"
                            + jobTXT
                        )
                        self.client_release(
                            expIdx, jobIdx, client_socket, params, reason=1
                        )
                        break  # cancel called by user
                    if _infoBox == None:
                        params["status"] = STATUS_user_cleared_job
                        self.client_release(
                            expIdx, jobIdx, client_socket, params, reason=2
                        )
                        break  # deleted by user - no longer needs updating
                    if _infoBox:
                        pass
                except Exception as exc:
                    print(exc.args)
                    params["status"] = STATUS_user_cleared_job
                    self.client_release(
                        expIdx, jobIdx, client_socket, params, reason=3
                    )
                    break  # deleted by user - no longer needs updating
                if self.JobsMgmt.has_submitted_job(
                    expIdx, jobIdx, mode="server"
                ):
                    if params["status"] in [STATUS_finished_job]:
                        self.client_release(
                            expIdx, jobIdx, client_socket, params, reason=4
                        )
                        break
                    elif params["status"] in [STATUS_errored_job]:
                        jobERR = self.JobsMgmt.check_for_jobID_File(
                            jobIdx, logs_folder_path, extension="err"
                        )
                        _infoBox.setText(
                            jobIdx + "\n" + params["desc"] + "\n\n" + jobERR
                        )
                        self.client_release(
                            expIdx, jobIdx, client_socket, params, reason=5
                        )
                        break
                    else:
                        jobTXT = self.JobsMgmt.check_for_jobID_File(
                            jobIdx, logs_folder_path, extension="out"
                        )
                        try:
                            if jobTXT == "":  # job file not created yet
                                # print(jobIdx + " not started yet")
                                time.sleep(2)
                                _tUpdateCount += 2
                                if (
                                    _tUpdateCount > 10
                                ):  # if out file is empty for 10s, check the err file to update user
                                    jobERR = self.JobsMgmt.check_for_jobID_File(
                                        jobIdx,
                                        logs_folder_path,
                                        extension="err",
                                    )
                                    if JOB_OOM_EVENT in jobERR:
                                        params["status"] = STATUS_errored_job
                                        _infoBox.setText(
                                            jobERR +
                                            "\n\n"
                                            + jobTXT
                                        )
                                        self.client_release(
                                            expIdx,
                                            jobIdx,
                                            client_socket,
                                            params,
                                            reason=0,
                                        )
                                        break
                                    _infoBox.setText(
                                        jobIdx
                                        + "\n"
                                        + params["desc"]
                                        + "\n\n"
                                        + jobERR
                                    )
                                    if _tUpdateCount > _tUpdateCountTimeout:
                                        self.client_release(
                                            expIdx,
                                            jobIdx,
                                            client_socket,
                                            params,
                                            reason=0,
                                        )
                                        break
                            elif params["status"] == STATUS_finished_job:
                                rowIdx = self.find_widget_row_in_layout(expIdx)
                                # check to ensure row deletion due to shrinking table
                                # if not deleted try to delete again
                                if rowIdx < 0:
                                    self.client_release(
                                        expIdx,
                                        jobIdx,
                                        client_socket,
                                        params,
                                        reason=6,
                                    )
                                    break
                                else:
                                    break
                            elif JOB_COMPLETION_STR in jobTXT:
                                params["status"] = STATUS_finished_job
                                _infoBox.setText(jobTXT)
                                # this is the only case where row deleting occurs
                                # we cant delete the row directly from this thread
                                # we will use the exp_id to identify and delete the row
                                # using Signal
                                # break - based on status
                            elif JOB_TRIGGERED_EXC in jobTXT:
                                params["status"] = STATUS_errored_job
                                jobERR = self.JobsMgmt.check_for_jobID_File(
                                    jobIdx, logs_folder_path, extension="err"
                                )
                                _infoBox.setText(
                                    jobIdx
                                    + "\n"
                                    + params["desc"]
                                    + "\n\n"
                                    + jobTXT
                                    + "\n\n"
                                    + jobERR
                                )
                                self.client_release(
                                    expIdx,
                                    jobIdx,
                                    client_socket,
                                    params,
                                    reason=0,
                                )
                                break
                            elif JOB_RUNNING_STR in jobTXT:
                                params["status"] = STATUS_running_job
                                _infoBox.setText(jobTXT)
                                _tUpdateCount += 1
                                if _tUpdateCount > 60:
                                    jobERR = self.JobsMgmt.check_for_jobID_File(
                                        jobIdx,
                                        logs_folder_path,
                                        extension="err",
                                    )
                                    if JOB_OOM_EVENT in jobERR:
                                        params["status"] = STATUS_errored_job
                                        _infoBox.setText(
                                            jobERR +
                                            "\n\n"
                                            + jobTXT
                                        )
                                        self.client_release(
                                            expIdx,
                                            jobIdx,
                                            client_socket,
                                            params,
                                            reason=0,
                                        )
                                        break
                                    elif _lastUpdate_jobTXT != jobTXT:
                                        # if there is an update reset counter
                                        _tUpdateCount = 0
                                        _lastUpdate_jobTXT = jobTXT
                                    else:
                                        _infoBox.setText(
                                            "Please check terminal output for Job status..\n\n"
                                            + jobTXT
                                        )
                                if _tUpdateCount > _tUpdateCountTimeout:
                                    self.client_release(
                                        expIdx,
                                        jobIdx,
                                        client_socket,
                                        params,
                                        reason=0,
                                    )
                                    break
                            else:
                                jobERR = self.JobsMgmt.check_for_jobID_File(
                                    jobIdx, logs_folder_path, extension="err"
                                )
                                _infoBox.setText(
                                    jobIdx
                                    + "\n"
                                    + params["desc"]
                                    + "\n\n"
                                    + jobERR
                                )
                                self.client_release(
                                    expIdx,
                                    jobIdx,
                                    client_socket,
                                    params,
                                    reason=0,
                                )
                                break
                        except Exception as exc:
                            print(exc.args)
                else:
                    self.client_release(
                        expIdx, jobIdx, client_socket, params, reason=0
                    )
                    break
        else:
            # this would occur when an exception happens on the pool side before or during job submission
            # we dont have a job ID and will update based on exp_ID/uID
            # if job submission was not successful we can assume the client is not listening
            # and does not require a clientRelease cmd
            for uID in self.results.keys():
                params = self.results[uID]["JobUNK"]
                if params["status"] in [STATUS_errored_pool]:
                    _infoBox = params["table_entry_infoBox"]
                    poolERR = params["error"]
                    _infoBox.setText(poolERR)

    def client_release(self, expIdx, jobIdx, client_socket, params, reason=0):
        # only need to release client from primary job
        # print("clientRelease Job: {job} expIdx: {expIdx} reason:{reason}".format(job=jobIdx, expIdx=expIdx, reason=reason))
        self.JobsMgmt.put_Job_completion_in_list(True, expIdx, jobIdx)
        showData_thread = None
        if params["primary"]:
            if "show" in params:
                if params["show"]:
                    # Read reconstruction data
                    showData_thread = ShowDataWorkerThread(
                        params["output_path"]
                    )
                    showData_thread.show_data_signal.connect(
                        self.tab_recon.show_dataset
                    )
                    showData_thread.start()

            # for multi-job expID we need to check completion for all of them
            while not self.JobsMgmt.check_all_ExpJobs_completion(expIdx):
                time.sleep(1)

            json_obj = {
                "uID": expIdx,
                "jID": jobIdx,
                "command": "clientRelease",
            }
            json_str = json.dumps(json_obj) + "\n"
            client_socket.send(json_str.encode())

            if reason != 0: # remove processing entry when exiting without error
                ROW_POP_QUEUE.append(expIdx)
            # print("FINISHED")

        if self.pool is not None:
            if self.pool._work_queue.qsize() == 0:
                self.pool.shutdown()
                self.pool = None

        if showData_thread is not None:
            while showData_thread.isRunning():
                time.sleep(3)

    def run_in_pool(self, params):
        if not self.isInitialized:
            self.initialize()

        self.start_pool()
        self.results[params["exp_id"]] = {}
        self.results[params["exp_id"]]["JobUNK"] = params
        self.results[params["exp_id"]]["JobUNK"][
            "status"
        ] = STATUS_running_pool
        self.results[params["exp_id"]]["JobUNK"]["error"] = ""

        try:
            # when a request on the listening port arrives with an empty path
            # we can assume the processing was initiated outside this application
            # we do not proceed with the processing and will display the results
            if params["input_path"] != "":
                f = self.pool.submit(self.run, params)
                self.futures.append(f)
        except Exception as exc:
            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_errored_pool
            self.results[params["exp_id"]]["JobUNK"]["error"] = str(
                "\n".join(exc.args)
            )
            self.table_update_and_cleaup_thread()

    def run_multi_in_pool(self, multi_params_as_list):
        self.start_pool()
        for params in multi_params_as_list:
            self.results[params["exp_id"]] = {}
            self.results[params["exp_id"]]["JobUNK"] = params
            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_submitted_pool
            self.results[params["exp_id"]]["JobUNK"]["error"] = ""
        try:
            self.pool.map(self.run, multi_params_as_list)
        except Exception as exc:
            for params in multi_params_as_list:
                self.results[params["exp_id"]]["JobUNK"][
                    "status"
                ] = STATUS_errored_pool
                self.results[params["exp_id"]]["JobUNK"]["error"] = str(
                    "\n".join(exc.args)
                )
            self.table_update_and_cleaup_thread()

    def get_results(self):
        return self.results

    def get_result(self, exp_id):
        return self.results[exp_id]

    def run(self, params):
        # thread where work is passed to CLI which will handle the
        # multi-processing aspects as Jobs
        if params["exp_id"] not in self.results.keys():
            self.results[params["exp_id"]] = {}
            self.results[params["exp_id"]]["JobUNK"] = params
            self.results[params["exp_id"]]["JobUNK"]["error"] = ""
            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_running_pool

        try:
            # does need further threading ? probably not !
            thread = threading.Thread(
                target=self.run_in_subprocess, args=(params,)
            )
            thread.start()

        except Exception as exc:
            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_errored_pool
            self.results[params["exp_id"]]["JobUNK"]["error"] = str(
                "\n".join(exc.args)
            )
            self.table_update_and_cleaup_thread()

    def run_in_subprocess(self, params):
        """function that initiates the processing on the CLI"""
        try:
            input_path = str(params["input_path"])
            config_path = str(params["config_path"])
            output_path = str(params["output_path"])
            uid = str(params["exp_id"])
            rx = str(params["rx"])
            mainfp = str(jobs_mgmt.FILE_PATH)

            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_submitted_job

            proc = subprocess.run(
                [
                    "python",
                    mainfp,
                    "reconstruct",
                    "-i",
                    input_path,
                    "-c",
                    config_path,
                    "-o",
                    output_path,
                    "-rx",
                    str(rx),
                    "-uid",
                    uid,
                ]
            )
            self.results[params["exp_id"]]["JobUNK"]["proc"] = proc
            if proc.returncode != 0:
                raise Exception(
                    "An error occurred in processing ! Check terminal output."
                )

        except Exception as exc:
            self.results[params["exp_id"]]["JobUNK"][
                "status"
            ] = STATUS_errored_pool
            self.results[params["exp_id"]]["JobUNK"]["error"] = str(
                "\n".join(exc.args)
            )
            self.table_update_and_cleaup_thread()


class ShowDataWorkerThread(QThread):
    """Worker thread for sending signal for adding component when request comes
    from a different thread"""

    show_data_signal = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        # Emit the signal to add the widget to the main thread
        self.show_data_signal.emit(self.path)

class AddOTFTableEntryWorkerThread(QThread):
    """Worker thread for sending signal for adding component when request comes
    from a different thread"""

    add_tableOTFentry_signal = Signal(str, bool, bool)

    def __init__(self, OTF_dir_path, bool_msg, doCheck=False):
        super().__init__()
        self.OTF_dir_path = OTF_dir_path
        self.bool_msg = bool_msg
        self.doCheck = doCheck

    def run(self):
        # Emit the signal to add the widget to the main thread
        self.add_tableOTFentry_signal.emit(
            self.OTF_dir_path, self.bool_msg, self.doCheck
        )

class AddTableEntryWorkerThread(QThread):
    """Worker thread for sending signal for adding component when request comes
    from a different thread"""

    add_tableentry_signal = Signal(str, str, dict)

    def __init__(self, expID, desc, params):
        super().__init__()
        self.expID = expID
        self.desc = desc
        self.params = params

    def run(self):
        # Emit the signal to add the widget to the main thread
        self.add_tableentry_signal.emit(self.expID, self.desc, self.params)

class AddWidgetWorkerThread(QThread):
    """Worker thread for sending signal for adding component when request comes
    from a different thread"""

    add_widget_signal = Signal(QVBoxLayout, str, str, str, str)

    def __init__(self, layout, expID, jID, desc, wellName):
        super().__init__()
        self.layout = layout
        self.expID = expID
        self.jID = jID
        self.desc = desc
        self.wellName = wellName

    def run(self):
        # Emit the signal to add the widget to the main thread
        self.add_widget_signal.emit(
            self.layout, self.expID, self.jID, self.desc, self.wellName
        )

class RowDeletionWorkerThread(QThread):
    """Searches for a row based on its ID and then
    emits a signal to QFormLayout on the main thread for deletion"""

    removeRowSignal = Signal(int, str)

    def __init__(self, formLayout):
        super().__init__()
        self.formLayout = formLayout

    def set_new_instances(self, formLayout):
        self.formLayout: QFormLayout = formLayout

    # we might deal with race conditions with a shrinking table
    # find out widget and return its index
    def find_widget_row_in_layout(self, strID):
        layout: QFormLayout = self.formLayout
        for idx in range(0, layout.rowCount()):
            widgetItem = layout.itemAt(idx)
            if widgetItem is not None:
                name_widget = widgetItem.widget()
                toolTip_string = str(name_widget.toolTip)
                if strID in toolTip_string:
                    name_widget.setParent(None)
                    return idx
        return -1

    def run(self):
        while True:
            if len(ROW_POP_QUEUE) > 0:
                stringID = ROW_POP_QUEUE.pop(0)
                # Emit the signal to remove the row
                deleteRow = self.find_widget_row_in_layout(stringID)
                if deleteRow > -1:
                    self.removeRowSignal.emit(int(deleteRow), str(stringID))
                time.sleep(1)
            else:
                time.sleep(5)

class DropButton(QPushButton):
    """A drag & drop PushButton to load model file(s)"""

    def __init__(self, text, parent=None, recon_tab: Ui_ReconTab_Form = None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.recon_tab = recon_tab

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = []
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            files.append(filepath)
        self.recon_tab.open_model_files(files)

class DropWidget(QWidget):
    """A drag & drop widget container to load model file(s)"""

    def __init__(self, recon_tab: Ui_ReconTab_Form = None):
        super().__init__()
        self.setAcceptDrops(True)
        self.recon_tab = recon_tab

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = []
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            files.append(filepath)
        self.recon_tab.open_model_files(files)

class ScrollableLabel(QScrollArea):
    """A scrollable label widget used for Job entry"""

    def __init__(self, text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setText(text)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        container = QWidget()
        container.setLayout(layout)
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.setWidget(container)
        self.setWidgetResizable(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setAlignment(Qt.AlignmentFlag.AlignTop)

    def setText(self, text):
        self.label.setText(text)

class MyWidget(QWidget):
    resized = Signal()

    def __init__(self):
        super().__init__()

    def resizeEvent(self, event):
        self.resized.emit()
        super().resizeEvent(event)

class CollapsibleBox(QWidget):
    """A collapsible widget"""

    def __init__(self, title="", parent=None, hasPydanticModel=False):
        super(CollapsibleBox, self).__init__(parent)

        self.hasPydanticModel = hasPydanticModel
        self.toggle_button = QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    def setNewName(self, name):
        self.toggle_button.setText(name)

    # @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if not checked
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Direction.Forward
            if not checked
            else QtCore.QAbstractAnimation.Direction.Backward
        )
        self.toggle_animation.start()
        if checked and self.hasPydanticModel:
            # do model verification on close
            pass

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)

# VScode debugging
if __name__ == "__main__":
    import napari

    napari.Viewer()
    napari.run()

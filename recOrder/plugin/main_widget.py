from recOrder.calib.Calibration import QLIPP_Calibration, LC_DEVICE_NAME
from pycromanager import Bridge
from qtpy.QtCore import Slot, Signal, Qt
from qtpy.QtWidgets import QWidget, QFileDialog, QSizePolicy, QSlider
from qtpy.QtGui import QPixmap, QColor
from superqt import QDoubleRangeSlider, QRangeSlider
from recOrder.calib import Calibration
from recOrder.calib.calibration_workers import (
    CalibrationWorker,
    BackgroundCaptureWorker,
    load_calibration,
)
from recOrder.acq.acquisition_workers import (
    PolarizationAcquisitionWorker,
    BFAcquisitionWorker,
)
from recOrder.plugin import gui
from recOrder.io.core_functions import set_lc_state, snap_and_average
from recOrder.io.metadata_reader import MetadataReader
from recOrder.io.utils import ret_ori_overlay
from waveorder.io.reader import WaveorderReader
from pathlib import Path, PurePath
from napari import Viewer
from napari.utils.notifications import show_warning
from numpydoc.docscrape import NumpyDocString
from packaging import version
import numpy as np
import os
from os.path import dirname
import json
import logging
import textwrap


class MainWidget(QWidget):
    """
    This is the main recOrder widget that houses all of the GUI components of recOrder.
    The GUI is designed in QT Designer in /recOrder/plugin/gui.ui and converted to a python file
    with the pyuic5 command.
    """

    # Initialize Custom Signals
    log_changed = Signal(str)

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI elements
        self.ui = gui.Ui_Form()
        self.ui.setupUi(self)

        # Disable buttons until connected to MM
        self._set_buttons_enabled(False)

        # Set up overlay sliders (Commenting for 0.3.0. Consider debugging or deleting for 1.0.0.)
        # self._promote_slider_init()

        ## Connect GUI elements to functions
        # Top bar
        self.ui.qbutton_connect_to_mm.clicked[bool].connect(
            self.toggle_mm_connection
        )

        # Calibration tab
        self.ui.qbutton_browse.clicked[bool].connect(self.browse_dir_path)
        self.ui.le_directory.editingFinished.connect(self.enter_dir_path)
        self.ui.le_directory.setText(str(Path.cwd()))

        self.ui.le_swing.editingFinished.connect(self.enter_swing)
        self.ui.le_swing.setText("0.1")
        self.enter_swing()

        self.ui.le_wavelength.editingFinished.connect(self.enter_wavelength)
        self.ui.le_wavelength.setText("532")
        self.enter_wavelength()

        self.ui.cb_calib_scheme.currentIndexChanged[int].connect(
            self.enter_calib_scheme
        )
        self.ui.cb_calib_mode.currentIndexChanged[int].connect(
            self.enter_calib_mode
        )
        self.ui.cb_lca.currentIndexChanged[int].connect(self.enter_dac_lca)
        self.ui.cb_lcb.currentIndexChanged[int].connect(self.enter_dac_lcb)
        self.ui.qbutton_calibrate.clicked[bool].connect(self.run_calibration)
        self.ui.qbutton_load_calib.clicked[bool].connect(self.load_calibration)
        self.ui.qbutton_calc_extinction.clicked[bool].connect(
            self.calc_extinction
        )
        self.ui.cb_config_group.currentIndexChanged[int].connect(
            self.enter_config_group
        )

        self.ui.le_bg_folder.editingFinished.connect(self.enter_bg_folder_name)
        self.ui.le_n_avg.editingFinished.connect(self.enter_n_avg)
        self.ui.qbutton_capture_bg.clicked[bool].connect(self.capture_bg)

        # Advanced tab
        self.ui.cb_loglevel.currentIndexChanged[int].connect(
            self.enter_log_level
        )
        self.ui.qbutton_push_note.clicked[bool].connect(self.push_note)

        # Acquisition tab
        self.ui.qbutton_browse_save_dir.clicked[bool].connect(
            self.browse_save_path
        )
        self.ui.le_save_dir.editingFinished.connect(self.enter_save_path)
        self.ui.le_save_dir.setText(str(Path.cwd()))
        self.ui.le_data_save_name.editingFinished.connect(self.enter_save_name)

        self.ui.le_zstart.editingFinished.connect(self.enter_zstart)
        self.ui.le_zstart.setText("-10")
        self.enter_zstart()

        self.ui.le_zend.editingFinished.connect(self.enter_zend)
        self.ui.le_zend.setText("10")
        self.enter_zend()

        self.ui.le_zstep.editingFinished.connect(self.enter_zstep)
        self.ui.le_zstep.setText("1")
        self.enter_zstep()

        self.ui.chb_use_gpu.stateChanged[int].connect(self.enter_use_gpu)
        self.ui.le_gpu_id.editingFinished.connect(self.enter_gpu_id)

        self.ui.cb_orientation_offset.stateChanged[int].connect(
            self.enter_orientation_offset
        )

        # This parameter seems to be wired differently than others...investigate later
        self.ui.le_recon_wavelength.setText("532")

        self.ui.le_obj_na.editingFinished.connect(self.enter_obj_na)
        self.ui.le_obj_na.setText("1.3")
        self.enter_obj_na()

        self.ui.le_cond_na.editingFinished.connect(self.enter_cond_na)
        self.ui.le_cond_na.setText("0.5")
        self.enter_cond_na()

        self.ui.le_mag.editingFinished.connect(self.enter_mag)
        self.ui.le_mag.setText("60")
        self.enter_mag()

        self.ui.le_ps.editingFinished.connect(self.enter_ps)
        self.ui.le_ps.setText("6.9")
        self.enter_ps()

        self.ui.le_n_media.editingFinished.connect(self.enter_n_media)
        self.ui.le_n_media.setText("1.3")
        self.enter_n_media()

        self.ui.le_pad_z.editingFinished.connect(self.enter_pad_z)
        self.ui.cb_acq_mode.currentIndexChanged[int].connect(
            self.enter_acq_mode
        )

        self.ui.cb_bg_method.currentIndexChanged[int].connect(
            self.enter_bg_correction
        )

        self.ui.le_bg_path.editingFinished.connect(self.enter_acq_bg_path)
        self.ui.qbutton_browse_bg_path.clicked[bool].connect(
            self.browse_acq_bg_path
        )
        self.ui.qbutton_acq_birefringence.clicked[bool].connect(
            self.acq_birefringence
        )
        self.ui.qbutton_acq_phase.clicked[bool].connect(self.acq_phase)
        self.ui.qbutton_acq_birefringence_phase.clicked[bool].connect(
            self.acq_birefringence_phase
        )
        # Commenting for 0.3.0. Consider debugging or deleting for 1.0.0.
        # self.ui.cb_colormap.currentIndexChanged[int].connect(
        #     self.enter_colormap
        # )
        # self.ui.chb_display_volume.stateChanged[int].connect(
        #     self.enter_use_full_volume
        # )
        # self.ui.le_overlay_slice.editingFinished.connect(
        #     self.enter_display_slice
        # )
        # self.ui.slider_value.sliderMoved[tuple].connect(
        #     self.handle_val_slider_move
        # )
        # self.ui.slider_saturation.sliderMoved[tuple].connect(
        #     self.handle_sat_slider_move
        # )

        # Display Tab (Commenting to suppress warnings. Consider debugging or deleting for 1.0.0.)
        # self.viewer.layers.events.inserted.connect(
        #    self._add_layer_to_display_boxes
        # )
        # self.viewer.layers.events.removed.connect(
        #    self._remove_layer_from_display_boxes
        # )
        # self.ui.qbutton_create_overlay.clicked[bool].connect(
        #    self.create_overlay
        # )
        # self.ui.cb_saturation.currentIndexChanged[int].connect(
        #    self.update_sat_scale
        # )
        # self.ui.cb_value.currentIndexChanged[int].connect(
        #    self.update_value_scale
        # )
        # self.ui.le_sat_max.editingFinished.connect(self.enter_sat_max)
        # self.ui.le_sat_min.editingFinished.connect(self.enter_sat_min)
        # self.ui.le_val_max.editingFinished.connect(self.enter_val_max)
        # self.ui.le_val_min.editingFinished.connect(self.enter_val_min)

        # Reconstruction tab
        self.ui.cb_phase_denoiser.currentIndexChanged[int].connect(
            self.enter_phase_denoiser
        )

        ## Initialize logging
        log_box = QtLogger(self.ui.te_log)
        log_box.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logging.getLogger().addHandler(log_box)
        logging.getLogger().setLevel(logging.INFO)

        ## Initialize attributes
        self.connected_to_mm = False
        self.bridge = None
        self.mm = None
        self.mmc = None
        self.calib = None
        self.current_dir_path = str(Path.cwd())
        self.current_save_path = str(Path.cwd())
        self.current_bg_path = str(Path.cwd())
        self.directory = str(Path.cwd())
        self.calib_scheme = "4-State"
        self.calib_mode = "MM-Retardance"
        self.interp_method = "schnoor_fit"
        self.config_group = "Channel"
        self.calib_channels = [
            "State0",
            "State1",
            "State2",
            "State3",
            "State4",
        ]
        self.last_calib_meta_file = None
        self.use_cropped_roi = False
        self.bg_folder_name = "BG"
        self.n_avg = 5
        self.intensity_monitor = []
        self.save_directory = str(Path.cwd())
        self.save_name = None
        self.bg_option = "None"
        self.acq_mode = "2D"
        self.gpu_id = 0
        self.use_gpu = False
        self.orientation_offset = False
        self.pad_z = 0
        self.phase_reconstructor = None
        self.acq_bg_directory = None
        self.auto_shutter = True
        self.lca_dac = None
        self.lcb_dac = None
        self.pause_updates = False
        self.method = "QLIPP"
        self.mode = "3D"
        self.calib_path = str(Path.cwd())
        self.data_dir = str(Path.cwd())
        self.config_path = str(Path.cwd())
        self.save_config_path = str(Path.cwd())
        self.colormap = "HSV"
        self.use_full_volume = False
        self.display_slice = 0
        self.last_p = 0
        self.reconstruction_data_path = None
        self.reconstruction_data = None
        self.calib_assessment_level = None
        recorder_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
        self.worker = None

        ## Initialize calibration plot
        self.plot_item = self.ui.plot_widget.getPlotItem()
        self.plot_item.enableAutoRange()
        self.plot_item.setLabel("left", "Intensity")
        self.ui.plot_widget.setBackground((32, 34, 40))
        self.plot_sequence = "Coarse"

        ## Initialize visuals
        # Initialiaze GUI Images (plotting legends, recOrder logo)
        jch_legend_path = os.path.join(
            recorder_dir, "docs/images/JCh_legend.png"
        )
        hsv_legend_path = os.path.join(
            recorder_dir, "docs/images/HSV_legend.png"
        )
        self.jch_pixmap = QPixmap(jch_legend_path)
        self.hsv_pixmap = QPixmap(hsv_legend_path)
        self.ui.label_orientation_image.setPixmap(self.hsv_pixmap)
        logo_path = os.path.join(
            recorder_dir, "docs/images/recOrder_plugin_logo.png"
        )
        logo_pixmap = QPixmap(logo_path)
        self.ui.label_logo.setPixmap(logo_pixmap)

        # Hide UI elements for popups
        # DAC mode popups
        self.ui.label_lca.hide()
        self.ui.label_lcb.hide()
        self.ui.cb_lca.hide()
        self.ui.cb_lcb.hide()

        # Background correction popups
        self.ui.label_bg_path.setHidden(True)
        self.ui.le_bg_path.setHidden(True)
        self.ui.qbutton_browse_bg_path.setHidden(True)

        # Reconstruction parameter popups
        self.ui.le_rho.setHidden(True)
        self.ui.label_phase_rho.setHidden(True)
        self.ui.le_itr.setHidden(True)
        self.ui.label_itr.setHidden(True)

        # Hide temporarily unsupported "Overlay" functions
        self.ui.tabWidget.setTabText(
            self.ui.tabWidget.indexOf(self.ui.Display), "Orientation Legend"
        )
        self.ui.label_orientation_legend.setHidden(True)
        self.ui.DisplayOptions.setHidden(True)

        # Set initial UI Properties
        self.ui.le_mm_status.setStyleSheet(
            "border: 1px solid rgb(200,0,0); color: rgb(200,0,0);"
        )
        self.ui.te_log.setStyleSheet("background-color: rgb(32,34,40);")
        self.ui.le_sat_min.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_sat_max.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_val_min.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_val_max.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.setStyleSheet("QTabWidget::tab-bar {alignment: center;}")
        self.red_text = QColor(200, 0, 0, 255)

        # Populate background correction GUI element
        for i in range(3):
            self.ui.cb_bg_method.removeItem(0)
        bg_options = ["None", "Measured", "Estimated", "Measured + Estimated"]
        tooltips = [
            "No background correction.",
            'Correct sample images with a background image acquired at an empty field of view, loaded from "Background Path".',
            "Estimate sample background by fitting a 2D surface to the sample images. Works well when structures are spatially distributed across the field of view and a clear background is unavailable.",
            'Apply "Measured" background correction and then "Estimated" background correction. Use to remove residual background after the sample retardance is corrected with measured background.',
        ]
        for i, bg_option in enumerate(bg_options):
            wrapped_tooltip = "\n".join(textwrap.wrap(tooltips[i], width=70))
            self.ui.cb_bg_method.addItem(bg_option)
            self.ui.cb_bg_method.setItemData(
                i, wrapped_tooltip, Qt.ToolTipRole
            )

        # Populate calibration modes from docstring
        cal_docs = NumpyDocString(
            Calibration.QLIPP_Calibration.__init__.__doc__
        )
        mode_docs = " ".join(cal_docs["Parameters"][3].desc).split("* ")[1:]
        for i, mode_doc in enumerate(mode_docs):
            mode_name, mode_tooltip = mode_doc.split(": ")
            wrapped_tooltip = "\n".join(textwrap.wrap(mode_tooltip, width=70))
            self.ui.cb_calib_mode.addItem(mode_name)
            self.ui.cb_calib_mode.setItemData(
                i, wrapped_tooltip, Qt.ToolTipRole
            )

        # Populate acquisition mode tooltips
        acq_tooltips = [
            "Acquires data to estimate parameters in a 2D plane. For birefringence acquisitions, this mode will acquire 2D data. For phase acquisitions, this mode will acquire 3D data.",
            "Acquires 3D data to estimate parameters in a 3D volume."
        ]
        for i, tooltip in enumerate(acq_tooltips):
            wrapped_tooltip = "\n".join(textwrap.wrap(tooltip, width=70))
            self.ui.cb_acq_mode.setItemData(i, wrapped_tooltip, Qt.ToolTipRole)
            
        # make sure the top says recOrder and not 'Form'
        self.ui.tabWidget.parent().setObjectName("recOrder")

        ## Set GUI behaviors
        # set focus to "Plot" tab by default
        self.ui.tabWidget_2.setCurrentIndex(0)

        # disable wheel events for combo boxes
        for attr_name in dir(self.ui):
            if "cb_" in attr_name:
                attr = getattr(self.ui, attr_name)
                attr.wheelEvent = lambda event: None

        # Display GUI using maximum resolution
        self.showMaximized()

    def _demote_slider_offline(self, ui_slider, range_):
        """
        This function converts a promoted superqt.QRangeSlider to a QSlider element

        Parameters
        ----------
        ui_slider       (superqt.QRangeSlider) QSlider UI element to demote
        range_          (tuple) initial range to set for the slider

        Returns
        -------

        """
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        # Get positioning information from regular sliders
        slider_idx = self.ui.gridLayout_26.indexOf(ui_slider)
        slider_position = self.ui.gridLayout_26.getItemPosition(slider_idx)
        slider_parent = ui_slider.parent().objectName()
        slider_name = ui_slider.objectName()

        # Remove regular sliders from the UI
        self.ui.gridLayout_26.removeWidget(ui_slider)

        # Add back the sliders as range sliders with the same properties
        ui_slider = QSlider(getattr(self.ui, slider_parent))
        sizePolicy.setHeightForWidth(
            ui_slider.sizePolicy().hasHeightForWidth()
        )
        ui_slider.setSizePolicy(sizePolicy)
        ui_slider.setOrientation(Qt.Horizontal)
        ui_slider.setObjectName(slider_name)
        self.ui.gridLayout_26.addWidget(
            ui_slider,
            slider_position[0],
            slider_position[1],
            slider_position[2],
            slider_position[3],
        )
        ui_slider.setRange(range_[0], range_[1])

    def _promote_slider_offline(self, ui_slider, range_):
        """
        This function converts a a QSlider element to a promoted superqt.QRangeSlider

        Parameters
        ----------
        ui_slider       (QT.Slider) QSlider UI element to demote
        range_          (tuple) initial range to set for the slider

        Returns
        -------

        """

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        # Get Information from regular sliders
        slider_idx = self.ui.gridLayout_26.indexOf(ui_slider)
        slider_position = self.ui.gridLayout_26.getItemPosition(slider_idx)
        slider_parent = ui_slider.parent().objectName()
        slider_name = ui_slider.objectName()

        # Remove regular sliders from the UI
        self.ui.gridLayout_26.removeWidget(ui_slider)

        # Add back the sliders as range sliders with the same properties
        ui_slider = QRangeSlider(getattr(self.ui, slider_parent))
        sizePolicy.setHeightForWidth(
            ui_slider.sizePolicy().hasHeightForWidth()
        )
        ui_slider.setSizePolicy(sizePolicy)
        ui_slider.setOrientation(Qt.Horizontal)
        ui_slider.setObjectName(slider_name)
        self.ui.gridLayout_26.addWidget(
            ui_slider,
            slider_position[0],
            slider_position[1],
            slider_position[2],
            slider_position[3],
        )
        ui_slider.setRange(range_[0], range_[1])

    def _promote_slider_init(self):

        """
        Used to promote the Display Tab sliders from QSlider to QDoubeRangeSlider with superqt
        Returns
        -------

        """

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        # Get Information from regular sliders
        value_slider_idx = self.ui.gridLayout_17.indexOf(self.ui.slider_value)
        value_slider_position = self.ui.gridLayout_17.getItemPosition(
            value_slider_idx
        )
        value_slider_parent = self.ui.slider_value.parent().objectName()
        saturation_slider_idx = self.ui.gridLayout_17.indexOf(
            self.ui.slider_saturation
        )
        saturation_slider_position = self.ui.gridLayout_17.getItemPosition(
            saturation_slider_idx
        )
        saturation_slider_parent = (
            self.ui.slider_saturation.parent().objectName()
        )

        # Remove regular sliders from the UI
        self.ui.gridLayout_17.removeWidget(self.ui.slider_value)
        self.ui.gridLayout_17.removeWidget(self.ui.slider_saturation)

        # Add back the sliders as range sliders with the same properties
        self.ui.slider_saturation = QDoubleRangeSlider(
            getattr(self.ui, saturation_slider_parent)
        )
        sizePolicy.setHeightForWidth(
            self.ui.slider_saturation.sizePolicy().hasHeightForWidth()
        )
        self.ui.slider_saturation.setSizePolicy(sizePolicy)
        self.ui.slider_saturation.setOrientation(Qt.Horizontal)
        self.ui.slider_saturation.setObjectName("slider_saturation")
        self.ui.gridLayout_17.addWidget(
            self.ui.slider_saturation,
            saturation_slider_position[0],
            saturation_slider_position[1],
            saturation_slider_position[2],
            saturation_slider_position[3],
        )
        self.ui.slider_saturation.setRange(0, 100)

        self.ui.slider_value = QDoubleRangeSlider(
            getattr(self.ui, value_slider_parent)
        )
        sizePolicy.setHeightForWidth(
            self.ui.slider_value.sizePolicy().hasHeightForWidth()
        )
        self.ui.slider_value.setSizePolicy(sizePolicy)
        self.ui.slider_value.setOrientation(Qt.Horizontal)
        self.ui.slider_value.setObjectName("slider_value")
        self.ui.gridLayout_17.addWidget(
            self.ui.slider_value,
            value_slider_position[0],
            value_slider_position[1],
            value_slider_position[2],
            value_slider_position[3],
        )
        self.ui.slider_value.setRange(0, 100)

    def _set_buttons_enabled(self, val):
        """
        enables/disables buttons that require a connection to MM
        """
        action_buttons = [
            self.ui.qbutton_calibrate,
            self.ui.qbutton_capture_bg,
            self.ui.qbutton_calc_extinction,
            self.ui.qbutton_acq_birefringence,
            self.ui.qbutton_acq_phase,
            self.ui.qbutton_acq_birefringence_phase,
            self.ui.qbutton_load_calib,
            self.ui.qbutton_create_overlay,
        ]

        for action_button in action_buttons:
            action_button.setEnabled(val)
            if val:
                action_button.setToolTip("")
                action_button.setStyleSheet("border: 1px solid rgb(32,34,40);")
            else:
                action_button.setToolTip(
                    "Action temporarily disabled. Connect to MM or wait for acquisition to finish."
                )
                action_button.setStyleSheet("border: 1px solid rgb(65,72,81);")

    def _enable_buttons(self):
        self._set_buttons_enabled(True)

    def _disable_buttons(self):
        self._set_buttons_enabled(False)

    def _handle_error(self, exc):
        """
        Handles errors from calibration and restores micromanager to its state prior to the start of calibration
        Parameters
        ----------
        exc:        (Error) Propogated error message to display

        Returns
        -------

        """

        self.ui.tb_calib_assessment.setText(f"Error: {str(exc)}")
        self.ui.tb_calib_assessment.setStyleSheet(
            "border: 1px solid rgb(200,0,0);"
        )

        # Reset ROI if it was cropped down during reconstruction
        if self.use_cropped_roi:
            self.mmc.clearROI()

        # Reset the autoshutter setting if errored during blacklevel calculation
        self.mmc.setAutoShutter(self.auto_shutter)

        # Reset the progress bar to 0
        self.ui.progress_bar.setValue(0)

        # Raise the error
        raise exc

    def _handle_calib_abort(self):
        if self.use_cropped_roi:
            self.mmc.clearROI()
        self.mmc.setAutoShutter(self.auto_shutter)
        self.ui.progress_bar.setValue(0)

    def _handle_acq_error(self, exc):
        raise exc

    def _handle_load_finished(self):
        """
        Updates the calibration assessment when the user loads a previous calibration metadata file.

        Returns
        -------

        """
        self.ui.tb_calib_assessment.setText(
            "Previous calibration successfully loaded"
        )
        self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid green;")
        self.ui.progress_bar.setValue(100)

    def _update_calib(self, val):
        self.calib = val

    # Commenting for 0.3.0. Consider debuging or removing for 1.0.0.
    # def _add_layer_to_display_boxes(self, val):
    #     """
    #     When a new napari layer is added to recOrder, update the Display Tab combo boxes with these layers.
    #     This will allow the user to then choose which layers it wants to use for the overlay.  Will skip over
    #     any layers that are already an 'Overlay'.  This function is connected to a napari.Layer signal

    #     Parameters
    #     ----------
    #     val:            (napari.Layer) layer that was added [not used]

    #     Returns
    #     -------

    #     """

    #     for layer in self.viewer.layers:
    #         if "Overlay" in layer.name:
    #             continue
    #         if layer.name not in [
    #             self.ui.cb_hue.itemText(i)
    #             for i in range(self.ui.cb_hue.count())
    #         ]:
    #             self.ui.cb_hue.addItem(layer.name)
    #         if layer.name not in [
    #             self.ui.cb_saturation.itemText(i)
    #             for i in range(self.ui.cb_saturation.count())
    #         ]:
    #             self.ui.cb_saturation.addItem(layer.name)
    #         if layer.name not in [
    #             self.ui.cb_value.itemText(i)
    #             for i in range(self.ui.cb_value.count())
    #         ]:
    #             self.ui.cb_value.addItem(layer.name)

    # def _remove_layer_from_display_boxes(self, val):
    #     """
    #     When a napari layer is removed from napari, remove the corresponding layer from Display Tab combo boxes.

    #     Parameters
    #     ----------
    #     val:            (napari.Layer) layer that was removed by the user

    #     Returns
    #     -------

    #     """

    #     for i in range(self.ui.cb_hue.count()):
    #         if val.value.name in self.ui.cb_hue.itemText(i):
    #             self.ui.cb_hue.removeItem(i)
    #         if val.value.name in self.ui.cb_saturation.itemText(i):
    #             self.ui.cb_saturation.removeItem(i)
    #         if val.value.name in self.ui.cb_value.itemText(i):
    #             self.ui.cb_value.removeItem(i)

    def _check_line_edit(self, name):
        """
        Convenience function used in checking whether a line edit is present or missing.  Will place a red border
        around the line edit if it is empty, otherwise it will remove the red border.

        Parameters
        ----------
        name:           (str) name of the LineEdit element as specified in QT Designer file.

        Returns
        -------

        """
        le = getattr(self.ui, f"le_{name}")
        text = le.text()

        if text == "":
            le.setStyleSheet("border: 1px solid rgb(200,0,0);")
            return False
        else:
            le.setStyleSheet("")
            return True

    def _check_requirements_for_acq(self, mode):
        """
        This function will loop through the parameters from a specific acquisition and make sure the user has
        specified the necessary parameters.  If it finds an empty or missing parameters, it will set missing fields red
        and stop the acquisition process.

        Parameters
        ----------
        mode:           (str) 'birefringence' or 'phase' which denotes the type of acquisition

        Returns
        -------

        """

        # initialize the variable to keep track of the success of the requirement check
        raise_error = False

        # define the fields required for the specific acquisition modes.  Matches LineEdit object names
        phase_required = {
            "wavelength",
            "mag",
            "cond_na",
            "obj_na",
            "n_media",
            "phase_strength",
            "ps",
            "zstep",
        }

        # Initalize all fields in their default style (not red).
        for field in phase_required:
            le = getattr(self.ui, f"le_{field}")
            le.setStyleSheet("")

        # Check generally required fields
        if mode == "birefringence" or mode == "phase":
            success = self._check_line_edit("save_dir")
            if not success:
                raise_error = True

            # check background path if 'Measured' or 'Measured + Estimated' is selected
            if self.bg_option == "local_fit+" or self.bg_option == "global":
                success = self._check_line_edit("bg_path")
                if not success:
                    raise_error = True

        # Check phase specific fields
        if mode == "phase":

            # add in extra requirement is user is acquiring PhaseFromBF
            if self.ui.chb_phase_from_bf.isChecked():
                cont = self._check_line_edit("recon_wavelength")
                tab = (
                    getattr(self.ui, f"le_recon_wavelength")
                    .parent()
                    .parent()
                    .objectName()
                )
                if not cont:
                    raise_error = True

            for field in phase_required:
                cont = self._check_line_edit(field)
                tab = (
                    getattr(self.ui, f"le_{field}")
                    .parent()
                    .parent()
                    .objectName()
                )
                if not cont:
                    raise_error = True
                else:
                    continue

        # Alert the user to check and enter in the missing parameters
        if raise_error:
            raise ValueError(
                "Please enter in all of the parameters necessary for the acquisition"
            )

    @Slot(bool)
    def toggle_mm_connection(self):
        """
        Toggles MM connection and updates the corresponding GUI elements.
        """
        if self.connected_to_mm:
            self.ui.qbutton_connect_to_mm.setText("Connect to MM")
            self.ui.le_mm_status.setText("Disconnected")
            self.ui.le_mm_status.setStyleSheet(
                "border: 1px solid rgb(200,0,0); color: rgb(200,0,0);"
            )
            self.connected_to_mm = False
            self._set_buttons_enabled(False)
            self.ui.cb_config_group.clear()

        else:
            try:
                self.connect_to_mm()
                self.ui.qbutton_connect_to_mm.setText("Disconnect from MM")
                self.ui.le_mm_status.setText("Connected")
                self.ui.le_mm_status.setStyleSheet(
                    "border: 1px solid green; color: green;"
                )
                self.connected_to_mm = True
                self._set_buttons_enabled(True)
            except:
                self.ui.le_mm_status.setText("Failed")
                self.ui.le_mm_status.setStyleSheet(
                    "border: 1px solid yellow; color: yellow;"
                )

    @Slot(bool)
    def connect_to_mm(self):
        """
        Establishes the python/java bridge to Micromanager.  Micromanager must be open with a config loaded
        in order for the connection to be successful.  On connection, it will populate all of the available config
        groups.  Config group choice is used to establish which config group the Polarization states live in.

        Returns
        -------

        """
        RECOMMENDED_MM = "20220920"
        ZMQ_TARGET_VERSION = "4.2.0"
        try:
            self.bridge = Bridge(convert_camel_case=False)
            self.mmc = self.bridge.get_core()
            self.mm = self.bridge.get_studio()
        except:
            print(
                (
                    "Could not establish pycromanager bridge.\n"
                    "Is micromanager open?\n"
                    "Is Tools > Options > Run server on port 4827 checked?\n"
                    f"Are you using nightly build {RECOMMENDED_MM}?"
                )
            )
            raise EnvironmentError

        # Warn the user if there is a MicroManager/ZMQ version mismatch
        self.bridge._main_socket.send({"command": "connect", "debug": False})
        reply_json = self.bridge._main_socket.receive(timeout=500)
        zmq_mm_version = reply_json["version"]
        if zmq_mm_version != ZMQ_TARGET_VERSION:
            upgrade_str = (
                "upgrade"
                if version.parse(zmq_mm_version)
                < version.parse(ZMQ_TARGET_VERSION)
                else "downgrade"
            )
            print(
                (
                    "WARNING: This version of Micromanager has not been tested with recOrder.\n"
                    f"Please {upgrade_str} to MicroManager nightly build {RECOMMENDED_MM}."
                )
            )

        # Find config group containing calibration channels
        # calib_channels is typically ['State0', 'State1', 'State2', ...]
        # config_list may be something line ['GFP', 'RFP', 'State0', 'State1', 'State2', ...]
        # config_list may also be of the form ['GFP', 'RFP', 'LF-State0', 'LF-State1', 'LF-State2', ...]
        # in this version of the code we correctly parse 'LF-State0', but these channels cannot be used
        # by the Calibration class.
        # A valid config group contains all channels in calib_channels
        # self.ui.cb_config_group.clear()    # This triggers the enter config we will clear when switching off
        groups = self.mmc.getAvailableConfigGroups()
        config_group_found = False
        for i in range(groups.size()):
            group = groups.get(i)
            configs = self.mmc.getAvailableConfigs(group)
            config_list = []
            for j in range(configs.size()):
                config_list.append(configs.get(j))
            if np.all(
                [
                    np.any([ch in config for config in config_list])
                    for ch in self.calib_channels
                ]
            ):
                if not config_group_found:
                    self.config_group = (
                        group  # set to first config group found
                    )
                    config_group_found = True
                self.ui.cb_config_group.addItem(group)
            # not entirely sure what this part does, but I left it in
            # I think it tried to find a channel such as 'BF'
            for ch in config_list:
                if ch not in self.calib_channels:
                    self.ui.cb_acq_channel.addItem(ch)
        if not config_group_found:
            msg = (
                f"No config group contains channels {self.calib_channels}. "
                "Please refer to the recOrder wiki on how to set up the config properly."
            )
            self.ui.cb_config_group.setStyleSheet(
                "border: 1px solid rgb(200,0,0);"
            )
            raise KeyError(msg)

        # set startup LC control mode
        _devices = self.mmc.getLoadedDevices()
        loaded_devices = [_devices.get(i) for i in range(_devices.size())]
        if LC_DEVICE_NAME in loaded_devices:
            config_desc = self.mmc.getConfigData(
                "Channel", "State0"
            ).getVerbose()
            if "String send to" in config_desc:
                self.calib_mode = "MM-Retardance"
                self.ui.cb_calib_mode.setCurrentIndex(0)
            if "Voltage (V)" in config_desc:
                self.calib_mode = "MM-Voltage"
                self.ui.cb_calib_mode.setCurrentIndex(1)
        else:
            self.calib_mode = "DAC"
            self.ui.cb_calib_mode.setCurrentIndex(2)

    @Slot(tuple)
    def handle_progress_update(self, value):
        self.ui.progress_bar.setValue(value[0])
        self.ui.label_progress.setText("Progress: " + value[1])

    @Slot(str)
    def handle_extinction_update(self, value):
        self.ui.le_extinction.setText(value)

    @Slot(object)
    def handle_plot_update(self, value):
        """
        handles the plotting of the intensity values during calibration.  Calibration class will emit a signal
        depending on which stage of the calibration process it is in and then we limit the scaling / range of the plot
        accordingly.  After the coarse search of extinction is done, the plot will shift the viewing range to only be
        that of the convex optimization.  Full plot will still exist if the user uses their mouse to zoom out.

        Parameters
        ----------
        value:          (float) new intensity value from calibration

        Returns
        -------

        """
        self.intensity_monitor.append(value)
        self.ui.plot_widget.plot(self.intensity_monitor)

        if self.plot_sequence[0] == "Coarse":
            self.plot_item.autoRange()
        else:
            self.plot_item.setRange(
                xRange=(self.plot_sequence[1], len(self.intensity_monitor)),
                yRange=(
                    0,
                    np.max(self.intensity_monitor[self.plot_sequence[1] :]),
                ),
                padding=0.1,
            )

    @Slot(str)
    def handle_calibration_assessment_update(self, value):
        self.calib_assessment_level = value

    @Slot(str)
    def handle_calibration_assessment_msg_update(self, value):
        self.ui.tb_calib_assessment.setText(value)

        if self.calib_assessment_level == "good":
            self.ui.tb_calib_assessment.setStyleSheet(
                "border: 1px solid green;"
            )
        elif self.calib_assessment_level == "okay":
            self.ui.tb_calib_assessment.setStyleSheet(
                "border: 1px solid rgb(252,190,3);"
            )
        elif self.calib_assessment_level == "bad":
            self.ui.tb_calib_assessment.setStyleSheet(
                "border: 1px solid rgb(200,0,0);"
            )
        else:
            pass

    @Slot(tuple)
    def handle_lc_states_emit(self, value: tuple[tuple, dict[str, list]]):
        """Receive and plot polarization state and calibrated LC retardance values from the calibration worker.

        Parameters
        ----------
        value : tuple[tuple, dict[str, list]]
            2-tuple consisting of a tuple of polarization state names and a dictionary of LC retardance values.
        """
        pol_states, lc_values = value

        # Calculate circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circ = self.swing * np.cos(theta) + lc_values["LCA"][0]
        y_circ = self.swing * np.sin(theta) + lc_values["LCB"][0]

        import matplotlib.pyplot as plt

        plt.close("all")
        with plt.rc_context(
            {
                "axes.spines.right": False,
                "axes.spines.top": False,
            }
        ) and plt.ion():
            plt.figure("Calibrated LC States")
            plt.scatter(lc_values["LCA"], lc_values["LCB"], c="r")
            plt.plot(x_circ, y_circ, "k--", alpha=0.25)
            plt.axis("equal")
            plt.xlabel("LCA retardance")
            plt.ylabel("LCB retardance")
            for i, pol in enumerate(pol_states):
                plt.annotate(
                    pol,
                    xy=(lc_values["LCA"][i], lc_values["LCB"][i]),
                    xycoords="data",
                    xytext=(10, 10),  # annotation offset
                    textcoords="offset points",
                )

    @Slot(object)
    def handle_bg_image_update(self, value):

        if "Background Images" in self.viewer.layers:
            self.viewer.layers["Background Images"].data = value
        else:
            self.viewer.add_image(
                value, name="Background Images", colormap="gray"
            )

    @Slot(object)
    def handle_bg_bire_image_update(self, value):

        # Separate Background Retardance and Background Orientation
        # Add new layer if none exists, otherwise update layer data
        if "Background Retardance" in self.viewer.layers:
            self.viewer.layers["Background Retardance"].data = value[0]
        else:
            self.viewer.add_image(
                value[0], name="Background Retardance", colormap="gray"
            )

        if "Background Orientation" in self.viewer.layers:
            self.viewer.layers["Background Orientation"].data = value[1]
        else:
            self.viewer.add_image(
                value[1], name="Background Orientation", colormap="gray"
            )

    @Slot(object)
    def handle_bire_image_update(self, value):

        channel_names = {
            "Orientation": 1,
            "Retardance": 0,
        }

        # Compute Overlay if birefringence acquisition is 2D
        if self.acq_mode == "2D":
            channel_names["BirefringenceOverlay"] = None
            overlay = ret_ori_overlay(
                retardance=value[0],
                orientation=value[1],
                ret_max=np.percentile(value[0], 99.99),
                cmap=self.colormap,
            )

        for key, chan in channel_names.items():
            if key == "BirefringenceOverlay":
                if key + self.acq_mode in self.viewer.layers:
                    self.viewer.layers[key + self.acq_mode].data = overlay
                else:
                    self.viewer.add_image(
                        overlay, name=key + self.acq_mode, rgb=True
                    )
            else:
                if key + self.acq_mode in self.viewer.layers:
                    self.viewer.layers[key + self.acq_mode].data = value[chan]
                else:
                    cmap = "gray" if key != "Orientation" else "hsv"
                    self.viewer.add_image(
                        value[chan],
                        name=key + self.acq_mode,
                        colormap=cmap,
                    )

    @Slot(object)
    def handle_phase_image_update(self, value):

        name = "Phase2D" if self.acq_mode == "2D" else "Phase3D"

        # Add new layer if none exists, otherwise update layer data
        if name in self.viewer.layers:
            self.viewer.layers[name].data = value
        else:
            self.viewer.add_image(value, name=name, colormap="gray")

        if "Phase" not in [
            self.ui.cb_saturation.itemText(i)
            for i in range(self.ui.cb_saturation.count())
        ]:
            self.ui.cb_saturation.addItem("Retardance")
        if "Phase" not in [
            self.ui.cb_value.itemText(i)
            for i in range(self.ui.cb_value.count())
        ]:
            self.ui.cb_value.addItem("Retardance")

    @Slot(object)
    def handle_qlipp_reconstructor_update(self, value):
        # Saves phase reconstructor to be re-used if possible
        self.phase_reconstructor = value

    @Slot(str)
    def handle_calib_file_update(self, value):
        self.last_calib_meta_file = value

    @Slot(str)
    def handle_plot_sequence_update(self, value):
        current_idx = len(self.intensity_monitor)
        self.plot_sequence = (value, current_idx)

    @Slot(tuple)
    def handle_sat_slider_move(self, value):
        self.ui.le_sat_min.setText(str(np.round(value[0], 3)))
        self.ui.le_sat_max.setText(str(np.round(value[1], 3)))

    @Slot(tuple)
    def handle_val_slider_move(self, value):
        self.ui.le_val_min.setText(str(np.round(value[0], 3)))
        self.ui.le_val_max.setText(str(np.round(value[1], 3)))

    @Slot(str)
    def handle_reconstruction_store_update(self, value):
        self.reconstruction_data_path = value

    @Slot(tuple)
    def handle_reconstruction_dim_update(self, value):
        p, t, c = value
        layer_name = self.worker.manager.config.data_save_name

        if p == 0 and t == 0 and c == 0:
            self.reconstruction_data = WaveorderReader(
                self.reconstruction_data_path, "zarr"
            )
            self.viewer.add_image(
                self.reconstruction_data.get_zarr(p),
                name=layer_name + f"_Pos_{p:03d}",
            )

            self.viewer.dims.set_axis_label(0, "T")
            self.viewer.dims.set_axis_label(1, "C")
            self.viewer.dims.set_axis_label(2, "Z")

        # Add each new position as a new layer in napari
        name = layer_name + f"_Pos_{p:03d}"
        if name not in self.viewer.layers:
            self.reconstruction_data = WaveorderReader(
                self.reconstruction_data_path, "zarr"
            )
            self.viewer.add_image(
                self.reconstruction_data.get_zarr(p), name=name
            )

        # update the napari dimension slider position if the user hasn't specified to pause updates
        if not self.pause_updates:
            self.viewer.dims.set_current_step(0, t)
            self.viewer.dims.set_current_step(1, c)

        self.last_p = p

    @Slot(bool)
    def browse_dir_path(self):
        result = self._open_file_dialog(self.current_dir_path, "dir")
        self.directory = result
        self.current_dir_path = result
        self.ui.le_directory.setText(result)
        self.ui.le_save_dir.setText(result)
        self.save_directory = result

    @Slot(bool)
    def browse_save_path(self):
        result = self._open_file_dialog(self.current_save_path, "dir")
        self.save_directory = result
        self.current_save_path = result
        self.ui.le_save_dir.setText(result)

    @Slot(bool)
    def browse_data_dir(self):
        path = self._open_file_dialog(self.data_dir, "dir")
        self.data_dir = path
        self.ui.le_data_dir.setText(self.data_dir)

    @Slot(bool)
    def browse_calib_meta(self):
        path = self._open_file_dialog(self.calib_path, "file")
        self.calib_path = path
        self.ui.le_calibration_metadata.setText(self.calib_path)

    @Slot()
    def enter_dir_path(self):
        path = self.ui.le_directory.text()
        if os.path.exists(path):
            self.directory = path
            self.save_directory = path
            self.ui.le_save_dir.setText(path)
        else:
            self.ui.le_directory.setText("Path Does Not Exist")

    @Slot()
    def enter_swing(self):
        self.swing = float(self.ui.le_swing.text())

    @Slot()
    def enter_wavelength(self):
        self.wavelength = int(self.ui.le_wavelength.text())

    @Slot()
    def enter_calib_scheme(self):
        index = self.ui.cb_calib_scheme.currentIndex()
        if index == 0:
            self.calib_scheme = "4-State"
        else:
            self.calib_scheme = "5-State"

    @Slot()
    def enter_calib_mode(self):
        index = self.ui.cb_calib_mode.currentIndex()
        if index == 0:
            self.calib_mode = "MM-Retardance"
            self.ui.label_lca.hide()
            self.ui.label_lcb.hide()
            self.ui.cb_lca.hide()
            self.ui.cb_lcb.hide()
        elif index == 1:
            self.calib_mode = "MM-Voltage"
            self.ui.label_lca.hide()
            self.ui.label_lcb.hide()
            self.ui.cb_lca.hide()
            self.ui.cb_lcb.hide()
        elif index == 2:
            self.calib_mode = "DAC"
            self.ui.cb_lca.clear()
            self.ui.cb_lcb.clear()
            self.ui.cb_lca.show()
            self.ui.cb_lcb.show()
            self.ui.label_lca.show()
            self.ui.label_lcb.show()

            cfg = self.mmc.getConfigData(self.config_group, "State0")

            # Update the DAC combo boxes with available DAC's from the config.  Necessary for the user
            # to specify which DAC output corresponds to which LC for voltage-space calibration
            memory = set()
            for i in range(cfg.size()):
                prop = cfg.getSetting(i)
                if "TS_DAC" in prop.getDeviceLabel():
                    dac = prop.getDeviceLabel()[-2:]
                    if dac not in memory:
                        self.ui.cb_lca.addItem("DAC" + dac)
                        self.ui.cb_lcb.addItem("DAC" + dac)
                        memory.add(dac)
                    else:
                        continue
            self.ui.cb_lca.setCurrentIndex(0)
            self.ui.cb_lcb.setCurrentIndex(1)

    @Slot()
    def enter_dac_lca(self):
        dac = self.ui.cb_lca.currentText()
        self.lca_dac = dac

    @Slot()
    def enter_dac_lcb(self):
        dac = self.ui.cb_lcb.currentText()
        self.lcb_dac = dac

    @Slot()
    def enter_config_group(self):
        """
        callback for changing the config group combo box.  User needs to specify a config group that has the
        hardcoded states 'State0', 'State1', ... , 'State4'.  Calibration will not work unless a proper config
        group is specific

        Returns
        -------

        """
        # if/else takes care of the clearing of config
        if self.ui.cb_config_group.count() != 0:
            self.mmc = self.bridge.get_core()
            self.mm = self.bridge.get_studio()

            # Gather config groups and their children
            self.config_group = self.ui.cb_config_group.currentText()
            config = self.mmc.getAvailableConfigs(self.config_group)

            channels = []
            for i in range(config.size()):
                channels.append(config.get(i))

            # Check to see if any states are missing
            states = ["State0", "State1", "State2", "State3", "State4"]
            missing = []
            for state in states:
                if state not in channels:
                    missing.append(state)

            # if states are missing, set the combo box red and alert the user
            if len(missing) != 0:
                msg = (
                    f"The chosen config group ({self.config_group}) is missing states: {missing}. "
                    "Please refer to the recOrder wiki on how to set up the config properly."
                )

                self.ui.cb_config_group.setStyleSheet(
                    "border: 1px solid rgb(200,0,0);"
                )
                raise KeyError(msg)
            else:
                self.ui.cb_config_group.setStyleSheet("")

    @Slot()
    def enter_bg_folder_name(self):
        self.bg_folder_name = self.ui.le_bg_folder.text()

    @Slot()
    def enter_n_avg(self):
        self.n_avg = int(self.ui.le_n_avg.text())

    @Slot()
    def enter_log_level(self):
        index = self.ui.cb_loglevel.currentIndex()
        if index == 0:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.DEBUG)

    @Slot()
    def enter_save_path(self):
        path = self.ui.le_save_dir.text()
        if os.path.exists(path):
            self.save_directory = path
            self.current_save_path = path
        else:
            self.ui.le_save_dir.setText("Path Does Not Exist")

    @Slot()
    def enter_save_name(self):
        name = self.ui.le_data_save_name.text()
        self.save_name = name

    @Slot()
    def enter_zstart(self):
        self.z_start = float(self.ui.le_zstart.text())

    @Slot()
    def enter_zend(self):
        self.z_end = float(self.ui.le_zend.text())

    @Slot()
    def enter_zstep(self):
        self.z_step = float(self.ui.le_zstep.text())

    @Slot()
    def enter_acq_mode(self):
        state = self.ui.cb_acq_mode.currentIndex()
        if state == 0:
            self.acq_mode = "2D"
        elif state == 1:
            self.acq_mode = "3D"

    @Slot()
    def enter_phase_denoiser(self):
        state = self.ui.cb_phase_denoiser.currentIndex()
        if state == 0:
            self.ui.label_itr.setHidden(True)
            self.ui.label_phase_rho.setHidden(True)
            self.ui.le_rho.setHidden(True)
            self.ui.le_itr.setHidden(True)

        elif state == 1:
            self.ui.label_itr.setHidden(False)
            self.ui.label_phase_rho.setHidden(False)
            self.ui.le_rho.setHidden(False)
            self.ui.le_itr.setHidden(False)

    @Slot()
    def enter_acq_bg_path(self):
        path = self.ui.le_bg_path.text()
        if os.path.exists(path):
            self.acq_bg_directory = path
            self.current_bg_path = path
        else:
            self.ui.le_bg_path.setText("Path Does Not Exist")

    @Slot(str)
    def handle_bg_path_update(self, value: str):
        """
        Handles the update of the most recent background folderpath from
        BackgroundWorker to display in the reconstruction texbox.

        Parameters
        ----------
        value : str
            most recent captured background folderpath
        """
        path = value
        if os.path.exists(path):
            self.acq_bg_directory = path
            self.current_bg_path = path
            self.ui.le_bg_path.setText(path)
        else:
            msg = """ 
                Background acquisition was not successful.
                Check latest background capture saving directory!
                """
            raise RuntimeError(msg)

    @Slot(bool)
    def browse_acq_bg_path(self):
        result = self._open_file_dialog(self.current_bg_path, "dir")
        self.acq_bg_directory = result
        self.current_bg_path = result
        self.ui.le_bg_path.setText(result)

    @Slot()
    def enter_bg_correction(self):
        state = self.ui.cb_bg_method.currentIndex()
        if state == 0:
            self.ui.label_bg_path.setHidden(True)
            self.ui.le_bg_path.setHidden(True)
            self.ui.qbutton_browse_bg_path.setHidden(True)
            self.bg_option = "None"
        elif state == 1:
            self.ui.label_bg_path.setHidden(False)
            self.ui.le_bg_path.setHidden(False)
            self.ui.qbutton_browse_bg_path.setHidden(False)
            self.bg_option = "global"
        elif state == 2:
            self.ui.label_bg_path.setHidden(True)
            self.ui.le_bg_path.setHidden(True)
            self.ui.qbutton_browse_bg_path.setHidden(True)
            self.bg_option = "local_fit"
        elif state == 3:
            self.ui.label_bg_path.setHidden(False)
            self.ui.le_bg_path.setHidden(False)
            self.ui.qbutton_browse_bg_path.setHidden(False)
            self.bg_option = "local_fit+"

    @Slot()
    def enter_gpu_id(self):
        self.gpu_id = int(self.ui.le_gpu_id.text())

    @Slot()
    def enter_use_gpu(self):
        state = self.ui.chb_use_gpu.checkState()
        if state == 2:
            self.use_gpu = True
        elif state == 0:
            self.use_gpu = False

    @Slot()
    def enter_orientation_offset(self):
        state = self.ui.cb_orientation_offset.checkState()
        if state == 2:
            self.orientation_offset = True
        elif state == 0:
            self.orientation_offset = False

    @Slot()
    def enter_obj_na(self):
        self.obj_na = float(self.ui.le_obj_na.text())

    @Slot()
    def enter_cond_na(self):
        self.cond_na = float(self.ui.le_cond_na.text())

    @Slot()
    def enter_mag(self):
        self.mag = float(self.ui.le_mag.text())

    @Slot()
    def enter_ps(self):
        self.ps = float(self.ui.le_ps.text())

    @Slot()
    def enter_n_media(self):
        self.n_media = float(self.ui.le_n_media.text())

    @Slot()
    def enter_pad_z(self):
        self.pad_z = int(self.ui.le_pad_z.text())

    @Slot()
    def enter_pause_updates(self):
        """
        pauses the updating of the dimension slider for offline reconstruction or live listening mode.

        Returns
        -------

        """
        state = self.ui.chb_pause_updates.checkState()
        if state == 2:
            self.pause_updates = True
        elif state == 0:
            self.pause_updates = False

    @Slot(int)
    def enter_method(self):
        """
        Handles the updating of UI elements depending on the method of offline reconstruction.

        Returns
        -------

        """

        idx = self.ui.cb_method.currentIndex()

        if idx == 0:
            self.method = "QLIPP"
            self.ui.label_bf_chan.hide()
            self.ui.le_bf_chan.hide()
            self.ui.label_chan_desc.setText(
                "Retardance, Orientation, BF, Phase3D, Phase2D, S0, S1, S2, S3"
            )

        elif idx == 1:
            self.method = "PhaseFromBF"
            self.ui.label_bf_chan.show()
            self.ui.le_bf_chan.show()
            self.ui.label_bf_chan.setText("Brightfield Channel Index")
            self.ui.le_bf_chan.setPlaceholderText("int")
            self.ui.label_chan_desc.setText("Phase3D, Phase2D")

    @Slot(int)
    def enter_mode(self):
        idx = self.ui.cb_mode.currentIndex()

        if idx == 0:
            self.mode = "3D"
            self.ui.label_focus_zidx.hide()
            self.ui.le_focus_zidx.hide()
        else:
            self.mode = "2D"
            self.ui.label_focus_zidx.show()
            self.ui.le_focus_zidx.show()

    @Slot()
    def enter_data_dir(self):
        entry = self.ui.le_data_dir.text()
        if not os.path.exists(entry):
            self.ui.le_data_dir.setStyleSheet(
                "border: 1px solid rgb(200,0,0);"
            )
            self.ui.le_data_dir.setText("Path Does Not Exist")
        else:
            self.ui.le_data_dir.setStyleSheet("")
            self.data_dir = entry

    @Slot()
    def enter_calib_meta(self):
        entry = self.ui.le_calibration_metadata.text()
        if not os.path.exists(entry):
            self.ui.le_calibration_metadata.setStyleSheet(
                "border: 1px solid rgb(200,0,0);"
            )
            self.ui.le_calibration_metadata.setText("Path Does Not Exist")
        else:
            self.ui.le_calibration_metadata.setStyleSheet("")
            self.calib_path = entry

    # Comment for 0.3.0. Consider debugging or deleting for 1.0.0.
    # @Slot()
    # def enter_colormap(self):
    #     """
    #     Handles the update of the display colormap.  Will display different png image legend
    #     depending on the colormap choice.

    #     Returns
    #     -------

    #     """

    #     prev_cmap = self.colormap
    #     state = self.ui.cb_colormap.currentIndex()
    #     if state == 0:
    #         self.ui.label_orientation_image.setPixmap(self.jch_pixmap)
    #         self.colormap = "JCh"
    #     else:
    #         self.ui.label_orientation_image.setPixmap(self.hsv_pixmap)
    #         self.colormap = "HSV"

    #     # Update the birefringence overlay to new colormap if the colormap has changed
    #     if prev_cmap != self.colormap:
    #         # TODO: Handle case where there are multiple snaps
    #         if "BirefringenceOverlay2D" in self.viewer.layers:
    #             if (
    #                 "Retardance2D" in self.viewer.layers
    #                 and "Orientation2D" in self.viewer.layers
    #             ):

    #                 overlay = ret_ori_overlay(
    #                     retardance=self.viewer.layers["Retardance2D"].data,
    #                     orientation=self.viewer.layers["Orientation2D"].data,
    #                     ret_max=np.percentile(
    #                         self.viewer.layers["Retardance2D"].data, 99.99
    #                     ),
    #                     cmap=self.colormap,
    #                 )

    #                 self.viewer.layers["BirefringenceOverlay2D"].data = overlay

    # @Slot(int)
    # def enter_use_full_volume(self):
    #     state = self.ui.chb_display_volume.checkState()

    #     if state == 2:
    #         self.ui.le_overlay_slice.clear()
    #         self.ui.le_overlay_slice.setEnabled(False)
    #         self.use_full_volume = False
    #     else:
    #         self.ui.le_overlay_slice.setEnabled(True)
    #         self.use_full_volume = True

    # @Slot()
    # def enter_display_slice(self):
    #     slice = int(self.ui.le_overlay_slice.text())
    #     self.display_slice = slice

    # @Slot()
    # def enter_sat_min(self):
    #     val = float(self.ui.le_sat_min.text())
    #     slider_val = self.ui.slider_saturation.value()
    #     self.ui.slider_saturation.setValue((val, slider_val[1]))

    # @Slot()
    # def enter_sat_max(self):
    #     val = float(self.ui.le_sat_max.text())
    #     slider_val = self.ui.slider_saturation.value()
    #     self.ui.slider_saturation.setValue((slider_val[0], val))

    # @Slot()
    # def enter_val_min(self):
    #     val = float(self.ui.le_val_min.text())
    #     slider_val = self.ui.slider_value.value()
    #     self.ui.slider_value.setValue((val, slider_val[1]))

    # @Slot()
    # def enter_val_max(self):
    #     val = float(self.ui.le_val_max.text())
    #     slider_val = self.ui.slider_value.value()
    #     self.ui.slider_value.setValue((slider_val[0], val))

    @Slot(bool)
    def push_note(self):
        """
        Pushes a note to the last calibration metadata file.

        Returns
        -------

        """

        # make sure the user has performed a calibration in this session (or loaded a previous one)
        if not self.last_calib_meta_file:
            raise ValueError(
                "No calibration has been performed yet so there is no previous metadata file"
            )
        else:
            note = self.ui.le_notes_field.text()

            # Open the existing calibration metadata file and append the notes
            with open(self.last_calib_meta_file, "r") as file:
                current_json = json.load(file)

            # Append note to the end of the old note (so we don't overwrite previous notes) or write a new
            # note in the blank notes field
            old_note = current_json["Notes"]
            if old_note is None or old_note == "" or old_note == note:
                current_json["Notes"] = note
            else:
                current_json["Notes"] = old_note + ", " + note

            # dump the contents into the metadata file
            with open(self.last_calib_meta_file, "w") as file:
                json.dump(current_json, file, indent=1)

    @Slot(bool)
    def calc_extinction(self):
        """
        Calculates the extinction when the user uses the Load Calibration functionality.  This if performed
        because the calibration file could be loaded in a different FOV which may require recalibration
        depending on the extinction quality.

        Returns
        -------

        """

        # Snap images from the extinction state and first elliptical state
        set_lc_state(self.mmc, self.config_group, "State0")
        extinction = snap_and_average(self.calib.snap_manager)
        set_lc_state(self.mmc, self.config_group, "State1")
        state1 = snap_and_average(self.calib.snap_manager)

        # Calculate extinction based off captured intensities
        extinction = self.calib.calculate_extinction(
            self.swing, self.calib.I_Black, extinction, state1
        )
        self.ui.le_extinction.setText(str(extinction))

    @Slot(bool)
    def load_calibration(self):
        """
        Uses previous JSON calibration metadata to load previous calibration
        """

        metadata_path = self._open_file_dialog(self.current_dir_path, "file")
        metadata = MetadataReader(metadata_path)

        # Update Properties
        self.wavelength = metadata.Wavelength
        self.swing = metadata.Swing

        # Initialize calibration class
        self.calib = QLIPP_Calibration(
            self.mmc,
            self.mm,
            group=self.config_group,
            lc_control_mode=self.calib_mode,
            interp_method=self.interp_method,
            wavelength=self.wavelength,
        )
        self.calib.swing = self.swing
        self.ui.le_swing.setText(str(self.swing))
        self.calib.wavelength = self.wavelength
        self.ui.le_wavelength.setText(str(self.wavelength))

        # Update Calibration Scheme Combo Box
        if metadata.Calibration_scheme == "4-State":
            self.ui.cb_calib_scheme.setCurrentIndex(0)
        else:
            self.ui.cb_calib_scheme.setCurrentIndex(1)

        self.last_calib_meta_file = metadata_path

        # Move the load calibration function to a separate thread
        self.worker = load_calibration(self.calib, metadata)

        def update_extinction(extinction):
            self.calib.extinction_ratio = float(extinction)

        # initialize worker properties for multi-threading
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)
        self.worker.yielded.connect(self.ui.le_extinction.setText)
        self.worker.yielded.connect(update_extinction)
        self.worker.returned.connect(self._update_calib)
        self.worker.errored.connect(self._handle_error)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.finished.connect(self._handle_load_finished)
        self.worker.start()

    @Slot(bool)
    def run_calibration(self):
        """
        Wrapper function to create calibration worker and move that worker to a thread.
        Calibration is then executed by the calibration worker
        """

        self._check_MM_config_setup()

        self.calib = QLIPP_Calibration(
            self.mmc,
            self.mm,
            group=self.config_group,
            lc_control_mode=self.calib_mode,
            interp_method=self.interp_method,
            wavelength=self.wavelength,
        )

        if self.calib_mode == "DAC":
            self.calib.set_dacs(self.lca_dac, self.lcb_dac)

        # Reset Styling
        self.ui.tb_calib_assessment.setText("")
        self.ui.tb_calib_assessment.setStyleSheet("")

        # Save initial autoshutter state for when we set it back later
        self.auto_shutter = self.mmc.getAutoShutter()

        logging.info("Starting Calibration")

        # Initialize displays + parameters for calibration
        self.ui.progress_bar.setValue(0)
        self.plot_item.clear()
        self.intensity_monitor = []
        self.calib.swing = self.swing
        self.calib.wavelength = self.wavelength
        self.calib.meta_file = os.path.join(
            self.directory, "calibration_metadata.txt"
        )

        # Make sure Live Mode is off
        if self.calib.snap_manager.getIsLiveModeOn():
            self.calib.snap_manager.setLiveModeOn(False)

        # Init Worker and Thread
        self.worker = CalibrationWorker(self, self.calib)

        # Connect Handlers
        self.worker.progress_update.connect(self.handle_progress_update)
        self.worker.extinction_update.connect(self.handle_extinction_update)
        self.worker.intensity_update.connect(self.handle_plot_update)
        self.worker.calib_assessment.connect(
            self.handle_calibration_assessment_update
        )
        self.worker.calib_assessment_msg.connect(
            self.handle_calibration_assessment_msg_update
        )
        self.worker.calib_file_emit.connect(self.handle_calib_file_update)
        self.worker.plot_sequence_emit.connect(
            self.handle_plot_sequence_update
        )
        self.worker.lc_states.connect(self.handle_lc_states_emit)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_error)
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)

        self.worker.start()

    @property
    def _channel_descriptions(self):
        return [
            self.mmc.getConfigData(
                self.config_group, calib_channel
            ).getVerbose()
            for calib_channel in self.calib_channels
        ]

    def _check_MM_config_setup(self):
        # Warns the user if the MM configuration is not correctly set up.
        desc = self._channel_descriptions
        if self.calib_mode == "MM-Retardance":
            if all("String send to" in s for s in desc) and not any(
                "Voltage (V)" in s for s in desc
            ):
                return
            else:
                msg = " \n".join(
                    textwrap.wrap(
                        "In 'MM-Retardance' mode each preset must include the "
                        "'String send to' property, and no 'Voltage' properties.",
                        width=40,
                    )
                )
                show_warning(msg)

        elif self.calib_mode == "MM-Voltage":
            if (
                all("Voltage (V) LC-A" in s for s in desc)
                and all("Voltage (V) LC-B" in s for s in desc)
                and not any("String send to" in s for s in desc)
            ):
                return
            else:
                msg = " \n".join(
                    textwrap.wrap(
                        "In 'MM-Voltage' mode each preset must include the 'Voltage (V) LC-A' "
                        "property, the 'Voltage (V) LC-B' property, and no 'String send to' properties.",
                        width=40,
                    )
                )
                show_warning(msg)

        elif self.calib_mode == "DAC":
            _devices = self.mmc.getLoadedDevices()
            loaded_devices = [_devices.get(i) for i in range(_devices.size())]
            if LC_DEVICE_NAME in loaded_devices:
                show_warning(
                    "In 'DAC' mode the MeadowLarkLC device adapter must not be loaded in MM."
                )

        else:
            raise ValueError(
                f"self.calib_mode = {self.calib_mode} is an unrecognized state."
            )

    @Slot(bool)
    def capture_bg(self):
        """
        Wrapper function to capture a set of background images.  Will snap images and display reconstructed
        birefringence.  Check connected handlers for napari display.

        Returns
        -------

        """

        if self.calib is None:
            no_calibration_message = """Capturing a background requires calibrated liquid crystals. \
                Please either run a calibration or load a calibration from file."""
            raise RuntimeError(no_calibration_message)

        # Init worker and thread
        self.worker = BackgroundCaptureWorker(self, self.calib)

        # Connect Handlers
        self.worker.bg_image_emitter.connect(self.handle_bg_image_update)
        self.worker.bire_image_emitter.connect(
            self.handle_bg_bire_image_update
        )

        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_error)
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)
        self.worker.aborted.connect(self._handle_calib_abort)

        # Connect to BG Correction Path
        self.worker.bg_path_update_emitter.connect(self.handle_bg_path_update)

        # Start Capture Background Thread
        self.worker.start()

    @Slot(bool)
    def acq_birefringence(self):
        """
        Wrapper function to acquire birefringence stack/image and plot in napari
        Returns
        -------

        """

        self._check_requirements_for_acq("birefringence")

        # Init Worker and thread
        self.worker = PolarizationAcquisitionWorker(
            self, self.calib, "birefringence"
        )

        # Connect Handlers
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)

        # Start Thread
        self.worker.start()

    @Slot(bool)
    def acq_phase(self):
        """
        Wrapper function to acquire phase stack and plot in napari
        """

        self._check_requirements_for_acq("phase")

        # Init worker and thread
        if self.ui.chb_phase_from_bf.isChecked():
            self.worker = BFAcquisitionWorker(self)
        else:
            self.worker = PolarizationAcquisitionWorker(
                self, self.calib, "phase"
            )

        # Connect Handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.phase_reconstructor_emitter.connect(
            self.handle_qlipp_reconstructor_update
        )
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)

        # Start thread
        self.worker.start()

    @Slot(bool)
    def acq_birefringence_phase(self):
        """
        Wrapper function to acquire both birefringence and phase stack and plot in napari
        """

        self._check_requirements_for_acq("phase")

        # Init worker and thread
        self.worker = PolarizationAcquisitionWorker(self, self.calib, "all")

        # connect handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.phase_reconstructor_emitter.connect(
            self.handle_qlipp_reconstructor_update
        )
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)
        self.ui.qbutton_stop_acq.clicked.connect(self.worker.quit)

        # Start Thread
        self.worker.start()

    @Slot(bool)
    def save_config(self):
        path = self._open_file_dialog(self.save_config_path, "save")
        self.save_config_path = path
        name = PurePath(self.save_config_path).name
        dir_ = self.save_config_path.strip(name)
        self._populate_config_from_app()

        if isinstance(self.config_reader.positions, tuple):
            pos = self.config_reader.positions
            self.config_reader.positions = (
                f"[!!python/tuple [{pos[0]},{pos[1]}]]"
            )
        if isinstance(self.config_reader.timepoints, tuple):
            t = self.config_reader.timepoints
            self.config_reader.timepoints = f"[!!python/tuple [{t[0]},{t[1]}]]"

        self.config_reader.save_yaml(dir_=dir_, name=name)

    # Commenting for 0.3.0. Consider debugging or deleting for 1.0.0.
    # @Slot(int)
    # def update_sat_scale(self):
    #     idx = self.ui.cb_saturation.currentIndex()
    #     if idx != -1:
    #         layer = self.ui.cb_saturation.itemText(idx)
    #         data = self.viewer.layers[layer].data
    #         min_, max_ = np.min(data), np.max(data)
    #         self.ui.slider_saturation.setMinimum(min_)
    #         self.ui.slider_saturation.setMaximum(max_)
    #         self.ui.slider_saturation.setSingleStep((max_ - min_) / 250)
    #         self.ui.slider_saturation.setValue((min_, max_))
    #         self.ui.le_sat_max.setText(str(np.round(max_, 3)))
    #         self.ui.le_sat_min.setText(str(np.round(min_, 3)))

    # @Slot(int)
    # def update_value_scale(self):
    #     idx = self.ui.cb_value.currentIndex()
    #     if idx != -1:
    #         layer = self.ui.cb_value.itemText(idx)
    #         data = self.viewer.layers[layer].data
    #         min_, max_ = np.min(data), np.max(data)
    #         self.ui.slider_value.setMinimum(min_)
    #         self.ui.slider_value.setMaximum(max_)
    #         self.ui.slider_value.setSingleStep((max_ - min_) / 250)
    #         self.ui.slider_value.setValue((min_, max_))
    #         self.ui.le_val_max.setText(str(np.round(max_, 3)))
    #         self.ui.le_val_min.setText(str(np.round(min_, 3)))

    # @Slot(bool)
    # def create_overlay(self):
    #     """
    #     Creates HSV or JCh overlay with the specified channels from the combo boxes.  Will compute and then
    #     display the overlay in napari.

    #     Returns
    #     -------

    #     """

    #     if (
    #         self.ui.cb_hue.count() == 0
    #         or self.ui.cb_saturation.count() == 0
    #         or self.ui.cb_value == 0
    #     ):
    #         raise ValueError(
    #             "Cannot create overlay until all 3 combo boxes are populated"
    #         )

    #     # Gather channel data
    #     H = self.viewer.layers[
    #         self.ui.cb_hue.itemText(self.ui.cb_hue.currentIndex())
    #     ].data
    #     S = self.viewer.layers[
    #         self.ui.cb_saturation.itemText(
    #             self.ui.cb_saturation.currentIndex()
    #         )
    #     ].data
    #     V = self.viewer.layers[
    #         self.ui.cb_value.itemText(self.ui.cb_value.currentIndex())
    #     ].data

    #     # TODO: this is a temp fix which handles on data with n-dimensions of 4, 3, or 2 which automatically
    #     # chooses the first timepoint
    #     if H.ndim > 2 or S.ndim > 2 or V.ndim > 2:
    #         if H.ndim == 4:
    #             # assumes this is a (T, Z, Y, X) array read from napari-ome-zarr
    #             H = (
    #                 H[0, self.display_slice]
    #                 if not self.use_full_volume
    #                 else H[0]
    #             )
    #         if S.ndim == 4:
    #             S = (
    #                 S[0, self.display_slice]
    #                 if not self.use_full_volume
    #                 else S[0]
    #             )
    #         if V.ndim == 4:
    #             V = (
    #                 V[0, self.display_slice]
    #                 if not self.use_full_volume
    #                 else V[0]
    #             )

    #         if H.ndim == 3:
    #             # assumes this is a (Z, Y, X) array collected from acquisition module
    #             H = H[self.display_slice] if not self.use_full_volume else H

    #         if S.ndim == 3:
    #             S = S[self.display_slice] if not self.use_full_volume else S

    #         if S.ndim == 3:
    #             S = S[self.display_slice] if not self.use_full_volume else S

    #     mode = "2D" if not self.use_full_volume else "3D"

    #     H_name = self.ui.cb_hue.itemText(self.ui.cb_hue.currentIndex())
    #     H_scale = (
    #         (np.min(H), np.max(H))
    #         if "Orientation" not in H_name
    #         else (0, np.pi)
    #     )
    #     S_scale = self.ui.slider_saturation.value()
    #     V_scale = self.ui.slider_value.value()

    #     hsv_image = generic_hsv_overlay(
    #         H, S, V, H_scale, S_scale, V_scale, mode=mode
    #     )

    #     # Create overlay layer name
    #     idx = 0
    #     while f"HSV_Overlay_{idx}" in self.viewer.layers:
    #         idx += 1

    #     # add overlay image to napari
    #     self.viewer.add_image(hsv_image, name=f"HSV_Overlay_{idx}", rgb=True)

    @Slot(tuple)
    def update_dims(self, dims):

        if not self.pause_updates:
            self.viewer.dims.set_current_step(0, dims[0])
            self.viewer.dims.set_current_step(1, dims[1])
            self.viewer.dims.set_current_step(3, dims[2])
        else:
            pass

    def _open_file_dialog(self, default_path, type):

        return self._open_dialog("select a directory", str(default_path), type)

    def _open_dialog(self, title, ref, type):
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

        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        if type == "dir":
            path = QFileDialog.getExistingDirectory(
                None, title, ref, options=options
            )
        elif type == "file":
            path = QFileDialog.getOpenFileName(
                None, title, ref, options=options
            )[0]
        elif type == "save":
            path = QFileDialog.getSaveFileName(
                None, "Choose a save name", ref, options=options
            )[0]
        else:
            raise ValueError("Did not understand file dialogue type")

        return path


class QtLogger(logging.Handler):
    """
    Class to changing logging handler to the napari log output display
    """

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    # emit function necessary to be considered a logging handler
    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

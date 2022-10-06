from recOrder.calib.Calibration import QLIPP_Calibration, LC_DEVICE_NAME
from pycromanager import Bridge
from qtpy.QtCore import Slot, Signal, Qt
from qtpy.QtWidgets import QWidget, QFileDialog, QSizePolicy, QSlider
from qtpy.QtGui import QPixmap, QColor
from superqt import QDoubleRangeSlider, QRangeSlider
from recOrder.calib import Calibration
from recOrder.plugin.workers.calibration_workers import CalibrationWorker, BackgroundCaptureWorker, load_calibration
from recOrder.plugin.workers.acquisition_workers import PolarizationAcquisitionWorker, ListeningWorker, BFAcquisitionWorker
from recOrder.plugin.workers.reconstruction_workers import ReconstructionWorker
from recOrder.plugin.qtdesigner import recOrder_ui
from recOrder.io.core_functions import set_lc_state, snap_and_average
from recOrder.io.metadata_reader import MetadataReader, get_last_metadata_file
from recOrder.io.utils import load_bg, ret_ori_overlay, generic_hsv_overlay
from recOrder.io.config_reader import ConfigReader, PROCESSING
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
import pathlib
import textwrap

class MainWidget(QWidget):
    """
    This is the main recOrder widget that houses all of the GUI components of recOrder.
    The GUI is designed in QT Designer in /recOrder/plugin/widget/qt_designer and converted to a python file
    with the pyuic5 command.
    """

    # Initialize Custom Signals
    mm_status_changed = Signal(bool)
    intensity_changed = Signal(float)
    log_changed = Signal(str)

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI Elements
        self.ui = recOrder_ui.Ui_Form()
        self.ui.setupUi(self)
        self._promote_slider_init()

        # Setup Connections between elements
        # Connect to MicroManager
        # I'm disconnecting and hiding this button for 0.2.0, TODO: reinstate for 1.0.0
        # In 1.0.0 we'll only have a single "Switch to Online/Offline" button in the "Calibration tab"
        
        # self.ui.qbutton_mm_connect.clicked[bool].connect(self.connect_to_mm)
        self.ui.qbutton_mm_connect.hide()
        
        # Calibration Tab

        # Remove QT creator calibration mode items
        self.ui.cb_calib_mode.removeItem(0)
        self.ui.cb_calib_mode.removeItem(0)

        # Populate calibration modes from docstring
        cal_docs = NumpyDocString(Calibration.QLIPP_Calibration.__init__.__doc__)
        mode_docs = ' '.join(cal_docs['Parameters'][3].desc).split('* ')[1:]
        for i, mode_doc in enumerate(mode_docs):
            mode_name, mode_tooltip = mode_doc.split(': ')
            wrapped_tooltip = '\n'.join(textwrap.wrap(mode_tooltip, width=70))
            self.ui.cb_calib_mode.addItem(mode_name)
            self.ui.cb_calib_mode.setItemData(i, wrapped_tooltip, Qt.ToolTipRole)

        self.ui.qbutton_browse.clicked[bool].connect(self.browse_dir_path)
        self.ui.le_directory.editingFinished.connect(self.enter_dir_path)
        self.ui.le_directory.setText(str(Path.cwd()))

        self.ui.le_swing.editingFinished.connect(self.enter_swing)
        self.ui.le_swing.setText('0.1')
        self.enter_swing()

        self.ui.le_wavelength.editingFinished.connect(self.enter_wavelength)
        self.ui.le_wavelength.setText('532')
        self.enter_wavelength()

        self.ui.cb_calib_scheme.currentIndexChanged[int].connect(self.enter_calib_scheme)
        self.ui.cb_calib_mode.currentIndexChanged[int].connect(self.enter_calib_mode)
        self.ui.cb_lca.currentIndexChanged[int].connect(self.enter_dac_lca)
        self.ui.cb_lcb.currentIndexChanged[int].connect(self.enter_dac_lcb)
        self.ui.chb_use_roi.stateChanged[int].connect(self.enter_use_cropped_roi)
        self.ui.qbutton_calibrate.clicked[bool].connect(self.run_calibration)
        self.ui.qbutton_load_calib.clicked[bool].connect(self.load_calibration)
        self.ui.qbutton_calc_extinction.clicked[bool].connect(self.calc_extinction)
        self.ui.cb_config_group.currentIndexChanged[int].connect(self.enter_config_group)

        # Capture Background
        self.ui.le_bg_folder.editingFinished.connect(self.enter_bg_folder_name)
        self.ui.le_n_avg.editingFinished.connect(self.enter_n_avg)
        self.ui.qbutton_capture_bg.clicked[bool].connect(self.capture_bg)

        # Advanced
        self.ui.cb_loglevel.currentIndexChanged[int].connect(self.enter_log_level)
        self.ui.qbutton_push_note.clicked[bool].connect(self.push_note)

        # Acquisition Tab
        self.ui.qbutton_gui_mode.clicked[bool].connect(self.change_gui_mode)
        self.ui.qbutton_browse_save_dir.clicked[bool].connect(self.browse_save_path)
        self.ui.le_save_dir.editingFinished.connect(self.enter_save_path)
        self.ui.le_save_dir.setText(str(Path.cwd()))
        self.ui.le_data_save_name.editingFinished.connect(self.enter_save_name)
        self.ui.qbutton_listen.clicked[bool].connect(self.listen_and_reconstruct)

        self.ui.le_zstart.editingFinished.connect(self.enter_zstart)
        self.ui.le_zstart.setText('-1')
        self.enter_zstart()

        self.ui.le_zend.editingFinished.connect(self.enter_zend)
        self.ui.le_zend.setText('1')
        self.enter_zend()

        self.ui.le_zstep.editingFinished.connect(self.enter_zstep)
        self.ui.le_zstep.setText('0.25')
        self.enter_zstep()

        self.ui.chb_use_gpu.stateChanged[int].connect(self.enter_use_gpu)
        self.ui.le_gpu_id.editingFinished.connect(self.enter_gpu_id)

        self.ui.le_recon_wavelength.setText('532') # This parameter seems to be wired differently than others...investigate later

        self.ui.le_obj_na.editingFinished.connect(self.enter_obj_na)
        self.ui.le_obj_na.setText('1.3')
        self.enter_obj_na()

        self.ui.le_cond_na.editingFinished.connect(self.enter_cond_na)
        self.ui.le_cond_na.setText('0.5')
        self.enter_cond_na()

        self.ui.le_mag.editingFinished.connect(self.enter_mag)
        self.ui.le_mag.setText('60')
        self.enter_mag()

        self.ui.le_ps.editingFinished.connect(self.enter_ps)
        self.ui.le_ps.setText('6.9')
        self.enter_ps()

        self.ui.le_n_media.editingFinished.connect(self.enter_n_media)
        self.ui.le_n_media.setText('1.3')
        self.enter_n_media()

        self.ui.le_pad_z.editingFinished.connect(self.enter_pad_z)
        self.ui.chb_pause_updates.stateChanged[int].connect(self.enter_pause_updates)
        self.ui.cb_birefringence.currentIndexChanged[int].connect(self.enter_birefringence_dim)
        self.ui.cb_phase.currentIndexChanged[int].connect(self.enter_phase_dim)

        # Populate background correction GUI element
        for i in range(3):
            self.ui.cb_bg_method.removeItem(0)
        bg_options = ['None','Measured','Estimated','Measured + Estimated']
        tooltips = ['No background correction.',
                    'Correct sample images with a background image acquired at an empty field of view, loaded from "Background Path".',
                    'Estimate sample background by fitting a 2D surface to the sample images. Works well when structures are spatially distributed across the field of view and a clear background is unavailable.',
                    'Apply "Measured" background correction and then "Estimated" background correction. Use to remove residual background after the sample retardance is corrected with measured background.']
        for i, bg_option in enumerate(bg_options):
            wrapped_tooltip = '\n'.join(textwrap.wrap(tooltips[i], width=70))
            self.ui.cb_bg_method.addItem(bg_option)
            self.ui.cb_bg_method.setItemData(i, wrapped_tooltip, Qt.ToolTipRole)
        self.ui.cb_bg_method.currentIndexChanged[int].connect(self.enter_bg_correction)

        self.ui.le_bg_path.editingFinished.connect(self.enter_acq_bg_path)
        self.ui.qbutton_browse_bg_path.clicked[bool].connect(self.browse_acq_bg_path)
        self.ui.qbutton_acq_birefringence.clicked[bool].connect(self.acq_birefringence)
        self.ui.qbutton_acq_phase.clicked[bool].connect(self.acq_phase)
        self.ui.qbutton_acq_birefringence_phase.clicked[bool].connect(self.acq_birefringence_phase)
        self.ui.cb_colormap.currentIndexChanged[int].connect(self.enter_colormap)
        self.ui.chb_display_volume.stateChanged[int].connect(self.enter_use_full_volume)
        self.ui.le_overlay_slice.editingFinished.connect(self.enter_display_slice)
        self.ui.slider_value.sliderMoved[tuple].connect(self.handle_val_slider_move)
        self.ui.slider_saturation.sliderMoved[tuple].connect(self.handle_sat_slider_move)

        # Display Tab
        self.viewer.layers.events.inserted.connect(self._add_layer_to_display_boxes)
        self.viewer.layers.events.removed.connect(self._remove_layer_from_display_boxes)
        self.ui.qbutton_create_overlay.clicked[bool].connect(self.create_overlay)
        self.ui.cb_saturation.currentIndexChanged[int].connect(self.update_sat_scale)
        self.ui.cb_value.currentIndexChanged[int].connect(self.update_value_scale)
        self.ui.le_sat_max.editingFinished.connect(self.enter_sat_max)
        self.ui.le_sat_min.editingFinished.connect(self.enter_sat_min)
        self.ui.le_val_max.editingFinished.connect(self.enter_val_max)
        self.ui.le_val_min.editingFinished.connect(self.enter_val_min)

        # Reconstruction
        self.ui.qbutton_browse_data_dir.clicked[bool].connect(self.browse_data_dir)
        self.ui.qbutton_browse_calib_meta.clicked[bool].connect(self.browse_calib_meta)
        self.ui.qbutton_load_config.clicked[bool].connect(self.load_config)
        self.ui.qbutton_save_config.clicked[bool].connect(self.save_config)
        self.ui.qbutton_load_default_config.clicked[bool].connect(self.load_default_config)
        self.ui.cb_method.currentIndexChanged[int].connect(self.enter_method)
        self.ui.cb_mode.currentIndexChanged[int].connect(self.enter_mode)
        self.ui.le_calibration_metadata.editingFinished.connect(self.enter_calib_meta)
        self.ui.qbutton_reconstruct.clicked[bool].connect(self.reconstruct)
        self.ui.cb_phase_denoiser.currentIndexChanged[int].connect(self.enter_phase_denoiser)

        # Logging
        log_box = QtLogger(self.ui.te_log)
        log_box.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_box)
        logging.getLogger().setLevel(logging.INFO)

        # Signal Emitters
        self.mm_status_changed.connect(self.handle_mm_status_update)

        # Instantiate Attributes:
        self.gui_mode = 'offline'
        self.bridge = None
        self.mm = None
        self.mmc = None
        self.calib = None
        self.current_dir_path = str(Path.cwd())
        self.current_save_path = str(Path.cwd())
        self.current_bg_path = str(Path.cwd())
        self.directory = str(Path.cwd())

        # Reconstruction / Calibration Parameter Defaults
        self.calib_scheme = '4-State'
        self.calib_mode = 'MM-Retardance'
        self.interp_method = 'schnoor_fit'
        self.config_group = 'Channel'
        self.calib_channels = ['State0', 'State1', 'State2', 'State3', 'State4']
        self.last_calib_meta_file = None
        self.use_cropped_roi = False
        self.bg_folder_name = 'BG'
        self.n_avg = 5
        self.intensity_monitor = []
        self.save_directory = str(Path.cwd())
        self.save_name = None
        self.bg_option = 'None'
        self.birefringence_dim = '2D'
        self.phase_dim = '2D'
        self.gpu_id = 0
        self.use_gpu = False
        self.pad_z = 0
        self.phase_reconstructor = None
        self.acq_bg_directory = None
        self.auto_shutter = True
        self.lca_dac = None
        self.lcb_dac = None
        self.pause_updates = False
        self.method = 'QLIPP'
        self.mode = '3D'
        self.calib_path = str(Path.cwd())
        self.data_dir = str(Path.cwd())
        self.config_path = str(Path.cwd())
        self.save_config_path = str(Path.cwd())
        self.colormap = 'HSV'
        self.use_full_volume = False
        self.display_slice = 0
        self.last_p = 0
        self.reconstruction_data_path = None
        self.reconstruction_data = None

        # Assessment attributes
        self.calib_assessment_level = None

        # Init Plot
        self.plot_item = self.ui.plot_widget.getPlotItem()
        self.plot_item.enableAutoRange()
        self.plot_item.setLabel('left', 'Intensity')
        self.ui.plot_widget.setBackground((32, 34, 40))
        self.plot_sequence = 'Coarse'

        # Init thread worker
        self.worker = None

        # Display/Initialiaze GUI Images (plotting legends, recOrder logo)
        recorder_dir = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
        jch_legend_path = os.path.join(recorder_dir, 'docs/images/JCh_legend.png')
        hsv_legend_path = os.path.join(recorder_dir, 'docs/images/HSV_legend.png')
        self.jch_pixmap = QPixmap(jch_legend_path)
        self.hsv_pixmap = QPixmap(hsv_legend_path)
        self.ui.label_orientation_image.setPixmap(self.hsv_pixmap)
        logo_path = os.path.join(recorder_dir, 'docs/images/recOrder_plugin_logo.png')
        logo_pixmap = QPixmap(logo_path)
        self.ui.label_logo.setPixmap(logo_pixmap)

        # Get default config file
        self.default_offline_config = os.path.join(recorder_dir, "recOrder/plugin/config_offline_default.yml")

        # Hide initial UI elements for later implementation or for later pop-up purposes
        self.ui.label_lca.hide()
        self.ui.label_lcb.hide()
        self.ui.cb_lca.hide()
        self.ui.cb_lcb.hide()
        self._hide_acquisition_ui(True)
        self.ui.label_bg_path.setHidden(True)
        self.ui.le_bg_path.setHidden(True)
        self.ui.qbutton_browse_bg_path.setHidden(True)
        self.ui.le_rho.setHidden(True)
        self.ui.label_phase_rho.setHidden(True)
        self.ui.le_itr.setHidden(True)
        self.ui.label_itr.setHidden(True)
        self.ui.le_bf_chan.setHidden(True)
        self.ui.label_bf_chan.setHidden(True)
        self.ui.label_focus_zidx.setHidden(True)
        self.ui.le_focus_zidx.setHidden(True)

        # Hide temporarily unsupported "Listen" functions
        self.ui.qbutton_listen.setHidden(True)
        self.ui.chb_pause_updates.setHidden(True)

        # Hide temporarily unsupported "Overlay" functions
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.Display), "Orientation Legend")
        self.ui.label_orientation_legend.setHidden(True)
        self.ui.DisplayOptions.setHidden(True)

        # Hide "Use Cropped ROI"
        self.ui.chb_use_roi.setHidden(True)

        # Set initial UI Properties
        self.ui.le_gui_mode.setStyleSheet("border: 1px solid rgb(200,0,0); color: rgb(200,0,0);")
        self.ui.te_log.setStyleSheet('background-color: rgb(32,34,40);')
        self.ui.le_mm_status.setText('Not Connected')
        self.ui.le_mm_status.setStyleSheet("border: 1px solid yellow;")
        self.ui.le_sat_min.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_sat_max.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_val_min.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.ui.le_val_max.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.setStyleSheet("QTabWidget::tab-bar {alignment: center;}")
        self.red_text = QColor(200, 0, 0, 255)
        self.original_tab_text = self.ui.tabWidget_3.tabBar().tabTextColor(0)
        self.ui.tabWidget.parent().setObjectName('recOrder') # make sure the top says recOrder and not 'Form'
        self.ui.tabWidget_2.setCurrentIndex(0) # set focus to "Plot" tab by default
        self.ui.tabWidget_3.setCurrentIndex(0) # set focus to "General" tab by default

        # No "Optional" text on offline mode's calibration metadata box
        self.ui.le_calibration_metadata.setPlaceholderText("")

        # disable wheel events for combo boxes
        for attr_name in dir(self.ui):
            if 'cb_' in attr_name:
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
        sizePolicy.setHeightForWidth(ui_slider.sizePolicy().hasHeightForWidth())
        ui_slider.setSizePolicy(sizePolicy)
        ui_slider.setOrientation(Qt.Horizontal)
        ui_slider.setObjectName(slider_name)
        self.ui.gridLayout_26.addWidget(ui_slider,
                                        slider_position[0],
                                        slider_position[1],
                                        slider_position[2],
                                        slider_position[3])
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
        sizePolicy.setHeightForWidth(ui_slider.sizePolicy().hasHeightForWidth())
        ui_slider.setSizePolicy(sizePolicy)
        ui_slider.setOrientation(Qt.Horizontal)
        ui_slider.setObjectName(slider_name)
        self.ui.gridLayout_26.addWidget(ui_slider,
                                        slider_position[0],
                                        slider_position[1],
                                        slider_position[2],
                                        slider_position[3])
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
        value_slider_position = self.ui.gridLayout_17.getItemPosition(value_slider_idx)
        value_slider_parent = self.ui.slider_value.parent().objectName()
        saturation_slider_idx = self.ui.gridLayout_17.indexOf(self.ui.slider_saturation)
        saturation_slider_position = self.ui.gridLayout_17.getItemPosition(saturation_slider_idx)
        saturation_slider_parent = self.ui.slider_saturation.parent().objectName()

        # Remove regular sliders from the UI
        self.ui.gridLayout_17.removeWidget(self.ui.slider_value)
        self.ui.gridLayout_17.removeWidget(self.ui.slider_saturation)

        # Add back the sliders as range sliders with the same properties
        self.ui.slider_saturation = QDoubleRangeSlider(getattr(self.ui, saturation_slider_parent))
        sizePolicy.setHeightForWidth(self.ui.slider_saturation.sizePolicy().hasHeightForWidth())
        self.ui.slider_saturation.setSizePolicy(sizePolicy)
        self.ui.slider_saturation.setOrientation(Qt.Horizontal)
        self.ui.slider_saturation.setObjectName("slider_saturation")
        self.ui.gridLayout_17.addWidget(self.ui.slider_saturation,
                                        saturation_slider_position[0],
                                        saturation_slider_position[1],
                                        saturation_slider_position[2],
                                        saturation_slider_position[3])
        self.ui.slider_saturation.setRange(0, 100)

        self.ui.slider_value = QDoubleRangeSlider(getattr(self.ui, value_slider_parent))
        sizePolicy.setHeightForWidth(self.ui.slider_value.sizePolicy().hasHeightForWidth())
        self.ui.slider_value.setSizePolicy(sizePolicy)
        self.ui.slider_value.setOrientation(Qt.Horizontal)
        self.ui.slider_value.setObjectName("slider_value")
        self.ui.gridLayout_17.addWidget(self.ui.slider_value,
                                        value_slider_position[0],
                                        value_slider_position[1],
                                        value_slider_position[2],
                                        value_slider_position[3])
        self.ui.slider_value.setRange(0, 100)

    def _hide_acquisition_ui(self, val: bool):
        """
        hides or shows the acquisition (online) UI elements.  Used when switching between online/offline mode

        Parameters
        ----------
        val:        (bool) True/False whether to hide (True) or show (False)

        Returns
        -------

        """
        self.ui.acq_settings.setHidden(val)
        self.ui.acquire.setHidden(val)

        # Calibration Tab
        self.ui.tabWidget.setTabEnabled(0, not val)
        if val:
            self.ui.tabWidget.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        else:
            self.ui.tabWidget.setStyleSheet("")
            self.ui.le_mm_status.setText('Not Connected')
            self.ui.le_mm_status.setStyleSheet("border: 1px solid yellow;")
            self.mmc = None
            self.mm = None
            # self.ui.cb_config_group.clear() # this might be the culprit because it clear the config
            self.ui.tabWidget.setCurrentIndex(0)

    def _hide_offline_ui(self, val: bool):
        """
        hides or shows the offline UI elements.  Used when switching between online/offline mode

        Parameters
        ----------
        val:        (bool) True/False whether to hide (True) or show (False)

        Returns
        -------

        """

        # General Settings
        self.ui.le_data_dir.setHidden(val)
        self.ui.label_data_dir.setHidden(val)
        self.ui.qbutton_browse_data_dir.setHidden(val)
        self.ui.le_calibration_metadata.setHidden(val)
        self.ui.label_calib_meta.setHidden(val)
        self.ui.qbutton_browse_calib_meta.setHidden(val)
        self.ui.qbutton_load_config.setHidden(val)
        self.ui.qbutton_save_config.setHidden(val)
        self.ui.qbutton_load_default_config.setHidden(val)
        self.ui.qbutton_reconstruct.setHidden(val)
        self.ui.qbutton_stop_reconstruct.setHidden(val)

        # Processing Settings
        self.ui.groupBox_2.setHidden(val)
        
    def _enable_buttons(self):
        """
        enables the buttons that were disabled during acquisition, calibration, or reconstruction

        Returns
        -------

        """

        self.ui.qbutton_calibrate.setEnabled(True)
        self.ui.qbutton_capture_bg.setEnabled(True)
        self.ui.qbutton_calc_extinction.setEnabled(True)
        self.ui.qbutton_acq_birefringence.setEnabled(True)
        self.ui.qbutton_acq_phase.setEnabled(True)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(True)
        self.ui.qbutton_load_calib.setEnabled(True)
        self.ui.qbutton_listen.setEnabled(True)
        self.ui.qbutton_create_overlay.setEnabled(True)
        self.ui.qbutton_reconstruct.setEnabled(True)
        self.ui.qbutton_load_config.setEnabled(True)
        self.ui.qbutton_load_default_config.setEnabled(True)

    def _disable_buttons(self):
        """
        disables the buttons during acquisition, calibration, or reconstruction.  This prevents the user from
        trying to do multiple actions at once (i.e. trying to use the acquisition features while calibration is running)

        Returns
        -------

        """

        self.ui.qbutton_calibrate.setEnabled(False)
        self.ui.qbutton_capture_bg.setEnabled(False)
        self.ui.qbutton_calc_extinction.setEnabled(False)
        self.ui.qbutton_acq_birefringence.setEnabled(False)
        self.ui.qbutton_acq_phase.setEnabled(False)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(False)
        self.ui.qbutton_load_calib.setEnabled(False)
        self.ui.qbutton_listen.setEnabled(False)
        self.ui.qbutton_create_overlay.setEnabled(False)
        self.ui.qbutton_reconstruct.setEnabled(False)
        self.ui.qbutton_load_config.setEnabled(False)
        self.ui.qbutton_load_default_config.setEnabled(False)

    def _handle_error(self, exc):
        """
        Handles errors from calibration and restores micromanager to its state prior to the start of calibration
        Parameters
        ----------
        exc:        (Error) Propogated error message to display

        Returns
        -------

        """

        self.ui.tb_calib_assessment.setText(f'Error: {str(exc)}')
        self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid rgb(200,0,0);")

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
        self.ui.tb_calib_assessment.setText('Previous calibration successfully loaded')
        self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid green;")
        self.ui.progress_bar.setValue(100)

    def _update_calib(self, val):
        self.calib = val

    def _add_layer_to_display_boxes(self, val):
        """
        When a new napari layer is added to recOrder, update the Display Tab combo boxes with these layers.
        This will allow the user to then choose which layers it wants to use for the overlay.  Will skip over
        any layers that are already an 'Overlay'.  This function is connected to a napari.Layer signal

        Parameters
        ----------
        val:            (napari.Layer) layer that was added [not used]

        Returns
        -------

        """

        for layer in self.viewer.layers:
            if 'Overlay' in layer.name:
                continue
            if layer.name not in [self.ui.cb_hue.itemText(i) for i in range(self.ui.cb_hue.count())]:
                self.ui.cb_hue.addItem(layer.name)
            if layer.name not in [self.ui.cb_saturation.itemText(i) for i in range(self.ui.cb_saturation.count())]:
                self.ui.cb_saturation.addItem(layer.name)
            if layer.name not in [self.ui.cb_value.itemText(i) for i in range(self.ui.cb_value.count())]:
                self.ui.cb_value.addItem(layer.name)

    def _remove_layer_from_display_boxes(self, val):
        """
        When a napari layer is removed from napari, remove the corresponding layer from Display Tab combo boxes.

        Parameters
        ----------
        val:            (napari.Layer) layer that was removed by the user

        Returns
        -------

        """

        for i in range(self.ui.cb_hue.count()):
            if val.value.name in self.ui.cb_hue.itemText(i):
                self.ui.cb_hue.removeItem(i)
            if val.value.name in self.ui.cb_saturation.itemText(i):
                self.ui.cb_saturation.removeItem(i)
            if val.value.name in self.ui.cb_value.itemText(i):
                self.ui.cb_value.removeItem(i)

    def _set_tab_red(self, name, state):
        """
        Convenience function to set a GUI tab red when there is a parameter missing for acquisiton or reconstruction

        Parameters
        ----------
        name:           (str) Name of the tab
        state:          (bool) True/False whether to set red (True) or not red (False)

        Returns
        -------

        """

        # this map corresponds to the tab index in the TabWidget GUI element
        name_map = {'General': 0,
                    'Processing': 1}

        index = name_map[name]

        if state:
            self.ui.tabWidget_3.tabBar().setTabTextColor(index, self.red_text)
        else:
            self.ui.tabWidget_3.tabBar().setTabTextColor(index, self.original_tab_text)

    def _check_line_edit(self, name):
        """
        Convencience function used in checking whether a line edit is present or missing.  Will place a red border
        around the line edit if it is empty, otherwise it will remove the red border.

        Parameters
        ----------
        name:           (str) name of the LineEdit element as specified in QT Designer file.

        Returns
        -------

        """
        le = getattr(self.ui, f'le_{name}')
        text = le.text()

        if text == '':
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

        # Initialize all tabs in their default style (not red)
        self._set_tab_red('General', False)
        self._set_tab_red('Processing', False)
        
        # initialize the variable to keep track of the success of the requirement check
        raise_error = False

        # define the fields required for the specific acquisition modes.  Matches LineEdit object names
        phase_required = {'wavelength', 'mag', 'cond_na', 'obj_na', 'n_media',
                          'phase_strength', 'ps', 'zstep'}

        # Initalize all fields in their default style (not red).
        for field in phase_required:
            le = getattr(self.ui, f'le_{field}')
            le.setStyleSheet("")
        
        # Check generally required fields
        if mode == 'birefringence' or mode == 'phase':
            success = self._check_line_edit('save_dir')
            if not success:
                raise_error = True
                self._set_tab_red('General', True)

            # check background path if 'Measured' or 'Measured + Estimated' is selected
            if self.bg_option == 'local_fit+' or self.bg_option == 'global':
                success = self._check_line_edit('bg_path')
                if not success:
                    raise_error = True
                    self._set_tab_red('General', True)

        # Check phase specific fields
        if mode == 'phase':

            # add in extra requirement is user is acquiring PhaseFromBF
            if self.ui.chb_phase_from_bf.isChecked():
                cont = self._check_line_edit('recon_wavelength')
                tab = getattr(self.ui, f'le_recon_wavelength').parent().parent().objectName()
                if not cont:
                    raise_error = True
                    self._set_tab_red(tab, True)

            for field in phase_required:
                cont = self._check_line_edit(field)
                tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                if not cont:
                    raise_error = True
                    if field != 'zstep':
                        self._set_tab_red(tab, True)
                else:
                    continue

        # Alert the user to check and enter in the missing parameters
        if raise_error:
            raise ValueError('Please enter in all of the parameters necessary for the acquisition')

    def _check_requirements_for_reconstruction(self):
        """
        This function will loop through the parameters for offline reconstruction and make sure the user has
        specified the necessary parameters.  If it finds an empty or missing parameters, it will set missing fields red
        and stop the reconstruction process.

        Returns
        -------

        """

        # Initalize all tab elements and reconstruct button to default state (not red)
        self._set_tab_red('General', False)
        self._set_tab_red('Processing', False)
        self.ui.qbutton_reconstruct.setStyleSheet("")

        # initalize the success variable of the requirement check
        success = True

        # gather the specified output channels (will determine which requirements are necessary to look at)
        output_channels = self.ui.le_output_channels.text()
        output_channels = output_channels.split(',')
        output_channels = [chan.replace(' ', '') for chan in output_channels]

        # intialize the reconstruction specific required fields
        always_required = {'data_dir', 'save_dir', 'positions', 'timepoints', 'output_channels'}
        birefringence_required = {'calibration_metadata', 'recon_wavelength'}
        phase_required = {'recon_wavelength', 'mag', 'obj_na', 'cond_na', 'n_media',
                          'phase_strength', 'ps'}
        
        # intialize all UI elements in the default state
        for field in always_required:
            le = getattr(self.ui, f'le_{field}')
            le.setStyleSheet("")
        for field in phase_required:
            le = getattr(self.ui, f'le_{field}')
            le.setStyleSheet("")
        for field in always_required:
            cont = self._check_line_edit(field)
            if not cont:
                success = False
                if field == 'data_dir' or field == 'save_dir':
                    self._set_tab_red('General', True)
                if field == 'positions' or field == 'timepoints' or field == 'output_channels':
                    self._set_tab_red('Processing', True)
            else:
                continue

        possible_channels = ['Retardance', 'Orientation', 'BF', 'S0', 'S1', 'S2', 'S3', 'Phase2D', 'Phase3D']
        bire_channels = ['Retardance', 'Orientation', 'BF', 'S0', 'S1', 'S2', 'S3']
        qlipp_channel_present = False
        bire_channel_present = False
        if self.method == 'QLIPP':
            for channel in output_channels:
                if channel in possible_channels:
                    qlipp_channel_present = True
                if channel in bire_channels:
                    bire_channel_present = True

            if qlipp_channel_present:
                if bire_channel_present:
                    for field in birefringence_required:
                        cont = self._check_line_edit(field)
                        tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                        if not cont:
                            if field != 'calibration_metadata':
                                self._set_tab_red(tab, True)
                            success = False
                        else:
                            # self._set_tab_red(tab, False)
                            continue

                if 'Phase2D' in output_channels or 'Phase3D' in output_channels:
                    for field in phase_required:
                        cont = self._check_line_edit(field)
                        tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                        if not cont:
                            self._set_tab_red(tab, True)
                            success = False
                        else:
                            self._set_tab_red(tab, False)
                            continue
                    if 'Phase2D' in output_channels and self.mode == '2D':
                        cont = self._check_line_edit('focus_zidx')
                        if not cont:
                            self._set_tab_red('Processing', True)
                            success = False
            else:
                self._set_tab_red('Processing', True)
                self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                print('User did not specify any QLIPP Specific Channels')
                success = False

        elif self.method == 'PhaseFromBF':
            cont = self._check_line_edit('bf_chan')
            tab = getattr(self.ui, f'le_bf_chan').parent().parent().objectName()
            if not cont:
                self._set_tab_red(tab, True)
                success = False

            if 'Phase2D' in output_channels or 'Phase3D' in output_channels:
                for field in phase_required:
                    cont = self._check_line_edit(field)
                    tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                    if not cont:
                        self._set_tab_red(tab, True)
                        success = False
                    else:
                        continue
                if 'Phase2D' in output_channels and self.mode == '2D':
                    cont = self._check_line_edit('focus_zidx')
                    if not cont:
                        self._set_tab_red('Processing', True)
                        success = False
            else:
                self._set_tab_red('Processing', True)
                self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                print('User did not specify any PhaseFromBF Specific Channels (Phase2D, Phase3D)')
                success = False

        else:
            print('Error in parameter checks')
            self.ui.qbutton_reconstruct.setStyleSheet("border: 1px solid rgb(200,0,0);")
            success = False

        return success

    def _populate_config_from_app(self):
        """
        This function will create a ConfigReader instance with the information from the UI elements.  This function
        is called prior to reconstruction or when the user uses the "Save Config" to file functionality.

        Returns
        -------

        """

        # ConfigReader is usually immutable but we need it to not be in this case to update its properties
        self.config_reader = ConfigReader(immutable=False)

        # Parse dataset fields manually
        self.config_reader.data_dir = self.ui.le_data_dir.text()
        self.data_dir = self.ui.le_data_dir.text()
        self.config_reader.save_dir = self.ui.le_save_dir.text()
        self.save_directory = self.ui.le_save_dir.text()
        self.config_reader.method = self.method
        self.config_reader.mode = self.mode
        self.config_reader.data_save_name = self.ui.le_data_save_name.text() if self.ui.le_data_save_name.text() != '' \
                                            else pathlib.PurePath(self.data_dir).name
        self.config_reader.calibration_metadata = self.ui.le_calibration_metadata.text()
        self.config_reader.background = self.ui.le_bg_path.text()
        self.config_reader.background_correction = self.bg_option

        # Assumes that positions/timepoints can either be 'all'; '[all]'; 1, 2, 3, N; (start, end)
        positions = self.ui.le_positions.text()
        positions = positions.replace(' ', '')
        if positions == 'all' or positions == "['all']" or positions == '[all]':
            self.config_reader.positions = ['all']
        elif positions.startswith('[') and positions.endswith(']'):
            vals = positions[1:-1].split(',')
            if len(vals) != 2:
                self._set_tab_red('Processing', True)
                self.ui.le_positions.setStyleSheet("border: 1px solid rgb(200,0,0);")
            else:
                self._set_tab_red('Processing', False)
                self.ui.le_positions.setStyleSheet("")
                self.config_reader.positions = [(int(vals[0]), int(vals[1]))]
        elif positions.startswith('(') and positions.endswith(')'):
            self.config_reader.positions = [eval(positions)]
        else:
            vals = positions.split(',')
            vals = map(lambda x: int(x), vals)
            self.config_reader.positions = list(vals)

        timepoints = self.ui.le_timepoints.text()
        timepoints = timepoints.replace(' ', '')
        if timepoints == 'all' or timepoints == "['all']" or timepoints == '[all]':
            self.config_reader.timepoints = ['all']
        elif timepoints.startswith('[') and timepoints.endswith(']'):
            vals = timepoints[1:-1].split(',')
            if len(vals) != 2:
                self._set_tab_red('Processing', True)
                self.ui.le_timepoints.setStyleSheet("border: 1px solid rgb(200,0,0);")
            else:
                self._set_tab_red('Processing', False)
                self.ui.le_timepoints.setStyleSheet("")
                self.config_reader.timepoints = [(int(vals[0]), int(vals[1]))]
        elif timepoints.startswith('(') and timepoints.endswith(')'):
            self.config_reader.timepoints = [eval(timepoints)]
        else:
            vals = timepoints.split(',')
            vals = map(lambda x: int(x), vals)
            self.config_reader.timepoints = list(vals)

        attrs = dir(self.ui)
        skip = ['wavelength', 'pixel_size', 'magnification', 'NA_objective', 'NA_condenser', 'n_objective_media']
        # TODO: Figure out how to catch errors in regularizer strength field
        for key, value in PROCESSING.items():
            if key not in skip:
                if key == 'background_correction':
                    bg_map = {0: 'None', 1: 'global', 2: 'local_fit', 3: 'local_fit+'}
                    setattr(self.config_reader, key, bg_map[self.ui.cb_bg_method.currentIndex()])

                elif key == 'output_channels':

                    # Reset style sheets
                    self.ui.le_output_channels.setStyleSheet("")
                    self.ui.cb_mode.setStyleSheet("")
                    self._set_tab_red('Processing', False)

                    # Make a list of the channels from the line edit string
                    field_text = self.ui.le_output_channels.text()
                    channels = field_text.split(',')
                    channels = [i.replace(' ', '') for i in channels]
                    setattr(self.config_reader, key, channels)

                    if 'Phase3D' in channels and 'Phase2D' in channels:
                        self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                        self._set_tab_red('Processing', True)
                        raise KeyError(
                            f'Both Phase3D and Phase2D cannot be specified in output_channels.  Please compute '
                            f'separately')

                    if 'Phase3D' in channels and self.mode == '2D':
                        self._set_tab_red('Processing', True)
                        self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                        self.ui.cb_mode.setStyleSheet("border: 1px solid rgb(200,0,0);")
                        raise KeyError(f'Specified mode is 2D and Phase3D was specified for reconstruction. '
                                       'Only 2D reconstructions can be performed in 2D mode')

                    if 'Phase2D' in channels and self.mode == '3D':
                        self._set_tab_red('Processing', True)
                        self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                        self.ui.cb_mode.setStyleSheet("border: 1px solid rgb(200,0,0);")
                        raise KeyError(f'Specified mode is 3D and Phase2D was specified for reconstruction. '
                                       'Only 3D reconstructions can be performed in 3D mode')

                elif key == 'pad_z':
                    val = self.ui.le_pad_z.text()
                    setattr(self.config_reader, key, int(val))

                elif key == 'gpu_id':
                    val = self.ui.le_gpu_id.text()
                    setattr(self.config_reader, key, int(val))

                elif key == 'use_gpu':
                    if self.ui.chb_use_gpu.isChecked():
                        setattr(self.config_reader, key, True)
                    else:
                        setattr(self.config_reader, key, False)

                elif key == 'focus_zidx':
                    val = self.ui.le_focus_zidx.text()
                    if val == '':
                        setattr(self.config_reader, key, None)
                    else:
                        setattr(self.config_reader, key, int(val))

                else:
                    attr_name = f'le_{key}'
                    if attr_name in attrs:
                        le = getattr(self.ui, attr_name)
                        try:
                            setattr(self.config_reader, key, float(le.text()))
                        except ValueError as err:
                            print(err)
                            tab = le.parent().parent().objectName()
                            self._set_tab_red(tab, True)
                            le.setStyleSheet("border: 1px solid rgb(200,0,0);")
                    else:
                        continue

        # Manually enter in phase regularization
        if self.mode == '3D':
            if self.ui.cb_phase_denoiser.currentIndex() == 0:
                setattr(self.config_reader, 'Tik_reg_ph_3D', float(self.ui.le_phase_strength.text()))
            else:
                setattr(self.config_reader, 'TV_reg_ph_3D', float(self.ui.le_phase_strength.text()))
                setattr(self.config_reader, 'rho_3D', float(self.ui.le_rho.text()))
                setattr(self.config_reader, 'itr_3D', int(self.ui.le_itr.text()))

        else:
            if self.ui.cb_phase_denoiser.currentIndex() == 0:
                setattr(self.config_reader, 'Tik_reg_ph_2D', float(self.ui.le_phase_strength.text()))
            else:
                setattr(self.config_reader, 'TV_reg_ph_2D', float(self.ui.le_phase_strength.text()))
                setattr(self.config_reader, 'rho_2D', float(self.ui.le_rho.text()))
                setattr(self.config_reader, 'itr_2D', int(self.ui.le_itr.text()))

        if self.method == 'PhaseFromBF':
            setattr(self.config_reader, 'wavelength', int(self.ui.le_recon_wavelength.text()))
            setattr(self.config_reader, 'brightfield_channel_index',
                      int(self.ui.le_bf_chan.text()))

            focus_zidx = self.ui.le_focus_zidx.text()
            setattr(self.config_reader, 'focus_zidx', int(focus_zidx) if focus_zidx != '' else None)

        else:
            setattr(self.config_reader, 'wavelength', int(self.ui.le_recon_wavelength.text()))

        # Parse name mismatch fields
        setattr(self.config_reader, 'NA_objective', float(self.ui.le_obj_na.text()) if self.ui.le_obj_na.text() != ''
                else None)
        setattr(self.config_reader, 'NA_condenser', float(self.ui.le_cond_na.text()) if self.ui.le_cond_na.text() != ''
                else None)
        setattr(self.config_reader, 'pixel_size', float(self.ui.le_ps.text()) if self.ui.le_ps.text() != ''
                else None)
        setattr(self.config_reader, 'n_objective_media', float(self.ui.le_n_media.text())
                if self.ui.le_n_media.text() != '' else None)
        setattr(self.config_reader, 'magnification', float(self.ui.le_mag.text()) if self.ui.le_mag.text() != ''
                else None)

    def _populate_from_config(self):
        """
        This function will take a previously defined config file and populate all of the UI elements.  Used in the
        Load Config workflow.

        Returns
        -------

        """
        # Parse dataset fields manually
        self.data_dir = self.config_reader.data_dir

        self.ui.le_data_dir.setText(self.config_reader.data_dir)
        self.save_directory = self.config_reader.save_dir
        self.ui.le_save_dir.setText(self.config_reader.save_dir)
        self.ui.le_data_save_name.setText(self.config_reader.data_save_name)
        self.ui.le_calibration_metadata.setText(self.config_reader.calibration_metadata)
        self.ui.le_bg_path.setText(self.config_reader.background)

        self.mode = self.config_reader.mode
        self.ui.cb_mode.setCurrentIndex(0) if self.mode == '3D' else self.ui.cb_mode.setCurrentIndex(1)
        self.method = self.config_reader.method
        if self.method == 'QLIPP':
            self.ui.cb_method.setCurrentIndex(0)
        elif self.method == 'PhaseFromBF':
            self.ui.cb_method.setCurrentIndex(1)
        else:
            print(f'Did not understand method from config: {self.method}')
            self.ui.cb_method.setStyleSheet("border: 1px solid rgb(200,0,0);")

        self.bg_option = self.config_reader.background_correction
        if self.bg_option == 'None':
            self.ui.cb_bg_method.setCurrentIndex(0)
        elif self.bg_option == 'global':
            self.ui.cb_bg_method.setCurrentIndex(1)
        elif self.bg_option == 'local_fit':
            self.ui.cb_bg_method.setCurrentIndex(2)
        elif self.bg_option == 'local_fit+':
            self.ui.cb_bg_method.setCurrentIndex(3)
        else:
            print(f'Did not understand method from config: {self.method}')
            self.ui.cb_method.setStyleSheet("border: 1px solid rgb(200,0,0);")

        if isinstance(self.config_reader.positions, list):
            positions = self.config_reader.positions
            text = ''
            for idx, pos in enumerate(positions):
                text += f'{pos}, ' if idx != len(positions) - 1 else f'{pos}'
                self.ui.le_positions.setText(text)
        else:
            self.ui.le_positions.setText(str(self.config_reader.positions))

        if isinstance(self.config_reader.timepoints, list):
            timepoints = self.config_reader.timepoints
            text = ''
            for idx, time in enumerate(timepoints):
                text += f'{time}, ' if idx != len(timepoints) - 1 else f'{time}'
                self.ui.le_timepoints.setText(text)
        else:
            self.ui.le_timepoints.setText(str(self.config_reader.timepoints))

        # Parse Processing name mismatch fields
        wavelengths = self.config_reader.wavelength
        if not isinstance(wavelengths, list):
            self.ui.le_recon_wavelength.setText(str(int(self.config_reader.wavelength)))
        else:
            text = ''
            for idx, chan in enumerate(wavelengths):
                text += f'{chan}, ' if idx != len(wavelengths) - 1 else f'{chan}'
            self.ui.le_recon_wavelength.setText(text)

        self.ui.le_obj_na.setText(str(self.config_reader.NA_objective))
        self.ui.le_cond_na.setText(str(self.config_reader.NA_condenser))
        self.ui.le_ps.setText(str(self.config_reader.pixel_size))
        self.ui.le_n_media.setText(str(self.config_reader.n_objective_media))
        self.ui.le_mag.setText(str(self.config_reader.magnification))

        # Parse PhaseFromBF
        if self.method == 'PhaseFromBF':
            self.ui.le_bf_chan.setText(str(self.config_reader.brightfield_channel_index))
        
        # Parse processing automatically
        denoiser = None
        for key, val in PROCESSING.items():
            if key == 'output_channels':
                channels = self.config_reader.output_channels
                text = ''
                for idx, chan in enumerate(channels):
                    text += f'{chan}, ' if idx != len(channels)-1 else f'{chan}'

                self.ui.le_output_channels.setText(text)

            elif key == 'focus_zidx':
                indices = self.config_reader.focus_zidx
                if isinstance(indices, int):
                    self.ui.le_focus_zidx.setText(str(indices))
                elif isinstance(indices, list):
                    text = ''
                    for idx, val in enumerate(indices):
                        text += f'{val}, ' if idx != len(indices) - 1 else f'{val}'

                    self.ui.le_focus_zidx.setText(text)

            elif key == 'use_gpu':
                state = getattr(self.config_reader, key)
                self.ui.chb_use_gpu.setChecked(state)

            elif key == 'gpu_id':
                val = str(int(getattr(self.config_reader, key)))
                self.ui.le_gpu_id.setText(val)

            elif key == 'pad_z':
                val = str(int(getattr(self.config_reader, key)))
                self.ui.le_pad_z.setText(val)

            elif key == 'focus_zidx':
                val = str(int(getattr(self.config_reader, key)))
                self.ui.le_focus_zidx.setText(val)

            elif hasattr(self.ui, f'le_{key}'):
                le = getattr(self.ui, f'le_{key}')
                le.setText(str(getattr(self.config_reader, key)) if not isinstance(getattr(self.config_reader, key),
                                                                                   str) else getattr(self.config_reader,
                                                                                                     key))
            elif hasattr(self.ui, f'cb_{key}'):
                cb = getattr(self.ui, f'cb_{key}')
                items = [cb.itemText(i) for i in range(cb.count())]
                cfg_attr = getattr(self.config_reader, key)
                self.ui.cb_mode.setCurrentIndex(items.index(cfg_attr))

            elif key == 'phase_denoiser_2D' or key == 'phase_denoiser_3D':
                cb = self.ui.cb_phase_denoiser
                cfg_attr = getattr(self.config_reader, f'phase_denoiser_{self.mode}')
                denoiser = cfg_attr
                cb.setCurrentIndex(0) if cfg_attr == 'Tikhonov' else cb.setCurrentIndex(1)
            else:
                if denoiser == 'Tikhonov':
                    strength = getattr(self.config_reader, f'Tik_reg_ph_{self.mode}')
                    self.ui.le_phase_strength.setText(str(strength))
                else:
                    strength = getattr(self.config_reader, f'TV_reg_ph_{self.mode}')
                    self.ui.le_phase_strength.setText(str(strength))
                    self.ui.le_rho.setText(str(getattr(self.config_reader, f'rho_{self.mode}')))
                    self.ui.le_itr.setText(str(getattr(self.config_reader, f'itr_{self.mode}')))

    @Slot(bool)
    def change_gui_mode(self):
        """
        Switches between offline/online mode and updates the corresponding GUI elements

        Returns
        -------

        """
        if self.gui_mode == 'offline':
            self.ui.qbutton_gui_mode.setText('Switch to Offline')
            self.ui.le_gui_mode.setText('Online')
            self.ui.le_gui_mode.setStyleSheet("border: 1px solid green; color: green;")
            self._hide_offline_ui(True)
            self._hide_acquisition_ui(False)
            self.gui_mode = 'online'
            self.connect_to_mm()

        else:
            self.ui.qbutton_gui_mode.setText('Switch to Online')
            self.ui.le_gui_mode.setText('Offline')
            self.ui.le_gui_mode.setStyleSheet("border: 1px solid rgb(200,0,0); color: rgb(200,0,0);")
            self._hide_offline_ui(False)
            self._hide_acquisition_ui(True)
            self.gui_mode = 'offline'
            self.ui.cb_config_group.clear()

            #Make sure button is still visible
            self.ui.qbutton_mm_connect.setEnabled(True)


    @Slot(bool)
    def connect_to_mm(self):
        """
        Function to establish the python/java bridge to MicroManager.  Micromanager must be open with a config loaded
        in order for the connection to be successful.  On connection, it will populate all of the available config
        groups.  Config group choice is used to establish which config group the Polarization states live in.

        Returns
        -------

        """
        RECOMMENDED_MM = '20220920'
        ZMQ_TARGET_VERSION = '4.2.0'
        try:
            # Try to open Bridge. Requires micromanager to be open with server running.
            # This does not fail gracefully, so I'm wrapping it in its own try-except block.
            try:
                self.bridge = Bridge(convert_camel_case=False)
                self.mmc = self.bridge.get_core()
                self.mm = self.bridge.get_studio()
            except:
                print(("Could not establish pycromanager bridge.\n"
                       "Is micromanager open?\n"
                       "Is Tools > Options > Run server on port 4827 checked?\n"
                       f"Are you using nightly build {RECOMMENDED_MM}?"))
                raise EnvironmentError

            # Warn the user if there is a MicroManager/ZMQ version mismatch
            self.bridge._main_socket.send({"command": "connect", "debug": False})
            reply_json = self.bridge._main_socket.receive(timeout=500)
            zmq_mm_version = reply_json['version']
            if zmq_mm_version != ZMQ_TARGET_VERSION:
                upgrade_str = 'upgrade' if version.parse(zmq_mm_version) < version.parse(ZMQ_TARGET_VERSION) else 'downgrade'
                print(("WARNING: This version of Micromanager has not been tested with recOrder.\n"
                      f"Please {upgrade_str} to MicroManager nightly build {RECOMMENDED_MM}."))

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
                if np.all([np.any([ch in config for config in config_list]) for ch in self.calib_channels]):
                    if not config_group_found:
                        self.config_group = group  # set to first config group found
                        config_group_found = True
                    self.ui.cb_config_group.addItem(group)
                # not entirely sure what this part does, but I left it in
                # I think it tried to find a channel such as 'BF'
                for ch in config_list:
                    if ch not in self.calib_channels:
                        self.ui.cb_acq_channel.addItem(ch)
            if not config_group_found:
                msg = f'No config group contains channels {self.calib_channels}. ' \
                      'Please refer to the recOrder wiki on how to set up the config properly.'
                self.ui.cb_config_group.setStyleSheet("border: 1px solid rgb(200,0,0);")
                raise KeyError(msg)


            # set startup LC control mode
            _devices = self.mmc.getLoadedDevices()
            loaded_devices = [_devices.get(i) for i in range(_devices.size())]
            if LC_DEVICE_NAME in loaded_devices:
                config_desc = self.mmc.getConfigData('Channel','State0').getVerbose()
                if 'String send to' in config_desc:
                    self.calib_mode = 'MM-Retardance'
                    self.ui.cb_calib_mode.setCurrentIndex(0)
                if 'Voltage (V)' in config_desc:
                    self.calib_mode = 'MM-Voltage'
                    self.ui.cb_calib_mode.setCurrentIndex(1)
            else:
                self.calib_mode = 'DAC'
                self.ui.cb_calib_mode.setCurrentIndex(2)


            self.mm_status_changed.emit(True)

        except:
            self.mm_status_changed.emit(False)

    @Slot(bool)
    def handle_mm_status_update(self, value):
        if value:
            self.ui.le_mm_status.setText('Success!')
            self.ui.le_mm_status.setStyleSheet("background-color: green;")
            #Disabling the button
            self.ui.qbutton_mm_connect.setEnabled(False)
        else:
            #Make sure button is still visible if it fails
            self.ui.qbutton_mm_connect.setEnabled(True)
            self.ui.le_mm_status.setText('Failed.')
            self.ui.le_mm_status.setStyleSheet("background-color: rgb(200,0,0);")

    @Slot(tuple)
    def handle_progress_update(self, value):
        self.ui.progress_bar.setValue(value[0])
        self.ui.label_progress.setText('Progress: ' + value[1])

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

        if self.plot_sequence[0] == 'Coarse':
            self.plot_item.autoRange()
        else:
            self.plot_item.setRange(xRange=(self.plot_sequence[1], len(self.intensity_monitor)),
                                    yRange=(0, np.max(self.intensity_monitor[self.plot_sequence[1]:])),
                                    padding=0.1)

    @Slot(str)
    def handle_calibration_assessment_update(self, value):
        self.calib_assessment_level = value

    @Slot(str)
    def handle_calibration_assessment_msg_update(self, value):
        self.ui.tb_calib_assessment.setText(value)

        if self.calib_assessment_level == 'good':
            self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid green;")
        elif self.calib_assessment_level == 'okay':
            self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid rgb(252,190,3);")
        elif self.calib_assessment_level == 'bad':
            self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid rgb(200,0,0);")
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
        theta = np.linspace(0, 2*np.pi, 100)
        x_circ = self.swing*np.cos(theta) + lc_values["LCA"][0]
        y_circ = self.swing*np.sin(theta) + lc_values["LCB"][0]

        import matplotlib.pyplot as plt
        plt.close('all')
        with plt.rc_context({
            "axes.spines.right": False,
            "axes.spines.top": False,
        }) and plt.ion():
            plt.figure("Calibrated LC States")
            plt.scatter(lc_values["LCA"], lc_values["LCB"], c='r')
            plt.plot(x_circ, y_circ, 'k--', alpha=0.25)
            plt.axis("equal")
            plt.xlabel("LCA retardance")
            plt.ylabel("LCB retardance")
            for i, pol in enumerate(pol_states):
                plt.annotate(
                    pol, 
                    xy=(lc_values["LCA"][i], lc_values["LCB"][i]),
                    xycoords='data',
                    xytext=(10,10), # annotation offset
                    textcoords='offset points'
            )

    @Slot(object)
    def handle_bg_image_update(self, value):

        if 'Background Images' in self.viewer.layers:
            self.viewer.layers['Background Images'].data = value
        else:
            self.viewer.add_image(value, name='Background Images', colormap='gray')

    @Slot(object)
    def handle_bg_bire_image_update(self, value):

        # Separate Background Retardance and Background Orientation
        # Add new layer if none exists, otherwise update layer data
        if 'Background Retardance' in self.viewer.layers:
            self.viewer.layers['Background Retardance'].data = value[0]
        else:
            self.viewer.add_image(value[0], name='Background Retardance', colormap='gray')

        if 'Background Orientation' in self.viewer.layers:
            self.viewer.layers['Background Orientation'].data = value[1]
        else:
            self.viewer.add_image(value[1], name='Background Orientation', colormap='gray')

    @Slot(object)
    def handle_bire_image_update(self, value):

        channel_names = {'Orientation': 1,
                         'Retardance': 0,
                         }

        # Compute Overlay if birefringence acquisition is 2D
        if self.birefringence_dim == '2D':
            channel_names['BirefringenceOverlay'] = None
            overlay = ret_ori_overlay(retardance=value[0],
                                      orientation=value[1],
                                      ret_max= np.percentile(value[0], 99.99),
                                      cmap=self.colormap)

        for key, chan in channel_names.items():
            if key == 'BirefringenceOverlay':
                if key+self.birefringence_dim in self.viewer.layers:
                    self.viewer.layers[key+self.birefringence_dim].data = overlay
                else:
                    self.viewer.add_image(overlay, name=key+self.birefringence_dim, rgb=True)
            else:
                if key+self.birefringence_dim in self.viewer.layers:
                    self.viewer.layers[key+self.birefringence_dim].data = value[chan]
                else:
                    cmap = 'gray' if key != 'Orientation' else 'hsv'
                    self.viewer.add_image(value[chan], name=key+self.birefringence_dim, colormap=cmap)

    @Slot(object)
    def handle_phase_image_update(self, value):

        name = 'Phase2D' if self.phase_dim == '2D' else 'Phase3D'

        # Add new layer if none exists, otherwise update layer data
        if name in self.viewer.layers:
            self.viewer.layers[name].data = value
        else:
            self.viewer.add_image(value, name=name, colormap='gray')

        if 'Phase' not in [self.ui.cb_saturation.itemText(i) for i in range(self.ui.cb_saturation.count())]:
            self.ui.cb_saturation.addItem('Retardance')
        if 'Phase' not in [self.ui.cb_value.itemText(i) for i in range(self.ui.cb_value.count())]:
            self.ui.cb_value.addItem('Retardance')

    @Slot(object)
    def handle_qlipp_reconstructor_update(self, value):
        # Saves phase reconstructor to be re-used if possible
        self.phase_reconstructor = value

    @Slot(dict)
    def handle_meta_update(self, meta):
        # Don't update microscope parameters saved in calibration metadata file

        # if self.last_calib_meta_file is None:
        #     print("\nWARNING: No calibration file has been loaded\n")
        #     return
        #
        # with open(self.last_calib_meta_file, 'r') as file:
        #     current_json = json.load(file)
        #
        # for key, value in current_json['Microscope Parameters'].items():
        #     if key in meta:
        #         current_json['Microscope Parameters'][key] = meta[key]
        #     else:
        #         current_json['Microscope Parameters'][key] = None
        #
        # with open(self.last_calib_meta_file, 'w') as file:
        #     json.dump(current_json, file, indent=1)

        pass

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
            self.reconstruction_data = WaveorderReader(self.reconstruction_data_path, 'zarr')
            self.viewer.add_image(self.reconstruction_data.get_zarr(p), name=layer_name + f'_Pos_{p:03d}')

            self.viewer.dims.set_axis_label(0, 'T')
            self.viewer.dims.set_axis_label(1, 'C')
            self.viewer.dims.set_axis_label(2, 'Z')


        # Add each new position as a new layer in napari
        name = layer_name + f'_Pos_{p:03d}'
        if name not in self.viewer.layers:
            self.reconstruction_data = WaveorderReader(self.reconstruction_data_path, 'zarr')
            self.viewer.add_image(self.reconstruction_data.get_zarr(p), name=name)

        # update the napari dimension slider position if the user hasn't specified to pause updates
        if not self.pause_updates:
            self.viewer.dims.set_current_step(0, t)
            self.viewer.dims.set_current_step(1, c)

        self.last_p = p

    @Slot(bool)
    def browse_dir_path(self):
        result = self._open_file_dialog(self.current_dir_path, 'dir')
        self.directory = result
        self.current_dir_path = result
        self.ui.le_directory.setText(result)
        self.ui.le_save_dir.setText(result)
        self.save_directory = result

    @Slot(bool)
    def browse_save_path(self):
        result = self._open_file_dialog(self.current_save_path, 'dir')
        self.save_directory = result
        self.current_save_path = result
        self.ui.le_save_dir.setText(result)

    @Slot(bool)
    def browse_data_dir(self):
        path = self._open_file_dialog(self.data_dir, 'dir')
        self.data_dir = path
        self.ui.le_data_dir.setText(self.data_dir)

    @Slot(bool)
    def browse_calib_meta(self):
        path = self._open_file_dialog(self.calib_path, 'file')
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
            self.ui.le_directory.setText('Path Does Not Exist')

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
            self.calib_scheme = '4-State'
        else:
            self.calib_scheme = '5-State'

    @Slot()
    def enter_calib_mode(self):
        index = self.ui.cb_calib_mode.currentIndex()
        if index == 0:
            self.calib_mode = 'MM-Retardance'
            self.ui.label_lca.hide()
            self.ui.label_lcb.hide()
            self.ui.cb_lca.hide()
            self.ui.cb_lcb.hide()
        elif index == 1:
            self.calib_mode = 'MM-Voltage'
            self.ui.label_lca.hide()
            self.ui.label_lcb.hide()
            self.ui.cb_lca.hide()
            self.ui.cb_lcb.hide()
        elif index == 2:
            self.calib_mode = 'DAC'
            self.ui.cb_lca.clear()
            self.ui.cb_lcb.clear()
            self.ui.cb_lca.show()
            self.ui.cb_lcb.show()
            self.ui.label_lca.show()
            self.ui.label_lcb.show()

            cfg = self.mmc.getConfigData(self.config_group, 'State0')

            # Update the DAC combo boxes with available DAC's from the config.  Necessary for the user
            # to specify which DAC output corresponds to which LC for voltage-space calibration
            memory = set()
            for i in range(cfg.size()):
                prop = cfg.getSetting(i)
                if 'TS_DAC' in prop.getDeviceLabel():
                    dac = prop.getDeviceLabel()[-2:]
                    if dac not in memory:
                        self.ui.cb_lca.addItem('DAC'+dac)
                        self.ui.cb_lcb.addItem('DAC'+dac)
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
        #if/else takes care of the clearing of config
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
            states = ['State0', 'State1', 'State2', 'State3', 'State4']
            missing = []
            for state in states:
                if state not in channels:
                    missing.append(state)

            # if states are missing, set the combo box red and alert the user
            if len(missing) != 0:
                msg = f'The chosen config group ({self.config_group}) is missing states: {missing}. '\
                    'Please refer to the recOrder wiki on how to set up the config properly.'

                self.ui.cb_config_group.setStyleSheet("border: 1px solid rgb(200,0,0);")
                raise KeyError(msg)
            else:
                self.ui.cb_config_group.setStyleSheet("")
       
    @Slot()
    def enter_use_cropped_roi(self):
        state = self.ui.chb_use_roi.checkState()
        if state == 2:
            self.use_cropped_roi = True
        elif state == 0:
            self.use_cropped_roi = False

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
            self.ui.le_save_dir.setText('Path Does Not Exist')

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
    def enter_birefringence_dim(self):
        state = self.ui.cb_birefringence.currentIndex()
        if state == 0:
            self.birefringence_dim = '2D'
        elif state == 1:
            self.birefringence_dim = '3D'

    @Slot()
    def enter_phase_dim(self):
        state = self.ui.cb_phase.currentIndex()
        if state == 0:
            self.phase_dim = '2D'
        elif state == 1:
            self.phase_dim = '3D'

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
            self.ui.le_bg_path.setText('Path Does Not Exist')

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
        result = self._open_file_dialog(self.current_bg_path, 'dir')
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
            self.bg_option = 'None'
        elif state == 1:
            self.ui.label_bg_path.setHidden(False)
            self.ui.le_bg_path.setHidden(False)
            self.ui.qbutton_browse_bg_path.setHidden(False)
            self.bg_option = 'global'
        elif state == 2:
            self.ui.label_bg_path.setHidden(True)
            self.ui.le_bg_path.setHidden(True)
            self.ui.qbutton_browse_bg_path.setHidden(True)
            self.bg_option = 'local_fit'
        elif state == 3:
            self.ui.label_bg_path.setHidden(False)
            self.ui.le_bg_path.setHidden(False)
            self.ui.qbutton_browse_bg_path.setHidden(False)
            self.bg_option = 'local_fit+'

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
            self.method = 'QLIPP'
            self.ui.label_bf_chan.hide()
            self.ui.le_bf_chan.hide()
            self.ui.label_chan_desc.setText('Retardance, Orientation, BF, Phase3D, Phase2D, S0, S1, S2, S3')

        elif idx == 1:
            self.method = 'PhaseFromBF'
            self.ui.label_bf_chan.show()
            self.ui.le_bf_chan.show()
            self.ui.label_bf_chan.setText('Brightfield Channel Index')
            self.ui.le_bf_chan.setPlaceholderText('int')
            self.ui.label_chan_desc.setText('Phase3D, Phase2D')

    @Slot(int)
    def enter_mode(self):
        idx = self.ui.cb_mode.currentIndex()

        if idx == 0:
            self.mode = '3D'
            self.ui.label_focus_zidx.hide()
            self.ui.le_focus_zidx.hide()
        else:
            self.mode = '2D'
            self.ui.label_focus_zidx.show()
            self.ui.le_focus_zidx.show()

    @Slot()
    def enter_data_dir(self):
        entry = self.ui.le_data_dir.text()
        if not os.path.exists(entry):
            self.ui.le_data_dir.setStyleSheet("border: 1px solid rgb(200,0,0);")
            self.ui.le_data_dir.setText('Path Does Not Exist')
        else:
            self.ui.le_data_dir.setStyleSheet("")
            self.data_dir = entry

    @Slot()
    def enter_calib_meta(self):
        entry = self.ui.le_calibration_metadata.text()
        if not os.path.exists(entry):
            self.ui.le_calibration_metadata.setStyleSheet("border: 1px solid rgb(200,0,0);")
            self.ui.le_calibration_metadata.setText('Path Does Not Exist')
        else:
            self.ui.le_calibration_metadata.setStyleSheet("")
            self.calib_path = entry

    @Slot()
    def enter_colormap(self):
        """
        Handles the update of the display colormap.  Will display different png image legend
        depending on the colormap choice.

        Returns
        -------

        """

        prev_cmap = self.colormap
        state = self.ui.cb_colormap.currentIndex()
        if state == 0:
            self.ui.label_orientation_image.setPixmap(self.jch_pixmap)
            self.colormap = 'JCh'
        else:
            self.ui.label_orientation_image.setPixmap(self.hsv_pixmap)
            self.colormap = 'HSV'

        # Update the birefringence overlay to new colormap if the colormap has changed
        if prev_cmap != self.colormap:
            #TODO: Handle case where there are multiple snaps
            if 'BirefringenceOverlay2D' in self.viewer.layers:
                if 'Retardance2D' in self.viewer.layers and 'Orientation2D' in self.viewer.layers:

                    overlay = ret_ori_overlay(retardance=self.viewer.layers['Retardance2D'].data,
                                              orientation=self.viewer.layers['Orientation2D'].data,
                                              ret_max= np.percentile(self.viewer.layers['Retardance2D'].data, 99.99),
                                              cmap=self.colormap)

                    self.viewer.layers['BirefringenceOverlay2D'].data = overlay

    @Slot(int)
    def enter_use_full_volume(self):
        state = self.ui.chb_display_volume.checkState()

        if state == 2:
            self.ui.le_overlay_slice.clear()
            self.ui.le_overlay_slice.setEnabled(False)
            self.use_full_volume = False
        else:
            self.ui.le_overlay_slice.setEnabled(True)
            self.use_full_volume = True

    @Slot()
    def enter_display_slice(self):
        slice = int(self.ui.le_overlay_slice.text())
        self.display_slice = slice

    @Slot()
    def enter_sat_min(self):
        val = float(self.ui.le_sat_min.text())
        slider_val = self.ui.slider_saturation.value()
        self.ui.slider_saturation.setValue((val, slider_val[1]))

    @Slot()
    def enter_sat_max(self):
        val = float(self.ui.le_sat_max.text())
        slider_val = self.ui.slider_saturation.value()
        self.ui.slider_saturation.setValue((slider_val[0], val))

    @Slot()
    def enter_val_min(self):
        val = float(self.ui.le_val_min.text())
        slider_val = self.ui.slider_value.value()
        self.ui.slider_value.setValue((val, slider_val[1]))

    @Slot()
    def enter_val_max(self):
        val = float(self.ui.le_val_max.text())
        slider_val = self.ui.slider_value.value()
        self.ui.slider_value.setValue((slider_val[0], val))

    @Slot(bool)
    def push_note(self):
        """
        Pushes a note to the last calibration metadata file.

        Returns
        -------

        """

        # make sure the user has performed a calibration in this session (or loaded a previous one)
        if not self.last_calib_meta_file:
            raise ValueError('No calibration has been performed yet so there is no previous metadata file')
        else:
            note = self.ui.le_notes_field.text()

            # Open the existing calibration metadata file and append the notes
            with open(self.last_calib_meta_file, 'r') as file:
                current_json = json.load(file)

            # Append note to the end of the old note (so we don't overwrite previous notes) or write a new
            # note in the blank notes field
            old_note = current_json['Notes']
            if old_note is None or old_note == '' or old_note == note:
                current_json['Notes'] = note
            else:
                current_json['Notes'] = old_note + ', ' + note

            # dump the contents into the metadata file
            with open(self.last_calib_meta_file, 'w') as file:
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
        set_lc_state(self.mmc, self.config_group, 'State0')
        extinction = snap_and_average(self.calib.snap_manager)
        set_lc_state(self.mmc, self.config_group, 'State1')
        state1 = snap_and_average(self.calib.snap_manager)

        # Calculate extinction based off captured intensities
        extinction = self.calib.calculate_extinction(self.swing, self.calib.I_Black, extinction, state1)
        self.ui.le_extinction.setText(str(extinction))

    @property
    def _microscope_params(self):
        """
        A dictionary containing microscope parameters from the current GUI
        Unused in 0.2.0 --- candidate for deletion
        """
        def _param_value(param_name: str, ui=self.ui):
            # refer to main widget class ui attribute names
            ui_attr_name = "le_" + param_name
            param_text = ui.__getattribute__(ui_attr_name).text()
            # handle blank string
            return float(param_text) if param_text != '' else None

        return {
            'n_objective_media': _param_value("n_media"),
            'objective_NA': _param_value("obj_na"),
            'condenser_NA': _param_value("cond_na"),
            'magnification': _param_value("mag"),
            'pixel_size': _param_value("ps")
        }

    @Slot(bool)
    def load_calibration(self):
        """
        Uses previous JSON calibration metadata to load previous calibration
        """

        metadata_path = self._open_file_dialog(self.current_dir_path, 'file')
        metadata = MetadataReader(metadata_path)

        # Update Properties
        self.wavelength = metadata.Wavelength
        self.swing = metadata.Swing

        # Initialize calibration class
        self.calib = QLIPP_Calibration(self.mmc, self.mm, group=self.config_group, lc_control_mode=self.calib_mode,
                                       interp_method=self.interp_method, wavelength=self.wavelength)
        self.calib.swing = self.swing
        self.ui.le_swing.setText(str(self.swing))
        self.calib.wavelength = self.wavelength
        self.ui.le_wavelength.setText(str(self.wavelength))

        # Update Calibration Scheme Combo Box
        if metadata.Calibration_scheme == '4-State':
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
        
        self.calib = QLIPP_Calibration(self.mmc, self.mm, group=self.config_group, lc_control_mode=self.calib_mode,
                                       interp_method=self.interp_method, wavelength=self.wavelength)

        if self.calib_mode == 'DAC':
            self.calib.set_dacs(self.lca_dac, self.lcb_dac)

        # Reset Styling
        self.ui.tb_calib_assessment.setText('')
        self.ui.tb_calib_assessment.setStyleSheet("")

        # Save initial autoshutter state for when we set it back later
        self.auto_shutter = self.mmc.getAutoShutter()

        logging.info('Starting Calibration')

        # Initialize displays + parameters for calibration
        self.ui.progress_bar.setValue(0)
        self.plot_item.clear()
        self.intensity_monitor = []
        self.calib.swing = self.swing
        self.calib.wavelength = self.wavelength
        self.calib.meta_file = os.path.join(self.directory, 'calibration_metadata.txt')

        # Make sure Live Mode is off
        if self.calib.snap_manager.getIsLiveModeOn():
            self.calib.snap_manager.setLiveModeOn(False)

        # Init Worker and Thread
        self.worker = CalibrationWorker(self, self.calib)

        # Connect Handlers
        self.worker.progress_update.connect(self.handle_progress_update)
        self.worker.extinction_update.connect(self.handle_extinction_update)
        self.worker.intensity_update.connect(self.handle_plot_update)
        self.worker.calib_assessment.connect(self.handle_calibration_assessment_update)
        self.worker.calib_assessment_msg.connect(self.handle_calibration_assessment_msg_update)
        self.worker.calib_file_emit.connect(self.handle_calib_file_update)
        self.worker.plot_sequence_emit.connect(self.handle_plot_sequence_update)
        self.worker.lc_states.connect(self.handle_lc_states_emit)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_error)
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)

        self.worker.start()

    @property
    def _channel_descriptions(self):
        return [
            self.mmc.getConfigData(self.config_group, calib_channel).getVerbose() 
            for calib_channel in self.calib_channels
        ]

    def _check_MM_config_setup(self):
        # Warns the user if the MM configuration is not correctly set up.
        desc = self._channel_descriptions
        if self.calib_mode == 'MM-Retardance':
            if all('String send to' in s for s in desc) and not any('Voltage (V)' in s for s in desc):
                return
            else:
                msg = ' \n'.join(textwrap.wrap("In \'MM-Retardance\' mode each preset must include the " \
                    "\'String send to\' property, and no \'Voltage\' properties.", width=40))
                show_warning(msg)
       
        elif self.calib_mode == 'MM-Voltage':
            if all('Voltage (V) LC-A' in s for s in desc) and all('Voltage (V) LC-B' in s for s in desc) and not any('String send to' in s for s in desc):
                return
            else:
                msg = ' \n'.join(textwrap.wrap("In \'MM-Voltage\' mode each preset must include the \'Voltage (V) LC-A\' " \
                    "property, the \'Voltage (V) LC-B\' property, and no \'String send to\' properties.", width=40))
                show_warning(msg)

        elif self.calib_mode == 'DAC':
            _devices = self.mmc.getLoadedDevices()
            loaded_devices = [_devices.get(i) for i in range(_devices.size())]
            if LC_DEVICE_NAME in loaded_devices:
                show_warning("In \'DAC\' mode the MeadowLarkLC device adapter must not be loaded in MM.")
        
        else:
            raise ValueError(f'self.calib_mode = {self.calib_mode} is an unrecognized state.')
        

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
        self.worker.bire_image_emitter.connect(self.handle_bg_bire_image_update)

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

        self._check_requirements_for_acq('birefringence')

        # Init Worker and thread
        self.worker = PolarizationAcquisitionWorker(self, self.calib, 'birefringence')

        # Connect Handlers
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.meta_emitter.connect(self.handle_meta_update)
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

        self._check_requirements_for_acq('phase')

        # Init worker and thread
        if self.ui.chb_phase_from_bf.isChecked():
            self.worker = BFAcquisitionWorker(self)
        else:
            self.worker = PolarizationAcquisitionWorker(self, self.calib, 'phase')

        # Connect Handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.phase_reconstructor_emitter.connect(self.handle_qlipp_reconstructor_update)
        self.worker.meta_emitter.connect(self.handle_meta_update)
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

        self._check_requirements_for_acq('phase')

        # Init worker
        # Init worker and thread
        self.worker = PolarizationAcquisitionWorker(self, self.calib, 'all')

        # connect handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.phase_reconstructor_emitter.connect(self.handle_qlipp_reconstructor_update)
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.meta_emitter.connect(self.handle_meta_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)
        self.ui.qbutton_stop_acq.clicked.connect(self.worker.quit)

        # Start Thread
        self.worker.start()

    @Slot(bool)
    def listen_and_reconstruct(self):
        """
        Wrapper function for on the fly data listening and reconstructing.  Only works if the user is acquiring
        polarization only data (cannot have any extra channels in the acquisition)

        Returns
        -------

        """

        # Init reconstructor
        if self.bg_option != 'None':
            metadata_file = get_last_metadata_file(self.current_bg_path)
            metadata = MetadataReader(metadata_file)
            roi = metadata.ROI
            height, width = roi[2], roi[3]
            bg_data = load_bg(self.current_bg_path, height, width, roi) # TODO: remove ROI for 1.0.0
        else:
            bg_data = None

        # Init worker
        self.worker = ListeningWorker(self, bg_data)

        # connect handlers
        self.worker.store_emitter.connect(self.add_listener_data)
        self.worker.dim_emitter.connect(self.update_dims)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.finished.connect(self._reset_listening)
        self.worker.errored.connect(self._handle_acq_error)
        self.ui.qbutton_stop_acq.clicked.connect(self.worker.quit)

        # Start Thread
        self.worker.start()

    @Slot(bool)
    def reconstruct(self):
        """
        Wrapper function for offline reconstruction.
        """

        success = self._check_requirements_for_reconstruction()
        if not success:
            raise ValueError('Please make sure all necessary parameters are set before reconstruction')

        self._populate_config_from_app()
        self.worker = ReconstructionWorker(self, self.config_reader)

        # connect handlers
        self.worker.dimension_emitter.connect(self.handle_reconstruction_dim_update)
        self.worker.store_emitter.connect(self.handle_reconstruction_store_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)
        self.ui.qbutton_stop_acq.clicked.connect(self.worker.quit)

        self.worker.start()

    @Slot(bool)
    def save_config(self):
        path = self._open_file_dialog(self.save_config_path, 'save')
        self.save_config_path = path
        name = PurePath(self.save_config_path).name
        dir_ = self.save_config_path.strip(name)
        self._populate_config_from_app()

        if isinstance(self.config_reader.positions, tuple):
            pos = self.config_reader.positions
            self.config_reader.positions = f'[!!python/tuple [{pos[0]},{pos[1]}]]'
        if isinstance(self.config_reader.timepoints, tuple):
            t = self.config_reader.timepoints
            self.config_reader.timepoints = f'[!!python/tuple [{t[0]},{t[1]}]]'

        self.config_reader.save_yaml(dir_=dir_, name=name)

    @Slot(bool)
    def load_config(self):
        """
        Populates the GUI elements with values from a pre-defined config file.

        Returns
        -------

        """
        path = self._open_file_dialog(self.save_config_path, 'file')
        if path == '':
            pass
        else:
            self.config_path = path
            self.config_reader = ConfigReader(self.config_path)
            self._populate_from_config()

    @Slot(bool)
    def load_default_config(self):
        self.config_reader = ConfigReader(self.default_offline_config)
        self._populate_from_config()

    @Slot(int)
    def update_sat_scale(self):
        idx = self.ui.cb_saturation.currentIndex()
        if idx != -1:
            layer = self.ui.cb_saturation.itemText(idx)
            data = self.viewer.layers[layer].data
            min_, max_ = np.min(data), np.max(data)
            self.ui.slider_saturation.setMinimum(min_)
            self.ui.slider_saturation.setMaximum(max_)
            self.ui.slider_saturation.setSingleStep((max_ - min_)/250)
            self.ui.slider_saturation.setValue((min_, max_))
            self.ui.le_sat_max.setText(str(np.round(max_, 3)))
            self.ui.le_sat_min.setText(str(np.round(min_, 3)))

    @Slot(int)
    def update_value_scale(self):
        idx = self.ui.cb_value.currentIndex()
        if idx != -1:
            layer = self.ui.cb_value.itemText(idx)
            data = self.viewer.layers[layer].data
            min_, max_ = np.min(data), np.max(data)
            self.ui.slider_value.setMinimum(min_)
            self.ui.slider_value.setMaximum(max_)
            self.ui.slider_value.setSingleStep((max_ - min_)/250)
            self.ui.slider_value.setValue((min_, max_))
            self.ui.le_val_max.setText(str(np.round(max_, 3)))
            self.ui.le_val_min.setText(str(np.round(min_, 3)))

    @Slot(bool)
    def create_overlay(self):
        """
        Creates HSV or JCh overlay with the specified channels from the combo boxes.  Will compute and then
        display the overlay in napari.

        Returns
        -------

        """

        if self.ui.cb_hue.count() == 0 or self.ui.cb_saturation.count() == 0 or self.ui.cb_value == 0:
            raise ValueError('Cannot create overlay until all 3 combo boxes are populated')

        # Gather channel data
        H = self.viewer.layers[self.ui.cb_hue.itemText(self.ui.cb_hue.currentIndex())].data
        S = self.viewer.layers[self.ui.cb_saturation.itemText(self.ui.cb_saturation.currentIndex())].data
        V = self.viewer.layers[self.ui.cb_value.itemText(self.ui.cb_value.currentIndex())].data


        #TODO: this is a temp fix which handles on data with n-dimensions of 4, 3, or 2 which automatically
        # chooses the first timepoint
        if H.ndim > 2 or S.ndim > 2 or V.ndim > 2:
            if H.ndim == 4:
                # assumes this is a (T, Z, Y, X) array read from napari-ome-zarr
                H = H[0, self.display_slice] if not self.use_full_volume else H[0]
            if S.ndim == 4:
                S = S[0, self.display_slice] if not self.use_full_volume else S[0]
            if V.ndim == 4:
                V = V[0, self.display_slice] if not self.use_full_volume else V[0]

            if H.ndim == 3:
                # assumes this is a (Z, Y, X) array collected from acquisition module
                H = H[self.display_slice] if not self.use_full_volume else H

            if S.ndim == 3:
                S = S[self.display_slice] if not self.use_full_volume else S

            if S.ndim == 3:
                S = S[self.display_slice] if not self.use_full_volume else S

        mode = '2D' if not self.use_full_volume else '3D'

        H_name = self.ui.cb_hue.itemText(self.ui.cb_hue.currentIndex())
        H_scale = (np.min(H), np.max(H)) if 'Orientation' not in H_name else (0, np.pi)
        S_scale = self.ui.slider_saturation.value()
        V_scale = self.ui.slider_value.value()

        hsv_image = generic_hsv_overlay(H, S, V, H_scale, S_scale, V_scale, mode=mode)

        # Create overlay layer name
        idx = 0
        while f'HSV_Overlay_{idx}' in self.viewer.layers:
            idx += 1

        # add overlay image to napari
        self.viewer.add_image(hsv_image, name=f'HSV_Overlay_{idx}', rgb=True)

    @Slot(object)
    def add_listener_data(self, store):

        self.viewer.add_image(store['Birefringence'], name=self.worker.prefix)
        self.viewer.dims.set_axis_label(0, 'P')
        self.viewer.dims.set_axis_label(1, 'T')
        self.viewer.dims.set_axis_label(2, 'C')
        self.viewer.dims.set_axis_label(3, 'Z')

    @Slot(tuple)
    def update_dims(self, dims):

        if not self.pause_updates:
            self.viewer.dims.set_current_step(0, dims[0])
            self.viewer.dims.set_current_step(1, dims[1])
            self.viewer.dims.set_current_step(3, dims[2])
        else:
            pass

    def _reset_listening(self):
        self.listening_reconstructor = None
        self.listening_store = None

    def _open_file_dialog(self, default_path, type):

        return self._open_dialog("select a directory",
                                 str(default_path),
                                 type)

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
        if type == 'dir':
            path = QFileDialog.getExistingDirectory(None,
                                                    title,
                                                    ref,
                                                    options=options)
        elif type == 'file':
            path = QFileDialog.getOpenFileName(None,
                                               title,
                                               ref,
                                               options=options)[0]
        elif type == 'save':
            path = QFileDialog.getSaveFileName(None,
                                               'Choose a save name',
                                               ref,
                                               options=options)[0]
        else:
            raise ValueError('Did not understand file dialogue type')

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

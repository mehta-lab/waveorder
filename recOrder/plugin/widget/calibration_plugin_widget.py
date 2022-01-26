from recOrder.calib.Calibration import QLIPP_Calibration
from pycromanager import Bridge
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap
from superqt import QLabeledRangeSlider, QRangeSlider
from recOrder.plugin.calibration.calibration_workers import CalibrationWorker, BackgroundCaptureWorker, load_calibration
from recOrder.plugin.acquisition.acquisition_workers import AcquisitionWorker, ListeningWorker
from recOrder.plugin.qtdesigner import recOrder_calibration_v5
from recOrder.postproc.post_processing import ret_ori_overlay
from recOrder.io.core_functions import set_lc_state, snap_and_average
from recOrder.io.utils import load_bg
from pathlib import Path
from napari import Viewer
import numpy as np
import os
import json
import logging


class Calibration(QWidget):

    # Initialize Signals
    mm_status_changed = pyqtSignal(bool)
    intensity_changed = pyqtSignal(float)
    log_changed = pyqtSignal(str)

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI Elements
        self.ui = recOrder_calibration_v5.Ui_Form()
        self.ui.setupUi(self)
        self._promote_slider()

        # Setup Connections between elements
        # Connect to MicroManager
        self.ui.qbutton_mm_connect.clicked[bool].connect(self.connect_to_mm)

        # Calibration Tab
        self.ui.qbutton_browse.clicked[bool].connect(self.browse_dir_path)
        self.ui.le_directory.editingFinished.connect(self.enter_dir_path)
        self.ui.le_swing.editingFinished.connect(self.enter_swing)
        self.ui.le_wavelength.editingFinished.connect(self.enter_wavelength)
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
        # self.ui.chb_save_imgs.stateChanged[int].connect(self.enter_save_imgs)
        self.ui.le_save_dir.editingFinished.connect(self.enter_save_path)
        self.ui.qbutton_listen.clicked[bool].connect(self.listen_and_reconstruct)
        self.ui.le_zstart.editingFinished.connect(self.enter_zstart)
        self.ui.le_zend.editingFinished.connect(self.enter_zend)
        self.ui.le_zstep.editingFinished.connect(self.enter_zstep)
        self.ui.chb_use_gpu.stateChanged[int].connect(self.enter_use_gpu)
        self.ui.le_gpu_id.editingFinished.connect(self.enter_gpu_id)
        self.ui.le_obj_na.editingFinished.connect(self.enter_obj_na)
        self.ui.le_cond_na.editingFinished.connect(self.enter_cond_na)
        self.ui.le_mag.editingFinished.connect(self.enter_mag)
        self.ui.le_ps.editingFinished.connect(self.enter_ps)
        self.ui.le_n_media.editingFinished.connect(self.enter_n_media)
        self.ui.le_pad_z.editingFinished.connect(self.enter_pad_z)
        self.ui.chb_pause_updates.stateChanged[int].connect(self.enter_pause_updates)
        self.ui.cb_birefringence.currentIndexChanged[int].connect(self.enter_birefringence_dim)
        self.ui.cb_phase.currentIndexChanged[int].connect(self.enter_phase_dim)
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

        # Logging
        log_box = QtLogger(self.ui.te_log)
        log_box.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_box)
        logging.getLogger().setLevel(logging.INFO)

        # Signal Emitters
        self.mm_status_changed.connect(self.handle_mm_status_update)

        # Instantiate Attributes:
        self.gui_mode = 'offline'
        self.mm = None
        self.mmc = None
        self.calib = None
        self.current_dir_path = str(Path.home())
        self.current_save_path = str(Path.home())
        self.current_bg_path = str(Path.home())
        self.directory = None

        # Reconstruction / Calibration Parameter Defaults
        self.swing = 0.1
        self.wavelength = 532
        self.calib_scheme = '4-State'
        self.calib_mode = 'retardance'
        self.config_group = 'Channel'
        self.last_calib_meta_file = None
        self.use_cropped_roi = False
        self.bg_folder_name = 'BG'
        self.n_avg = 5
        self.intensity_monitor = []
        self.save_imgs = False
        self.save_directory = None
        self.bg_option = 'None'
        self.birefringence_dim = '2D'
        self.phase_dim = '2D'
        self.z_start = None
        self.z_end = None
        self.z_step = None
        self.gpu_id = 0
        self.use_gpu = False
        self.obj_na = None
        self.cond_na = None
        self.mag = None
        self.ps = None
        self.n_media = 1.003
        self.pad_z = 0
        self.phase_reconstructor = None
        self.acq_bg_directory = None
        self.auto_shutter = True
        self.lca_dac = None
        self.lcb_dac = None
        self.pause_updates = False
        self.colormap = 'JCh'
        self.use_full_volume = False
        self.display_slice = None

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

        # Display Options
        recorder_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        jch_legend_path = os.path.join(recorder_dir, 'docs/images/JCh_legend.png')
        hsv_legend_path = os.path.join(recorder_dir, 'docs/images/HSV_legend.png')
        self.jch_pixmap = QPixmap(jch_legend_path)
        self.hsv_pixmap = QPixmap(hsv_legend_path)
        self.ui.label_orientation_image.setPixmap(self.jch_pixmap)

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
        self.ui.le_itr.setHidden(True)

        # Set initial UI Properties
        self.ui.le_gui_mode.setStyleSheet("border: 1px solid rgb(200,0,0); color: rgb(200,0,0);")
        self.ui.te_log.setStyleSheet('background-color: rgb(32,34,40);')

        # disable wheel events for combo boxes
        for attr_name in dir(self.ui):
            if 'cb_' in attr_name:
                attr = getattr(self.ui, attr_name)
                attr.wheelEvent = lambda event: None

    def _promote_slider(self):

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
        self.ui.slider_saturation = QRangeSlider(getattr(self.ui, saturation_slider_parent))
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

        self.ui.slider_value = QRangeSlider(getattr(self.ui, value_slider_parent))
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
        self.ui.acq_settings.setHidden(val)
        self.ui.acquire.setHidden(val)

        # Calibration Tab
        self.ui.tabWidget.setTabEnabled(0, not val)
        if val:
            self.ui.tabWidget.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        else:
            self.ui.tabWidget.setStyleSheet("")
            self.mm_status_changed.emit(False)
            self.mmc = None
            self.mm = None
            self.ui.cb_config_group.clear()

    def _hide_offline_ui(self, val: bool):

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

        # Processing Settings
        self.ui.tabWidget_3.setTabEnabled(1, not val)
        if val:
            self.ui.tabWidget_3.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        else:
            self.ui.tabWidget_3.setStyleSheet("")

        # Physical Parameters
        self.ui.le_recon_wavelength.setHidden(val)
        self.ui.label_recon_wavelength.setHidden(val)

        # Regularization
        self.ui.fluorescence.setHidden(val)

        # Pre/Post Processing
        self.ui.tabWidget_3.setTabEnabled(4, not val)
        if val:
            self.ui.tabWidget_3.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        else:
            self.ui.tabWidget_3.setStyleSheet("")

        self.ui.tabWidget_3.setTabEnabled(5, not val)
        if val:
            self.ui.tabWidget_3.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        else:
            self.ui.tabWidget_3.setStyleSheet("")

    def _enable_buttons(self):

        self.ui.qbutton_calibrate.setEnabled(True)
        self.ui.qbutton_capture_bg.setEnabled(True)
        self.ui.qbutton_calc_extinction.setEnabled(True)
        self.ui.qbutton_acq_birefringence.setEnabled(True)
        self.ui.qbutton_acq_phase.setEnabled(True)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(True)
        self.ui.qbutton_load_calib.setEnabled(True)
        self.ui.qbutton_listen.setEnabled(True)
        self.ui.qbutton_create_overlay.setEnabled(True)

    def _disable_buttons(self):
        self.ui.qbutton_calibrate.setEnabled(False)
        self.ui.qbutton_capture_bg.setEnabled(False)
        self.ui.qbutton_calc_extinction.setEnabled(False)
        self.ui.qbutton_acq_birefringence.setEnabled(False)
        self.ui.qbutton_acq_phase.setEnabled(False)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(False)
        self.ui.qbutton_load_calib.setEnabled(False)
        self.ui.qbutton_listen.setEnabled(False)
        self.ui.qbutton_create_overlay.setEnabled(False)

    def _handle_error(self, exc):
        self.ui.tb_calib_assessment.setText(f'Error: {str(exc)}')
        self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid rgb(200,0,0);")

        if self.use_cropped_roi:
            self.mmc.clearROI()

        self.mmc.setAutoShutter(self.auto_shutter)
        self.ui.progress_bar.setValue(0)
        raise exc

    def _handle_calib_abort(self):
        if self.use_cropped_roi:
            self.mmc.clearROI()
        self.mmc.setAutoShutter(self.auto_shutter)
        self.ui.progress_bar.setValue(0)

    def _handle_acq_error(self, exc):
        raise exc

    def _handle_load_finished(self):
        self.ui.tb_calib_assessment.setText('Previous calibration successfully loaded')
        self.ui.tb_calib_assessment.setStyleSheet("border: 1px solid green;")
        self.ui.progress_bar.setValue(100)

    def _update_calib(self, val):
        self.calib = val

    def _add_layer_to_display_boxes(self, val):
        for layer in self.viewer.layers:
            if layer.name not in [self.ui.cb_hue.itemText(i) for i in range(self.ui.cb_hue.count())]:
                self.ui.cb_hue.addItem(layer.name)
            if layer.name not in [self.ui.cb_saturation.itemText(i) for i in range(self.ui.cb_saturation.count())]:
                self.ui.cb_saturation.addItem(layer.name)
            if layer.name not in [self.ui.cb_value.itemText(i) for i in range(self.ui.cb_value.count())]:
                self.ui.cb_value.addItem(layer.name)

    def _remove_layer_from_display_boxes(self, val):

        for i in range(self.ui.cb_hue.count()):
            if val.value.name in self.ui.cb_hue.itemText(i):
                self.ui.cb_hue.removeItem(i)
            if val.value.name in self.ui.cb_saturation.itemText(i):
                self.ui.cb_saturation.removeItem(i)
            if val.value.name in self.ui.cb_value.itemText(i):
                self.ui.cb_value.removeItem(i)

    @pyqtSlot(bool)
    def change_gui_mode(self):
        if self.gui_mode == 'offline':
            self.ui.qbutton_gui_mode.setText('Switch to Offline')
            self.ui.le_gui_mode.setText('Online')
            self.ui.le_gui_mode.setStyleSheet("border: 1px solid green; color: green;")
            self._hide_offline_ui(True)
            self._hide_acquisition_ui(False)
            self.gui_mode = 'online'
        else:
            self.ui.qbutton_gui_mode.setText('Switch to Online')
            self.ui.le_gui_mode.setText('Offline')
            self.ui.le_gui_mode.setStyleSheet("border: 1px solid rgb(200,0,0); color: rgb(200,0,0);")
            self._hide_offline_ui(False)
            self._hide_acquisition_ui(True)
            self.gui_mode = 'offline'

    @pyqtSlot(bool)
    def connect_to_mm(self):
        try:
            bridge = Bridge(convert_camel_case=False)
            self.mmc = bridge.get_core()
            self.mm = bridge.get_studio()
            self.ui.cb_config_group.clear()
            groups = self.mmc.getAvailableConfigGroups()
            group_list = []
            for i in range(groups.size()):
                group_list.append(groups.get(i))
            self.ui.cb_config_group.addItems(group_list)

            self.mm_status_changed.emit(True)
        except:
            self.mm_status_changed.emit(False)

    @pyqtSlot(bool)
    def handle_mm_status_update(self, value):
        if value:
            self.ui.le_mm_status.setText('Sucess!')
            self.ui.le_mm_status.setStyleSheet("background-color: green;")
            self.gui_mode = 'online'
            self._hide_acquisition_ui(False)
            self._hide_offline_ui(True)
            self.ui.qbutton_gui_mode.setText('Switch to Offline')
            self.ui.le_gui_mode.setText('Online')
        else:
            self.ui.le_mm_status.setText('Failed.')
            self.ui.le_mm_status.setStyleSheet("background-color: rgb(200,0,0);")

    @pyqtSlot(int)
    def handle_progress_update(self, value):
        self.ui.progress_bar.setValue(value)

    @pyqtSlot(str)
    def handle_extinction_update(self, value):
        self.ui.le_extinction.setText(value)

    @pyqtSlot(object)
    def handle_plot_update(self, value):
        self.intensity_monitor.append(value)
        self.ui.plot_widget.plot(self.intensity_monitor)

        #
        # state = self.plot_item.getViewBox().getState()
        # view_range = state['viewRange']
        #
        # print(state)

        if self.plot_sequence[0] == 'Coarse':
            self.plot_item.autoRange()
        else:
            self.plot_item.setRange(xRange=(self.plot_sequence[0], len(self.intensity_monitor)),
                                    yRange=(0, np.max(self.intensity_monitor[self.plot_sequence[0]:])))
            # self.plot_item.autoRange()

    @pyqtSlot(str)
    def handle_calibration_assessment_update(self, value):
        self.calib_assessment_level = value

    @pyqtSlot(str)
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

    @pyqtSlot(object)
    def handle_bg_image_update(self, value):

        if 'Background Images' in self.viewer.layers:
            self.viewer.layers['Background Images'].data = value
        else:
            self.viewer.add_image(value, name='Background Images', colormap='gray')

    @pyqtSlot(object)
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

    @pyqtSlot(object)
    def handle_bire_image_update(self, value):

        channel_names = {'Orientation': 1,
                         'Retardance': 0,
                         }

        # Compute Overlay if birefringence acquisition is 2D
        if self.birefringence_dim == '2D':
            channel_names['BirefringenceOverlay'] = None
            overlay = ret_ori_overlay(retardance=value[0],
                                      orientation=value[1],
                                      scale=(0, np.percentile(value[0], 99.99)),
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

        # if self.ui.DisplayOptions.isHidden():
        #     self.ui.DisplayOptions.show()

        if 'Orientation' not in [self.ui.cb_hue.itemText(i) for i in range(self.ui.cb_hue.count())]:
            self.ui.cb_hue.addItem('Orientation')
        if 'Retardance' not in [self.ui.cb_saturation.itemText(i) for i in range(self.ui.cb_saturation.count())]:
            self.ui.cb_saturation.addItem('Retardance')
        if 'Retardance' not in [self.ui.cb_value.itemText(i) for i in range(self.ui.cb_value.count())]:
            self.ui.cb_value.addItem('Retardance')

    @pyqtSlot(object)
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

    @pyqtSlot(object)
    def handle_reconstructor_update(self, value):
        # Saves phase reconstructor to be re-used if possible
        self.phase_reconstructor = value

    @pyqtSlot(str)
    def handle_calib_file_update(self, value):
        self.last_calib_meta_file = value

    @pyqtSlot(str)
    def handle_plot_sequence_update(self, value):
        current_idx = len(self.intensity_monitor)
        self.plot_sequence = (value, current_idx)

    @pyqtSlot(tuple)
    def handle_sat_slider_move(self, value):
        self.ui.label_sat_min.setText(str(value[0]))
        self.ui.label_sat_max.setText(str(value[1]))

    @pyqtSlot(tuple)
    def handle_val_slider_move(self, value):
        self.ui.label_val_min.setText(str(value[0]))
        self.ui.label_val_max.setText(str(value[1]))

    @pyqtSlot(bool)
    def browse_dir_path(self):
        result = self._open_browse_dialog(self.current_dir_path)
        self.directory = result
        self.current_dir_path = result
        self.ui.le_directory.setText(result)

    @pyqtSlot(bool)
    def browse_save_path(self):
        result = self._open_browse_dialog(self.current_save_path)
        self.save_directory = result
        self.current_save_path = result
        self.ui.le_save_path.setText(result)

    @pyqtSlot()
    def enter_dir_path(self):
        path = self.ui.le_directory.text()
        if os.path.exists(path):
            self.directory = path
        else:
            self.ui.le_directory.setText('Path Does Not Exist')

    @pyqtSlot()
    def enter_swing(self):
        self.swing = float(self.ui.le_swing.text())

    @pyqtSlot()
    def enter_wavelength(self):
        self.wavelength = int(self.ui.le_wavelength.text())

    @pyqtSlot()
    def enter_calib_scheme(self):
        index = self.ui.cb_calib_scheme.currentIndex()
        if index == 0:
            self.calib_scheme = '4-State'
        else:
            self.calib_scheme = '5-State'

    @pyqtSlot()
    def enter_calib_mode(self):
        index = self.ui.cb_calib_mode.currentIndex()
        if index == 0:
            self.calib_mode = 'retardance'
            self.ui.label_lca.hide()
            self.ui.label_lcb.hide()
            self.ui.cb_lca.hide()
            self.ui.cb_lcb.hide()
        else:
            self.calib_mode = 'voltage'
            self.ui.cb_lca.clear()
            self.ui.cb_lcb.clear()
            self.ui.cb_lca.show()
            self.ui.cb_lcb.show()
            self.ui.label_lca.show()
            self.ui.label_lcb.show()

            cfg = self.mmc.getConfigData('Channel', 'State0')

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

    @pyqtSlot()
    def enter_dac_lca(self):
        dac = self.ui.cb_lca.currentText()
        self.lca_dac = dac

    @pyqtSlot()
    def enter_dac_lcb(self):
        dac = self.ui.cb_lcb.currentText()
        self.lcb_dac = dac

    @pyqtSlot()
    def enter_config_group(self):
        self.config_group = self.ui.cb_config_group.currentText()

    @pyqtSlot()
    def enter_use_cropped_roi(self):
        state = self.ui.chb_use_roi.checkState()
        if state == 2:
            self.use_cropped_roi = True
        elif state == 0:
            self.use_cropped_roi = False

    @pyqtSlot()
    def enter_bg_folder_name(self):
        self.bg_folder_name = self.ui.le_bg_folder.text()

    @pyqtSlot()
    def enter_n_avg(self):
        self.n_avg = int(self.ui.le_n_avg.text())

    @pyqtSlot()
    def enter_log_level(self):
        index = self.ui.cb_loglevel.currentIndex()
        if index == 0:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.DEBUG)

    @pyqtSlot()
    def enter_save_imgs(self):
        state = self.ui.chb_save_imgs.checkState()
        if state == 2:
            self.save_imgs = True
        elif state == 0:
            self.save_imgs = False

    @pyqtSlot()
    def enter_save_path(self):
        path = self.ui.le_save_path.text()
        if os.path.exists(path):
            self.save_directory = path
            self.current_save_path = path
        else:
            self.ui.le_save_path.setText('Path Does Not Exist')

    @pyqtSlot()
    def enter_zstart(self):
        self.z_start = float(self.ui.le_zstart.text())

    @pyqtSlot()
    def enter_zend(self):
        self.z_end = float(self.ui.le_zend.text())

    @pyqtSlot()
    def enter_zstep(self):
        self.z_step = float(self.ui.le_zstep.text())

    @pyqtSlot()
    def enter_birefringence_dim(self):
        state = self.ui.cb_birefringence.currentIndex()
        if state == 0:
            self.birefringence_dim = '2D'
        elif state == 1:
            self.birefringence_dim = '3D'

    @pyqtSlot()
    def enter_phase_dim(self):
        state = self.ui.cb_phase.currentIndex()
        if state == 0:
            self.phase_dim = '2D'
        elif state == 1:
            self.phase_dim = '3D'

    @pyqtSlot()
    def enter_acq_bg_path(self):
        path = self.ui.le_bg_path.text()
        if os.path.exists(path):
            self.acq_bg_directory = path
            self.current_bg_path = path
        else:
            self.ui.le_bg_path.setText('Path Does Not Exist')

    @pyqtSlot(bool)
    def browse_acq_bg_path(self):
        result = self._open_browse_dialog(self.current_bg_path)
        self.acq_bg_directory = result
        self.current_bg_path = result
        self.ui.le_bg_path.setText(result)

    @pyqtSlot()
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
            self.bg_option = 'Global'
        elif state == 2:
            self.ui.label_bg_path.setHidden(False)
            self.ui.le_bg_path.setHidden(False)
            self.ui.qbutton_browse_bg_path.setHidden(False)
            self.bg_option = 'local_fit'

    @pyqtSlot()
    def enter_gpu_id(self):
        self.gpu_id = int(self.ui.le_gpu_id.text())

    @pyqtSlot()
    def enter_use_gpu(self):
        state = self.ui.chb_use_gpu.checkState()
        if state == 2:
            self.use_gpu = True
        elif state == 0:
            self.use_gpu = False

    @pyqtSlot()
    def enter_obj_na(self):
        self.obj_na = float(self.ui.le_obj_na.text())

    @pyqtSlot()
    def enter_cond_na(self):
        self.cond_na = float(self.ui.le_cond_na.text())

    @pyqtSlot()
    def enter_mag(self):
        self.mag = float(self.ui.le_mag.text())

    @pyqtSlot()
    def enter_ps(self):
        self.ps = float(self.ui.le_ps.text())

    @pyqtSlot()
    def enter_n_media(self):
        self.n_media = float(self.ui.le_n_media.text())

    @pyqtSlot()
    def enter_pad_z(self):
        self.pad_z = int(self.ui.le_pad_z.text())

    @pyqtSlot()
    def enter_pause_updates(self):
        state = self.ui.chb_pause_updates.checkState()
        if state == 2:
            self.pause_updates = True
        elif state == 0:
            self.pause_updates = False

    @pyqtSlot()
    def enter_colormap(self):
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
                                              scale=(0, np.percentile(self.viewer.layers['Retardance2D'].data, 99.99)),
                                              cmap=self.colormap)

                    self.viewer.layers['BirefringenceOverlay2D'].data = overlay

    @pyqtSlot(int)
    def enter_use_full_volume(self):
        state = self.ui.chb_display_volume.checkState()

        if state == 2:
            self.ui.le_overlay_slice.clear()
            self.ui.le_overlay_slice.setEnabled(False)
            self.use_full_volume = False
        else:
            self.ui.le_overlay_slice.setEnabled(True)
            self.use_full_volume = True

    @pyqtSlot()
    def enter_display_slice(self):
        slice = int(self.ui.le_overlay_slice.text())
        self.display_slice = slice

    @pyqtSlot(bool)
    def push_note(self):

        if not self.last_calib_meta_file:
            raise ValueError('No calibration has been performed yet so there is no previous metadata file')
        else:
            note = self.ui.le_notes_field.text()

            with open(self.last_calib_meta_file, 'r') as file:
                current_json = json.load(file)

            old_note = current_json['Notes']
            if old_note is None:
                current_json['Notes'] = note
            else:
                current_json['Notes'] = old_note + ',' + note

            with open(self.last_calib_meta_file, 'w') as file:
                json.dump(current_json, file, indent=1)

    @pyqtSlot(bool)
    def calc_extinction(self):

        # Snap images from the extinction state and first elliptical state
        set_lc_state(self.mmc, self.config_group, 'State0')
        extinction = snap_and_average(self.calib.snap_manager)
        set_lc_state(self.mmc, self.config_group, 'State1')
        state1 = snap_and_average(self.calib.snap_manager)

        # Calculate extinction based off captured intensities
        extinction = self.calib.calculate_extinction(self.swing, self.calib.I_Black, extinction, state1)
        self.ui.le_extinction.setText(str(extinction))

    @pyqtSlot(bool)
    def load_calibration(self):
        """
        Uses previous JSON calibration metadata to load previous calibration
        """
        result = self._open_browse_dialog(self.current_dir_path, file=True)
        with open(result, 'r') as file:
            meta = json.load(file)

        # Update Properties
        self.wavelength = meta['Summary']['Wavelength (nm)']
        self.swing = meta['Summary']['Swing (fraction)']

        # Initialize calibration class
        self.calib = QLIPP_Calibration(self.mmc, self.mm, group=self.config_group)
        self.calib.swing = self.swing
        self.ui.le_swing.setText(str(self.swing))
        self.calib.wavelength = self.wavelength
        self.ui.le_wavelength.setText(str(self.wavelength))

        # Update Calibration Scheme Combo Box
        if meta['Summary']['Acquired Using'] == '4-State':
            self.ui.cb_calib_scheme.setCurrentIndex(0)
        else:
            self.ui.cb_calib_scheme.setCurrentIndex(1)

        self.last_calib_meta_file = result

        # Move the load calibration function to a separate thread
        self.worker = load_calibration(self.calib, meta)

        # initialize worker properties
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)
        self.worker.yielded.connect(self.ui.le_extinction.setText)
        self.worker.returned.connect(self._update_calib)
        self.worker.errored.connect(self._handle_error)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.finished.connect(self._handle_load_finished)
        self.worker.start()

    @pyqtSlot(bool)
    def run_calibration(self):
        """
        Wrapper function to create calibration worker and move that worker to a thread.
        Calibration is then executed by the calibration worker
        """

        self.calib = QLIPP_Calibration(self.mmc, self.mm, mode=self.calib_mode)

        if self.calib_mode == 'voltage':
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
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_error)
        self.ui.qbutton_stop_calib.clicked.connect(self.worker.quit)

        self.worker.start()

    @pyqtSlot(bool)
    def capture_bg(self):
        """
        Wrapper function to capture a set of background images.  Will snap images and display reconstructed
        birefringence.  Check connected handlers for napari display.

        Returns
        -------

        """

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

        # Start Capture Background Thread
        self.worker.start()

    @pyqtSlot(bool)
    def acq_birefringence(self):
        """
        Wrapper function to acquire birefringence stack/image and plot in napari
        Returns
        -------

        """

        # Init Worker and thread
        self.worker = AcquisitionWorker(self, self.calib, 'birefringence')

        # Connect Handler
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)

        # Start Thread
        self.worker.start()

    @pyqtSlot(bool)
    def acq_phase(self):
        """
        Wrapper function to acquire phase stack and plot in napari
        """

        # Init worker and thread
        self.worker = AcquisitionWorker(self, self.calib, 'phase')

        # Connect Handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.phase_reconstructor_emitter.connect(self.handle_reconstructor_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)

        # Start thread
        self.worker.start()

    @pyqtSlot(bool)
    def acq_birefringence_phase(self):
        """
        Wrapper function to acquire both birefringence and phase stack and plot in napari
        """

        # Init worker
        self.worker = AcquisitionWorker(self, self.calib, 'all')

        # connect handlers
        self.worker.phase_image_emitter.connect(self.handle_phase_image_update)
        self.worker.bire_image_emitter.connect(self.handle_bire_image_update)
        self.worker.phase_reconstructor_emitter.connect(self.handle_reconstructor_update)
        self.worker.started.connect(self._disable_buttons)
        self.worker.finished.connect(self._enable_buttons)
        self.worker.errored.connect(self._handle_acq_error)
        self.ui.qbutton_stop_acq.clicked.connect(self.worker.quit)

        # Start Thread
        self.worker.start()

    @pyqtSlot(bool)
    def listen_and_reconstruct(self):

        # Init reconstructor
        if self.bg_option != 'None':
            with open(os.path.join(self.current_bg_path, 'calibration_metadata.txt')) as file:
                js = json.load(file)
                roi = js['Summary']['ROI Used (x, y, width, height)']
                height, width = roi[2], roi[3]
            bg_data = load_bg(self.current_bg_path, height, width, roi)
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

    @pyqtSlot(bool)
    def create_overlay(self):

        if self.display_slice is None and not self.use_full_volume:
            raise ValueError('Please specify a slice to display or choose to use the entire volume')
        else:
            if self.ui.cb_hue.count() == 0 or self.ui.cb_saturation.count() != 2 or self.ui.cb_value != 2:
                raise ValueError('Cannot create overlay until Orientation, Retardance, and Phase are available')
            else:
                pass

    @pyqtSlot(object)
    def add_listener_data(self, store):

        self.viewer.add_image(store['Birefringence'], name=self.worker.prefix)
        self.viewer.dims.set_axis_label(0, 'P')
        self.viewer.dims.set_axis_label(1, 'T')
        self.viewer.dims.set_axis_label(2, 'C')
        self.viewer.dims.set_axis_label(3, 'Z')

    @pyqtSlot(tuple)
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

    def _open_browse_dialog(self, default_path, file=False):

        if not file:
            return self._open_dir_dialog("select a directory",
                                         default_path)
        else:
            return self._open_file_dialog('Please select a file',
                                          default_path)

    def _open_dir_dialog(self, title, ref):
        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,
                                                title,
                                                ref,
                                                options=options)
        return path

    def _open_file_dialog(self, title, ref):
        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getOpenFileName(None,
                                           title,
                                           ref,
                                           options=options)[0]
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

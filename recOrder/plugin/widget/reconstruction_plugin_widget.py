from pycromanager import Bridge
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QLineEdit, QCheckBox
from PyQt5.QtGui import QColor
from recOrder.plugin.widget.thread_worker import ThreadWorker
from recOrder.plugin.qtdesigner import recOrder_reconstruction
from recOrder.plugin.widget.loading_widget import Overlay
from recOrder.io.config_reader import ConfigReader, DATASET, PROCESSING, PREPROCESSING, POSTPROCESSING
from pathlib import Path
from napari import Viewer
import os


class Reconstruction(QWidget):

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI Elements
        self.ui = recOrder_reconstruction.Ui_Form()
        self.ui.setupUi(self)

        self.overlay = Overlay(self)
        self.overlay.hide()

        #QT Styling
        self.red_text = QColor(200, 0, 0, 255)
        self.original_tab_text = self.ui.tabWidget_4.tabBar().tabTextColor(0)

        # variables
        self.data_dir = Path.home()
        self.save_dir = Path.home()
        self.bg_path = Path.home()
        self.calib_path = Path.home()
        self.config_path = Path.home()
        self.save_config_path = Path.home()
        self.mode = '3D'
        self.method = 'QLIPP'

        self.config_reader = None

        # self.ui.qb_reconstruct.setStyleSheet("QPushButton {background-color: 1px solid rgb(62,161,193)}"
        #                                      "QPushButton:pressed {background-color: 1px solid rgb(99,179,205)}")

        # Setup Connections between elements
        # Recievers

        # =================================
        # File Dialog Buttons
        self.ui.qb_browse_data_dir.clicked[bool].connect(self.browse_data_dir)
        self.ui.qb_browse_save_dir.clicked[bool].connect(self.browse_save_dir)
        self.ui.qb_browse_bg_path.clicked[bool].connect(self.browse_bg_path)
        self.ui.qb_browse_calib_meta.clicked[bool].connect(self.browse_calib_meta)
        self.ui.qb_load_config.clicked[bool].connect(self.load_config)
        self.ui.qb_save_config.clicked[bool].connect(self.save_config)

        # Other Buttons
        self.ui.qb_load_default_config.clicked[bool].connect(self.load_default_config)
        self.ui.qb_reconstruct.clicked[bool].connect(self.run_reconstruction)


        # Line Edit Checks
        self.ui.le_data_dir.editingFinished.connect(self.enter_data_dir)
        self.ui.le_save_dir.editingFinished.connect(self.enter_save_dir)
        self.ui.le_calibration_metadata.editingFinished.connect(self.enter_calib_meta)
        self.ui.le_background.editingFinished.connect(self.enter_bg_path)

        # Combo Box Checks
        self.ui.cb_method.currentIndexChanged[int].connect(self.enter_method)
        self.ui.cb_mode.currentIndexChanged[int].connect(self.enter_mode)

        # Line Edits
        self.attrs = dir(self.ui)
        for attr in self.attrs:
            if attr.startswith('le_'):
                le = getattr(self.ui, attr)
                le.editingFinished.connect(self.reset_le_style)

    def resizeEvent(self, event):
        self.overlay.resize(event.size())
        event.accept()

    def _open_file_dialog(self, default_path, type):

        return self._open_dialog("select a directory",
                                 str(default_path),
                                 type)

    def _open_dialog(self, title, ref, type):
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

    def _set_tab_red(self, name, state):
        name_map = {'Processing': 0,
                    'Physical': 1,
                    'Regularization': 2,
                    'preprocessing': 3,
                    'postprocessing': 4}

        index = name_map[name]

        if state:
            self.ui.tabWidget_4.tabBar().setTabTextColor(index, self.red_text)
        else:
            self.ui.tabWidget_4.tabBar().setTabTextColor(index, self.original_tab_text)


    def _check_line_edit(self, name):

        le = getattr(self.ui, f'le_{name}')
        text = le.text()

        if text == '':
            le.setStyleSheet("border: 1px solid rgb(200,0,0);")
            return False
        else:
            return True


    def _check_requirements(self):
        output_channels = self.ui.le_output_channels.text()
        output_channels.replace(' ', '')
        output_channels = output_channels.strip(',')

        always_required = {'data_dir', 'save_dir', 'positions', 'timepoints', 'output_channels'}
        birefringence_required = {'calibration_metadata', 'wavelength'}
        phase_required = {'wavelength', 'magnification', 'NA_objective', 'NA_condenser', 'n_objective_media',
                          'phase_strength'}
        fluor_decon_required = {'wavelength', 'magnification', 'NA_objective', 'n_objective_media', 'reg'}

        cont = True
        for field in always_required:
            cont = self._check_line_edit(field)
            if not cont:
                return False
            else:
                continue

        if self.method == 'QLIPP':
            if 'Retardance' in output_channels or 'Orientation' in output_channels or 'BF' in output_channels:
                for field in birefringence_required:
                    cont = self._check_line_edit(field)
                    tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                    if not cont:
                        self._set_tab_red(tab, True)
                        return False
                    else:
                        continue
            elif 'Phase2D' in output_channels or 'Phase3D' in output_channels:
                for field in phase_required:
                    cont = self._check_line_edit(field)
                    tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                    if not cont:
                        self._set_tab_red(tab, True)
                        return False
                    else:
                        continue
                if 'Phase2D' in output_channels:
                    cont = self._check_line_edit('focus_zidx')
                    if not cont:
                        self._set_tab_red('Physical', True)
                        return False

            else:
                self._set_tab_red('Processing', True)
                self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                print('User did not specify any QLIPP Specific Channels')
                return False

        elif self.method == 'PhaseFromBF':
            if 'Phase2D' in output_channels or 'Phase3D' in output_channels:
                for field in phase_required:
                    cont = self._check_line_edit(field)
                    tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                    if not cont:
                        self._set_tab_red(tab, True)
                        return False
                    else:
                        continue
                if 'Phase2D' in output_channels:
                    cont = self._check_line_edit('focus_zidx')
                    if not cont:
                        self._set_tab_red('Physical', True)
                        return False
            else:
                self._set_tab_red('Processing', True)
                self.ui.le_output_channels.setStyleSheet("border: 1px solid rgb(200,0,0);")
                print('User did not specify any PhaseFromBF Specific Channels (Phase2D, Phase3D)')
                return False

        elif self.method == 'FluorDeconv':
            for field in fluor_decon_required:
                cont = self._check_line_edit(field)
                tab = getattr(self.ui, f'le_{field}').parent().parent().objectName()
                if not cont:
                    self._set_tab_red(tab, True)
                    return False
                else:
                    continue

        else:
            print('Error in parameter checks')
            self.ui.qb_reconstruct.setStyleSheet("border: 1px solid rgb(200,0,0);")

    def _populate_config_from_app(self):
        cfg_dict = {'dataset': {},
                     'pre_processing': {},
                     'processing': {},
                     'post_processing': {}}

        # Parse dataset fields manually
        self.cfg_dict['dataset']['data_dir'] = self.data_dir
        self.cfg_dict['dataset']['save_dir'] = self.save_dir
        self.cfg_dict['dataset']['method'] = self.method
        self.cfg_dict['dataset']['mode'] = self.mode
        self.cfg_dict['dataset']['data_save_name'] = self.data_save_name
        self.cfg_dict['dataset']['calibration_metadata'] = self.ui.le_calibration_metadata.text()
        self.cfg_dict['dataset']['background'] = self.ui.le_calibration_metadata.text()

        #TODO: Figure out how to parse positions/timepoints
        positions = self.ui.le_positions.text()
        timepoints = self.ui.le_timepoints.text()

        for key, value in PREPROCESSING.items():
            if isinstance(value, dict):
                cfg_dict['pre_processing'][key] = {}
                for key_child, value_child in PREPROCESSING[key].items():
                    field = getattr(self.ui, f'le_preproc_{key}_{key_child}')
                    if key_child == 'use':
                        cfg_dict['pre_processing'][key][key_child] = field.checkState()
                    else:
                        cfg_dict['pre_processing'][key][key_child] = field.text()
            else:
                cfg_dict['pre_processing'][key] = getattr(self, key)

        for key, value in PROCESSING.items():

            if key == 'background_correction':
                bg_map = {0: 'None', 1: 'global', 2: 'local_fit'}
                cfg_dict['processing'][key] = bg_map[self.ui.cb_background_correction.currentIndex()]
            elif key == 'output_channels':
                field_text = self.ui.le_output_channels.text()
                channels = field_text.replace(' ', '')
                channels = channels.split(',')
                cfg_dict['processing'][key] = channels
            else:
                attr_name = f'le_{key}'
                if attr_name in self.attrs:
                    le = getattr(self.ui, attr_name)
                    cfg_dict['processing'][key] = int(le.text())

                else:
                    continue


        # Parse Postprocessing automatically
        for key, val in POSTPROCESSING.items():
            for key_child, val_child in val.items():
                if key == 'deconvolution':
                    if key_child == 'use':
                        cb = getattr(self.ui, f'chb_postproc_fluor_{key_child}')
                        cfg_dict['post_processing']['deconvolution'][key_child] = cb.checkState()
                    if hasattr(self.ui, f'le_postproc_fluor_{key_child}'):
                        # TODO: Parse wavelengths and channel indices in a smart manor
                        le = getattr(self.ui, f'le_postproc_fluor_{key_child}')
                        cfg_dict['post_processing'][key][key_child] = le.text()

                elif key == 'registration':
                    if key_child == 'use':
                        cb = getattr(self.ui, 'chb_postproc_registration_use')
                        cfg_dict['post_processing']['registration'][key_child] = cb.checkState()
                    else:
                        le = getattr(self.ui, f'le_postproc_reg_{key_child}')
                        cfg_dict['post_processing'][key][key_child] = le.text()

                elif key == 'denoise':
                    if key_child == 'use':
                        cb = getattr(self.ui, 'chb_postproc_denoise_use')
                        cfg_dict['post_processing']['denoise'][key_child] = cb.checkState()
                    else:
                        le = getattr(self.ui, f'le_postproc_denoise_{key_child}')
                        cfg_dict['post_processing'][key][key_child] = le.text()


    def _populate_from_config(self):

        # Parse dataset fields manually
        self.data_dir = self.config_reader.data_dir
        self.ui.le_data_dir.setText(self.config_reader.data_dir)
        self.save_dir = self.config_reader.save_dir
        self.ui.le_save_dir.setText(self.config_reader.save_dir)
        self.ui.le_data_save_name.setText(self.config_reader.data_save_name)
        self.ui.le_calibration_metadata.setText(self.config_reader.calibration_metadata)
        self.ui.le_background.setText(self.config_reader.background)

        self.mode = self.config_reader.mode
        self.ui.cb_mode.setCurrentIndex(0) if self.mode == '3D' else self.ui.cb_mode.setCurrentIndex(1)
        self.method = self.config_reader.method
        if self.method == 'QLIPP':
            self.ui.cb_method.setCurrentIndex(0)
        elif self.method == 'PhaseFromBF':
            self.ui.cb_method.setCurrentIndex(1)
        elif self.method == 'FluorDeconv':
            self.ui.cb_method.setCurrentIndex(2)
        else:
            print(f'Did not understand method from config: {self.method}')
            self.ui.cb_method.setStyleSheet("border: 1px solid rgb(200,0,0);")

        self.ui.le_positions.setText(str(self.config_reader.positions))
        self.ui.le_timepoints.setText(str(self.config_reader.timepoints))

        # Parse Preprocessing automatically
        for key, val in PREPROCESSING.items():
            for key_child, val_child in val.items():
                if key_child == 'use':
                    attr = getattr(self.config_reader.preprocessing, 'denoise_use')
                    self.ui.chb_preproc_denoise_use.setCheckState(attr)
                else:
                    le = getattr(self.ui, f'le_preproc_denoise_{key_child}')
                    le.setText(str(getattr(self.config_reader.preprocessing, f'denoise_{key_child}')))

        # Parse processing automatically
        for key, val in PROCESSING.items():
            if hasattr(self.ui, f'le_{key}'):
                le = getattr(self.ui, f'le_{key}')
                le.setText(str(getattr(self.config_reader, key)))

            elif hasattr(self.ui, f'cb_{key}'):
                cb = getattr(self.ui, f'cb_{key}')
                items = [cb.itemText(i) for i in range(cb.count())]
                cfg_attr = getattr(self.config_reader, key)
                self.ui.cb_mode.setCurrentIndex(items.index(cfg_attr))
            elif key == 'phase_denoiser_2D' or key == 'phase_denosier_3D':
                cb = self.ui.cb_phase_denoiser
                cfg_attr = getattr(self.config_reader, f'phase_denoiser_{self.mode}')
                cb.setCurrentIndex(0) if cfg_attr == 'Tikhonov' else cb.setCurrentIndex(1)
            #TODO: DENOISER STRENGTH PARSING
            else:
                pass

        # Parse Postprocessing automatically
        for key, val in POSTPROCESSING.items():
            for key_child, val_child in val.items():
                if key == 'deconvolution':
                    if key_child == 'use':
                        attr = getattr(self.config_reader.postprocessing, 'registration_use')
                        self.ui.chb_preproc_denoise_use.setCheckState(attr)
                    if hasattr(self.ui, f'le_postproc_fluor_{key_child}'):
                        le = getattr(self.ui, f'le_postproc_fluor_{key_child}')
                        attr = str(getattr(self.config_reader.postprocessing, f'{key}_{key_child}'))
                        le.setText(attr)

                elif key == 'registration':
                    if key_child == 'use':
                        attr = getattr(self.config_reader.postprocessing, 'registration_use')
                        self.ui.chb_postproc_reg_use.setCheckState(attr)
                    else:
                        le = getattr(self.ui, f'le_postproc_reg_{key_child}')
                        attr = str(getattr(self.config_reader.postprocessing, f'registration_{key_child}'))
                        le.setText(attr)

                elif key == 'denoise':
                    if key_child == 'use':
                        attr = getattr(self.config_reader.postprocessing, 'denoise_use')
                        self.ui.chb_postproc_denoise_use.setCheckState(attr)
                    else:
                        le = getattr(self.ui, f'le_postproc_denoise_{key_child}')
                        attr = str(getattr(self.config_reader.postprocessing, f'denoise_{key_child}'))
                        le.setText(attr)

    @pyqtSlot()
    def reset_style(self):
        sender = self.sender()
        sender.setStyleSheet("")

    @pyqtSlot(bool)
    def run_reconstruction(self):
        le = self.ui.le_wavelength
        # group_box = self.ui.ReconstructionParams.setStyle()
        print(le.parent().parent().objectName())
        # self.ui.tabWidget_4.tabBar().
        tab_bar = self.ui.tabWidget_4.tabBar().setTabTextColor(0, QColor(200, 0, 0, 255))
        le.setStyleSheet("border: 1px solid rgb(200,0,0);")
        # tab.setTabTextColor(0, 'red')
        # tab.setStyleSheet("QTabWidget {border: 1px solid rgb(200,0,0)};")
        # self.overlay.show()
        # self._populate_config_from_app()


    @pyqtSlot(bool)
    def browse_data_dir(self):
        path = self._open_file_dialog(self.data_dir, 'dir')
        self.data_dir = path
        self.ui.le_data_dir.setText(self.data_dir)

    @pyqtSlot(bool)
    def browse_save_dir(self):
        path = self._open_file_dialog(self.save_dir, 'dir')
        self.save_dir = path
        self.ui.le_save_dir.setText(self.save_dir)

    @pyqtSlot(bool)
    def browse_bg_path(self):
        path = self._open_file_dialog(self.bg_path, 'dir')
        self.bg_path = path
        self.ui.le_background.setText(self.bg_path)

    @pyqtSlot(bool)
    def browse_calib_meta(self):
        path = self._open_file_dialog(self.calib_path, 'dir')
        self.calib_path = path
        self.ui.le_calibration_metadata.setText(self.calib_path)

    @pyqtSlot(bool)
    def save_config(self):
        path = self._open_file_dialog(self.save_config_path, 'save')
        self.save_config_path = path

    @pyqtSlot(bool)
    def load_config(self):
        path = self._open_file_dialog(self.save_config_path, 'file')
        if path == '':
            pass
        else:
            self.config_path = path
            self.config_reader = ConfigReader(self.config_path)
            self._populate_from_config()

    @pyqtSlot(bool)
    def load_default_config(self):
        self.config_reader = ConfigReader(mode='3D', method='QLIPP')
        self._populate_from_config()

    @pyqtSlot()
    def enter_data_dir(self):
        entry = self.ui.le_data_dir.text()
        if not os.path.exists(entry):
            self.ui.le_data_dir.setStyleSheet("border: 1px solid rgb(200,0,0);")
            self.ui.le_data_dir.setText('Path Does Not Exist')
        else:
            self.ui.le_data_dir.setStyleSheet("")
            self.data_dir = entry

    @pyqtSlot()
    def enter_save_dir(self):
        entry = self.ui.le_data_dir.text()
        self.save_dir = entry

    @pyqtSlot()
    def enter_calib_meta(self):
        entry = self.ui.le_calibration_metadata.text()
        if not os.path.exists(entry):
            self.ui.le_calibration_metadata.setStyleSheet("border: 1px solid rgb(200,0,0);")
            self.ui.le_calibration_metadata.setText('Path Does Not Exist')
        else:
            self.ui.le_calibration_metadata.setStyleSheet("")
            self.calib_path = entry

    @pyqtSlot()
    def enter_bg_path(self):
        entry = self.ui.le_background.text()
        if not os.path.exists(entry):
            self.ui.le_background.setStyleSheet("border: 1px solid rgb(200,0,0);")
            self.ui.le_background.setText('Path Does Not Exist')
        else:
            self.ui.le_background.setStyleSheet("")
            self.bg_path = entry

    @pyqtSlot(int)
    def enter_method(self):
        idx = self.ui.cb_method.currentIndex()

        if idx == 0:
            self.mode = 'QLIPP'
        elif idx == 1:
            self.mode = 'PhaseFromBF'
        else:
            self.mode = 'FluorDeconv'

    @pyqtSlot(int)
    def enter_mode(self):
        idx = self.ui.cb_mode.currentIndex()

        if idx == 0:
            self.mode = '3D'
        else:
            self.mode = '2D'



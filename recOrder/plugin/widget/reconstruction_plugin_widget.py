from pycromanager import Bridge
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from recOrder.plugin.widget.thread_worker import ThreadWorker
from recOrder.plugin.qtdesigner import recOrder_reconstruction
from recOrder.io.config_reader import ConfigReader, DATASET, PROCESSING, PREPROCESSING, POSTPROCESSING
from pathlib import Path
from napari import Viewer
import os
import logging


class Reconstruction(QWidget):

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Setup GUI Elements
        self.ui = recOrder_reconstruction.Ui_Form()
        self.ui.setupUi(self)

        # variables
        self.data_dir = Path.home()
        self.save_dir = Path.home()
        self.bg_path = Path.home()
        self.calib_path = Path.home()
        self.config_path = Path.home()
        self.save_config_path = Path.home()

        self.config_reader = None

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


        # File Settings Line Edits
        # self.ui.le_data_dir.editingFinished

    def _open_file_dialog(self, default_path, type):

        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        if type == 'dir':
            path = QFileDialog.getExistingDirectory(None,
                                                    'Select a directory',
                                                    default_path,
                                                    options=options)
            return path

        elif type == 'file':
            path = QFileDialog.getOpenFileName(None,
                                               'Select a file',
                                               default_path,
                                               options=options)[0]
            return path


        elif type == 'save':
            path = QFileDialog.getSaveFileName(None,
                                               'Choose a save name',
                                               default_path,
                                               options=options
                                               )[0]

            return path

        else:
            raise ValueError('Did not understand file dialogue type')


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
        self.ui.cb_mode.setCurrentIndex(0) if self.mode == '3D' else self.cb.setCurrentIndex(1)
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
                    le.setText(str(getattr(self.config_reader.preprocessing, f'denoise_{key_child')))

        # Parse processing automatically
        for key, val in PROCESSING.items():
            if hasattr(self.ui, f'le_{key}'):
                le = getattr(self.ui, f'le_{key}')
                le.setText(getattr(self.config_reader, key))

            elif hasattr(self.ui, f'cb_{key}'):
                cb = getattr(self.ui, f'cb_{key}')
                items = [cb.itemText(i) for i in range(cb.count())]
                cfg_attr = getattr(self.config_reader, key)
                self.ui.cb_mode.setCurrentIndex(items.index(cfg_attr))
            elif key == 'phase_denoiser_2D' or key == 'phase_denosier_3D':
                cb = self.ui.cb_phase_denoiser
                cfg_attr = getattr(self.config_reader, f'phase_denoiser_{self.mode}')
                cb.setCurrentIndex(0) if cfg_attr == 'Tikhonov' else cb.setCurrentIndex(1)
            # elif key == ''



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
        self.ui.le_bg_path.setText(self.bg_path)

    @pyqtSlot(bool)
    def browse_calib_meta(self):
        path = self._open_file_dialog(self.calib_path, 'dir')
        self.calib_path = path
        self.ui.le_calib_meta.setText(self.calib_path)

    @pyqtSlot(bool)
    def save_config(self):
        path = self._open_file_dialog(self.save_config_path, 'save')
        self.save_config_path = path

    @pyqtSlot(bool)
    def load_config(self):
        path = self._open_file_dialog(self.save_config_path, 'file')
        self.config_path = path
        self.config_reader = ConfigReader(self.config_path)
        self._populate_config()


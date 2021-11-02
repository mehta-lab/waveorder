from pycromanager import Bridge
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from recOrder.plugin.widget.thread_worker import ThreadWorker
from recOrder.plugin.qtdesigner import recOrder_reconstruction
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

        # Setup Connections between elements
        # Recievers
        # =================================
        # Connect to Micromanager
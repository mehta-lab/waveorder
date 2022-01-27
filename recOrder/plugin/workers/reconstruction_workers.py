from PyQt5.QtCore import pyqtSignal
from recOrder.compute.qlipp_compute import initialize_reconstructor, \
    reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, reconstruct_phase2D, reconstruct_phase3D
from recOrder.acq.acq_functions import generate_acq_settings, acquire_from_settings
from recOrder.io.utils import load_bg, extract_reconstruction_parameters
from recOrder.compute import QLIPPBirefringenceCompute
from napari.qt.threading import WorkerBaseSignals, WorkerBase
import logging
from waveorder.io.writer import WaveorderWriter
import tifffile as tiff
import json
import numpy as np
import os
import zarr
import shutil
import time
import glob

class ReconstructionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    phase_image_emitter = pyqtSignal(object)
    bire_image_emitter = pyqtSignal(object)
    phase_reconstructor_emitter = pyqtSignal(object)
    meta_emitter = pyqtSignal(dict)
    aborted = pyqtSignal()

# class ReconstructionWorker(WorkerBase):
#
#     def __init__(self):




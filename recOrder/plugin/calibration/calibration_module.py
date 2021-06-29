from recOrder.calib.Calibration import QLIPP_Calibration
from pycromanager import Bridge
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from recOrder.plugin.qtdesigner.recOrder_calibration_v4 import Ui_Form

class CalibrationModule(QtCore.QObject):
    # emitters
    progress_value_changed = QtCore.pyqtSignal(int)
    extinction_value_changed = QtCore.pyqtSignal(float)
    mm_status_changed = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.mm = None
        self.mmc = None
        self.calib_working = False
        self.calib = QLIPP_Calibration(mmc=None, mm=None, print_details=True)

        # Connections
        self.ui.qbutton_mm_connect.clicked.connect(self.connect_to_mm)
        self.mm_status_changed.connect(self._handle_mm_status_update)

    @pyqtSlot(bool)
    def connect_to_mm(self):
        print('here')
        try:
            bridge = Bridge(convert_camel_case=False)
            self.mmc = bridge.get_core()
            self.mm = bridge.get_studio()

            self.calib_working = True
            self.mm_status_changed.emit(self.calib_working)
        except:
            self.calib_working = False
            self.mm_status_changed.emit(self.calib_working)

    @pyqtSlot(bool)
    def _handle_mm_status_update(self, value):
        if value:
            self.ui.le_mm_status.setText('Sucess!')
        else:
            self.ui.le_mm_status.setText('Failed.')

    def browse_dir_path(self):
        pass

    def set_calib_roi(self):
        pass

    def set_dir_path(self):
        pass

    def set_swing(self):
        pass

    def set_wavelength(self):
        pass

    def set_calib_scheme(self):
        pass

    def run_calibration(self):
        pass

    def handle_progress_update(self):
        pass

    def handle_extinction_update(self):
        pass


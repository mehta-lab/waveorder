from recOrder.plugin.qtdesigner.recOrder_calibration_v4 import Ui_Form
from recOrder.plugin.qtdesigner import recOrder_calibration_v4
from recOrder.plugin.calibration.calibration_module import CalibrationModule
from PyQt5.QtCore import pyqtSlot
from qtpy.QtWidgets import QWidget

class SignalManager(QWidget):
    """
    manages signal connections between certain GUI elements and their corresponding functions

    """
    def __init__(self, ui):
        self.ui = ui
        self.ui.qbutton_mm_connect.clicked[bool].connect(self.connect_to_mm)
        funcs_module.mm_status_changed.connect(funcs_module._handle_mm_status_update)

    @pyqtSlot(bool)
    def connect_to_mm(self):
        print('here')
        # try:
        #     bridge = Bridge(convert_camel_case=False)
        #     self.mmc = bridge.get_core()
        #     self.mm = bridge.get_studio()
        #
        #     # self.calib_working = True
        #     self.mm_status_changed.emit(True)
        # except:
        #     # self.calib_working = False
        #     self.mm_status_changed.emit(False)



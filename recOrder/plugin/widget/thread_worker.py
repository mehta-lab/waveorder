from PyQt5.QtCore import QThread
from PyQt5 import QtCore


class ThreadWorker(QtCore.QObject):

    def __init__(self, widget, worker):
        super().__init__()
        self.ui = widget
        self.worker = worker
        self.thread = None

    def initalize(self):
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.thread.started.connect(self._enable_buttons)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def _enable_buttons(self):
        self.ui.qbutton_calibrate.setEnabled(True)
        self.ui.qbutton_capture_bg.setEnabled(True)
        self.ui.qbutton_calc_extinction.setEnabled(True)
        self.ui.qbutton_acq_birefringence.setEnabled(True)
        self.ui.qbutton_acq_phase.setEnabled(True)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(True)

    def _disable_buttons(self):
        self.ui.qbutton_calibrate.setEnabled(False)
        self.ui.qbutton_capture_bg.setEnabled(False)
        self.ui.qbutton_calc_extinction.setEnabled(False)
        self.ui.qbutton_acq_birefringence.setEnabled(False)
        self.ui.qbutton_acq_phase.setEnabled(False)
        self.ui.qbutton_acq_birefringence_phase.setEnabled(False)
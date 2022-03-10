from qtpy.QtCore import Qt, QRect
from qtpy.QtGui import QPalette, QPainter, QBrush, QColor, QPen
from qtpy.QtWidgets import QWidget, QPushButton

class Overlay(QWidget):
    """
    ADAPTED FROM AYDIN https://github.com/royerlab/aydin/blob/master/aydin/gui/_qt/custom_widgets/overlay.py
    """
    def __init__(self, parent=None):

        # super().__init__(self, parent)
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(self.parent.ui.qb_reconstruct.rect(), QBrush(QColor(40, 45, 60, 197)))
        # text_rect = QRect(self.width()//2, self.height()//2,
        #                   self.width()//2, self.height()//3)
        # text_pen = QPen()
        # text_pen.setWidth(100)
        # text_pen.setColor(QColor(0, 191, 255))
        # painter.setPen(text_pen)
        #
        # text = "Reconstructing..."
        # painter.drawText(self.width()//2,
        #                  self.height()//2,
        #                  "Reconstructing...")
        # painter.drawText(text_rect,
        #                  Qt.AlignVCenter,
        #                  "Reconstructing...")

        painter.setPen(QPen(Qt.NoPen))
        for i in range(5):
            if (self.counter / 5) % 5 >= i:
                painter.setBrush(QBrush(QColor(0, 191, 255)))
            else:
                painter.setBrush(QBrush(QColor(197, 197, 197)))
            painter.drawEllipse(
                self.width() // 2 + 50 * i - 100, self.height() // 2, 20, 20
            )

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(100)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()

    def hideEvent(self, event):
        self.killTimer(self.timer)
        self.hide()
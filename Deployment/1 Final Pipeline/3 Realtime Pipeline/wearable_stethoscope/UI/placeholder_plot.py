from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

class PlaceholderPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor(12, 14, 18)); self.setPalette(pal)
        label = QtWidgets.QLabel()
        label.setStyleSheet("color: #7aa2f7; font-size: 15px;"); label.setAlignment(Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self); layout.addStretch(); layout.addWidget(label); layout.addStretch()

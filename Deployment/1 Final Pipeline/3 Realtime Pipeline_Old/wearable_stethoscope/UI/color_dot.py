from PyQt5 import QtWidgets, QtGui

class ColorDot(QtWidgets.QFrame):
    def __init__(self, width=120, height=40, parent=None):  # bigger to visually "fit"
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.set_off()
    def set_on(self):
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor(230, 40, 40)); self.setPalette(pal); self.update()
    def set_off(self):
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor(250, 250, 250)); self.setPalette(pal); self.update()

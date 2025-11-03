from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

class DiagnosisTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Class", "Prob. (%)"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setStyleSheet("QHeaderView::section{font-weight:600;}")
        for cls in ["Healthy", "Wheezing", "Crackle", "Bronchi"]:
            self.add_row(cls, 0)
        #self._fit_height()
    def add_row(self, cls_name, prob):
        r = self.rowCount(); self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(cls_name))
        pitem = QtWidgets.QTableWidgetItem(f"{prob:>3d}"); pitem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(r, 1, pitem)
        #self._fit_height()
    def set_probs(self, probs_dict):
        for r in range(self.rowCount()):
            cls_name = self.item(r, 0).text()
            prob = int(round(probs_dict.get(cls_name, 0) * 100))
            self.item(r, 1).setText(str(prob))
    def _fit_height(self):
        self.resizeRowsToContents()
        header_h = self.horizontalHeader().height()
        rows_h = sum(self.rowHeight(i) for i in range(self.rowCount()))
        total = header_h + rows_h + self.frameWidth()*2 + 2
        self.setFixedHeight(total)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

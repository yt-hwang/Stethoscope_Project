import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from BLE.bluetooth_manager import BluetoothManager
from Streaming.stream_manager import StreamManager
from UI.color_dot import ColorDot
from ML_Diagnosis.diagnosis_table import DiagnosisTable


class WearableStethoUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wearable Stethoscope — UI Mockup (Wide Plot, v2)")
        self.resize(1140, 400)
        self._build_ui(); self._apply_style()

        self.ble = BluetoothManager()
        self.ble.devices_found.connect(self._on_devices_found)
        self.ble.connected.connect(lambda addr: print(f"[✓] Connected to {addr}"))
        self.ble.error.connect(lambda msg: print(f"[!] {msg}"))
        self.devices = {}
        self.stream = StreamManager(
            ble_manager=self.ble,
            ax=self.ax,
            line=self.line,
            canvas=self.canvas
        )

        self.btn_scan.clicked.connect(lambda: self.ble.scan_devices(timeout=3.0))
        self.btn_connect.clicked.connect(self._connect_selected)
        self.btn_start.clicked.connect(self.stream.start_stream)
        self.btn_stop.clicked.connect(self.stream.stop_stream)
        self.btn_save.clicked.connect(self.stream.save_data)

    def _build_ui(self):
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(12)
        root.setVerticalSpacing(6)

        # ── Section 1: Top controls ──────────────────────────────────────────────
        self.cmb_ble = QtWidgets.QComboBox()
        self.cmb_ble.setFixedWidth(400)
        self.btn_scan = QtWidgets.QPushButton("scan")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_start   = QtWidgets.QPushButton("Start")
        self.btn_stop    = QtWidgets.QPushButton("Stop")
        self.btn_save    = QtWidgets.QPushButton("Save")

        sec1 = QtWidgets.QHBoxLayout()
        sec1.addWidget(self.cmb_ble)
        sec1.addSpacing(8)
        sec1.addWidget(self.btn_scan)
        sec1.addSpacing(8)
        sec1.addWidget(self.btn_connect)
        sec1.addSpacing(8)
        sec1.addWidget(self.btn_start)
        sec1.addSpacing(8)
        sec1.addWidget(self.btn_stop)
        sec1.addSpacing(8)
        sec1.addWidget(self.btn_save)
        sec1.addStretch()
        root.addLayout(sec1, 0, 0, 1, 2)

        # ── Sections 2~5 ────────────────────────────────────────────────────────
        # Section 2: Signal
        box2 = QtWidgets.QGroupBox("Signal (real-time)")
        box2.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v2 = QtWidgets.QVBoxLayout(box2)
        v2.setContentsMargins(6, 6, 6, 6)

        # streaming Plot in section 2
        self.fig = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot([], [], lw=1, color="#f8e803")

        # Theme of plot
        self.ax.set_facecolor("#0b0c10")
        self.fig.patch.set_facecolor("#0b0c10")
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        for side in ['top', 'right']:
            self.ax.spines[side].set_visible(False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Amplitude (V)", color='white')

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        v2.addWidget(self.canvas, stretch=1)


        # Section 3: Diagnosis
        box3 = QtWidgets.QGroupBox("Real-time Diagnosis")
        box3.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v3 = QtWidgets.QVBoxLayout(box3)
        v3.setSpacing(1)
        self.tbl_diag = DiagnosisTable()
        self.tbl_diag.setMaximumHeight(180)   # 얇게
        v3.addWidget(self.tbl_diag, stretch=20)
        box3.setMinimumWidth(260)

        # Section 4: Interactive
        box4 = QtWidgets.QGroupBox("Interactive Feedback")
        box4.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v4 = QtWidgets.QVBoxLayout(box4)
        v4.setContentsMargins(8, 8, 8, 8)
        v4.setSpacing(8)
        row_alert = QtWidgets.QHBoxLayout()
        row_alert.setSpacing(6)
        row_alert.setContentsMargins(0, 0, 0, 0)
        self.lbl_alert = QtWidgets.QLabel("Alert")
        self.lbl_alert.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.dot_alert = ColorDot(width=200, height=40)
        row_alert.addWidget(self.lbl_alert)
        row_alert.addSpacing(8)
        row_alert.addWidget(self.dot_alert)
        row_alert.addStretch()
        v4.addLayout(row_alert)
        self.btn_tap    = QtWidgets.QPushButton("Tap")
        self.btn_music  = QtWidgets.QPushButton("Music")
        self.btn_guided = QtWidgets.QPushButton("Guided Breath")
        v4.addWidget(self.btn_tap)
        v4.addWidget(self.btn_music)
        v4.addWidget(self.btn_guided)

        # Section 5: Vitals
        box5 = QtWidgets.QGroupBox("Vitals (real-time)")
        box5.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        h5 = QtWidgets.QHBoxLayout(box5)
        h5.setContentsMargins(6, 6, 6, 6)
        self.card_hr = self._metric_card("Heart Rate", "-- bpm")
        self.card_br = self._metric_card("Breathing Rate", "-- bpm")
        h5.addWidget(self.card_hr, 1)
        h5.addWidget(self.card_br, 1)

        # Splitters (keep references)
        left_split  = QtWidgets.QSplitter(Qt.Vertical)   # (Signal | Vitals)
        right_split = QtWidgets.QSplitter(Qt.Vertical)   # (Diagnosis | Interactive)
        left_split.addWidget(box2)
        left_split.addWidget(box5)
        right_split.addWidget(box3)
        right_split.addWidget(box4)

        main_split = QtWidgets.QSplitter(Qt.Horizontal)  # (Left | Right)
        main_split.addWidget(left_split)
        main_split.addWidget(right_split)

        self.box2, self.box3, self.box4, self.box5 = box2, box3, box4, box5
        self.left_split, self.right_split, self.main_split = left_split, right_split, main_split
        root.addWidget(main_split, 1, 0, 1, 2)

        # initial sizes
        SIGNAL_H   = 405
        VITALS_H   = 30
        DIAG_H     = 160
        INTER_H    = 80
        LEFT_W     = 860
        RIGHT_W    = 200
        self.left_split.setSizes([SIGNAL_H, VITALS_H])
        self.right_split.setSizes([DIAG_H, INTER_H])
        self.main_split.setSizes([LEFT_W, RIGHT_W])
        self.resize(LEFT_W + RIGHT_W, SIGNAL_H + VITALS_H + 20)

    def _metric_card(self, title, value_text):
        card = QtWidgets.QFrame(); card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        v = QtWidgets.QVBoxLayout(card)
        lab_title = QtWidgets.QLabel(title); lab_title.setStyleSheet("font-weight:600; color:#c0caf5;")
        lab_val = QtWidgets.QLabel(value_text); lab_val.setObjectName("value"); lab_val.setStyleSheet("font-size:22px; font-weight:700;")
        v.addWidget(lab_title); v.addWidget(lab_val); v.addStretch()
        return card

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #0b0c10; color: #e5e9f0; }
            QGroupBox { border: 1px solid #2b3a4a; border-radius: 8px; margin-top: 10px; padding: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; top: -4px; color: #8aadf4; }
            QPushButton { background-color: #1f2833; border: 1px solid #3a4b5c; border-radius: 8px; padding: 8px 14px; }
            QPushButton:hover { border-color: #6b8fb3; }
            QComboBox { background-color: #1f2833; border: 1px solid #3a4b5c; border-radius: 8px; padding: 6px 10px; }
            QLabel#value { color: #a6e3a1; }
            QTableWidget { background-color: #12151b; border: 1px solid #2b3a4a; }
            QHeaderView::section { background: #1b222c; color: #d8dee9; }
        """)

    def _on_devices_found(self, dev_map: dict):
        self.devices = dev_map or {}
        self.cmb_ble.clear()
        if self.devices:
            self.cmb_ble.addItems(self.devices.keys())
        else:
            self.cmb_ble.addItem("No devices")

    def _connect_selected(self):
        name = self.cmb_ble.currentText()
        addr = self.devices.get(name)
        if addr:
            self.ble.connect_device(addr)   

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = WearableStethoUI(); ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

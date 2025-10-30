
import sys
import asyncio
import threading
import os
import time
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from bleak import BleakScanner, BleakClient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE = "0000eef3-0000-1000-8000-00805f9b34fb"
PACKET_LENGTH = 180
SAMPLE_RATE = 4000
WINDOW_DURATION = 4.0  # seconds
WINDOW_SIZE = int (SAMPLE_RATE * WINDOW_DURATION)
DT = 1.0 / SAMPLE_RATE


def parse_24bit_signed(data):
    values = []
    for i in range(0, len(data), 3):
        raw = data[i:i+3]
        if raw[0] & 0x80:
            val = int.from_bytes(b'\xFF' + raw, byteorder='big', signed=True)
        else:
            val = int.from_bytes(b'\x00' + raw, byteorder='big', signed=True)
        values.append(val)
    return np.array(values)


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
        # frameWidth*2 는 위/아래 테두리, +2 는 여유 픽셀
        total = header_h + rows_h + self.frameWidth()*2 + 2
        self.setFixedHeight(total)
        # 내용만큼만 보이게 (세로 확장 금지)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

class PlaceholderPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor(12, 14, 18)); self.setPalette(pal)
        label = QtWidgets.QLabel()
        label.setStyleSheet("color: #7aa2f7; font-size: 15px;"); label.setAlignment(Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self); layout.addStretch(); layout.addWidget(label); layout.addStretch()

class WearableStethoUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wearable Stethoscope — UI Mockup (Wide Plot, v2)")
        self.resize(1140, 400)
        self._build_ui(); self._apply_style()
        
        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0
        
        # Setting timer
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(100)  # every 100ms (10Hz)
        self.plot_timer.timeout.connect(self.update_plot)

        self.streaming = False
        
        self.full_time = []
        self.full_audio = []

        self.sample_counter = 0
        self.last_rate_check = time.time()

    
        self.devices = {}
        self.client = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.notify_started = False

        self.updater = QtCore.QTimer()
        self.updater.timeout.connect(self.update_plot)

        self.btn_scan.clicked.connect(self.scan_devices)
        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_save.clicked.connect(self.save_data)
    
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
        
        #streaming Plot in section 2
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
        #v3.setContentsMargins(4, 4, 4, 4)
        v3.setSpacing(1)
        self.tbl_diag = DiagnosisTable()
        self.tbl_diag.setMaximumHeight(180)   # 얇게
        v3.addWidget(self.tbl_diag, stretch=20)
        #v3.addStretch(1)
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


        # ── Splitters (manual resize) ────────────────────────────────────────────
        left_split  = QtWidgets.QSplitter(Qt.Vertical)   # (Signal | Vitals)
        right_split = QtWidgets.QSplitter(Qt.Vertical)   # (Diagnosis | Interactive)
        left_split.addWidget(box2)
        left_split.addWidget(box5)
        right_split.addWidget(box3)
        right_split.addWidget(box4)

        main_split = QtWidgets.QSplitter(Qt.Horizontal)  # (Left | Right)
        main_split.addWidget(left_split)
        main_split.addWidget(right_split)

        # 스플리터/컨테이너를 self에 보관해 수명 보장 (중요)
        self.box2, self.box3, self.box4, self.box5 = box2, box3, box4, box5
        self.left_split, self.right_split, self.main_split = left_split, right_split, main_split

        root.addWidget(main_split, 1, 0, 1, 2)

        # ── 기본 레이아웃 초기 크기(숫자만 바꿔 조절) ─────────────────────────────
        SIGNAL_H   = 405
        VITALS_H   = 30
        DIAG_H     = 160
        INTER_H    = 80
        LEFT_W     = 860
        RIGHT_W    = 200

        self.left_split.setSizes([SIGNAL_H, VITALS_H])
        self.right_split.setSizes([DIAG_H, INTER_H])
        self.main_split.setSizes([LEFT_W, RIGHT_W])

        # 창 전체 초기 크기 (필요 시 조절)
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
        
    def update_plot(self):
        # 1) Signal plot
        self.line.set_data(self.x_buffer, self.audio_buffer) 
        self.ax.relim() 
        self.ax.autoscale_view() 
        self.canvas.draw()
        
    def scan_devices(self):
        async def run_scan():
            devices = await BleakScanner.discover(timeout=3)
            self.devices = {f"{d.name} [{d.address}]": d.address for d in devices if d.name}
            self.cmb_ble.clear()
            self.cmb_ble.addItems(self.devices.keys())

        asyncio.run_coroutine_threadsafe(run_scan(), self.loop)
        
    def connect_device(self):
        addr = self.devices.get(self.cmb_ble.currentText())
        if not addr:
            return

        async def run_connect():
            self.client = BleakClient(addr, loop=self.loop)
            try:
                await self.client.connect()
                if self.client.is_connected:
                    print(f"[✓] Connected to {addr}")
            except Exception as e:
                print(f"[!] Connect failed: {e}")

        asyncio.run_coroutine_threadsafe(run_connect(), self.loop)

    def start_stream(self):
        if not self.client or not self.client.is_connected:
            return

        async def run_notify():
            try:
                await self.client.start_notify(CHAR_UUID_STREAM, self.handle_data)
                self.notify_started = True
                print("[✓] Started streaming")
            except Exception as e:
                print(f"[!] Notify error: {e}")

        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.time_counter = 0.0
        self.full_time = []
        self.full_audio = []
        self.updater.start(50)
        asyncio.run_coroutine_threadsafe(run_notify(), self.loop)
        self.streaming = True

    def stop_stream(self):
        if not self.client or not self.notify_started:
            return

        async def stop():
            try:
                await self.client.stop_notify(CHAR_UUID_STREAM)
                await self.client.disconnect()
                if hasattr(self.client, "set_disconnected_callback"):
                    self.client.set_disconnected_callback(None)
                print("[✓] Stopped stream and disconnected")
                self.notify_started = False
            except Exception as e:
                print(f"[!] Stop error: {e}")

        self.updater.stop()
        asyncio.run_coroutine_threadsafe(stop(), self.loop)
        self.streaming = False
        
    def handle_data(self, handle, data):
        if not self.streaming:
            return

        parsed = parse_24bit_signed(data)
        ADC_SCALE = 2.4 / (2**23)  # ≈ 2.861e-7 V per count
        parsed = parsed * ADC_SCALE
        
        # ---  filtering ---
        fc = 350.0  
        f_low = 30.0  # cutoff Hz
        f_high = 300
        #parsed = self.highpass_iir(parsed, fc=fc, fs=SAMPLE_RATE)
        # ---------------------------

        if parsed.size == 0:
            return

        shift_len = len(parsed)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = parsed

        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(
            self.time_counter - WINDOW_DURATION,
            self.time_counter,
            WINDOW_SIZE
        )

        # Save to full buffer for CSV saving
        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array)
        self.full_audio.extend(parsed)

    
    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save")
            return

        # 1) array + time set to 0 
        time_array = np.array(self.full_time)
        time_array = time_array - time_array[0]
        audio_array = np.array(self.full_audio)


        # 3) Save table
        #    time, amplitude(Volt), energy, energy_smooth(Guassian)
        data_to_save = np.column_stack((time_array, audio_array))

        # 4) Directory / Name
        save_dir = "C:/Users/dhtpd/Downloads/sound"
        os.makedirs(save_dir, exist_ok=True)
        base_filename = "recorded_data"
        extension = ".csv"
        full_path = os.path.join(save_dir, base_filename + extension)

        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")
            counter += 1

        # 5) Save
        try:
            header = "Time(s),Amplitude(V)"
            np.savetxt(full_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.9f")
            print(f"[✓] Data saved to {full_path}")
        except Exception as e:
            print(f"[!] Save error: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = WearableStethoUI(); ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 통합본: 원본 stetho_ui_mock_wideplot_v2.py UI 유지 + 2초 실시간 추론 주입

import sys
import asyncio
import threading
import os
import time
import json
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from bleak import BleakScanner, BleakClient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# === 추가 라이브러리 (추론/리샘플) ===
from joblib import load as joblib_load
from scipy.signal import resample_poly
import librosa

# ----------------- 원본 상수 (그대로 유지) -----------------
CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE = "0000eef3-0000-1000-8000-00805f9b34fb"
PACKET_LENGTH = 180
SAMPLE_RATE = 4000
WINDOW_DURATION = 4.0  # seconds
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
DT = 1.0 / SAMPLE_RATE

# ----------------- 모델/특징 설정 -----------------
#MODEL_DIR = r"D:\Stethoscope_Project\Deployment\ Group Split\model\run_20251008_172910".replace(" ", " ")  # 공백 깨짐 방지
#MODEL_DIR = r"D:\Stethoscope_Project\Deployment\Group Split\model\run_20251008_172910"  # 최종 적용
MODEL_DIR = r"/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/model/run_20251029_235024"  # 최종 적용

TARGET_SR = 16000
SEG_SECONDS = 2.0
SEG_SAMPLES_16K = int(TARGET_SR * SEG_SECONDS)

N_MELS = 64
WIN_MS = 64.0
HOP_MS = 32.0
N_FFT = int(TARGET_SR * (WIN_MS / 1000.0))
HOP_LENGTH = int(TARGET_SR * (HOP_MS / 1000.0))
FMIN, FMAX = 50, 7900

ADC_SCALE = 2.4 / (2 ** 23)  # ≈2.861e-7 V per count

# ----------------- 원본 함수 (그대로) -----------------
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


# ----------------- 추가: DSP/추론 유틸 -----------------
def segment_logmel_64(x_16k: np.ndarray) -> np.ndarray:
    """2초 파형(16k, mono) -> (T, 64) log-mel / per-segment z-norm"""
    if x_16k.ndim > 1:
        x_16k = np.mean(x_16k, axis=0)
    S = librosa.feature.melspectrogram(
        y=x_16k, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max).T.astype(np.float32)  # (T,64)
    m = S_db.mean(axis=0, keepdims=True)
    s = S_db.std(axis=0, keepdims=True) + 1e-6
    return (S_db - m) / s  # (T,64)


class RealtimeInferenceEngine:
    """scaler + LR + MLP 로드, class_names.json 없이 동작"""
    def __init__(self, model_dir: str):
        sc_path = os.path.join(model_dir, "scaler.pkl")
        lr_path = os.path.join(model_dir, "model_lr.pkl")
        mlp_path = os.path.join(model_dir, "model_mlp.pkl")
        for p in (sc_path, lr_path, mlp_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"필수 모델 파일이 없습니다: {p}")
        self.scaler = joblib_load(sc_path)
        self.model_lr = joblib_load(lr_path)
        self.model_mlp = joblib_load(mlp_path)

        # 모델 내부 classes_에서 순서 복원
        if hasattr(self.model_lr, "classes_"):
            self.class_names = [str(c) for c in self.model_lr.classes_]
        elif hasattr(self.model_mlp, "classes_"):
            self.class_names = [str(c) for c in self.model_mlp.classes_]
        else:
            self.class_names = ["Healthy", "Wheezing", "Crackle", "Rhonchi"]  # 최후 fallback

        # UI 표기와의 라벨 매핑 (UI는 "Bronchi" 표기)
        self.ui_names = ["Healthy", "Wheezing", "Crackle", "Bronchi"]
        self.alias = {"Rhonchi": "Bronchi", "Bronchi": "Bronchi"}

    def predict_proba(self, x2s_16k: np.ndarray) -> dict:
        """2초(16k) 입력 -> {UI_class_name: prob(0~1)}"""
        M = segment_logmel_64(x2s_16k)           # (T,64)
        feat64 = M.mean(axis=0, keepdims=False)  # (64,)
        Xs = self.scaler.transform(feat64.reshape(1, -1))
        p_lr = self.model_lr.predict_proba(Xs)[0]
        p_mlp = self.model_mlp.predict_proba(Xs)[0]
        p = (p_lr + p_mlp) / 2.0
        p = np.maximum(p, 1e-12); p = p / p.sum()

        # 모델 라벨 → UI 라벨로 매핑/집계
        out = {name: 0.0 for name in self.ui_names}
        for cls_name, prob in zip(self.class_names, p):
            ui_name = self.alias.get(cls_name, cls_name)
            if ui_name in out:
                out[ui_name] += float(prob)
        # 정규화(혹시 모를 누적/누락 대비)
        s = sum(out.values()) or 1.0
        for k in out:
            out[k] /= s
        return out


# ----------------- 원본 UI 위젯 (그대로) -----------------
class ColorDot(QtWidgets.QFrame):
    def __init__(self, width=120, height=40, parent=None):
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
    def add_row(self, cls_name, prob):
        r = self.rowCount(); self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(cls_name))
        pitem = QtWidgets.QTableWidgetItem(f"{prob:>3d}")
        pitem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(r, 1, pitem)
    def set_probs(self, probs_dict):
        # probs_dict: {class_name: 0~1}
        for r in range(self.rowCount()):
            cls_name = self.item(r, 0).text()
            prob = int(round(probs_dict.get(cls_name, 0.0) * 100))
            self.item(r, 1).setText(str(prob))

class PlaceholderPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor(12, 14, 18)); self.setPalette(pal)
        label = QtWidgets.QLabel()
        label.setStyleSheet("color: #7aa2f7; font-size: 15px;"); label.setAlignment(Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self); layout.addStretch(); layout.addWidget(label); layout.addStretch()


# ----------------- 원본 메인 UI 클래스 + 추론 주입 -----------------
class WearableStethoUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wearable Stethoscope — UI Mockup (Wide Plot, v2)")
        self.resize(1140, 400)
        self._build_ui(); self._apply_style()

        # ---- plot buffers (원본 그대로) ----
        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(100)
        self.plot_timer.timeout.connect(self.update_plot)

        self.streaming = False
        self.full_time = []
        self.full_audio = []
        self.sample_counter = 0
        self.last_rate_check = time.time()

        # ---- BLE (원본) ----
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

        # ---- 추가: 추론 엔진/버퍼/타이머 ----
        self.engine = RealtimeInferenceEngine(MODEL_DIR)
        self.seg_buffer_16k = np.zeros(0, dtype=np.float32)

        self.seg_timer = QtCore.QTimer()
        self.seg_timer.setInterval(100)  # 10 Hz: 2초분 도달 여부 확인
        self.seg_timer.timeout.connect(self.consume_segments)
        self.seg_timer.start()

    def _build_ui(self):
        # === 원본 그대로 ===
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(12)
        root.setVerticalSpacing(6)

        # Top controls
        self.cmb_ble = QtWidgets.QComboBox()
        self.cmb_ble.setFixedWidth(400)
        self.btn_scan = QtWidgets.QPushButton("scan")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_start   = QtWidgets.QPushButton("Start")
        self.btn_stop    = QtWidgets.QPushButton("Stop")
        self.btn_save    = QtWidgets.QPushButton("Save")
        sec1 = QtWidgets.QHBoxLayout()
        sec1.addWidget(self.cmb_ble); sec1.addSpacing(8)
        sec1.addWidget(self.btn_scan); sec1.addSpacing(8)
        sec1.addWidget(self.btn_connect); sec1.addSpacing(8)
        sec1.addWidget(self.btn_start); sec1.addSpacing(8)
        sec1.addWidget(self.btn_stop); sec1.addSpacing(8)
        sec1.addWidget(self.btn_save); sec1.addStretch()
        root.addLayout(sec1, 0, 0, 1, 2)

        # Section 2: Signal
        box2 = QtWidgets.QGroupBox("Signal (real-time)")
        box2.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v2 = QtWidgets.QVBoxLayout(box2)
        v2.setContentsMargins(6, 6, 6, 6)
        self.fig = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot([], [], lw=1, color="#f8e803")
        self.ax.set_facecolor("#0b0c10")
        self.fig.patch.set_facecolor("#0b0c10")
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        for side in ['top', 'right']: self.ax.spines[side].set_visible(False)
        self.ax.spines['bottom'].set_color('white'); self.ax.spines['left'].set_color('white')
        self.ax.set_xlabel("Time (s)", color='white'); self.ax.set_ylabel("Amplitude (V)", color='white')
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.canvas.updateGeometry()
        v2.addWidget(self.canvas, stretch=1)

        # Section 3: Diagnosis
        box3 = QtWidgets.QGroupBox("Real-time Diagnosis")
        box3.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v3 = QtWidgets.QVBoxLayout(box3); v3.setSpacing(1)
        self.tbl_diag = DiagnosisTable()
        self.tbl_diag.setMaximumHeight(180)
        v3.addWidget(self.tbl_diag, stretch=20)
        box3.setMinimumWidth(260)

        # Section 4: Interactive
        box4 = QtWidgets.QGroupBox("Interactive Feedback")
        box4.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v4 = QtWidgets.QVBoxLayout(box4)
        v4.setContentsMargins(8, 8, 8, 8); v4.setSpacing(8)
        row_alert = QtWidgets.QHBoxLayout(); row_alert.setSpacing(6); row_alert.setContentsMargins(0, 0, 0, 0)
        self.lbl_alert = QtWidgets.QLabel("Alert"); self.lbl_alert.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.dot_alert = ColorDot(width=200, height=40)
        row_alert.addWidget(self.lbl_alert); row_alert.addSpacing(8); row_alert.addWidget(self.dot_alert); row_alert.addStretch()
        v4.addLayout(row_alert)
        self.btn_tap    = QtWidgets.QPushButton("Tap")
        self.btn_music  = QtWidgets.QPushButton("Music")
        self.btn_guided = QtWidgets.QPushButton("Guided Breath")
        v4.addWidget(self.btn_tap); v4.addWidget(self.btn_music); v4.addWidget(self.btn_guided)

        # Section 5: Vitals
        box5 = QtWidgets.QGroupBox("Vitals (real-time)")
        box5.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        h5 = QtWidgets.QHBoxLayout(box5); h5.setContentsMargins(6, 6, 6, 6)
        self.card_hr = self._metric_card("Heart Rate", "-- bpm")
        self.card_br = self._metric_card("Breathing Rate", "-- bpm")
        h5.addWidget(self.card_hr, 1); h5.addWidget(self.card_br, 1)

        # Splitters
        left_split  = QtWidgets.QSplitter(Qt.Vertical)
        right_split = QtWidgets.QSplitter(Qt.Vertical)
        left_split.addWidget(box2); left_split.addWidget(box5)
        right_split.addWidget(box3); right_split.addWidget(box4)
        main_split = QtWidgets.QSplitter(Qt.Horizontal)
        main_split.addWidget(left_split); main_split.addWidget(right_split)
        self.box2, self.box3, self.box4, self.box5 = box2, box3, box4, box5
        self.left_split, self.right_split, self.main_split = left_split, right_split, main_split
        root.addWidget(main_split, 1, 0, 1, 2)

        # Default sizes (원본 그대로)
        SIGNAL_H, VITALS_H, DIAG_H, INTER_H = 405, 30, 160, 80
        LEFT_W, RIGHT_W = 860, 200
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

    # ----------------- 원본 plot 업데이트 -----------------
    def update_plot(self):
        self.line.set_data(self.x_buffer, self.audio_buffer)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()

    # ----------------- 원본 BLE 핸들러 -----------------
    def scan_devices(self):
        async def run_scan():
            devices = await BleakScanner.discover(timeout=3)
            self.devices = {f"{d.name} [{d.address}]": d.address for d in devices if d.name}
            self.cmb_ble.clear(); self.cmb_ble.addItems(self.devices.keys())
        asyncio.run_coroutine_threadsafe(run_scan(), self.loop)

    def connect_device(self):
        addr = self.devices.get(self.cmb_ble.currentText())
        if not addr: return
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
        if not self.client or not self.client.is_connected: return
        async def run_notify():
            try:
                await self.client.start_notify(CHAR_UUID_STREAM, self.handle_data)
                self.notify_started = True
                print("[✓] Started streaming")
            except Exception as e:
                print(f"[!] Notify error: {e}")
        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.time_counter = 0.0
        self.full_time = []; self.full_audio = []
        self.updater.start(50)
        asyncio.run_coroutine_threadsafe(run_notify(), self.loop)
        self.streaming = True

    def stop_stream(self):
        if not self.client or not self.notify_started: return
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

    # ----------------- 실시간 수신 + (추가) 추론 버퍼 적재 -----------------
    def handle_data(self, handle, data):
        if not self.streaming: return
        parsed = parse_24bit_signed(data).astype(np.float32) * ADC_SCALE
        if parsed.size == 0: return

        # 1) Plot용 4s 윈도우 갱신 (원본)
        shift_len = len(parsed)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = parsed
        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION, self.time_counter, WINDOW_SIZE)

        # 2) CSV 저장용 누적 (원본)
        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array); self.full_audio.extend(parsed)

        # 3) (추가) 추론용 16k 버퍼 적재: 패킷마다 resample → 16k 누적
        #    up=4, down=1 (4k -> 16k)
        x_16k = resample_poly(parsed, up=TARGET_SR, down=SAMPLE_RATE)
        self.seg_buffer_16k = np.concatenate([self.seg_buffer_16k, x_16k.astype(np.float32)])

    # ----------------- 2초(16k) 단위로 추론 수행 -----------------
    def consume_segments(self):
        while self.seg_buffer_16k.size >= SEG_SAMPLES_16K:
            seg = self.seg_buffer_16k[:SEG_SAMPLES_16K].copy()
            self.seg_buffer_16k = self.seg_buffer_16k[SEG_SAMPLES_16K:]
            try:
                probs_ui = self.engine.predict_proba(seg)  # {UI_name: prob}
                self.tbl_diag.set_probs(probs_ui)          # 우상단 테이블 업데이트
            except Exception as e:
                print(f"[!] Inference error: {e}")

    # ----------------- 원본 저장 -----------------
    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save"); return
        time_array = np.array(self.full_time); time_array = time_array - time_array[0]
        audio_array = np.array(self.full_audio)
        data_to_save = np.column_stack((time_array, audio_array))
        save_dir = "C:/Users/dhtpd/Downloads/sound"
        os.makedirs(save_dir, exist_ok=True)
        base_filename = "recorded_data"; extension = ".csv"
        full_path = os.path.join(save_dir, base_filename + extension)
        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}"); counter += 1
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

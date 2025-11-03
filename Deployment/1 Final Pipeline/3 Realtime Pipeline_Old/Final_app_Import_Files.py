# -*- coding: utf-8 -*-
# Final_app_Import_Files.py
# UI/레이아웃/컬러는 stetho_ui_mock_wideplot_v2.py 그대로 유지
# - BLE 스캔/연결 (원본 유지)
# - 파일 임포트 -> 실시간처럼 50ms 단위 재생 + 소리 출력 + 2초 단위 추론/소프트맥스 표시
# - 테이블에 "현재 윈도우(초)" 표시(간단 디버깅용)
# - 모델 경로는 절대경로 고정 (네 폴더)

import sys
import os
import json
import time
import asyncio
import threading
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QFileDialog, QMessageBox

from bleak import BleakScanner, BleakClient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from joblib import load as joblib_load
from scipy.signal import resample_poly
import librosa

# ---- optional: audio playback ----
try:
    import sounddevice as sd
except Exception:
    sd = None

# =================== DEVICE/BLE CONSTANTS (원본) ===================
CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE    = "0000eef3-0000-1000-8000-00805f9b34fb"
PACKET_LENGTH = 180
SAMPLE_RATE   = 4000               # 디바이스/플롯 측 레이트
WINDOW_DURATION = 4.0              # seconds
WINDOW_SIZE  = int(SAMPLE_RATE * WINDOW_DURATION)
DT = 1.0 / SAMPLE_RATE
ADC_SCALE = 2.4 / (2 ** 23)        # 24bit ADC -> Volt

# =================== MODEL/FEATURE CONSTANTS ===================
# --- 네 환경의 절대 경로 ---
MODEL_DIR   = r"/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/model/run_20251102_120824"
#MODEL_DIR   = r"D:\Stethoscope_Project\Deployment\Group Split\model\run_20251009_092633"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LR_PATH     = os.path.join(MODEL_DIR, "model_lr.pkl")
MLP_PATH    = os.path.join(MODEL_DIR, "model_mlp.pkl")
LE_PATH     = os.path.join(MODEL_DIR, "label_encoder.pkl")  # optional

# 추론 입력 파라미터
TARGET_SR = 16000
SEG_SECONDS = 2.0
SEG_SAMPLES_16K = int(TARGET_SR * SEG_SECONDS)

# Log-mel 파라미터 (학습과 동일)
N_MELS = 64
WIN_MS = 64.0
HOP_MS = 32.0
N_FFT = int(TARGET_SR * (WIN_MS / 1000.0))
HOP_LENGTH = int(TARGET_SR * (HOP_MS / 1000.0))
FMIN, FMAX = 50, 7900

# 전처리 스위치 (디버깅용)
USE_SEGMENT_ZNORM = False      # 사용자 요청: False 케이스도 검증 가능
USE_REF_MAX_DB    = True       # librosa.power_to_db의 ref=np.max
BYPASS_SCALER     = False      # True면 저장된 StandardScaler 무시

# CANONICAL 클래스(라벨 인코더 없거나, 모델 classes_를 안전히 UI로 매핑해야 할 때 사용)
CANONICAL_UI = ["Healthy", "Crackle", "Rhonchi", "Wheezing", "Non-Breathing"]

# ---- debug switches ----
DEBUG_VERBOSE = True           # 세부 로그 전체 on/off
LOG_CONSUME_IDLE = False       # seg_buf < 2초일 때의 [CONSUME] 아이들 로그 표시 여부
LOG_ACC = False                # [ACC] push 로그 표시 여부
LOG_TICK = False               # [TICK] 타이머 틱 로그 표시 여부

# =================== 유틸 ===================
def parse_24bit_signed(data: bytes) -> np.ndarray:
    """3바이트(24bit) signed big-endian -> int32 배열"""
    values = []
    for i in range(0, len(data), 3):
        raw = data[i:i+3]
        if len(raw) < 3:
            break
        if raw[0] & 0x80:
            val = int.from_bytes(b'\xFF' + raw, byteorder='big', signed=True)
        else:
            val = int.from_bytes(b'\x00' + raw, byteorder='big', signed=True)
        values.append(val)
    return np.asarray(values, dtype=np.int32)


def mels_64_from_2s(x_16k: np.ndarray) -> np.ndarray:
    """2초(16k) 파형 -> (T,64) log-mel (per-seg z-norm은 스위치로 제어)"""
    if x_16k.ndim > 1:
        x_16k = np.mean(x_16k, axis=0)
    S = librosa.feature.melspectrogram(
        y=x_16k, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    if USE_REF_MAX_DB:
        S_db = librosa.power_to_db(S, ref=np.max).T.astype(np.float32)
    else:
        S_db = librosa.power_to_db(S, ref=1.0).T.astype(np.float32)

    if USE_SEGMENT_ZNORM:
        m = S_db.mean(axis=0, keepdims=True)
        s = S_db.std(axis=0, keepdims=True) + 1e-6
        S_db = (S_db - m) / s
    return S_db  # (T,64)


# =================== 추론 엔진 ===================
class RealtimeInferenceEngine:
    """scaler + LR + MLP, 라벨인코더 없을 때 CANONICAL 매핑으로 안전 동작"""
    def __init__(self):
        for p in (SCALER_PATH, LR_PATH, MLP_PATH):
            if not os.path.exists(p):
                raise FileNotFoundError(f"모델 파일이 없습니다: {p}")

        self.scaler = joblib_load(SCALER_PATH) if not BYPASS_SCALER else None
        self.model_lr  = joblib_load(LR_PATH)
        self.model_mlp = joblib_load(MLP_PATH)

        # 라벨 인코더는 optional
        self.label_encoder = None
        if os.path.exists(LE_PATH):
            try:
                self.label_encoder = joblib_load(LE_PATH)
            except Exception:
                self.label_encoder = None

        # 모델 classes_ 확보 (없으면 CANONICAL 길이로 더미)
        def model_classes(m):
            if hasattr(m, "classes_"):
                return list(m.classes_)
            return list(range(len(CANONICAL_UI)))

        self.lr_classes  = model_classes(self.model_lr)
        self.mlp_classes = model_classes(self.model_mlp)

        # 각 모델 로컬 클래스 -> 문자열 라벨로 매핑
        self.lr_labels  = self._to_str_labels(self.lr_classes)
        self.mlp_labels = self._to_str_labels(self.mlp_classes)

        # CANONICAL 순서에 맞춘 reindex
        self.lr_reorder  = [self.lr_labels.index(c)  for c in CANONICAL_UI]
        self.mlp_reorder = [self.mlp_labels.index(c) for c in CANONICAL_UI]

        if self.label_encoder is None:
            print(f"[WRN] inverse_transform fail: '{type(None).__name__}' object has no attribute 'inverse_transform'")
        print("[MDL] canonical UI classes:", CANONICAL_UI)
        print(f"[MDL] LR.local={self.lr_labels}  -> reorder={self.lr_reorder}")
        print(f"[MDL] MLP.local={self.mlp_labels} -> reorder={self.mlp_reorder}")
        if self.scaler is not None:
            print(f"[MDL] scaler mean shape={self.scaler.mean_.shape}  scale shape={self.scaler.scale_.shape}")
        else:
            print("[MDL] scaler BYPASSED")
        print(f"[MDL] switches -> ZNORM={USE_SEGMENT_ZNORM}, REF_MAX={USE_REF_MAX_DB}, BYPASS_SCALER={BYPASS_SCALER}")

    def _to_str_labels(self, classes_local):
        """모델 classes_ (숫자/문자) -> 문자열 라벨
           - label_encoder 있으면 inverse_transform
           - 없으면 CANONICAL 길이==classes 길이 가정하고 순서대로 맵핑"""
        if self.label_encoder is not None:
            try:
                inv = self.label_encoder.inverse_transform(np.array(classes_local))
                return [str(x) for x in inv]
            except Exception as e:
                print(f"[WRN] inverse_transform fail: {e}")
        if len(classes_local) == len(CANONICAL_UI):
            print("[WRN] cannot map via encoder. Falling back to CANONICAL order guess.")
            return list(CANONICAL_UI)
        print("[WRN] classes length mismatch. Using raw stringified labels.")
        return [str(c) for c in classes_local]

    def predict_proba_ui(self, seg2s_16k: np.ndarray, win_a: float, win_b: float) -> dict:
        """2초(16k) -> CANONICAL_UI 순서의 확률 dict 반환"""
        # 간단 에너지 로그
        rms = float(np.sqrt(np.mean(seg2s_16k**2) + 1e-12))
        peak = float(np.max(np.abs(seg2s_16k)) + 1e-12)
        if DEBUG_VERBOSE:
            print(f"[MDL] seg energy: rms={rms:.6f}, peak={peak:.6f}")

        M = mels_64_from_2s(seg2s_16k)           # (T,64)
        feat64 = M.mean(axis=0, keepdims=False)  # (64,)
        X = feat64.reshape(1, -1)

        # scaler
        if (self.scaler is not None) and (not BYPASS_SCALER) and hasattr(self.scaler, "transform"):
            Xs = self.scaler.transform(X)
        else:
            Xs = X

        # 각각 예측
        p_lr_local  = self.model_lr.predict_proba(Xs)[0]
        p_mlp_local = self.model_mlp.predict_proba(Xs)[0]

        # CANONICAL 순서로 재정렬
        try:
            p_lr  = p_lr_local[self.lr_reorder]
            p_mlp = p_mlp_local[self.mlp_reorder]
        except Exception:
            p_lr  = np.zeros(len(CANONICAL_UI), dtype=float)
            p_mlp = np.zeros(len(CANONICAL_UI), dtype=float)

        p = (p_lr + p_mlp) / 2.0
        p = np.maximum(p, 1e-12)
        p = p / p.sum()

        return {cls: float(prob) for cls, prob in zip(CANONICAL_UI, p)}


# =================== UI WIDGETS (원본 스타일) ===================
class ColorDot(QtWidgets.QFrame):
    def __init__(self, width=120, height=40, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.set_off()
    def set_on(self):
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(230, 40, 40))
        self.setPalette(pal); self.update()
    def set_off(self):
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(250, 250, 250))
        self.setPalette(pal); self.update()


class DiagnosisTable(QtWidgets.QTableWidget):
    """원본 2열 구조 유지 + 디버깅용 't(s)' 좁은 열 추가"""
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Class", "Prob. (%)", "t(s)"])
        self.horizontalHeader().setStretchLastSection(False)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setStyleSheet("QHeaderView::section{font-weight:600;}")
        # 초기행
        for cls in CANONICAL_UI:
            self._add_row(cls, 0, "")
        # 컬럼 폭
        self.setColumnWidth(0, 120)
        self.setColumnWidth(1, 80)
        self.setColumnWidth(2, 80)

    def _add_row(self, cls_name, prob, tstr):
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(cls_name))
        pitem = QtWidgets.QTableWidgetItem(f"{int(prob):>3d}")
        pitem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(r, 1, pitem)
        titem = QtWidgets.QTableWidgetItem(tstr)
        titem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setItem(r, 2, titem)

    def set_probs(self, probs_dict, tstr):
        """확률 + 해당 윈도우 문자열(예: ' 2.00– 4.00')"""
        for r in range(self.rowCount()):
            cls_name = self.item(r, 0).text()
            prob = int(round(probs_dict.get(cls_name, 0.0) * 100))
            self.item(r, 1).setText(str(prob))
            self.item(r, 2).setText(tstr)


class PlaceholderPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(12, 14, 18))
        self.setPalette(pal)
        label = QtWidgets.QLabel()
        label.setStyleSheet("color: #7aa2f7; font-size: 15px;")
        label.setAlignment(Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch(); layout.addWidget(label); layout.addStretch()


# =================== 메인 UI ===================
class WearableStethoUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wearable Stethoscope — UI Mockup (Wide Plot, v2)")
        self.resize(1140, 400)
        self._build_ui()
        self._apply_style()

        # plot buffers
        self.audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(100)
        self.plot_timer.timeout.connect(self.update_plot)

        self.streaming = False
        self.full_time = []
        self.full_audio = []

        # BLE
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

        # inference
        self.engine = RealtimeInferenceEngine()
        self.seg_buffer_16k = np.zeros(0, dtype=np.float32)

        self.seg_timer = QtCore.QTimer()
        self.seg_timer.setInterval(100)  # 10Hz: 2초 도달 확인
        self.seg_timer.timeout.connect(self.consume_segments)
        self.seg_timer.start()

        # file playback
        self.playback_mode = False
        self.play_timer = QtCore.QTimer()
        self.play_timer.setInterval(50)  # 20Hz
        self.play_timer.timeout.connect(self._playback_tick)

        self.file_plot_4k = None
        self.file_inf_16k = None
        self.file_idx_plot = 0
        self.file_idx_inf  = 0

        # audio out stream for file mode
        self.sd_stream = None

        # 재생 기준 시간(초)
        self.play_start_time = None

        # per-window probs CSV (로그)
        self.csv_path = os.path.join(os.path.expanduser("~"), "Stethoscope_Project", "live_probs.csv")
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", encoding="utf-8") as f:
                    header = ["win_start_s", "win_end_s"] + CANONICAL_UI
                    f.write(",".join(header) + "\n")
            print(f"[LOG] writing per-window probs to: {self.csv_path}")
        except Exception as e:
            print(f"[LOG] CSV init failed: {e}")

    # -------- UI build/style --------
    def _build_ui(self):
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(12)
        root.setVerticalSpacing(6)

        # Top controls
        self.cmb_ble = QtWidgets.QComboBox()
        self.cmb_ble.setFixedWidth(400)
        self.btn_scan    = QtWidgets.QPushButton("scan")
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

        # Signal plot
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
        for side in ['top', 'right']:
            self.ax.spines[side].set_visible(False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Amplitude (V)", color='white')
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        v2.addWidget(self.canvas, stretch=1)

        # Diagnosis
        box3 = QtWidgets.QGroupBox("Real-time Diagnosis")
        box3.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        v3 = QtWidgets.QVBoxLayout(box3); v3.setSpacing(1)
        self.tbl_diag = DiagnosisTable()
        self.tbl_diag.setMaximumHeight(180)
        v3.addWidget(self.tbl_diag, stretch=20)
        box3.setMinimumWidth(260)

        # Interactive Feedback (자리만)
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

        # Vitals (자리만)
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

        root.addWidget(main_split, 1, 0, 1, 2)

        # default sizes (원본)
        SIGNAL_H, VITALS_H, DIAG_H, INTER_H = 405, 30, 160, 80
        LEFT_W, RIGHT_W = 860, 200
        left_split.setSizes([SIGNAL_H, VITALS_H])
        right_split.setSizes([DIAG_H, INTER_H])
        main_split.setSizes([LEFT_W, RIGHT_W])
        self.resize(LEFT_W + RIGHT_W, SIGNAL_H + VITALS_H + 20)

        # expose for reuse
        self.box2, self.box3, self.box4, self.box5 = box2, box3, box4, box5
        self.left_split, self.right_split, self.main_split = left_split, right_split, main_split

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

    # -------- plotting --------
    def update_plot(self):
        self.line.set_data(self.x_buffer, self.audio_buffer)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()

    # -------- BLE --------
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

    # -------- stream control --------
    def start_stream(self):
        """디바이스 연결시 BLE 스트림, 아니면 파일 선택"""
        if self.client and getattr(self.client, "is_connected", False):
            async def run_notify():
                try:
                    await self.client.start_notify(CHAR_UUID_STREAM, self.handle_data)
                    self.notify_started = True
                    print("[✓] Started streaming (BLE)")
                except Exception as e:
                    print(f"[!] Notify error: {e}")
            self._reset_buffers(reason="start_stream(BLE)")
            self.updater.start(50)
            self.seg_timer.start(100)
            asyncio.run_coroutine_threadsafe(run_notify(), self.loop)
            self.streaming = True
            self.playback_mode = False
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open audio file (WAV or CSV)", "",
                "Audio/CSV Files (*.wav *.WAV *.csv *.CSV)"
            )
            if not path:
                return
            ok = self._prepare_playback_from_file(path)
            if not ok:
                QMessageBox.warning(self, "Open failed", "파일을 읽지 못했습니다.")
                return
            self._start_playback()

    def stop_stream(self):
        """BLE/파일 재생 모두 정지 + 상태 리셋"""
        # file playback stop
        if self.playback_mode:
            self.play_timer.stop()
            self.updater.stop()
            if self.sd_stream is not None:
                try: self.sd_stream.stop(); self.sd_stream.close()
                except Exception: pass
                self.sd_stream = None
            self.playback_mode = False
            print("[✓] Stopped file playback")

        # BLE stop
        if self.client and self.notify_started:
            async def stop():
                try:
                    await self.client.stop_notify(CHAR_UUID_STREAM)
                    await self.client.disconnect()
                    print("[✓] Stopped stream and disconnected")
                except Exception as e:
                    print(f"[!] Stop error: {e}")
            self.updater.stop()
            asyncio.run_coroutine_threadsafe(stop(), self.loop)
            self.streaming = False

        self._reset_buffers(reason="stop_stream(file)")
        print("[RST] inference state reset: stop_stream")

    # -------- BLE data callback --------
    def handle_data(self, handle, data):
        if not self.streaming: return
        parsed = parse_24bit_signed(data).astype(np.float32) * ADC_SCALE
        if parsed.size == 0: return
        self._push_plot_4k(parsed, SAMPLE_RATE)
        self._accumulate_for_inference(parsed, SAMPLE_RATE)

    # -------- File playback --------
    def _prepare_playback_from_file(self, path: str) -> bool:
        """WAV/CSV -> 4k plot 버퍼, 16k inference 버퍼 준비"""
        try:
            # 로드
            if path.lower().endswith((".wav", ".wave")):
                x, sr = librosa.load(path, sr=None, mono=True)  # float32 [-1,1]
            else:
                arr = np.genfromtxt(path, delimiter=",", names=True)
                if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype.names:
                    if "Amplitude(V)" in arr.dtype.names:
                        x = np.asarray(arr["Amplitude(V)"], dtype=np.float32)
                    else:
                        x = np.asarray(arr[list(arr.dtype.names)[1]], dtype=np.float32)
                    if "Time(s)" in arr.dtype.names:
                        t = np.asarray(arr["Time(s)"], dtype=np.float64)
                        dt = np.diff(t); sr = int(round(1.0/np.median(dt)))
                    else:
                        sr = SAMPLE_RATE
                else:
                    raw = np.loadtxt(path, delimiter=",")
                    if raw.ndim == 1:
                        x = raw.astype(np.float32); sr = SAMPLE_RATE
                    else:
                        t = raw[:,0]; x = raw[:,1].astype(np.float32)
                        dt = np.diff(t); sr = int(round(1.0/np.median(dt)))

            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # 리셋
            self._reset_buffers(reason="prepare_playback")
            # 리샘플
            self.file_plot_4k = resample_poly(x, up=SAMPLE_RATE, down=sr).astype(np.float32)
            self.file_inf_16k = resample_poly(x, up=TARGET_SR,  down=sr).astype(np.float32)
            self.file_idx_plot = 0
            self.file_idx_inf  = 0

            print(f"[✓] Loaded file: {os.path.basename(path)} (sr={sr}, len={len(x)})")
            self.update_plot()
            return True
        except Exception as e:
            print(f"[!] File open error: {e}")
            return False

    def _start_playback(self):
        self.seg_timer.start(100)
        self.playback_mode = True
        self.updater.start(50)
        self.play_start_time = time.time()

        # 오디오 출력(선택)
        if sd is not None and self.sd_stream is None:
            try:
                self.sd_stream = sd.OutputStream(samplerate=TARGET_SR, channels=1, dtype='float32')
                self.sd_stream.start()
            except Exception as e:
                print(f"[!] Audio stream open failed: {e}")
                self.sd_stream = None
        else:
            if sd is None:
                print("[i] sounddevice 미설치: 오디오는 재생하지 않습니다.")

        self.play_timer.start()
        self._playback_tick()
        print("[✓] Started file playback (tick armed)")

    def _playback_tick(self):
        if not self.playback_mode:
            return
        n_plot = int(0.050 * SAMPLE_RATE)  # 200 @ 4k
        n_inf  = int(0.050 * TARGET_SR)    # 800 @ 16k

        # plot
        if self.file_idx_plot < len(self.file_plot_4k):
            end_p = min(self.file_idx_plot + n_plot, len(self.file_plot_4k))
            chunk_p = self.file_plot_4k[self.file_idx_plot:end_p]
            self.file_idx_plot = end_p
            self._push_plot_4k(chunk_p, SAMPLE_RATE)

        # inference + audio out
        if self.file_idx_inf < len(self.file_inf_16k):
            end_i = min(self.file_idx_inf + n_inf, len(self.file_inf_16k))
            chunk_i = self.file_inf_16k[self.file_idx_inf:end_i]
            self.file_idx_inf = end_i
            self._accumulate_for_inference(chunk_i, TARGET_SR)
            if self.sd_stream is not None:
                try:
                    self.sd_stream.write(chunk_i.astype(np.float32))
                except Exception as e:
                    print(f"[!] Audio write failed: {e}")

        remain_plot = len(self.file_plot_4k) - self.file_idx_plot if self.file_plot_4k is not None else 0
        remain_inf  = len(self.file_inf_16k) - self.file_idx_inf  if self.file_inf_16k  is not None else 0
        if DEBUG_VERBOSE and LOG_TICK:
            print(f"[TICK] idx_plot={self.file_idx_plot}, remain_plot={remain_plot}, "
                  f"idx_inf={self.file_idx_inf}, remain_inf={remain_inf}, seg_buf={self.seg_buffer_16k.size}")

        # 종료
        if self.file_idx_plot >= len(self.file_plot_4k) and self.file_idx_inf >= len(self.file_inf_16k):
            self.play_timer.stop()
            self.updater.stop()
            if self.sd_stream is not None:
                try: self.sd_stream.stop(); self.sd_stream.close()
                except Exception: pass
                self.sd_stream = None
            self.playback_mode = False
            print("[✓] Stopped file playback")

    # -------- 공통: plot 4k push --------
    def _push_plot_4k(self, samples: np.ndarray, sr: int):
        if samples.size == 0:
            return
        if sr != SAMPLE_RATE:
            samples = resample_poly(samples, up=SAMPLE_RATE, down=sr).astype(np.float32)
        shift_len = len(samples)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = samples
        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION, self.time_counter, WINDOW_SIZE)

        # CSV 저장용 원시 스트림
        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array); self.full_audio.extend(samples)

    # -------- 공통: inference buffer accumulate --------
    def _accumulate_for_inference(self, samples: np.ndarray, sr: int):
        if sr != TARGET_SR:
            samples = resample_poly(samples, up=TARGET_SR, down=sr).astype(np.float32)
        self.seg_buffer_16k = np.concatenate([self.seg_buffer_16k, samples])

        if DEBUG_VERBOSE and LOG_ACC:
            print(f"[ACC] pushed {len(samples)} @16k | seg_buf={self.seg_buffer_16k.size}")

        if self.seg_buffer_16k.size >= SEG_SAMPLES_16K:
            self.consume_segments()

    # -------- 2초 단위 소모/추론 --------
    def consume_segments(self):
        # 아이들 상태: 버퍼가 2초 미만이면 즉시 리턴
        if self.seg_buffer_16k.size < SEG_SAMPLES_16K:
            if DEBUG_VERBOSE and LOG_CONSUME_IDLE:
                print(f"[CONSUME] idle | seg_buf={self.seg_buffer_16k.size}")
            return

        updated = False
        windows_done = 0

        while self.seg_buffer_16k.size >= SEG_SAMPLES_16K:
            seg = self.seg_buffer_16k[:SEG_SAMPLES_16K].copy()
            self.seg_buffer_16k = self.seg_buffer_16k[SEG_SAMPLES_16K:]
            windows_done += 1

            # 현재 윈도우 시간 계산(파일모드: 재생 시작 기준 경과시간, BLE: time_counter 사용)
            if self.playback_mode and self.play_start_time is not None:
                elapsed = time.time() - self.play_start_time
                win_b = elapsed
                win_a = max(0.0, win_b - SEG_SECONDS)
            else:
                win_b = max(0.0, self.time_counter)
                win_a = max(0.0, win_b - SEG_SECONDS)

            try:
                probs_ui = self.engine.predict_proba_ui(seg, win_a, win_b)
                # 테이블 업데이트 (시간 문자열 포함)
                tstr = f"{win_a:5.2f}–{win_b:5.2f}"
                self.tbl_diag.set_probs(probs_ui, tstr)
                updated = True

                # 콘솔 요약
                top1 = max(probs_ui.items(), key=lambda kv: kv[1])
                if DEBUG_VERBOSE:
                    print(f"[INF] softmax updated | window={win_a:6.2f}–{win_b:6.2f}s | top1={top1[0]:<12} {top1[1]*100:5.1f}%")

                # CSV 저장 (윈도우별 확률)
                try:
                    if self.csv_path:
                        with open(self.csv_path, "a", encoding="utf-8") as f:
                            row = [f"{win_a:.2f}", f"{win_b:.2f}"] + [f"{probs_ui[c]:.6f}" for c in CANONICAL_UI]
                            f.write(",".join(row) + "\n")
                except Exception as e:
                    print(f"[LOG] CSV append failed: {e}")

            except Exception as e:
                print(f"[!] Inference error: {e}")

        if DEBUG_VERBOSE:
            print(f"[CONSUME] processed windows={windows_done} | seg_buf_remain={self.seg_buffer_16k.size}")

        if updated:
            self.tbl_diag.viewport().update()

    # -------- 저장 --------
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
            full_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")
            counter += 1
        try:
            header = "Time(s),Amplitude(V)"
            np.savetxt(full_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.9f")
            print(f"[✓] Data saved to {full_path}")
        except Exception as e:
            print(f"[!] Save error: {e}")

    # -------- 내부 리셋 --------
    def _reset_buffers(self, reason=""):
        self.audio_buffer[:] = 0.0
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0
        self.full_time = []; self.full_audio = []
        self.seg_buffer_16k = np.zeros(0, dtype=np.float32)
        self.file_idx_plot = 0; self.file_idx_inf = 0
        self.file_plot_4k = None; self.file_inf_16k = None
        self.play_start_time = None
        print(f"[RST] inference state reset: {reason}")


# =================== MAIN (GUI 우선, CLI 선택) ===================
import argparse

def _parse_cli_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("wav", nargs="?", help="(선택) 실행 시 자동 재생할 WAV/CSV 경로")
    p.add_argument("--max", type=float, default=None, help="(선택) 자동 재생 제한 시간(초)")
    return p.parse_known_args()

def main():
    args, unknown = _parse_cli_args()
    if unknown:
        sys.argv = [sys.argv[0]] + unknown

    app = QtWidgets.QApplication(sys.argv)
    ui = WearableStethoUI()
    ui.show()

    # CLI로 파일 지정 시 자동 재생
    if args.wav:
        ok = ui._prepare_playback_from_file(args.wav)
        if ok:
            ui._start_playback()
            if args.max is not None:
                QtCore.QTimer.singleShot(int(args.max * 1000), ui.stop_stream)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
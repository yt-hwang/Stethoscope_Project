# -*- coding: utf-8 -*-
import sys
import asyncio
import threading
import os
import time
import json
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QSizePolicy, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent   # 사운드 재생
from bleak import BleakScanner, BleakClient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# === 추가 의존성 ===
import soundfile as sf
import librosa
from joblib import load as joblib_load

# ─────────────────────────────────────────────────────────
# 고정 하드코딩 경로
# ─────────────────────────────────────────────────────────
TEST_WAV_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Test for Realtime Deployment")
MODEL_DIR    = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/model/run_20251102_225046")

# ─────────────────────────────────────────────────────────
# 기존 BLE/Plot 파라미터 (원본 유지)
# ─────────────────────────────────────────────────────────
CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE = "0000eef3-0000-1000-8000-00805f9b34fb"
PACKET_LENGTH = 180
SAMPLE_RATE = 4000
WINDOW_DURATION = 4.0  # seconds
WINDOW_SIZE = int (SAMPLE_RATE * WINDOW_DURATION)
DT = 1.0 / SAMPLE_RATE

# ─────────────────────────────────────────────────────────
# 모델 아답터 (윈도우 2.0s / 홉 1.0s → 1초 간격 분류)
# ─────────────────────────────────────────────────────────
class RealtimeModelAdapter:
    SR = 16000
    WIN_S = 2.0
    HOP_S = 1.0  # ★ 1초 간격으로 결과 생성 (슬라이딩)
    N_MELS = 64
    FMIN, FMAX = 50, 7900
    WIN_MS, HOP_MS = 64, 32

    # UI 표시 순서(모델 thresholds.json의 표기와 일치)
    UI_ORDER = ['Healthy', 'Crackle', 'Rhonchi', 'Wheezing', 'Non-breathing']

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._load_models_build_reorder()
        self._reset_state()

    def reset_stream(self):
        self._reset_state()

    def push_audio(self, samples_16k: np.ndarray):
        if samples_16k is None or samples_16k.ndim == 0:
            return
        self.buf = np.concatenate([self.buf, samples_16k.astype(np.float32)], axis=0)
        self._consume_ready_hops()

    def has_new_result(self) -> bool:
        return self._has_new

    def get_latest_probs_dict_for_ui(self) -> dict:
        p_ui = self.latest_probs_ui.copy()  # UI_ORDER 순서
        d = {
            'Healthy':       float(p_ui[self._idx_ui('Healthy')]),
            'Wheezing':      float(p_ui[self._idx_ui('Wheezing')]),
            'Crackle':       float(p_ui[self._idx_ui('Crackle')]),
            'Bronchi':       float(p_ui[self._idx_ui('Rhonchi')]),      # 표시는 Bronchi
            'Non-breathing': float(p_ui[self._idx_ui('Non-breathing')]),
        }
        self._has_new = False
        return d

    def get_latest_label_ui(self) -> str:
        idx = int(np.argmax(self.latest_probs_ui - self.thresholds_ui))
        return self.UI_ORDER[idx]

    def get_ui_classes(self):
        return list(self.UI_ORDER)

    def _idx_ui(self, name: str) -> int:
        return self.UI_ORDER.index(name)

    def _load_models_build_reorder(self):
        self.scaler = joblib_load(self.model_dir / "scaler.pkl")
        self.model_lr = joblib_load(self.model_dir / "model_lr.pkl")
        self.model_mlp = joblib_load(self.model_dir / "model_mlp.pkl")
        with open(self.model_dir / "thresholds.json", "r", encoding="utf-8") as f:
            th = json.load(f)
        local_names = list(th.get("class_names", []))
        thresholds_local = np.array(th.get("thresholds", []), dtype=np.float32)

        reorder = []
        for c in self.UI_ORDER:
            key = c  # 'Non-breathing' 포함 동일 표기
            try:
                reorder.append(local_names.index(key))
            except ValueError:
                raise RuntimeError(f"class '{c}' not found in model classes {local_names}")
        self.reorder_local_to_ui = np.array(reorder, dtype=int)
        self.thresholds_ui = thresholds_local[self.reorder_local_to_ui]

        print(f"[MDL] local classes={local_names}")
        print(f"[MDL] UI_ORDER     ={self.UI_ORDER}")
        print(f"[MDL] reorder local->ui = {self.reorder_local_to_ui.tolist()}")
        print(f"[MDL] thresholds(ui)    = {self.thresholds_ui.tolist()}")

    def _reset_state(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.n_consumed = 0
        self.latest_probs_ui = np.zeros(len(self.UI_ORDER), dtype=np.float32)
        self._has_new = False
        self._win = int(self.WIN_S * self.SR)
        self._hop = int(self.HOP_S * self.SR)

    def _consume_ready_hops(self):
        # 버퍼 길이가 (n_consumed*h + w) 이상이면 한 스텝 추론
        needed = self.n_consumed * self._hop + self._win
        while self.buf.shape[0] >= needed:
            start = self.n_consumed * self._hop
            end   = start + self._win
            seg = self.buf[start:end]
            self._infer_one(seg)
            self.n_consumed += 1
            self._has_new = True
            needed = self.n_consumed * self._hop + self._win
        # 앞부분 절삭(메모리 관리) — 최근 5초 남김
        cut_at = max(0, self.n_consumed * self._hop - int(5 * self.SR))
        if cut_at > 0:
            self.buf = self.buf[cut_at:]
            self.n_consumed -= cut_at // self._hop
            if self.n_consumed < 0: self.n_consumed = 0

    def _infer_one(self, seg: np.ndarray):
        # Log-Mel 64 → 시간평균(64,) → per-sample 표준화 → scaler
        n_fft = int(self.SR*self.WIN_MS/1000)
        hop_l = int(self.SR*self.HOP_MS/1000)
        m = librosa.feature.melspectrogram(
            y=seg, sr=self.SR, n_mels=self.N_MELS,
            n_fft=n_fft, hop_length=hop_l,
            fmin=self.FMIN, fmax=self.FMAX, power=2.0
        )
        lm = librosa.power_to_db(m, ref=np.max)
        v = lm.mean(axis=1).astype(np.float32)
        mu, sd = float(v.mean()), float(v.std()) + 1e-8
        v = (v - mu) / sd
        xs = self.scaler.transform(v.reshape(1, -1))
        p_lr  = self.model_lr.predict_proba(xs)[0]
        p_mlp = self.model_mlp.predict_proba(xs)[0]
        p_loc = 0.5 * p_lr + 0.5 * p_mlp
        self.latest_probs_ui = p_loc[self.reorder_local_to_ui].astype(np.float32)

# ─────────────────────────────────────────────────────────
# 기존 유틸(원본 유지)
# ─────────────────────────────────────────────────────────
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
        # 5행: Non-breathing 추가
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Class", "Prob. (%)"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setStyleSheet("QHeaderView::section{font-weight:600;}")
        for cls in ["Healthy", "Wheezing", "Crackle", "Bronchi", "Non-breathing"]:
            self.add_row(cls, 0)
    def add_row(self, cls_name, prob):
        r = self.rowCount(); self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(cls_name))
        pitem = QtWidgets.QTableWidgetItem(f"{prob:>3d}"); pitem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(r, 1, pitem)
    def set_probs(self, probs_dict):
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

# ─────────────────────────────────────────────────────────
# UI 메인
# ─────────────────────────────────────────────────────────
class WearableStethoUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wearable Stethoscope — UI Mockup (Wide Plot, v2)")
        self.resize(1180, 420)
        self._build_ui(); self._apply_style()

        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0

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
        self.btn_import.clicked.connect(self.play_from_file)  # 상단 Import: 파일 임포트 재생
        self.btn_music.clicked.connect(self.play_from_file)   # 기존 Music도 파일 재생 유지

        self.adapter = RealtimeModelAdapter(MODEL_DIR)
        self.adapter.reset_stream()

        # 오디오 플레이어 (파일 임포트 재생 시 실제 소리 출력)
        self.player = QMediaPlayer(self)
        self.player.setVolume(80)

        self.full_time = []
        self.full_audio = []

        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self._playback_tick)
        self.playing = False
        self.play_data_4k = None
        self.play_data_16k = None
        self.play_pos_4k = 0
        self.play_pos_16k = 0

    def _build_ui(self):
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(12)
        root.setVerticalSpacing(6)

        # Section 1: Top controls  (Import 버튼을 Save 옆)
        self.cmb_ble = QtWidgets.QComboBox()
        self.cmb_ble.setFixedWidth(400)
        self.btn_scan   = QtWidgets.QPushButton("scan")
        self.btn_connect= QtWidgets.QPushButton("Connect")
        self.btn_start  = QtWidgets.QPushButton("Start")
        self.btn_stop   = QtWidgets.QPushButton("Stop")
        self.btn_save   = QtWidgets.QPushButton("Save")
        self.btn_import = QtWidgets.QPushButton("Import")  # 추가

        sec1 = QtWidgets.QHBoxLayout()
        sec1.addWidget(self.cmb_ble); sec1.addSpacing(8)
        sec1.addWidget(self.btn_scan); sec1.addSpacing(8)
        sec1.addWidget(self.btn_connect); sec1.addSpacing(8)
        sec1.addWidget(self.btn_start); sec1.addSpacing(8)
        sec1.addWidget(self.btn_stop); sec1.addSpacing(8)
        sec1.addWidget(self.btn_save); sec1.addSpacing(8)
        sec1.addWidget(self.btn_import)  # Save 옆
        sec1.addStretch()
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
        self.tbl_diag.setMaximumHeight(205)
        v3.addWidget(self.tbl_diag, stretch=20)
        box3.setMinimumWidth(280)

        # Section 4: Interactive Feedback (원본 유지)
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
        row_alert.addWidget(self.lbl_alert); row_alert.addSpacing(8); row_alert.addWidget(self.dot_alert); row_alert.addStretch()
        v4.addLayout(row_alert)
        self.btn_tap    = QtWidgets.QPushButton("Tap")
        self.btn_music  = QtWidgets.QPushButton("Music")
        self.btn_guided = QtWidgets.QPushButton("Guided Breath")
        v4.addWidget(self.btn_tap); v4.addWidget(self.btn_music); v4.addWidget(self.btn_guided)

        # Section 5: Vitals
        box5 = QtWidgets.QGroupBox("Vitals (real-time)")
        box5.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        h5 = QtWidgets.QHBoxLayout(box5)
        h5.setContentsMargins(6, 6, 6, 6)
        self.card_hr = self._metric_card("Heart Rate", "-- bpm")
        self.card_br = self._metric_card("Breathing Rate", "-- bpm")
        h5.addWidget(self.card_hr, 1); h5.addWidget(self.card_br, 1)

        left_split  = QtWidgets.QSplitter(Qt.Vertical)
        right_split = QtWidgets.QSplitter(Qt.Vertical)
        left_split.addWidget(box2); left_split.addWidget(box5)
        right_split.addWidget(box3); right_split.addWidget(box4)

        main_split = QtWidgets.QSplitter(Qt.Horizontal)
        main_split.addWidget(left_split); main_split.addWidget(right_split)

        self.box2, self.box3, self.box4, self.box5 = box2, box3, box4, box5
        self.left_split, self.right_split, self.main_split = left_split, right_split, main_split
        root.addWidget(main_split, 1, 0, 1, 2)

        SIGNAL_H   = 405; VITALS_H = 30; DIAG_H = 185; INTER_H = 80
        LEFT_W     = 880; RIGHT_W = 220
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

    # ───────────── Plot 업데이트 ─────────────
    def update_plot(self):
        self.line.set_data(self.x_buffer, self.audio_buffer)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()

        if self.adapter.has_new_result():
            probs_dict = self.adapter.get_latest_probs_dict_for_ui()
            self.tbl_diag.set_probs(probs_dict)
            _ = self.adapter.get_latest_label_ui()

    # ───────────── BLE 스캔/연결 ─────────────
    def scan_devices(self):
        async def run_scan():
            devices = await BleakScanner.discover(timeout=3)
            self.devices = {f"{d.name} [{d.address}]": d.address for d in devices if d.name}
            self.cmb_ble.clear()
            self.cmb_ble.addItems(self.devices.keys())
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

    # ───────────── BLE 스트리밍 ─────────────
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
        self.full_time = []; self.full_audio = []
        self.adapter.reset_stream()

        self.updater.start(50)
        asyncio.run_coroutine_threadsafe(run_notify(), self.loop)

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

    # ───────────── BLE 데이터 수신 ─────────────
    def handle_data(self, handle, data):
        parsed = parse_24bit_signed(data)
        ADC_SCALE = 2.4 / (2**23)
        parsed = parsed * ADC_SCALE
        if parsed.size == 0: return

        shift_len = len(parsed)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = parsed
        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION, self.time_counter, WINDOW_SIZE)

        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array)
        self.full_audio.extend(parsed)

        # 4kHz → 16kHz 업샘플 후 2초 윈도우 / 1초 홉 추론
        up = librosa.resample(parsed.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=RealtimeModelAdapter.SR, res_type="kaiser_best")
        self.adapter.push_audio(up)

    # ───────────── 파일 임포트/재생 ─────────────
    def play_from_file(self):
        start_dir = str(TEST_WAV_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV", start_dir, "WAV Files (*.wav)")
        if not path:
            print("[!] No file selected.")
            return
        wav_path = Path(path)
        try:
            x, sr = sf.read(str(wav_path))
        except Exception as e:
            print(f"[!] Failed to read wav: {e}")
            return

        # 시각화/모델용 리샘플
        if sr != SAMPLE_RATE:
            x4 = librosa.resample((x.mean(axis=1) if x.ndim>1 else x).astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE, res_type="kaiser_best")
        else:
            x4 = (x.mean(axis=1) if x.ndim>1 else x).astype(np.float32)

        if sr != RealtimeModelAdapter.SR:
            x16 = librosa.resample((x.mean(axis=1) if x.ndim>1 else x).astype(np.float32), orig_sr=sr, target_sr=RealtimeModelAdapter.SR, res_type="kaiser_best")
        else:
            x16 = (x.mean(axis=1) if x.ndim>1 else x).astype(np.float32)

        # 내부 상태 초기화
        self.play_data_4k = x4
        self.play_data_16k = x16
        self.play_pos_4k = 0
        self.play_pos_16k = 0
        self.adapter.reset_stream()
        self.full_time = []; self.full_audio = []
        self.time_counter = 0.0
        self.audio_buffer = np.zeros(WINDOW_SIZE)

        # 실제 소리 재생
        media = QMediaContent(QUrl.fromLocalFile(str(wav_path)))
        self.player.setMedia(media)
        self.player.play()

        # 타이머로(50ms) 플로팅/모델에 청크 공급
        self.play_timer.start(50)
        self.updater.start(50)
        self.playing = True
        print(f"[PLAY] {wav_path.name}  len4k={len(x4)}  len16k={len(x16)}  (audio: playing)")

    def _playback_tick(self):
        if not self.playing: return
        if self.play_data_4k is None or self.play_data_16k is None:
            self._stop_playback()
            return

        c4 = int(0.05 * SAMPLE_RATE)
        c16 = int(0.05 * RealtimeModelAdapter.SR)

        if self.play_pos_4k >= len(self.play_data_4k) or self.play_pos_16k >= len(self.play_data_16k):
            self._stop_playback()
            return

        end4 = min(len(self.play_data_4k), self.play_pos_4k + c4)
        end16 = min(len(self.play_data_16k), self.play_pos_16k + c16)

        chunk4 = self.play_data_4k[self.play_pos_4k:end4]
        chunk16 = self.play_data_16k[self.play_pos_16k:end16]

        # Plot 버퍼 갱신 (4k)
        shift_len = len(chunk4)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = chunk4
        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION, self.time_counter, WINDOW_SIZE)

        # 전체 저장용(4k 기준 시간)
        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array)
        self.full_audio.extend(chunk4)

        # 모델 아답터에 16k 푸시 (2s 윈도우 / 1s 홉)
        self.adapter.push_audio(chunk16)

        # 위치 전진
        self.play_pos_4k = end4
        self.play_pos_16k = end16

    def _stop_playback(self):
        if self.playing:
            self.play_timer.stop()
            print("[PLAY] finished")
        self.playing = False
        # 재생 중지
        if self.player.state() != QMediaPlayer.StoppedState:
            self.player.stop()

    # ───────────── 저장(원본 유지) ─────────────
    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save")
            return
        time_array = np.array(self.full_time)
        time_array = time_array - time_array[0]
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

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = WearableStethoUI(); ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

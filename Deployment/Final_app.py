import sys
import asyncio
import threading
import os
import time
import json
import numpy as np

from collections import deque

from PyQt5 import QtWidgets, QtCore, QtGui
from bleak import BleakScanner, BleakClient

import librosa
import joblib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# ===== BLE / Stream constants =====
CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE    = "0000eef3-0000-1000-8000-00805f9b34fb"
SAMPLE_RATE      = 4000
WINDOW_DURATION  = 4.0  # seconds
WINDOW_SIZE      = int(SAMPLE_RATE * WINDOW_DURATION)
DT               = 1.0 / SAMPLE_RATE

# ===== Energy plot params =====
ENERGY_WIN_SEC   = 0.05     # boxcar window for MA of squared signal
GAUSS_SIGMA_SEC  = 0.02     # gaussian smoothing sigma (sec)


# ---------- Utility: parse 24-bit signed big-endian ----------
def parse_24bit_signed(data: bytes) -> np.ndarray:
    """
    Parse a stream of 24-bit big-endian signed ints into int32 array.
    """
    values = []
    for i in range(0, len(data), 3):
        raw = data[i:i+3]
        if len(raw) < 3:
            break
        # Sign-extend to 32 bits:
        if raw[0] & 0x80:
            val = int.from_bytes(b'\xFF' + raw, byteorder='big', signed=True)
        else:
            val = int.from_bytes(b'\x00' + raw, byteorder='big', signed=True)
        values.append(val)
    return np.array(values, dtype=np.int32)


# ---------- Real-time feature extraction + inference ----------
class RealtimeFeatureAndInference:
    """
    Real-time windowing, feature extraction, and softmax inference.

    Stream SR: 4 kHz
    Window: 1.0 s (4000 samples), Hop: 0.25 s (1000 samples)
    Feature: Resample to 16 kHz -> Log-Mel(64 mel bands, 64ms win, 32ms hop, 50–7900 Hz)
             -> per-sample standardization -> temporal mean pooling -> (64,) feature vector
    Inference: StandardScaler -> LR & MLP predict_proba -> average softmax
               (optional) thresholds.json for per-class biasing before argmax
    """
    def __init__(
        self,
        model_dir: str,
        sr_stream: int = 4000,
        win_sec: float = 1.0,
        hop_sec: float = 0.25,
        sr_feat: int = 16000,
        n_mels: int = 64,
        win_ms: float = 64.0,
        hop_ms: float = 32.0,
        fmin: int = 50,
        fmax: int = 7900
    ):
        # Stream-side window/hop (4 kHz domain)
        self.sr_stream = sr_stream
        self.win_n = int(sr_stream * win_sec)   # e.g., 4000
        self.hop_n = int(sr_stream * hop_sec)   # e.g., 1000
        self.buf = deque(maxlen=self.win_n)
        self._hop_counter = 0

        # Feature params (must match training)
        self.sr_feat = sr_feat
        self.n_mels = n_mels
        self.win_len = int(sr_feat * (win_ms / 1000.0))
        self.hop_len = int(sr_feat * (hop_ms / 1000.0))
        self.fmin, self.fmax = fmin, fmax

        # Load scaler and models
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.lr = joblib.load(os.path.join(model_dir, "model_lr.pkl"))
        self.mlp = joblib.load(os.path.join(model_dir, "model_mlp.pkl"))

        # Try to load thresholds + class_names
        self.thresholds = None
        self.class_names = None
        thr_path = os.path.join(model_dir, "thresholds.json")
        if os.path.exists(thr_path):
            with open(thr_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.class_names = obj.get("class_names", None)
            thr_vals = obj.get("thresholds", None)
            if thr_vals is not None:
                self.thresholds = np.array(thr_vals, dtype=np.float32)

        # If class_names missing, try to infer from model classes_
        # (Note: order must match models; ideally both LR/MLP classes_ are identical.)
        if self.class_names is None:
            try:
                self.class_names = [str(x) for x in self.lr.classes_]
            except Exception:
                pass

    def _extract_feat_vec(self, x_4k: np.ndarray) -> np.ndarray:
        """
        Convert 1s 4 kHz signal into a (64,) feature vector, following training pipeline.
        """
        # Resample 4 kHz -> 16 kHz
        x16 = librosa.resample(x_4k.astype(np.float32), orig_sr=self.sr_stream, target_sr=self.sr_feat)

        # Log-Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=x16, sr=self.sr_feat, n_fft=2048,
            hop_length=self.hop_len, win_length=self.win_len,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=2.0
        )
        logmel = librosa.power_to_db(S, ref=np.max)

        # Per-sample normalization (mean/std)
        m = logmel.mean()
        s = logmel.std() + 1e-6
        logmel = (logmel - m) / s

        # Temporal mean pooling -> (64,)
        feat_vec = logmel.mean(axis=1).astype(np.float32)
        return feat_vec

    def push(self, new_samples_4k: np.ndarray):
        """
        Push new 4 kHz samples and return zero or more inference results.

        Returns:
            list of (pred_idx: int, probs: np.ndarray[K,], class_names: list[str] or None)
        """
        out = []
        for v in new_samples_4k:
            self.buf.append(float(v))
            self._hop_counter += 1

            if len(self.buf) == self.win_n and self._hop_counter >= self.hop_n:
                # A 1s window is ready at hop interval
                x_win = np.array(self.buf, dtype=np.float32)  # (4000,)
                feat = self._extract_feat_vec(x_win).reshape(1, -1)  # (1, 64)

                # Scale
                feat_sc = self.scaler.transform(feat)

                # Softmax probs (average LR and MLP)
                p_lr = self.lr.predict_proba(feat_sc)
                p_mlp = self.mlp.predict_proba(feat_sc)
                probs = (p_lr + p_mlp) / 2.0  # shape (1, K)

                # Optional thresholds adjustment before argmax
                if self.thresholds is not None:
                    adj = probs - self.thresholds[None, :]
                    pred = int(np.argmax(adj, axis=1)[0])
                else:
                    pred = int(np.argmax(probs, axis=1)[0])

                out.append((pred, probs.flatten(), self.class_names))
                self._hop_counter = 0
        return out


# ---------- Main Application ----------
class BluetoothSoundApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLE Sensor Viewer")
        self.resize(1300, 700)

        # ====== Top control bar ======
        self.device_selector = QtWidgets.QComboBox()
        self.device_selector.setFixedWidth(300)
        self.scan_button = QtWidgets.QPushButton("Scan")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save")

        self.cue_button  = QtWidgets.QPushButton("Cue1")
        self.cue1_button = QtWidgets.QPushButton("Cue2")

        self.freq_input = QtWidgets.QLineEdit()
        self.amp_input  = QtWidgets.QLineEdit()
        self.freq_input.setPlaceholderText("freq (0-65535)")
        self.amp_input.setPlaceholderText("amp (0-255)")

        # Cue3 (tone) pressed/released
        self.cue3_button = QtWidgets.QPushButton("Cue3 (hold)")
        self.cue3_button.setCheckable(False)
        self.cue3_button.pressed.connect(lambda: self.send_cue(3))
        self.cue3_button.released.connect(lambda: self.send_cue(4))

        self.byte_rate_label = QtWidgets.QLabel("Sample rate: 0 samples/s")

        # Top bar layout
        h_layout_1 = QtWidgets.QHBoxLayout()
        h_layout_1.addWidget(self.device_selector)
        v_layout_1 = QtWidgets.QVBoxLayout()
        v_layout_1.addWidget(self.scan_button)
        v_layout_1.addWidget(self.start_button)
        v_layout_2 = QtWidgets.QVBoxLayout()
        v_layout_2.addWidget(self.connect_button)
        v_layout_2.addWidget(self.stop_button)
        v_layout_3 = QtWidgets.QVBoxLayout()
        v_layout_3.addWidget(self.save_button)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Freq:", self.freq_input)
        form_layout.addRow("Amp:", self.amp_input)

        v_layout_4 = QtWidgets.QVBoxLayout()
        v_layout_4.addWidget(self.cue_button)
        v_layout_4.addWidget(self.cue1_button)
        v_layout_4.addWidget(self.cue3_button)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(QtWidgets.QLabel("Select Device:"))
        top_layout.addLayout(h_layout_1)
        top_layout.addLayout(v_layout_1)
        top_layout.addLayout(v_layout_2)
        top_layout.addLayout(v_layout_3)
        top_layout.addLayout(v_layout_4)
        top_layout.addLayout(form_layout)
        top_layout.addStretch(1)
        top_layout.addWidget(self.byte_rate_label)

        # ====== Left: Plots (signal + energy) ======
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.15)
        self.ax_sig = self.fig.add_subplot(gs[0])
        self.ax_eng = self.fig.add_subplot(gs[1], sharex=self.ax_sig)

        self.line, = self.ax_sig.plot([], [], lw=1, color='yellow')
        self.ax_sig.set_ylabel("Voltage (V)")
        self.ax_sig.set_xlabel("")

        self.energy_line, = self.ax_eng.plot([], [], lw=2)
        self.ax_eng.set_ylabel("Energy")
        self.ax_eng.set_xlabel("Time (s)")
        self.ax_eng.set_ylim(0, 0.00003)
        self.ax_eng.set_autoscale_on(False)

        for ax in (self.ax_sig, self.ax_eng):
            ax.set_facecolor('black')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
        self.fig.patch.set_facecolor('black')

        # Buffers for plotting
        self.audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE, dtype=np.float32)
        self.time_counter = 0.0

        # Energy kernels
        self.energy_win_size = max(1, int(ENERGY_WIN_SEC * SAMPLE_RATE))
        self.energy_box_kernel = np.ones(self.energy_win_size, dtype=float) / self.energy_win_size
        sigma_samples = max(1, int(GAUSS_SIGMA_SEC * SAMPLE_RATE))
        self.gauss_radius = int(3 * sigma_samples)
        gx = np.arange(-self.gauss_radius, self.gauss_radius + 1)
        self.gauss_kernel = np.exp(-(gx**2) / (2 * (sigma_samples**2)))
        self.gauss_kernel = self.gauss_kernel / self.gauss_kernel.sum()

        # ====== Right: Classification + Actuator panels ======
        right_panel = QtWidgets.QVBoxLayout()

        # Classification group
        cls_group = QtWidgets.QGroupBox("Classification")
        cls_layout = QtWidgets.QVBoxLayout()

        self.pred_label = QtWidgets.QLabel("Prediction: (waiting)")
        self.pred_label.setStyleSheet("font-weight: bold;")

        self.prob_table = QtWidgets.QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Class", "Prob (%)"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.prob_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.prob_table.setFixedWidth(350)

        cls_layout.addWidget(self.pred_label)
        cls_layout.addWidget(self.prob_table)
        cls_group.setLayout(cls_layout)

        # Actuator status group
        act_group = QtWidgets.QGroupBox("Actuator Status")
        act_layout = QtWidgets.QVBoxLayout()

        self.last_cue_label = QtWidgets.QLabel("Last Cue: (none)")
        self.tone_status_label = QtWidgets.QLabel("Tone (Cue3): Inactive")
        self.tone_status_label.setStyleSheet("color: gray; font-weight: bold;")

        act_layout.addWidget(self.last_cue_label)
        act_layout.addWidget(self.tone_status_label)
        act_layout.addStretch(1)
        act_group.setLayout(act_layout)

        right_panel.addWidget(cls_group)
        right_panel.addWidget(act_group)
        right_panel.addStretch(1)

        # ====== Assemble main layout (left plots + right panel) ======
        left_column = QtWidgets.QVBoxLayout()
        left_column.addLayout(top_layout)
        left_column.addWidget(self.canvas)

        main_h = QtWidgets.QHBoxLayout()
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_column)
        main_h.addWidget(left_container, stretch=1)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        main_h.addWidget(right_container, stretch=0)

        self.setLayout(main_h)

        # ====== Streaming / BLE / timers ======
        self.streaming = False
        self.full_time = []
        self.full_audio = []
        self.sample_counter = 0
        self.last_rate_check = time.time()

        # BLE / asyncio loop thread
        self.devices = {}
        self.client = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.notify_started = False

        # Plot updater
        self.updater = QtCore.QTimer(self)  # QTimer는 QtCore 소속, parent로 self 주면 자동 정리도 됨
        self.updater.timeout.connect(self.update_plot)

        # Connect UI signals
        self.scan_button.clicked.connect(self.scan_devices)
        self.connect_button.clicked.connect(self.connect_device)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.cue_button.clicked.connect(lambda: self.send_cue(1))
        self.cue1_button.clicked.connect(lambda: self.send_cue(2))
        self.save_button.clicked.connect(self.save_data)

        # Simple IIR state vars
        self.lp_prev = 0.0
        self.hp_lp_prev = 0.0
        self.bp_lp_prev = 0.0
        self.bp_hp_prev = 0.0

        # Inference engine (set your actual model directory)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.model_dir = r"D:\Stethoscope_Project\Deployment\Group Split\model\run_20251008_172910"  # TODO: change to your path
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if os.path.isdir(self.model_dir):
            try:
                self.infer = RealtimeFeatureAndInference(self.model_dir, sr_stream=SAMPLE_RATE)
            except Exception as e:
                self.infer = None
                QtWidgets.QMessageBox.warning(self, "Model Load Error", str(e))
        else:
            self.infer = None
            QtWidgets.QMessageBox.information(self, "Model not found",
                                              "Please set a valid model_dir path in the code.")

        # Actuator state
        self.tone_active = False   # Cue3 pressed -> True, released (Cue4) -> False

        # Initialize prob table if class names known
        if self.infer and self.infer.class_names:
            self._init_prob_table(self.infer.class_names)

    # ---------- Filtering (1st-order IIR) ----------
    def lowpass_iir(self, x: np.ndarray, fc: float, fs: float) -> np.ndarray:
        """
        1st order low-pass:
        y[n] = y[n-1] + alpha * (x[n] - y[n-1]), alpha = 1 - exp(-2*pi*fc/fs)
        """
        if fc <= 0:
            return x
        alpha = 1.0 - np.exp(-2.0 * np.pi * fc / fs)
        y = np.empty_like(x)
        y_prev = self.lp_prev
        for i, xi in enumerate(x):
            y_prev = y_prev + alpha * (xi - y_prev)
            y[i] = y_prev
        self.lp_prev = y_prev
        return y

    def highpass_iir(self, x: np.ndarray, fc: float, fs: float) -> np.ndarray:
        """
        1st order high-pass via y = x - LP(x)
        """
        if fc <= 0:
            return x
        alpha = 1.0 - np.exp(-2.0 * np.pi * fc / fs)
        y = np.empty_like(x)
        lp = self.hp_lp_prev
        for i, xi in enumerate(x):
            lp = lp + alpha * (xi - lp)
            y[i] = xi - lp
        self.hp_lp_prev = lp
        return y

    def bandpass_iir(self, x: np.ndarray, f_low: float, f_high: float, fs: float) -> np.ndarray:
        """
        1st order band-pass: HP(f_low) -> LP(f_high)
        """
        y = x
        if f_low is not None and f_low > 0:
            alpha_hp = 1.0 - np.exp(-2.0 * np.pi * f_low / fs)
            hp = np.empty_like(y)
            hp_prev = self.bp_hp_prev
            for i, xi in enumerate(y):
                hp_prev = hp_prev + alpha_hp * (xi - hp_prev)
                hp[i] = xi - hp_prev
            self.bp_hp_prev = hp_prev
            y = hp

        if f_high is not None and f_high > 0:
            f_high_eff = min(f_high, 0.49 * fs)
            alpha_lp = 1.0 - np.exp(-2.0 * np.pi * f_high_eff / fs)
            lp = np.empty_like(y)
            lp_prev = self.bp_lp_prev
            for i, xi in enumerate(y):
                lp_prev = lp_prev + alpha_lp * (xi - lp_prev)
                lp[i] = lp_prev
            self.bp_lp_prev = lp_prev
            y = lp
        return y

    # ---------- Plot update ----------
    def update_plot(self):
        # Update signal line
        self.line.set_data(self.x_buffer, self.audio_buffer)

        # Compute energy curve (square -> boxcar MA -> gaussian)
        sq = self.audio_buffer ** 2
        energy = np.convolve(sq, self.energy_box_kernel, mode='same')
        energy_smooth = np.convolve(energy, self.gauss_kernel, mode='same')
        self.energy_line.set_data(self.x_buffer, energy_smooth)

        # Refresh axes
        self.ax_sig.relim(); self.ax_sig.autoscale_view()
        # Energy axis is fixed unless you change autoscale policy

        self.canvas.draw()

    # ---------- BLE scanning / connection ----------
    def scan_devices(self):
        async def run_scan():
            devices = await BleakScanner.discover(timeout=3)
            self.devices = {f"{d.name} [{d.address}]": d.address for d in devices if d.name}
            self.device_selector.clear()
            self.device_selector.addItems(self.devices.keys())
        asyncio.run_coroutine_threadsafe(run_scan(), self.loop)

    def connect_device(self):
        addr = self.devices.get(self.device_selector.currentText())
        if not addr:
            return

        async def run_connect():
            # Note: Do not pass 'loop=' on recent bleak versions
            self.client = BleakClient(addr)
            try:
                await self.client.connect()
                if self.client.is_connected:
                    print(f"[✓] Connected to {addr}")
            except Exception as e:
                print(f"[!] Connect failed: {e}")

        asyncio.run_coroutine_threadsafe(run_connect(), self.loop)

    # ---------- Start / stop streaming ----------
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

        # Reset buffers
        self.audio_buffer[:] = 0.0
        self.time_counter = 0.0
        self.full_time = []
        self.full_audio = []

        # Start UI updater
        self.updater.start(50)  # 20 Hz UI refresh

        asyncio.run_coroutine_threadsafe(run_notify(), self.loop)
        self.streaming = True

    def stop_stream(self):
        if not self.client or not self.notify_started:
            return

        async def stop():
            try:
                await self.client.stop_notify(CHAR_UUID_STREAM)
                await self.client.disconnect()
                print("[✓] Stopped stream and disconnected")
                self.notify_started = False
            except Exception as e:
                print(f"[!] Stop error: {e}")

        self.updater.stop()
        asyncio.run_coroutine_threadsafe(stop(), self.loop)
        self.streaming = False

    # ---------- BLE notification handler ----------
    def handle_data(self, handle, data: bytes):
        if not self.streaming:
            return

        # Parse 24-bit -> volts
        parsed = parse_24bit_signed(data)
        ADC_SCALE = 2.4 / (2**23)  # volts per count
        parsed = parsed.astype(np.float32) * ADC_SCALE

        # Keep a raw copy for feature extraction (must match training domain)
        raw_for_feat = parsed.copy()

        # For plotting, apply a HPF to clean low drifts (visual only)
        fc = 350.0
        parsed = self.highpass_iir(parsed, fc=fc, fs=SAMPLE_RATE)

        if parsed.size == 0:
            return

        # Slide 4s window for plotting
        shift_len = len(parsed)
        self.audio_buffer = np.roll(self.audio_buffer, -shift_len)
        self.audio_buffer[-shift_len:] = parsed

        self.time_counter += shift_len / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION,
                                    self.time_counter, WINDOW_SIZE, dtype=np.float32)

        # Save full stream for CSV
        start_time = self.time_counter - shift_len / SAMPLE_RATE
        full_time_array = start_time + np.arange(shift_len) / SAMPLE_RATE
        self.full_time.extend(full_time_array.tolist())
        self.full_audio.extend(parsed.tolist())

        # Sample rate label update each ~1 s
        self.sample_counter += shift_len
        now = time.time()
        elapsed = now - self.last_rate_check
        if elapsed >= 1.0:
            rate_sps = self.sample_counter / elapsed
            self.byte_rate_label.setText(f"Sample rate: {rate_sps:.1f} samples/s")
            self.sample_counter = 0
            self.last_rate_check = now

        # ===== Real-time inference (softmax) =====
        if self.infer is not None:
            results = self.infer.push(raw_for_feat.astype(np.float32))
            if results:
                pred_idx, probs, class_names = results[-1]
                # Update right-panel UI
                self._update_classification_ui(pred_idx, probs, class_names)

    # ---------- Save CSV ----------
    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save")
            return

        time_array = np.array(self.full_time, dtype=np.float64)
        time_array = time_array - time_array[0]  # start at 0
        audio_array = np.array(self.full_audio, dtype=np.float32)

        sq = audio_array ** 2
        energy = np.convolve(sq, self.energy_box_kernel, mode='same')
        energy_smooth = np.convolve(energy, self.gauss_kernel, mode='same')

        data_to_save = np.column_stack((time_array, audio_array, energy, energy_smooth))

        # Save path (change if needed)
        save_dir = os.path.join(os.path.expanduser("~"), "Downloads", "sound")
        os.makedirs(save_dir, exist_ok=True)
        base_filename = "recorded_data"
        extension = ".csv"
        full_path = os.path.join(save_dir, base_filename + extension)

        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")
            counter += 1

        try:
            header = "Time(s),Amplitude(V),Energy(boxcar),Energy_smooth(gauss)"
            np.savetxt(full_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.9f")
            print(f"[✓] Data saved to {full_path}")
        except Exception as e:
            print(f"[!] Save error: {e}")

    # ---------- Cue writer + Actuator status ----------
    def send_cue(self, cue_id=1):
        """
        Send cue command over BLE and update Actuator Status UI.
        - Cue1 (1): momentary action
        - Cue2 (2): momentary action
        - Cue3 (3): start tone (active until Cue4)
        - Cue4 (4): stop tone
        """
        if not self.client or not self.client.is_connected:
            return

        async def send():
            try:
                if cue_id == 1:
                    payload = b'\x01'
                    self._set_last_cue_ui("Cue1")
                elif cue_id == 2:
                    payload = b'\x02'
                    self._set_last_cue_ui("Cue2")
                elif cue_id == 3:
                    # Parse freq/amp safely
                    try:
                        freq = int(self.freq_input.text())
                        amp  = int(self.amp_input.text())
                        if not (0 <= freq <= 65535 and 0 <= amp <= 255):
                            raise ValueError
                    except Exception:
                        QtWidgets.QMessageBox.warning(self, "Invalid input",
                            "Freq must be 0..65535, Amp must be 0..255")
                        return
                    payload = bytes([0x03, freq & 0xFF, (freq >> 8) & 0xFF, amp])
                    self.tone_active = True
                    self._set_last_cue_ui(f"Cue3 (tone ON, f={freq}, a={amp})")
                    self._update_tone_status_ui()
                elif cue_id == 4:
                    payload = bytes([0x04])
                    self.tone_active = False
                    self._set_last_cue_ui("Cue4 (tone OFF)")
                    self._update_tone_status_ui()
                else:
                    payload = b'\x00'
                    self._set_last_cue_ui("Unknown")
                await self.client.write_gatt_char(CHAR_UUID_CUE, payload, response=True)
                print(f"[✓] Cue {cue_id} sent")
            except Exception as e:
                print(f"[!] Cue send error: {e}")

        asyncio.run_coroutine_threadsafe(send(), self.loop)

    # ---------- UI helpers (right panel) ----------
    def _init_prob_table(self, class_names):
        """
        Initialize probability table with class names.
        """
        K = len(class_names)
        self.prob_table.setRowCount(K)
        for i, name in enumerate(class_names):
            name_item = QtWidgets.QTableWidgetItem(str(name))
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            prob_item = QtWidgets.QTableWidgetItem("0.00")
            prob_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            prob_item.setFlags(prob_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.prob_table.setItem(i, 0, name_item)
            self.prob_table.setItem(i, 1, prob_item)
        self.prob_table.resizeColumnsToContents()

    def _update_classification_ui(self, pred_idx: int, probs: np.ndarray, class_names):
        """
        Update prediction label and per-class percentage probabilities.
        """
        # Initialize table lazily if not set
        if self.prob_table.rowCount() == 0:
            names = class_names if class_names else [f"Class {i}" for i in range(len(probs))]
            self._init_prob_table(names)

        # Prediction label
        if class_names and pred_idx < len(class_names):
            pred_text = f"Prediction: {class_names[pred_idx]}"
        else:
            pred_text = f"Prediction: Class {pred_idx}"
        self.pred_label.setText(pred_text)

        # Fill probability table (percentage with two decimals)
        K = len(probs)
        if self.prob_table.rowCount() != K:
            # Re-init if class count changed
            names = class_names if class_names else [f"Class {i}" for i in range(K)]
            self._init_prob_table(names)
        for i in range(K):
            val = f"{probs[i]*100.0:0.2f}"
            item = self.prob_table.item(i, 1)
            if item is None:
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.prob_table.setItem(i, 1, item)
            else:
                item.setText(val)

        # Optional: visually highlight the predicted row
        for r in range(self.prob_table.rowCount()):
            for c in range(self.prob_table.columnCount()):
                it = self.prob_table.item(r, c)
                if it:
                    if r == pred_idx:
                        it.setBackground(QtGui.QBrush(QtGui.QColor(35, 80, 160)))
                        it.setForeground(QtGui.QBrush(QtGui.QColor("white")))
                    else:
                        it.setBackground(QtGui.QBrush(QtGui.QColor("white")))
                        it.setForeground(QtGui.QBrush(QtGui.QColor("black")))

        self.prob_table.resizeColumnsToContents()

    def _set_last_cue_ui(self, text: str):
        """
        Update last sent cue label with timestamp.
        """
        ts = time.strftime("%H:%M:%S")
        self.last_cue_label.setText(f"Last Cue: {text} @ {ts}")

    def _update_tone_status_ui(self):
        """
        Update tone status label color/text.
        """
        if self.tone_active:
            self.tone_status_label.setText("Tone (Cue3): Active")
            self.tone_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.tone_status_label.setText("Tone (Cue3): Inactive")
            self.tone_status_label.setStyleSheet("color: gray; font-weight: bold;")


# ---------- Entry point ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BluetoothSoundApp()
    window.show()
    sys.exit(app.exec_())

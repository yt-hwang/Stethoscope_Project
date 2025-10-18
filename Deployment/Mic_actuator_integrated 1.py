import sys
import asyncio
import threading
import os
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
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


# ===== [NEW] Energy plot params =====
ENERGY_WIN_SEC = 0.05                   # Moving average(sum of square) window (sec)
GAUSS_SIGMA_SEC = 0.02                # Guassian smoothing sigma (sec)



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

class BluetoothSoundApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLE Sensor Viewer")
        self.resize(1000, 600)

        
        self.device_selector = QtWidgets.QComboBox()
        self.device_selector.setFixedWidth(300)  # Set width to 300 pixels
        self.scan_button = QtWidgets.QPushButton("Scan")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save")
        self.cue_button = QtWidgets.QPushButton("Cue1")
        self.cue1_button = QtWidgets.QPushButton("Cue2")
        
        self.freq_input = QtWidgets.QLineEdit()
        self.amp_input = QtWidgets.QLineEdit()

        freq_layout = QtWidgets.QHBoxLayout()
        freq_label = QtWidgets.QLabel("Freq:")
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)

        amp_layout = QtWidgets.QHBoxLayout()
        amp_label = QtWidgets.QLabel("Amp:")
        amp_layout.addWidget(amp_label)
        amp_layout.addWidget(self.amp_input)

        
        self.cue3_button = QtWidgets.QPushButton("Cue3")
        self.cue3_button.setEnabled(True)
        self.cue3_button.pressed.connect(lambda: self.send_cue(3))
        self.cue3_button.released.connect(lambda: self.send_cue(4))
        
        self.byte_rate_label = QtWidgets.QLabel("Sample rate: 0 samples/s")

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
        
        v_layout_4 = QtWidgets.QVBoxLayout()
        v_layout_4.addWidget(self.cue_button)
        v_layout_4.addWidget(self.cue1_button)
        
        v_layout_5 = QtWidgets.QVBoxLayout()
        v_layout_5.addLayout(freq_layout)
        v_layout_5.addLayout(amp_layout)
        v_layout_5.addWidget(self.cue3_button)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(QtWidgets.QLabel("Select Device:"))
        top_layout.addLayout(h_layout_1)
        top_layout.addLayout(v_layout_1)
        top_layout.addLayout(v_layout_2)
        top_layout.addLayout(v_layout_3)
        top_layout.addLayout(v_layout_4)
        top_layout.addLayout(v_layout_5)
        top_layout.addWidget(self.byte_rate_label)

        
        # ===== [CHANGED] Figure: 2-row layout =====
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # 2개의 서브플롯 (Signal 위, Energy 아래)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.15)
        self.ax_sig = self.fig.add_subplot(gs[0])
        self.ax_eng = self.fig.add_subplot(gs[1], sharex=self.ax_sig)

        # Signal line
        self.line, = self.ax_sig.plot([], [], lw=1, color='yellow')
        self.ax_sig.set_ylabel("Voltage (V)")
        self.ax_sig.set_xlabel("")  

        # Energy line
        self.energy_line, = self.ax_eng.plot([], [], lw=2)
        self.ax_eng.set_ylabel("Energy")
        self.ax_eng.set_xlabel("Time (s)")
        self.ax_eng.set_ylim(0, 0.00003)  
        self.ax_eng.set_autoscale_on(False)  # off auto scale

        # 다크 테마 스타일 통일
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

        # ===== [NEW] Buffers =====
        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)

        # ===== [NEW] Precompute kernels for energy/smoothing =====
        self.energy_win_size = max(1, int(ENERGY_WIN_SEC * SAMPLE_RATE))
        self.energy_box_kernel = np.ones(self.energy_win_size, dtype=float) / self.energy_win_size

        sigma_samples = max(1, int(GAUSS_SIGMA_SEC * SAMPLE_RATE))
        self.gauss_radius = int(3 * sigma_samples)
        gx = np.arange(-self.gauss_radius, self.gauss_radius + 1)
        self.gauss_kernel = np.exp(-(gx**2) / (2 * (sigma_samples**2)))
        self.gauss_kernel = self.gauss_kernel / self.gauss_kernel.sum()

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)

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

        self.scan_button.clicked.connect(self.scan_devices)
        self.connect_button.clicked.connect(self.connect_device)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.cue_button.clicked.connect(lambda : self.send_cue(1))
        self.cue1_button.clicked.connect(lambda : self.send_cue(2)  )
        self.save_button.clicked.connect(self.save_data)
        
        self.lp_prev = 0.0
        self.hp_lp_prev = 0.0
        self.bp_lp_prev = 0.0
        self.bp_hp_prev = 0.0
    
    def lowpass_iir(self, x: np.ndarray, fc: float, fs: float) -> np.ndarray:
        """
        1차 IIR 로우패스
        y[n] = y[n-1] + alpha * (x[n] - y[n-1])
        alpha = 1 - exp(-2*pi*fc/fs)
        """
        if fc <= 0:
            return x  # raw signal if cut-off freq is 0
        
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
        1st order high pass: y = x - LP(x)
        LP: y_lp[n] = y_lp[n-1] + alpha*(x[n] - y_lp[n-1])
        alpha = 1 - exp(-2*pi*fc/fs)
        """
        if fc <= 0:
            return x
        alpha = 1.0 - np.exp(-2.0 * np.pi * fc / fs)

        y = np.empty_like(x)
        lp = self.hp_lp_prev
        for i, xi in enumerate(x):
            lp = lp + alpha * (xi - lp)   # Low pass
            y[i] = xi - lp                # High pass = Input - Low pass
        self.hp_lp_prev = lp
        return y

    def bandpass_iir(self, x: np.ndarray, f_low: float, f_high: float, fs: float) -> np.ndarray:
        """
        1st order bandpass: HP(f_low) → LP(f_high)
        - f_low > 0 : High pass
        - f_high > 0 : Low pass
        """
        y = x

        # --- High-pass stage (remove below f_low) ---
        if f_low is not None and f_low > 0:
            alpha_hp = 1.0 - np.exp(-2.0 * np.pi * f_low / fs)
            hp = np.empty_like(y)
            hp_prev = self.bp_hp_prev
            for i, xi in enumerate(y):
                hp_prev = hp_prev + alpha_hp * (xi - hp_prev)  # LP
                hp[i] = xi - hp_prev                            # HP = x - LP(x)
            self.bp_hp_prev = hp_prev
            y = hp

        # --- Low-pass stage (remove above f_high) ---
        if f_high is not None and f_high > 0:
            # Not exceeding Nyquist (fs/2 slightly lower than)
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


    def custom_scale(self, y, t=0.015, gamma=1.5, min_factor=0.2):
        """
        When |y| < t, the scaling factor increases in proportion to |y|:
        factor = min_factor + (1 - min_factor) * (|y|/t)^gamma
        => The smaller |y| is, the closer the factor is to min_factor, so the value is reduced more strongly.
        When |y| >= t, the factor is 1 (unchanged).

        t: threshold (0.015)
        gamma: curvature (>1 makes the decrease sharper, <1 makes it gentler)
        min_factor: minimum factor at |y| = 0 (range 0–1)
        """
        y_scaled = y.copy()
        abs_y = np.abs(y_scaled)

        # Default factor 1
        factor = np.ones_like(y_scaled)

        # Apply continuous scaling only for the region smaller than the threshold
        mask = abs_y < t
        # Smoothly increase from min_factor -> 1 over 0 to t
        factor[mask] = min_factor + (1.0 - min_factor) * (abs_y[mask] / t) ** gamma

        return y_scaled * factor

    
    def custom_root_scale(self, y, t=0.03, min_factor=0.01):
        """
        |y| < t: factor = min_factor + (1 - min_factor) * sqrt(|y|/t)
        |y| >= t: factor = 1
        """
        y_scaled = y.copy()
        abs_y = np.abs(y_scaled)
        factor = np.ones_like(y_scaled)

        mask = abs_y < t
        factor[mask] = min_factor + (1.0 - min_factor) * np.sqrt(abs_y[mask] / t)

        return y_scaled * factor

    def custom_log_scale(self, y, t=0.03, min_factor=0.2):
        """
        |y| < t: factor = min_factor + (1 - min_factor) * log1p(|y|/t) / log(2)
        |y| >= t: factor = 1
        
        """
        y_scaled = y.copy()
        abs_y = np.abs(y_scaled)
        factor = np.ones_like(y_scaled)

        mask = abs_y < t
        factor[mask] = min_factor + (1.0 - min_factor) * np.log1p(abs_y[mask] / t) / np.log(2)

        return y_scaled * factor

    def update_plot(self):
        # 1) Signal plot
        self.line.set_data(self.x_buffer, self.audio_buffer)

        # 2) Calculation of energy (Square → Moving average) + Guassian Smoothing
        #    same mode same length as x_buffer
        sq = self.audio_buffer ** 2
        energy = np.convolve(sq, self.energy_box_kernel, mode='same')
        energy_smooth = np.convolve(energy, self.gauss_kernel, mode='same')

        # 3) Update energy plot
        self.energy_line.set_data(self.x_buffer, energy_smooth)

        # 4) axis scale auto update
        self.ax_sig.relim(); self.ax_sig.autoscale_view()
        self.ax_eng.relim(); self.ax_eng.autoscale_view()

        # 5) plot
        self.canvas.draw()


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
        parsed = self.highpass_iir(parsed, fc=fc, fs=SAMPLE_RATE)
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

        self.sample_counter += shift_len
        now = time.time()
        elapsed = now - self.last_rate_check
        if elapsed >= 1.0:
            rate_sps = self.sample_counter / elapsed
            self.byte_rate_label.setText(f"Sample rate: {rate_sps:.1f} samples/s")
            self.sample_counter = 0
            self.last_rate_check = now


        

    
    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save")
            return

        # 1) array + time set to 0 
        time_array = np.array(self.full_time)
        time_array = time_array - time_array[0]
        audio_array = np.array(self.full_audio)


        sq = audio_array ** 2
        energy = np.convolve(sq, self.energy_box_kernel, mode='same')
        energy_smooth = np.convolve(energy, self.gauss_kernel, mode='same')

        # 3) Save table
        #    time, amplitude(Volt), energy, energy_smooth(Guassian)
        data_to_save = np.column_stack((time_array, audio_array, energy, energy_smooth))

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
            header = "Time(s),Amplitude(V),Energy(boxcar),Energy_smooth(gauss)"
            np.savetxt(full_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.9f")
            print(f"[✓] Data saved to {full_path}")
        except Exception as e:
            print(f"[!] Save error: {e}")

    





    def send_cue(self, cue_id=1):
        if not self.client or not self.client.is_connected:
            return

        async def send():
            try:
                if cue_id == 1:
                    payload = b'\x01'  # Cue 1
                elif cue_id == 2:
                    payload = b'\x02'  # Cue 2
                elif cue_id == 3:
                    freq = int(self.freq_input.text())
                    amp = int(self.amp_input.text())
                    payload = bytes([0x03, freq & 0xFF, (freq >> 8) & 0xFF, amp])  # Cue 3
                elif cue_id == 4:
                    payload = bytes([0x04])
                else:
                    payload = b'\x00'  # Unknown / stop signal

                await self.client.write_gatt_char(CHAR_UUID_CUE, payload, response=True)
                print(f"[✓] Cue {cue_id} sent")
            except Exception as e:
                print(f"[!] Cue send error: {e}")

        asyncio.run_coroutine_threadsafe(send(), self.loop)
        









if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BluetoothSoundApp()
    window.show()
    sys.exit(app.exec_())

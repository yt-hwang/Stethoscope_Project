# Streaming/stream_manager.py
import os, numpy as np
from PyQt5 import QtCore
from Utils.utils import (
    CHAR_UUID_STREAM, SAMPLE_RATE, WINDOW_DURATION, WINDOW_SIZE, parse_24bit_signed
)

class StreamManager(QtCore.QObject):
    """start_stream / stop_stream / handle_data / update_plot / save_data methods"""
    def __init__(self, ble_manager, ax, line, canvas, parent=None):
        super().__init__(parent)
        self.ble = ble_manager        # BLE.BluetoothManager
        self.ax = ax
        self.line = line
        self.canvas = canvas

        # State/buffers 
        self.audio_buffer = np.zeros(WINDOW_SIZE)
        self.x_buffer = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
        self.time_counter = 0.0
        self.full_time = []
        self.full_audio = []
        self.streaming = False
        self.notify_started = False

        # Periodic plot update
        self.updater = QtCore.QTimer(self)
        self.updater.setInterval(50)
        self.updater.timeout.connect(self.update_plot)

    def start_stream(self):
        client = getattr(self.ble, "client", None)
        if not client or not client.is_connected:
            return

        async def run_notify():
            try:
                await client.start_notify(CHAR_UUID_STREAM, self.handle_data)
                self.notify_started = True
                print("[✓] Started streaming")
            except Exception as e:
                print(f"[!] Notify error: {e}")

        # Buffer reset
        self.audio_buffer[:] = 0.0
        self.time_counter = 0.0
        self.full_time.clear()
        self.full_audio.clear()

        self.updater.start()
        import asyncio
        asyncio.run_coroutine_threadsafe(run_notify(), self.ble.loop)
        self.streaming = True

    def stop_stream(self):
        client = getattr(self.ble, "client", None)
        if not client or not self.notify_started:
            return

        async def stop():
            try:
                await client.stop_notify(CHAR_UUID_STREAM)
                await client.disconnect()
                if hasattr(client, "set_disconnected_callback"):
                    client.set_disconnected_callback(None)
                print("[✓] Stopped stream and disconnected")
                self.notify_started = False
            except Exception as e:
                print(f"[!] Stop error: {e}")

        self.updater.stop()
        import asyncio
        asyncio.run_coroutine_threadsafe(stop(), self.ble.loop)
        self.streaming = False

    def handle_data(self, handle, data):
        if not self.streaming:
            return
   
        parsed = parse_24bit_signed(data)
        ADC_SCALE = 2.4 / (2**23)  # ≈ 2.861e-7 V per count
        parsed = parsed * ADC_SCALE
        if parsed.size == 0:
            return

        # Window buffer update
        n = len(parsed)
        self.audio_buffer = np.roll(self.audio_buffer, -n)
        self.audio_buffer[-n:] = parsed

        # x-scale time update
        self.time_counter += n / SAMPLE_RATE
        self.x_buffer = np.linspace(self.time_counter - WINDOW_DURATION,
                                    self.time_counter, WINDOW_SIZE)

        # Full data accumulation
        start_time = self.time_counter - n / SAMPLE_RATE
        full_time_array = start_time + np.arange(n) / SAMPLE_RATE
        self.full_time.extend(full_time_array)
        self.full_audio.extend(parsed)

    def update_plot(self):
        self.line.set_data(self.x_buffer, self.audio_buffer)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def save_data(self):
        if not self.full_time or not self.full_audio:
            print("[!] No full data to save")
            return

        time_array = np.array(self.full_time); time_array -= time_array[0]
        audio_array = np.array(self.full_audio)
        data_to_save = np.column_stack((time_array, audio_array))

        save_dir = "C:/Users/dhtpd/Downloads/sound"
        os.makedirs(save_dir, exist_ok=True)
        base, ext = "recorded_data", ".csv"
        full_path = os.path.join(save_dir, base + ext)
        i = 1
        while os.path.exists(full_path):
            full_path = os.path.join(save_dir, f"{base}_{i}{ext}")
            i += 1

        try:
            np.savetxt(full_path, data_to_save, delimiter=",",
                       header="Time(s),Amplitude(V)", comments="", fmt="%.9f")
            print(f"[✓] Data saved to {full_path}")
        except Exception as e:
            print(f"[!] Save error: {e}")

# BLE/bluetooth_manager.py
import asyncio, threading
from PyQt5 import QtCore
from bleak import BleakScanner, BleakClient

class BluetoothManager(QtCore.QObject):
    devices_found = QtCore.pyqtSignal(dict)   # {"Name [addr]": "addr"}
    connected     = QtCore.pyqtSignal(str)    # address
    error         = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.client = None


    def scan_devices(self, timeout: float = 3.0):
        async def run_scan():
            devs = await BleakScanner.discover(timeout=timeout)
            return {f"{d.name} [{d.address}]": d.address for d in devs if d.name}

        fut = asyncio.run_coroutine_threadsafe(run_scan(), self.loop)

        def _on_done(f):
            try:
                self.devices_found.emit(f.result())
            except Exception as e:
                self.error.emit(f"Scan error: {e}")
        fut.add_done_callback(_on_done)

    def connect_device(self, addr: str):
        if not addr:
            return

        async def run_connect():
            try:
                self.client = BleakClient(addr, loop=self.loop)
                await self.client.connect()
                if self.client.is_connected:
                    self.connected.emit(addr)
            except Exception as e:
                self.error.emit(f"Connect failed: {e}")

        asyncio.run_coroutine_threadsafe(run_connect(), self.loop)

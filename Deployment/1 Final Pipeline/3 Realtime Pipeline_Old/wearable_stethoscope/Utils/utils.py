import numpy as np

# Constants (from original)
CHAR_UUID_STREAM = "0000eef2-0000-1000-8000-00805f9b34fb"
CHAR_UUID_CUE = "0000eef3-0000-1000-8000-00805f9b34fb"
PACKET_LENGTH = 180
SAMPLE_RATE = 4000
WINDOW_DURATION = 4.0  # seconds
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
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

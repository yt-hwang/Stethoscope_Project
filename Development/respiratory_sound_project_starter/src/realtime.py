"""Streaming-friendly inference skeleton.

- Maintains a rolling buffer of audio samples
- Computes log-mel features on sliding windows
- Calls a trained model (exported SmallCNN) in a causal way

NOTE: Replace 'get_audio_chunk()' with your actual streaming source.
"""
import time, numpy as np, torch, librosa
from .config import SR, HOP_LEN
from .features import logmel
from .models.cnn_small import SmallCNN

class RollingBuffer:
    def __init__(self, seconds=3.0):
        self.n = int(SR*seconds)
        self.buf = np.zeros(self.n, dtype=np.float32)

    def push(self, x):
        L = len(x)
        if L >= self.n:
            self.buf = x[-self.n:]
        else:
            self.buf = np.concatenate([self.buf[L:], x])

    def get(self):
        return self.buf.copy()

def get_audio_chunk():
    # TODO: Replace with microphone or device input
    return np.zeros(int(0.1*SR), dtype=np.float32)

@torch.inference_mode()
def main():
    model = SmallCNN(num_classes=3)
    model.load_state_dict(torch.load('best.pt', map_location='cpu'))
    model.eval()

    rb = RollingBuffer(seconds=3.0)
    while True:
        x = get_audio_chunk()
        rb.push(x)
        y = rb.get()
        M = logmel(y)  # (n_mels, T)
        # last ~1s window for decision (adjust as needed)
        T = M.shape[1]
        win = 100
        if T >= win:
            crop = M[:, -win:]
            inp = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = model(inp)
            pred = int(torch.argmax(logits, dim=1))
            label = ['breathing','wheezing','noise'][pred]
            print("pred:", label)
        time.sleep(0.1)

if __name__ == '__main__':
    main()

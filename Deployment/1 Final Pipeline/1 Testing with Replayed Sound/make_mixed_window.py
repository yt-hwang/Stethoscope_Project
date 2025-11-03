# tests/make_mixed_window.py
from pathlib import Path
import soundfile as sf
import numpy as np

SR = 16000
OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Test for Realtime Deployment")

# 1) 훈련에서 만든 2초 세그먼트 두 개 경로 (서로 다른 라벨)
A = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Segments_2s_hop500ms/Crackle/KP005_WWS_B004_8.25-9.67_0.00-2.00.wav")     # 예: Crackle 2.00s
B = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Segments_2s_hop500ms/Healthy/H001_B001_6.23-7.52_0.00-2.00.wav")     # 예: Healthy 2.00s

def load2s(p):
    x, sr = sf.read(p)
    assert sr == SR and len(x) == 2*SR, f"{p.name} must be 2.00s@16k"
    return x.astype(np.float32)

a = load2s(A); b = load2s(B)
mixed = np.concatenate([a[:SR], b[-SR:]])  # 앞 1초 + 뒤 1초 = 2초

out = OUT_DIR / f"mixed_{A.stem}_{B.stem}_1s1s.wav"
sf.write(out, mixed, SR)
print("[DONE] made:", out)

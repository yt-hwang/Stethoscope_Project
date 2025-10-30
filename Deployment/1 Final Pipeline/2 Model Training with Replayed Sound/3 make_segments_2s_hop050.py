# make_segments_2s_hop050.py
# -----------------------------------------------------------------------------
# 1) Existing JSON segments created from 16kHz WAV files to 2.0s/0.5s hop
# 2) Include option to remove silent (low energy) segments
# 3) Output/metadata saved under D:\Stethoscope_Project\Deployment\1 Final Pipeline\2 Model Training with Replayed Sound under Segments_2s_hop500ms
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd

# ==== Path settings ====
DEPLOY_ROOT = Path(r"D:\\Stethoscope_Project\\Deployment\\1 Final Pipeline\\2 Model Training with Replayed Sound")
# Location of existing segments (input): Change if needed (e.g. already created Segments_from_JSON)
RAW_SEGMENTS_SRC = Path(r"D:\\Stethoscope_Project\\Deployment\\1 Final Pipeline\\2 Model Training with Replayed Sound\\Output\\Segments_from_JSON")

# Output root (fixed)
OUT_DIR = DEPLOY_ROOT / r"D:\\Stethoscope_Project\\Deployment\\1 Final Pipeline\\2 Model Training with Replayed Sound\\Output\\Segments_2s_hop500ms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Parameters ====
SR = 16000
WIN = 2.0      # seconds
HOP = 0.5      # seconds (75% overlap)
MIN_RMS = 5e-4 # Silent filter. Turn off by None or 0.

def rms_energy(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))

rows = []
for label_dir in RAW_SEGMENTS_SRC.iterdir():
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

    for wav_path in label_dir.glob("*.wav"):
        try:
            x, sr = sf.read(wav_path)
        except Exception as e:
            print(f"[WARN] read fail: {wav_path} -> {e}")
            continue

        if sr != SR:
            # If needed, use librosa.resample. Here we assume SR=16k.
            pass

        dur = len(x) / SR

        # dur <= WIN: Pad and create only one segment
        if dur <= WIN:
            pad = int(WIN * SR) - len(x)
            seg = np.pad(x, (0, pad))
            if (MIN_RMS is None) or (rms_energy(seg) >= MIN_RMS):
                out_name = f"{wav_path.stem}_0.00-{WIN:.2f}.wav"
                out_path = OUT_DIR / label / out_name
                sf.write(out_path, seg, SR)
                rows.append(dict(
                    id=out_name, wav_path=str(out_path),
                    label=label, group="UNK",
                    start=0.00, end=round(WIN, 2),
                    source_file=wav_path.name, parent_seg=wav_path.name
                ))
            continue

        # 2s window, 0.5s hop
        t = 0.0
        while t + WIN <= dur + 1e-6:
            i0 = int(t * SR); i1 = i0 + int(WIN * SR)
            seg = x[i0:i1]

            if (MIN_RMS is None) or (rms_energy(seg) >= MIN_RMS):
                out_name = f"{wav_path.stem}_{t:.2f}-{t+WIN:.2f}.wav"
                out_path = OUT_DIR / label / out_name
                sf.write(out_path, seg, SR)
                rows.append(dict(
                    id=out_name, wav_path=str(out_path),
                    label=label, group="UNK",
                    start=round(t, 2), end=round(t + WIN, 2),
                    source_file=wav_path.name, parent_seg=wav_path.name
                ))
            t += HOP

df = pd.DataFrame(rows)
meta_path = OUT_DIR / "metadata_2s_hop500ms.csv"
df.to_csv(meta_path, index=False)

print(f"[DONE] segments: {len(df)}")
print(f"[OUT ] dir     : {OUT_DIR}")
print(f"[OUT ] meta    : {meta_path}")

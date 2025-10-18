# make_segments_1s_hop025.py
# -----------------------------------------------------------------------------
# 1) 기존 JSON 세그먼트로 만든 16kHz WAV들을 1.0s/0.25s hop으로 재윈도잉
# 2) 무음(저에너지) 구간 제거 옵션 포함
# 3) 출력/메타 모두 D:\Stethoscope_Project\Deployment 하위로 저장
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd

# ==== 경로 설정 ====
DEPLOY_ROOT = Path(r"D:\Stethoscope_Project\Deployment")
# 기존 세그먼트(입력) 위치: 필요시 바꿔줘 (예: 네가 이미 만들어 둔 Segments_from_JSON)
RAW_SEGMENTS_SRC = Path(r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\data\audio\Segments_from_JSON")

# 출력 루트 (고정)
OUT_DIR = DEPLOY_ROOT / r"data\Segments_1s_hop250ms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 파라미터 ====
SR = 16000
WIN = 1.0      # seconds
HOP = 0.25     # seconds (오버랩 75%)
MIN_RMS = 5e-4 # 무음 필터. 끄려면 None 또는 0으로.

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
            # 필요하면 librosa.resample 사용 가능. 여기선 SR=16k 가정.
            pass

        dur = len(x) / SR

        # dur <= 1s: 패딩해서 1개만 생성
        if dur <= WIN:
            pad = int(WIN * SR) - len(x)
            seg = np.pad(x, (0, pad))
            if (MIN_RMS is None) or (rms_energy(seg) >= MIN_RMS):
                out_name = f"{wav_path.stem}_0.00-1.00.wav"
                out_path = OUT_DIR / label / out_name
                sf.write(out_path, seg, SR)
                rows.append(dict(
                    id=out_name, wav_path=str(out_path),
                    label=label, group="UNK",
                    start=0.00, end=1.00,
                    source_file=wav_path.name, parent_seg=wav_path.name
                ))
            continue

        # 1s 창, 0.25s hop
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
meta_path = OUT_DIR / "metadata_1s_hop250ms.csv"
df.to_csv(meta_path, index=False)

print(f"[DONE] segments: {len(df)}")
print(f"[OUT ] dir     : {OUT_DIR}")
print(f"[OUT ] meta    : {meta_path}")

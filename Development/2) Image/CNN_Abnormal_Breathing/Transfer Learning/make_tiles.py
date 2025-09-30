#!/usr/bin/env python3
# make_tiles.py — generate 1.0 s tiles (mel or scalogram) only from BREATHING intervals

from pathlib import Path
import json, csv
import numpy as np
import librosa, matplotlib.pyplot as plt
import pywt

# ---------- CONFIG ----------
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")

MODE = "mel"  # "mel" or "scalo"
OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/Tiles_1s_mel"
               if MODE=="mel" else
               "/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/Tiles_1s_scalo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "tiles_manifest.csv"

# audio / feature
SR = 4000
TILE_SEC = 1.0
HOP_SEC  = 0.5

# mel params
N_FFT=1024; HOP=256; N_MELS=64; FMAX=4000
# scalogram params
WAVELET="morl"; NUM_SCALES=128; FMAX_SCALO=4000

# robust display scaling
VMIN_PCT=5.0; VMAX_PCT=99.5

# ---------- HELPERS ----------
def load_json(p: Path): 
    return json.loads(p.read_text())

def load_audio(p: Path, sr=SR):
    y, _ = librosa.load(str(p), sr=sr, mono=True)
    return y

def norm_label(diag: str):
    s = (diag or "").lower()
    if "crack" in s: return "Crackle"
    if "wheez" in s: return "Wheezing"
    if "bronch" in s: return "Bronchi"
    if "rhonch" in s: return "Rhonchi"
    return "Healthy"

def patient_id_from_name(name: str):
    # e.g., "KP005_WWS_1" -> "KP005"
    return name.split("_")[0]

def mel_image(seg, sr=SR):
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmax=FMAX, power=2.0
    )
    Sdb = librosa.power_to_db(S, ref=np.max)
    vmin = float(np.percentile(Sdb, VMIN_PCT))
    vmax = float(np.percentile(Sdb, VMAX_PCT))
    return Sdb, vmin, vmax

def scalogram_image(seg, sr=SR):
    cf = pywt.central_frequency(WAVELET)
    min_scale = max(1, int(sr * cf / max(1.0, FMAX_SCALO)))
    scales = np.arange(min_scale, min_scale + NUM_SCALES)
    coefs, _ = pywt.cwt(seg, scales, WAVELET, sampling_period=1.0/sr)
    P = 10.0*np.log10(np.maximum(np.abs(coefs)**2, 1e-12))
    vmin = float(np.percentile(P, VMIN_PCT))
    vmax = float(np.percentile(P, VMAX_PCT))
    return P, vmin, vmax

def save_img(arr, out_png, vmin, vmax, cmap="plasma"):
    plt.figure(figsize=(4,4), dpi=224/4)  # ~224x224 output
    plt.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------- MAIN ----------
def main():
    meta = load_json(JSON_FILE)
    rows = []
    n_saved = 0
    for fname, rec in meta.items():
        diag = norm_label(rec.get("diagnosis",""))
        pid = patient_id_from_name(fname)
        wav = AUDIO_DIR / f"{fname}.wav"
        if not wav.exists():
            continue
        y = load_audio(wav)

        tile = int(TILE_SEC*SR); hop = int(HOP_SEC*SR)
        for (s,e) in rec.get("breathing", []):   # only breathing intervals
            s_i = int(s*SR); e_i = int(e*SR)
            for st in range(s_i, max(s_i, e_i - tile + 1), hop):
                seg = y[st:st+tile]
                if len(seg) < tile:
                    break

                if MODE=="mel":
                    img, vmin, vmax = mel_image(seg)
                else:
                    img, vmin, vmax = scalogram_image(seg)

                t0 = st / SR
                out_png = OUT_DIR / f"{fname}_{MODE}_{t0:.2f}s.png"
                save_img(img, out_png, vmin, vmax)
                rows.append([str(out_png), diag, pid, fname, t0, t0+TILE_SEC])
                n_saved += 1

    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label","patient_id","orig_file","t_start","t_end"])
        w.writerows(rows)
    print(f"✅ Saved {n_saved} tiles to {OUT_DIR}")
    print(f"🗂️ Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()

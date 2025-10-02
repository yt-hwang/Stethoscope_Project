# segment_audio.py
import os
import json
import numpy as np
import soundfile as sf
import librosa

JSON_PATH = 'D:\\Stethoscope_Project\\Development\\9.2) Reference\\breathing_nonbreathing_intervals.json'  # Edit as needed
AUDIO_ROOT = 'D:\\Stethoscope_Project\\Audio shared\\ML test sound list\\RAW sound_ML test sound list'
SEGMENT_LENGTH_SEC = 2.0    # Adjustable parameter!
TARGET_SR = 16000
OUTPUT_DIR = f'segments_{int(SEGMENT_LENGTH_SEC*1000)}ms'  # output will go here
MIN_SEGMENT_SEC = 0.1       # skip segments < 0.1s

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_wav(segment, sr, out_path):
    sf.write(out_path, segment, sr)

def split_and_pad(audio, sr, start, end, window_sec):
    # Returns list of segment arrays for an interval
    segs = []
    i_start = int(start * sr)
    i_end = int(end * sr)
    interval = audio[i_start:i_end]
    win_samples = int(window_sec * sr)
    L = len(interval)
    if L < MIN_SEGMENT_SEC * sr:
        return []

    # Iterate over full-length chunks
    offset = 0
    while offset + win_samples <= L:
        seg = interval[offset:offset+win_samples]
        segs.append(seg)
        offset += win_samples
    # Handle tail < window length (pad to length, if at least MIN_SEGMENT_SEC)
    if L - offset >= MIN_SEGMENT_SEC * sr:
        last = interval[offset:]
        pad_width = win_samples - len(last)
        segs.append(np.pad(last, (0, pad_width), mode='constant'))
    return segs

def main():
    with open(JSON_PATH, 'r') as f:
        meta = json.load(f)
    total_saved = 0
    for fid, info in meta.items():
        audio_path = os.path.join(AUDIO_ROOT, f"{fid}.wav")
        if not os.path.exists(audio_path):
            print(f"Missing file: {audio_path}")
            continue
        audio, orig_sr = librosa.load(audio_path, sr=None)
        if orig_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr, TARGET_SR)
            sr = TARGET_SR
        else:
            sr = orig_sr

        # Breathing
        for i, (st, et) in enumerate(info['breathing']):
            for j, segment in enumerate(split_and_pad(audio, sr, st, et, SEGMENT_LENGTH_SEC)):
                outname = f"{fid}_breath_{i:03d}_{j:02d}.wav"
                label = info['diagnosis']
                outpath = os.path.join(OUTPUT_DIR, outname)
                save_wav(segment, sr, outpath)
                with open(outpath + '.lab', 'w') as f_lab:
                    f_lab.write(label)
                total_saved += 1

        # Non-breathing
        for i, (st, et) in enumerate(info['non_breathing']):
            for j, segment in enumerate(split_and_pad(audio, sr, st, et, SEGMENT_LENGTH_SEC)):
                outname = f"{fid}_nonbreath_{i:03d}_{j:02d}.wav"
                label = "Non-breathing"
                outpath = os.path.join(OUTPUT_DIR, outname)
                save_wav(segment, sr, outpath)
                with open(outpath + '.lab', 'w') as f_lab:
                    f_lab.write(label)
                total_saved += 1

    print(f"Saved {total_saved} segments in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

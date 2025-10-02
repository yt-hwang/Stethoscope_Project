# check-audio.py

import os
import json
import soundfile as sf

JSON_PATH = 'D:\\Stethoscope_Project\\Development\\9.2) Reference\\breathing_nonbreathing_intervals.json'  # Edit as needed
AUDIO_ROOT = 'D:\\Stethoscope_Project\\Audio shared\\ML test sound list\\RAW sound_ML test sound list'

def check_single_audio(audio_path):
    try:
        info = sf.info(audio_path)
        print(f"File: {os.path.basename(audio_path)}")
        print(f"  Sample Rate: {info.samplerate} Hz")
        print(f"  Channels: {info.channels}")
        print(f"  Duration: {info.duration:.2f} seconds")
        print(f"  Format: {info.format}")
        print(f"  Subtype: {info.subtype}")
        return info.samplerate, info.channels
    except Exception as e:
        print(f"❌ Error reading {audio_path}: {e}")
        return None, None

def main():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    file_ids = list(data.keys())
    for file_id in file_ids[:3]:  # Check the first 3 as sample
        audio_path = os.path.join(AUDIO_ROOT, f"{file_id}.wav")
        if os.path.exists(audio_path):
            check_single_audio(audio_path)
        else:
            print(f"❌ File not found: {audio_path}")

if __name__ == "__main__":
    main()

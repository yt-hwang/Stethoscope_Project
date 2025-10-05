# /home/un_wang/my_stethoscope_project/scripts/counting_segments_per_class.py

import os
from glob import glob
from collections import Counter

# === 사용자 설정 ===
# segment_audio.py에서 생성한 세그먼트가 저장된 디렉토리
OUTPUT_DIR = "/home/un_wang/my_stethoscope_project/data/audio/segments_2000ms"

def count_segments_by_class(output_dir: str):
    # .lab 파일 전부 찾기
    lab_files = glob(os.path.join(output_dir, "*.lab"))

    if not lab_files:
        print(f"No .lab files found in: {output_dir}")
        return {}

    labels = []
    for lab_file in lab_files:
        with open(lab_file, "r") as f:
            label = f.read().strip()
            labels.append(label)

    counts = Counter(labels)

    print(f"\n=== Segment Counts by Class ===")
    for label, cnt in sorted(counts.items(), key=lambda x: x[0]):
        print(f"{label}: {cnt}")

    print(f"\nTotal segments: {sum(counts.values())}")
    return counts

if __name__ == "__main__":
    count_segments_by_class(OUTPUT_DIR)

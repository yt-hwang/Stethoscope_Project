from BreathSound_Analysis_Pipeline import run_pipeline, batch_process_audio_files, Config
import os
import glob

# =====================
# SINGLE FILE PROCESSING
# =====================
print("Processing single file...")
my_cfg = Config(
    audio_path="/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound/WEBSS003_6/WEBSS-003 TP6_60sec.wav",
    output_dir="/Users/yunhwang/Desktop/Stethoscope_Project/Signal Processing/Output"
)

summary = run_pipeline(my_cfg)
print("Single file result:", summary)

# =====================
# BATCH PROCESSING EXAMPLE
# =====================
print("\n" + "="*60)
print("BATCH PROCESSING EXAMPLE")
print("="*60)

# Find all WAV files in the hospital sound directory
audio_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound"
audio_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)

print(f"Found {len(audio_files)} audio files:")
for i, file in enumerate(audio_files, 1):
    print(f"  {i}. {os.path.basename(file)}")

# Process ALL files in the hospital sound directory
if len(audio_files) > 0:
    print(f"\nProcessing ALL {len(audio_files)} files...")
    results = batch_process_audio_files(
        audio_files,  # 모든 파일 처리
        output_base_dir="/Users/yunhwang/Desktop/Stethoscope_Project/Signal Processing/Output",
        config_overrides={
            "lowcut_hz": 100.0,
            "highcut_hz": 1800.0,
            "thresh_mad_scale": 0.6,
            # wheeze 감지 설정 추가 (더 민감한 감지)
            "flatness_max": 0.6,
            "centroid_min_hz": 100.0,
            "centroid_max_hz": 1400.0,
            "peakiness_min": 0.15,
            "wheeze_min_dur_sec": 0.25
        }
    )
    
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    for i, result in enumerate(results, 1):
        if "error" in result:
            print(f"File {i}: ❌ ERROR - {result['error']}")
        else:
            print(f"File {i}: ✅ {result['num_cycles']} cycles, {result['estimated_breaths_per_min']:.1f} breaths/min, {result['num_wheeze_candidates']} wheeze candidates")
else:
    print("Not enough audio files found for batch processing example.")

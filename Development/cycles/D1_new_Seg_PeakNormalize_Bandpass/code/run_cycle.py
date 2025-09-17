#!/usr/bin/env python3
"""
Orchestrate the execution of D1_new: Seg + PeakNormalize + Bandpass
"""

import subprocess
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
AUDIO_DIR = '/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_script(script_name, *args):
    """Run a script and handle errors."""
    script_path = os.path.join(BASE_DIR, script_name)
    command = [sys.executable, str(script_path)] + list(args)
    print(f"Running: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stdout)
        print(result.stderr)
        raise Exception(f"Script {script_name} failed.")
    
    print("✓ Success")
    return result.stdout

def main():
    print("="*80)
    print("D1_new: Seg + PeakNormalize + Bandpass")
    print("="*80)
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Preprocessing: Peak Normalization + Bandpass Filter (100-2000 Hz)")
    print("Combines: Amplitude normalization + Frequency focusing")

    try:
        print("\n1. Extracting features...")
        run_script('extract_features.py', '--audio_dir', AUDIO_DIR, '--output_dir', OUTPUT_DIR)

        print("\n2. Running clustering...")
        run_script('run_clustering.py', '--output_dir', OUTPUT_DIR)

        print("\n3. Creating visualizations...")
        run_script('make_visuals.py', '--output_dir', OUTPUT_DIR)

        print("\n4. Logging to Excel...")
        run_script('log_to_excel.py', '--output_dir', OUTPUT_DIR)

        print("\n" + "="*80)
        print("D1_new COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Output directory: {OUTPUT_DIR}")
        print("Next: Run robust evaluation to compare with other cycles")

    except Exception as e:
        print(f"\n" + "="*80)
        print(f"D1_new FAILED: {e}")
        print(f"="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()

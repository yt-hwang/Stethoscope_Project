#!/usr/bin/env python3
"""
Cycle C4: Seg + PeakNormalize - Main Runner
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_path, args=None):
    """Run a Python script and return success status"""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Success")
        return True
    else:
        print("✗ Failed")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = Path(__file__).parent.parent / "outputs"
    audio_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec"
    
    print("="*60)
    print("CYCLE C4: Seg + PeakNormalize")
    print("="*60)
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Extract features
    print("\n1. Extracting features...")
    if not run_script(script_dir / "extract_features.py", 
                     ["--audio_dir", audio_dir, "--output_dir", str(output_dir)]):
        print("CYCLE C4 FAILED!")
        return
    
    # Step 2: Run clustering
    print("\n2. Running clustering...")
    if not run_script(script_dir / "run_clustering.py", 
                     ["--output_dir", str(output_dir)]):
        print("CYCLE C4 FAILED!")
        return
    
    # Step 3: Create visualizations
    print("\n3. Creating visualizations...")
    if not run_script(script_dir / "make_visuals.py", 
                     ["--output_dir", str(output_dir)]):
        print("CYCLE C4 FAILED!")
        return
    
    # Step 4: Log to Excel
    print("\n4. Logging to Excel...")
    if not run_script(script_dir / "log_to_excel.py", 
                     ["--output_dir", str(output_dir)]):
        print("CYCLE C4 FAILED!")
        return
    
    print("\n" + "="*60)
    print("CYCLE C4 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("Next: Run robust evaluation to compare with A4 and C0")

if __name__ == "__main__":
    main()

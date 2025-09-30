#!/usr/bin/env python3
"""
Main cycle runner for Cycle A0: NoSeg + NoPreprocess baseline
"""

import os
import sys
import subprocess
import argparse
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
        if result.stdout:
            print(result.stdout)
    else:
        print("✗ Failed")
        if result.stderr:
            print(result.stderr)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run Cycle A0: NoSeg + NoPreprocess baseline')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
                       help='Path to output directory')
    parser.add_argument('--excel_path', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Experiment_Tracking_System_Final.xlsx',
                       help='Path to Excel tracking file')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip feature extraction step')
    parser.add_argument('--skip_clustering', action='store_true',
                       help='Skip clustering step')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--skip_logging', action='store_true',
                       help='Skip Excel logging step')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("CYCLE A2: NoSeg + SpectralGating")
    print("="*60)
    
    success = True
    
    # Step 1: Feature Extraction
    if not args.skip_extraction:
        print("\n1. Feature Extraction...")
        success &= run_script(
            script_dir / 'extract_features.py',
            ['--audio_dir', args.audio_dir, '--output_dir', args.output_dir]
        )
    
    # Step 2: Clustering
    if success and not args.skip_clustering:
        print("\n2. Clustering...")
        success &= run_script(
            script_dir / 'run_clustering.py',
            ['--output_dir', args.output_dir]
        )
    
    # Step 3: Visualization
    if success and not args.skip_visualization:
        print("\n3. Visualization...")
        success &= run_script(
            script_dir / 'make_visuals.py',
            ['--output_dir', args.output_dir]
        )
    
    # Step 4: Excel Logging
    if success and not args.skip_logging:
        print("\n4. Excel Logging...")
        success &= run_script(
            script_dir / 'log_to_excel.py',
            ['--output_dir', args.output_dir, '--excel_path', args.excel_path]
        )
    
    if success:
        print("\n" + "="*60)
        print("CYCLE A2 COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("CYCLE A2 FAILED!")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

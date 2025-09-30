#!/usr/bin/env python3
"""
New Run Script - Creates a new experiment run
No data assumptions - pure run initialization
"""

import argparse
import sys
import pkg_resources
from pathlib import Path
from datetime import datetime
import platform

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from run_manager import RunManager
from excel_logger import ExcelLogger

def get_versions():
    """Get library versions for reproducibility."""
    versions = []
    versions.append(f"Python: {sys.version}")
    
    # Core packages to check
    packages = [
        'numpy', 'pandas', 'scipy', 'torch', 'torchaudio', 
        'scikit-learn', 'umap-learn', 'matplotlib', 'tqdm', 
        'soundfile', 'librosa', 'openpyxl'
    ]
    
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            versions.append(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            versions.append(f"{package}: Not installed")
    
    return "\n".join(versions)

def create_minimal_manifest():
    """Create minimal manifest with system info."""
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'working_directory': str(Path.cwd())
    }
    
    # Add GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            manifest['gpu_available'] = True
            manifest['gpu_name'] = torch.cuda.get_device_name(0)
            manifest['cuda_version'] = torch.version.cuda
        else:
            manifest['gpu_available'] = False
    except ImportError:
        manifest['gpu_available'] = 'unknown (torch not installed)'
    
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Create a new experiment run")
    parser.add_argument("--base_dir", default="results", 
                       help="Base results directory")
    
    args = parser.parse_args()
    
    print("🚀 Creating new experiment run...")
    
    # Initialize managers
    run_manager = RunManager(args.base_dir)
    excel_logger = ExcelLogger(args.base_dir)
    
    # Create run
    run_id = run_manager.create_run_id()
    run_id, run_dir = run_manager.init_run_folders(run_id)
    
    print(f"📁 Run ID: {run_id}")
    print(f"📁 Run directory: {run_dir}")
    
    # Create and save manifest
    manifest = create_minimal_manifest()
    manifest_path = run_manager.save_manifest(run_id, manifest)
    print(f"📝 Manifest saved: {manifest_path}")
    
    # Save versions
    versions = get_versions()
    versions_path = run_dir / "artifacts" / "versions.txt"
    with open(versions_path, 'w') as f:
        f.write(versions)
    print(f"📝 Versions saved: {versions_path}")
    
    # Create minimal config (empty for now)
    minimal_config = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(),
        'status': 'initialized'
    }
    config_path = run_manager.dump_config(run_id, minimal_config)
    print(f"📝 Config saved: {config_path}")
    
    # Log to Excel runs sheet
    excel_data = {
        'run_id': run_id,
        'status': 'initialized',
        'hostname': manifest['hostname'],
        'gpu_available': manifest.get('gpu_available', 'unknown'),
        'gpu_name': manifest.get('gpu_name', ''),
        'python_version': sys.version.split()[0],  # Just version number
        'run_dir': str(run_dir),
        'manifest_path': str(manifest_path),
        'config_path': str(config_path),
        'versions_path': str(versions_path)
    }
    
    try:
        row_num = excel_logger.append_row('runs', excel_data)
        print(f"📊 Logged to Excel: row {row_num}")
    except Exception as e:
        print(f"⚠️ Excel logging failed: {e}")
    
    print(f"\n✅ Run {run_id} created successfully!")
    print(f"📁 Directory structure:")
    for subdir in ["00_setup", "01_features", "02_train", "03_eval", "04_cluster", "05_transfer", "artifacts"]:
        print(f"   {run_dir / subdir}")
    
    print(f"\n🎯 Run ID: {run_id}")
    print("STOP - Run initialization complete.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run Manager - Core utilities for experiment run management
No data assumptions - pure scaffolding for multi-step pipeline
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import sys

class RunManager:
    """Manages experiment runs with structured folder layout and artifacts."""
    
    def __init__(self, base_results_dir="results"):
        self.base_results_dir = Path(base_results_dir)
        self.experiments_dir = self.base_results_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
    def create_run_id(self):
        """Create run_id following convention: YYYYMMDD-HHMMSS__auto"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}__auto"
    
    def ensure_unique_run_id(self, run_id):
        """Ensure run_id is unique, append _v2, _v3 etc if needed."""
        original_run_id = run_id
        version = 1
        
        while (self.experiments_dir / run_id).exists():
            version += 1
            run_id = f"{original_run_id}_v{version}"
            
        return run_id
    
    def init_run_folders(self, run_id):
        """Create structured subfolder layout for a run."""
        run_id = self.ensure_unique_run_id(run_id)
        run_dir = self.experiments_dir / run_id
        
        # Create required subfolders
        subdirs = [
            "00_setup", "01_features", "02_train", "03_eval", 
            "04_cluster", "05_transfer", "artifacts"
        ]
        
        for subdir in subdirs:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        return run_id, run_dir
    
    def save_manifest(self, run_id, manifest_dict):
        """Save manifest.json to artifacts folder."""
        run_dir = self.experiments_dir / run_id
        manifest_path = run_dir / "artifacts" / "manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)
        
        return manifest_path
    
    def dump_config(self, run_id, cfg_dict):
        """Save config_dump.yaml to artifacts folder."""
        run_dir = self.experiments_dir / run_id
        config_path = run_dir / "artifacts" / "config_dump.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, indent=2)
        
        return config_path
    
    def get_run_dir(self, run_id):
        """Get run directory path."""
        return self.experiments_dir / run_id
    
    def list_runs(self):
        """List all existing run IDs."""
        if not self.experiments_dir.exists():
            return []
        
        runs = []
        for item in self.experiments_dir.iterdir():
            if item.is_dir():
                runs.append(item.name)
        
        return sorted(runs)
    
    def create_manifest_dict(self):
        """Create basic manifest dictionary with system info."""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'working_directory': str(Path.cwd()),
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
            manifest['gpu_available'] = 'unknown'
            manifest['torch_not_installed'] = True
        
        # Add git info if available
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                           stderr=subprocess.DEVNULL).decode().strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                           stderr=subprocess.DEVNULL).decode().strip()
            manifest['git_commit'] = commit
            manifest['git_branch'] = branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            manifest['git_info'] = 'not available'
        
        return manifest

# Convenience functions
def create_run_id():
    """Create a new run ID (convenience function)."""
    manager = RunManager()
    return manager.create_run_id()

def init_run_folders(run_id):
    """Initialize run folders (convenience function)."""
    manager = RunManager()
    return manager.init_run_folders(run_id)

def save_manifest(run_id, manifest_dict):
    """Save manifest (convenience function)."""
    manager = RunManager()
    return manager.save_manifest(run_id, manifest_dict)

def dump_config(run_id, cfg_dict):
    """Dump config (convenience function)."""
    manager = RunManager()
    return manager.dump_config(run_id, cfg_dict)
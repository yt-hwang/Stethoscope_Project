#!/usr/bin/env python3
"""
Environment Check Script - Verifies environment and OPERA-CT availability
Records all results without failing on missing components
"""

import argparse
import sys
import platform
import os
from pathlib import Path
from datetime import datetime
import subprocess

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from run_manager import RunManager
from excel_logger import ExcelLogger

def check_python_version():
    """Check Python version."""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    return {
        'python_version': version_str,
        'python_full': sys.version,
        'python_executable': sys.executable
    }

def check_os_info():
    """Check operating system information."""
    return {
        'os_platform': platform.platform(),
        'os_system': platform.system(),
        'os_release': platform.release(),
        'hostname': platform.node(),
        'architecture': platform.architecture()[0]
    }

def check_cuda_availability():
    """Check CUDA availability and GPU information."""
    cuda_info = {
        'cuda_available': False,
        'gpu_name': '',
        'cuda_version': '',
        'torch_cuda_available': False
    }
    
    try:
        import torch
        cuda_info['torch_installed'] = True
        cuda_info['torch_version'] = torch.__version__
        
        if torch.cuda.is_available():
            cuda_info['torch_cuda_available'] = True
            cuda_info['cuda_available'] = True
            cuda_info['gpu_name'] = torch.cuda.get_device_name(0)
            cuda_info['cuda_version'] = torch.version.cuda
            cuda_info['gpu_count'] = torch.cuda.device_count()
        
    except ImportError as e:
        cuda_info['torch_installed'] = False
        cuda_info['torch_import_error'] = str(e)
    
    return cuda_info

def check_opera_ct_availability():
    """Check OPERA-CT availability under multiple possible import names."""
    opera_info = {
        'opera_ct_available': False,
        'opera_ct_version': '',
        'import_attempts': [],
        'successful_import': '',
        'import_errors': []
    }
    
    # Try multiple possible import names
    import_names = [
        'opera',
        'opera_ct', 
        'OPERA',
        'operact',
        'src.model.models_cola',  # From OPERA repo structure
        'src.benchmark.model_util',
        'models_cola',
        'model_util'
    ]
    
    for import_name in import_names:
        try:
            if '.' in import_name:
                # Handle module.submodule imports
                module_parts = import_name.split('.')
                module = __import__(import_name, fromlist=[module_parts[-1]])
            else:
                module = __import__(import_name)
            
            opera_info['opera_ct_available'] = True
            opera_info['successful_import'] = import_name
            
            # Try to get version if available
            if hasattr(module, '__version__'):
                opera_info['opera_ct_version'] = module.__version__
            elif hasattr(module, 'version'):
                opera_info['opera_ct_version'] = module.version
            else:
                opera_info['opera_ct_version'] = 'unknown'
            
            opera_info['import_attempts'].append(f"{import_name}: SUCCESS")
            break
            
        except ImportError as e:
            opera_info['import_attempts'].append(f"{import_name}: {str(e)}")
            opera_info['import_errors'].append(str(e))
        except Exception as e:
            opera_info['import_attempts'].append(f"{import_name}: UNEXPECTED_ERROR - {str(e)}")
            opera_info['import_errors'].append(str(e))
    
    return opera_info

def check_core_dependencies():
    """Check availability of core dependencies."""
    deps_info = {}
    
    core_deps = [
        'numpy', 'pandas', 'scipy', 'torch', 'torchaudio',
        'sklearn', 'umap', 'librosa', 'soundfile', 'matplotlib', 
        'tqdm', 'openpyxl'
    ]
    
    for dep in core_deps:
        try:
            if dep == 'sklearn':
                import sklearn
                module = sklearn
            elif dep == 'umap':
                import umap
                module = umap
            else:
                module = __import__(dep)
            
            version = getattr(module, '__version__', 'unknown')
            deps_info[f'{dep}_available'] = True
            deps_info[f'{dep}_version'] = version
            
        except ImportError as e:
            deps_info[f'{dep}_available'] = False
            deps_info[f'{dep}_error'] = str(e)
    
    return deps_info

def main():
    parser = argparse.ArgumentParser(description="Check environment and OPERA-CT availability")
    parser.add_argument("--run_id", required=True, help="Run ID to log results")
    
    args = parser.parse_args()
    
    print(f"🔍 Environment check for run: {args.run_id}")
    print("=" * 50)
    
    # Initialize logging
    run_manager = RunManager()
    excel_logger = ExcelLogger()
    
    # Get run directory
    run_dir = run_manager.get_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Create log file path
    setup_dir = run_dir / "00_setup"
    setup_dir.mkdir(exist_ok=True)
    log_path = setup_dir / "install_log.txt"
    
    # Collect all environment information
    print("🔍 Checking Python version...")
    python_info = check_python_version()
    
    print("🔍 Checking OS information...")
    os_info = check_os_info()
    
    print("🔍 Checking CUDA availability...")
    cuda_info = check_cuda_availability()
    
    print("🔍 Checking core dependencies...")
    deps_info = check_core_dependencies()
    
    print("🔍 Checking OPERA-CT availability...")
    opera_info = check_opera_ct_availability()
    
    # Combine all information
    all_info = {
        **python_info,
        **os_info, 
        **cuda_info,
        **deps_info,
        **opera_info
    }
    
    # Create detailed log content
    log_content = []
    log_content.append(f"Environment Check Log - {datetime.now().isoformat()}")
    log_content.append("=" * 60)
    log_content.append("")
    
    log_content.append("PYTHON INFORMATION:")
    log_content.append(f"Version: {python_info['python_version']}")
    log_content.append(f"Executable: {python_info['python_executable']}")
    log_content.append(f"Full version: {python_info['python_full']}")
    log_content.append("")
    
    log_content.append("SYSTEM INFORMATION:")
    log_content.append(f"Platform: {os_info['os_platform']}")
    log_content.append(f"Hostname: {os_info['hostname']}")
    log_content.append(f"Architecture: {os_info['architecture']}")
    log_content.append("")
    
    log_content.append("CUDA/GPU INFORMATION:")
    log_content.append(f"PyTorch installed: {cuda_info.get('torch_installed', False)}")
    log_content.append(f"CUDA available: {cuda_info['cuda_available']}")
    log_content.append(f"GPU name: {cuda_info['gpu_name']}")
    log_content.append(f"CUDA version: {cuda_info['cuda_version']}")
    log_content.append("")
    
    log_content.append("CORE DEPENDENCIES:")
    for dep in ['numpy', 'pandas', 'torch', 'librosa', 'sklearn']:
        available = deps_info.get(f'{dep}_available', False)
        version = deps_info.get(f'{dep}_version', 'unknown')
        status = "✅" if available else "❌"
        log_content.append(f"{status} {dep}: {version}")
    log_content.append("")
    
    log_content.append("OPERA-CT AVAILABILITY:")
    log_content.append(f"Available: {opera_info['opera_ct_available']}")
    log_content.append(f"Successful import: {opera_info['successful_import']}")
    log_content.append(f"Version: {opera_info['opera_ct_version']}")
    log_content.append("")
    log_content.append("Import attempts:")
    for attempt in opera_info['import_attempts']:
        log_content.append(f"  {attempt}")
    
    # Save log file
    with open(log_path, 'w') as f:
        f.write("\n".join(log_content))
    
    print(f"💾 Environment log saved: {log_path}")
    
    # Prepare Excel row data
    excel_row = {
        'run_id': args.run_id,
        'python': python_info['python_version'],
        'torch': cuda_info.get('torch_version', 'not_installed'),
        'cuda': cuda_info['cuda_available'],
        'gpu_name': cuda_info['gpu_name'] or 'none',
        'opera_ct_available': opera_info['opera_ct_available'],
        'opera_ct_version': opera_info['opera_ct_version'] or 'unknown',
        'notes': f"Import attempts: {len(opera_info['import_attempts'])}, Successful: {opera_info['successful_import']}",
        'log_path': str(log_path)
    }
    
    # Log to Excel
    try:
        row_num = excel_logger.append_row('runs', excel_row)
        print(f"📊 Updated Excel runs sheet: row {row_num}")
    except Exception as e:
        print(f"⚠️ Excel logging failed: {e}")
    
    # Print summary
    print("\n📊 ENVIRONMENT CHECK SUMMARY:")
    print("=" * 40)
    print(f"Python: {python_info['python_version']} ✅")
    print(f"PyTorch: {cuda_info.get('torch_version', 'NOT INSTALLED')} {'✅' if cuda_info.get('torch_installed') else '❌'}")
    print(f"CUDA: {cuda_info['cuda_available']} {'✅' if cuda_info['cuda_available'] else '❌'}")
    print(f"GPU: {cuda_info['gpu_name'] or 'None detected'}")
    print(f"OPERA-CT: {opera_info['opera_ct_available']} {'✅' if opera_info['opera_ct_available'] else '❌'}")
    
    if opera_info['opera_ct_available']:
        print(f"OPERA-CT Import: {opera_info['successful_import']}")
        print(f"OPERA-CT Version: {opera_info['opera_ct_version']}")
    else:
        print("OPERA-CT Status: Not available")
        print(f"Tried {len(opera_info['import_attempts'])} import methods")
    
    print(f"\n💾 Detailed log: {log_path}")
    print(f"📊 Excel updated: results/experiment_log.xlsx")
    print("\nSTOP - Environment check complete.")

if __name__ == "__main__":
    main()

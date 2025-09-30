#!/usr/bin/env python3
"""
OPERA-CT Verification Script
Verifies OPERA-CT is usable with timeout and comprehensive logging
"""

import argparse
import sys
import os
import json
import time
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

class TimeoutException(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutException("Operation timed out")

def timed_operation(func, description, timeout_sec=30):
    """Execute function with timing and timeout."""
    print(f"🔍 {description}...")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    start_time = time.time()
    try:
        result = func()
        end_time = time.time()
        duration = end_time - start_time
        
        signal.alarm(0)  # Cancel timeout
        print(f"✅ {description} completed in {duration:.2f}s")
        return True, duration, result, None
        
    except TimeoutException:
        signal.alarm(0)
        error_msg = f"Timeout after {timeout_sec}s"
        print(f"⏰ {description} timed out after {timeout_sec}s")
        return False, timeout_sec, None, error_msg
        
    except Exception as e:
        signal.alarm(0)
        end_time = time.time()
        duration = end_time - start_time
        error_msg = str(e)
        print(f"❌ {description} failed after {duration:.2f}s: {error_msg}")
        return False, duration, None, error_msg

def verify_basic_imports():
    """Verify basic imports."""
    import torch
    import transformers
    return {
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__
    }

def verify_opera_imports():
    """Verify OPERA-CT specific imports."""
    from src.benchmark.model_util import extract_opera_feature, initialize_pretrained_model
    from src.model.models_cola import Cola
    return {
        'extract_opera_feature': 'available',
        'initialize_pretrained_model': 'available', 
        'Cola': 'available'
    }

def verify_model_initialization():
    """Verify model can be initialized."""
    from src.benchmark.model_util import initialize_pretrained_model
    model = initialize_pretrained_model("operaCT")
    return {
        'model_type': str(type(model)),
        'model_initialized': True
    }

def verify_feature_extraction():
    """Verify feature extraction with dummy audio."""
    from src.benchmark.model_util import extract_opera_feature
    
    # Create dummy 0.5 second audio at 16kHz
    dummy_audio = np.random.randn(int(0.5 * 16000)).astype(np.float32)
    
    # Save dummy audio temporarily
    import soundfile as sf
    temp_audio_path = "/tmp/dummy_audio.wav"
    sf.write(temp_audio_path, dummy_audio, 16000)
    
    try:
        # Extract features
        features = extract_opera_feature([temp_audio_path], pretrain="operaCT", input_sec=1, dim=768)
        
        # Clean up
        os.remove(temp_audio_path)
        
        return {
            'feature_extraction': 'success',
            'embedding_shape': features.shape if hasattr(features, 'shape') else str(type(features)),
            'embedding_type': str(type(features))
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise e

def main():
    parser = argparse.ArgumentParser(description="Verify OPERA-CT installation and functionality")
    parser.add_argument("--run_id", required=True, help="Run ID for logging")
    
    args = parser.parse_args()
    
    print(f"🔍 OPERA-CT Verification for run: {args.run_id}")
    print("=" * 50)
    
    # Set up environment
    opera_path = Path.cwd() / "setup" / "OPERA"
    if not opera_path.exists():
        print(f"❌ OPERA directory not found: {opera_path}")
        sys.exit(1)
    
    # Extend PYTHONPATH
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f"{original_pythonpath}:{opera_path}"
    sys.path.append(str(opera_path))
    
    # Set tokenizers parallelism
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print(f"🔧 PYTHONPATH extended with: {opera_path}")
    print(f"🔧 TOKENIZERS_PARALLELISM set to false")
    
    # Initialize logging
    excel_logger = ExcelLogger()
    
    # Results tracking
    verification_results = {
        'run_id': args.run_id,
        'timestamp': datetime.now().isoformat(),
        'opera_path': str(opera_path),
        'python_version': sys.version.split()[0]
    }
    
    # Set overall timeout
    overall_start = time.time()
    overall_timeout = 60  # 60 seconds total
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(overall_timeout)
        
        # Step 1: Basic imports
        success, duration, result, error = timed_operation(
            verify_basic_imports, "Basic imports (torch, transformers)", 15
        )
        verification_results['basic_imports_ok'] = success
        verification_results['basic_imports_sec'] = duration
        if success:
            verification_results.update(result)
        else:
            verification_results['basic_imports_error'] = error
        
        if not success:
            raise Exception(f"Basic imports failed: {error}")
        
        # Step 2: OPERA imports
        success, duration, result, error = timed_operation(
            verify_opera_imports, "OPERA-CT imports", 15
        )
        verification_results['opera_imports_ok'] = success
        verification_results['opera_imports_sec'] = duration
        if success:
            verification_results.update(result)
        else:
            verification_results['opera_imports_error'] = error
        
        if not success:
            raise Exception(f"OPERA imports failed: {error}")
        
        # Step 3: Model initialization
        success, duration, result, error = timed_operation(
            verify_model_initialization, "Model initialization", 20
        )
        verification_results['init_ok'] = success
        verification_results['init_sec'] = duration
        if success:
            verification_results.update(result)
        else:
            verification_results['init_error'] = error
        
        if not success:
            raise Exception(f"Model initialization failed: {error}")
        
        # Step 4: Feature extraction
        success, duration, result, error = timed_operation(
            verify_feature_extraction, "Feature extraction (0.5s dummy audio)", 20
        )
        verification_results['feature_ok'] = success
        verification_results['feature_sec'] = duration
        if success:
            verification_results.update(result)
        else:
            verification_results['feature_error'] = error
        
        # Cancel overall timeout
        signal.alarm(0)
        
        # Final status
        overall_duration = time.time() - overall_start
        verification_results['total_duration'] = overall_duration
        verification_results['overall_success'] = success
        
        if success:
            print(f"\n✅ OPERA-CT verified: init={verification_results['init_sec']:.1f}s, feature={verification_results['feature_sec']:.1f}s, emb_shape={verification_results.get('embedding_shape', 'unknown')}")
        else:
            print(f"\n❌ Verification failed: {error}")
            
    except TimeoutException:
        signal.alarm(0)
        overall_duration = time.time() - overall_start
        error_msg = f"Overall timeout after {overall_timeout}s"
        print(f"\n⏰ {error_msg}")
        
        verification_results['overall_success'] = False
        verification_results['total_duration'] = overall_duration
        verification_results['timeout_error'] = error_msg
        
    except Exception as e:
        signal.alarm(0)
        overall_duration = time.time() - overall_start
        error_msg = str(e)
        print(f"\n❌ Verification failed: {error_msg}")
        
        verification_results['overall_success'] = False
        verification_results['total_duration'] = overall_duration
        verification_results['general_error'] = error_msg
    
    # Save detailed results to JSON
    run_dir = Path("results/experiments") / args.run_id
    setup_dir = run_dir / "00_setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = setup_dir / "verify_log.json"
    with open(json_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"💾 Detailed verification log: {json_path}")
    
    # Prepare Excel row
    excel_row = {
        'run_id': args.run_id,
        'opera_import_ok': verification_results.get('overall_success', False),
        'init_sec': verification_results.get('init_sec', 0),
        'feature_sec': verification_results.get('feature_sec', 0),
        'emb_shape': verification_results.get('embedding_shape', 'unknown'),
        'notes': verification_results.get('general_error') or 
                verification_results.get('timeout_error') or 
                verification_results.get('feature_error') or 
                verification_results.get('init_error') or 
                verification_results.get('opera_imports_error') or
                verification_results.get('basic_imports_error') or
                'Success',
        'log_path': str(json_path)
    }
    
    # Log to Excel
    try:
        row_num = excel_logger.append_row('runs', excel_row)
        print(f"📊 Updated Excel: row {row_num}")
    except Exception as e:
        print(f"⚠️ Excel logging failed: {e}")
    
    print("\nSTOP - OPERA-CT verification complete.")

if __name__ == "__main__":
    main()

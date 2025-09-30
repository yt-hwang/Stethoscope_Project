#!/usr/bin/env python3
"""
Run All Methods Script
Systematically execute all 16 preprocessing methods for comprehensive validation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import yaml

# All 16 methods from our unsupervised analysis
ALL_METHODS = [
    # A-Series: Individual NoSeg methods
    'A0', 'A1', 'A2', 'A3', 'A4',
    # B-Series: Combination NoSeg methods  
    'B0', 'B1', 'B2',
    # C-Series: Individual Seg methods
    'C0', 'C1', 'C2', 'C3', 'C4',
    # D-Series: Combination Seg methods
    'D0', 'D1', 'D2'
]

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return None, e.stderr

def create_run(method, dataset_name, features_backend, train_head, seed):
    """Create a new run for a specific method."""
    cmd = [
        'python', 'scripts/new_run.py',
        f'--data.dataset_name={dataset_name}',
        f'--preprocess.method={method}',
        f'--features.backend={features_backend}',
        f'--train.head={train_head}',
        f'--seed={seed}'
    ]
    
    stdout, stderr = run_command(cmd, f"Creating run for method {method}")
    
    if stdout:
        # Extract run_id from output
        lines = stdout.strip().split('\n')
        for line in lines:
            if 'Created run:' in line:
                run_id = line.split('Created run: ')[-1].strip()
                return run_id
    
    return None

def extract_features(run_id):
    """Extract features for a run."""
    cmd = ['python', 'scripts/extract_features.py', f'--run_id={run_id}']
    return run_command(cmd, f"Extracting features for {run_id}")

def train_model(run_id):
    """Train model for a run."""
    cmd = ['python', 'scripts/train.py', f'--run_id={run_id}']
    return run_command(cmd, f"Training model for {run_id}")

def evaluate_model(run_id):
    """Evaluate model for a run."""
    cmd = ['python', 'scripts/eval.py', f'--run_id={run_id}']
    return run_command(cmd, f"Evaluating model for {run_id}")

def cluster_analysis(run_id):
    """Run clustering analysis for a run."""
    cmd = ['python', 'scripts/cluster_eval.py', f'--run_id={run_id}']
    return run_command(cmd, f"Clustering analysis for {run_id}")

def run_single_method(method, config):
    """Run complete pipeline for a single method."""
    print(f"\n{'='*60}")
    print(f"🎯 PROCESSING METHOD: {method}")
    print(f"{'='*60}")
    
    # Create run
    run_id = create_run(
        method=method,
        dataset_name=config['dataset_name'],
        features_backend=config['features_backend'],
        train_head=config['train_head'],
        seed=config['seed']
    )
    
    if not run_id:
        print(f"❌ Failed to create run for method {method}")
        return None, "Failed to create run"
    
    print(f"📁 Run ID: {run_id}")
    
    # Execute pipeline phases
    phases = [
        (extract_features, "Feature Extraction"),
        (train_model, "Model Training"),
        (evaluate_model, "Model Evaluation"),
        (cluster_analysis, "Clustering Analysis")
    ]
    
    results = {'run_id': run_id, 'method': method}
    
    for phase_func, phase_name in phases:
        print(f"\n📊 Phase: {phase_name}")
        stdout, stderr = phase_func(run_id)
        
        if stderr:
            print(f"⚠️ Phase {phase_name} had issues for {method}")
            results[f'{phase_name.lower().replace(" ", "_")}_error'] = stderr
        else:
            print(f"✅ Phase {phase_name} completed for {method}")
            results[f'{phase_name.lower().replace(" ", "_")}_success'] = True
    
    print(f"\n🎉 Method {method} pipeline completed!")
    return run_id, results

def main():
    parser = argparse.ArgumentParser(description="Run all 16 preprocessing methods")
    parser.add_argument("--methods", nargs='+', choices=ALL_METHODS, 
                       default=ALL_METHODS, help="Methods to run (default: all)")
    parser.add_argument("--dataset_name", default="respiratory", 
                       help="Dataset name for run naming")
    parser.add_argument("--features_backend", default="opera_ct", 
                       choices=["opera_ct", "mel_fallback"],
                       help="Feature extraction backend")
    parser.add_argument("--train_head", default="mlp", 
                       choices=["linear", "mlp"],
                       help="Classification head type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue processing other methods if one fails")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be run without executing")
    
    args = parser.parse_args()
    
    config = {
        'dataset_name': args.dataset_name,
        'features_backend': args.features_backend,
        'train_head': args.train_head,
        'seed': args.seed
    }
    
    print(f"🎯 OPERA-CT COMPREHENSIVE METHOD VALIDATION")
    print(f"{'='*60}")
    print(f"Methods to process: {len(args.methods)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Backend: {config['features_backend']}")
    print(f"Head: {config['train_head']}")
    print(f"Seed: {config['seed']}")
    print(f"Continue on error: {args.continue_on_error}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - No actual execution")
        for method in args.methods:
            print(f"  Would process: {method}")
        return
    
    # Track results
    all_results = []
    successful_runs = []
    failed_runs = []
    
    start_time = time.time()
    
    for i, method in enumerate(args.methods, 1):
        print(f"\n{'🔄' * 20}")
        print(f"Progress: {i}/{len(args.methods)} methods")
        print(f"Current: {method}")
        print(f"{'🔄' * 20}")
        
        try:
            run_id, results = run_single_method(method, config)
            
            if run_id:
                successful_runs.append((method, run_id))
                all_results.append(results)
                print(f"✅ {method} completed successfully: {run_id}")
            else:
                failed_runs.append((method, results))
                print(f"❌ {method} failed")
                
                if not args.continue_on_error:
                    print(f"🛑 Stopping due to failure (use --continue_on_error to continue)")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error processing {method}: {e}")
            failed_runs.append((method, str(e)))
            
            if not args.continue_on_error:
                break
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'🎉' * 20}")
    print(f"COMPREHENSIVE VALIDATION COMPLETE")
    print(f"{'🎉' * 20}")
    print(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"✅ Successful runs: {len(successful_runs)}")
    print(f"❌ Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ SUCCESSFUL METHODS:")
        for method, run_id in successful_runs:
            print(f"  {method}: {run_id}")
    
    if failed_runs:
        print(f"\n❌ FAILED METHODS:")
        for method, error in failed_runs:
            print(f"  {method}: {error}")
    
    # Update global results index
    print(f"\n📊 Updating global results index...")
    try:
        subprocess.run(['python', 'scripts/summarize_runs.py'], check=True)
        print(f"✅ Results index updated")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to update results index: {e}")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Check results/index.csv for comprehensive comparison")
    print(f"2. Analyze results/latest for most recent run")
    print(f"3. Compare with unsupervised clustering findings")
    print(f"4. Generate comprehensive validation report")

if __name__ == "__main__":
    main()

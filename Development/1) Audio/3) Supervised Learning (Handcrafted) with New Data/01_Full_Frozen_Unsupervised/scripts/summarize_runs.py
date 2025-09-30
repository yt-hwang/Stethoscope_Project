#!/usr/bin/env python3
"""
Run Summarizer - Update global Excel and maintain latest link
Compiles key results for quick comparison across runs
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

def scan_experiment_results(experiments_dir):
    """Scan all experiment directories for results."""
    experiments_dir = Path(experiments_dir)
    
    if not experiments_dir.exists():
        return []
    
    results = []
    
    for run_dir in experiments_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_id = run_dir.name
        run_result = {'run_id': run_id}
        
        # Look for training metrics
        train_metrics_path = run_dir / "02_train" / "metrics_train_val.json"
        if train_metrics_path.exists():
            try:
                with open(train_metrics_path, 'r') as f:
                    train_data = json.load(f)
                run_result['val_best_f1'] = train_data.get('best_val_f1')
                run_result['best_epoch'] = train_data.get('best_epoch')
            except Exception as e:
                print(f"Warning: Could not read train metrics for {run_id}: {e}")
        
        # Look for test metrics
        test_metrics_path = run_dir / "03_eval" / "metrics_test.json"
        if test_metrics_path.exists():
            try:
                with open(test_metrics_path, 'r') as f:
                    test_data = json.load(f)
                run_result['test_macro_f1'] = test_data.get('f1_macro')
                run_result['test_accuracy'] = test_data.get('accuracy')
            except Exception as e:
                print(f"Warning: Could not read test metrics for {run_id}: {e}")
        
        # Look for clustering metrics
        cluster_report_path = run_dir / "04_cluster" / "clustering_report.json"
        if cluster_report_path.exists():
            try:
                with open(cluster_report_path, 'r') as f:
                    cluster_data = json.load(f)
                run_result['cluster_k'] = cluster_data.get('chosen_k')
                run_result['cluster_silhouette'] = cluster_data.get('silhouette')
                run_result['cluster_nmi'] = cluster_data.get('nmi')
                run_result['cluster_ari'] = cluster_data.get('ari')
            except Exception as e:
                print(f"Warning: Could not read cluster metrics for {run_id}: {e}")
        
        # Look for feature stats
        feature_stats_path = run_dir / "01_features" / "stats.json"
        if feature_stats_path.exists():
            try:
                with open(feature_stats_path, 'r') as f:
                    feature_data = json.load(f)
                run_result['embedding_dim'] = feature_data.get('embedding_dim')
                run_result['feature_backend'] = feature_data.get('feature_backend')
                run_result['n_segments'] = feature_data.get('n_segments_created')
            except Exception as e:
                print(f"Warning: Could not read feature stats for {run_id}: {e}")
        
        # Only add if we found some metrics
        if any(key in run_result for key in ['val_best_f1', 'test_macro_f1', 'cluster_silhouette', 'embedding_dim']):
            results.append(run_result)
    
    return results

def update_excel_with_metrics(excel_logger, results):
    """Update the main Excel runs sheet with compiled metrics."""
    
    for result in results:
        run_id = result['run_id']
        
        # Prepare update data (exclude run_id from update)
        update_data = {k: v for k, v in result.items() if k != 'run_id'}
        
        try:
            excel_logger.update_run_row('runs', run_id, update_data)
        except Exception as e:
            print(f"Warning: Could not update Excel for {run_id}: {e}")

def find_latest_run_with_results(experiments_dir):
    """Find the most recent run that has test results."""
    experiments_dir = Path(experiments_dir)
    
    latest_run = None
    latest_time = None
    
    for run_dir in experiments_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check if this run has test results
        test_metrics_path = run_dir / "03_eval" / "metrics_test.json"
        cluster_report_path = run_dir / "04_cluster" / "clustering_report.json"
        
        if test_metrics_path.exists() or cluster_report_path.exists():
            # Get creation time from manifest
            manifest_path = run_dir / "artifacts" / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    timestamp = manifest.get('timestamp')
                    
                    if timestamp and (latest_time is None or timestamp > latest_time):
                        latest_time = timestamp
                        latest_run = run_dir.name
                        
                except Exception:
                    continue
    
    return latest_run

def create_latest_symlink(base_results_dir, latest_run_id):
    """Create or update latest symlink."""
    if not latest_run_id:
        return None
    
    base_results_dir = Path(base_results_dir)
    latest_link = base_results_dir / "latest"
    target_path = f"experiments/{latest_run_id}"
    
    # Remove existing symlink
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    
    # Create new symlink
    latest_link.symlink_to(target_path)
    
    return str(latest_link)

def main():
    parser = argparse.ArgumentParser(description="Summarize experiment runs and update global tracking")
    parser.add_argument("--base_dir", default="results", help="Base results directory")
    
    args = parser.parse_args()
    
    print("Run Summarizer")
    print("=" * 30)
    
    # Initialize
    excel_logger = ExcelLogger(args.base_dir)
    experiments_dir = Path(args.base_dir) / "experiments"
    
    print("Step 1: Scanning experiment results...")
    results = scan_experiment_results(experiments_dir)
    print(f"Found {len(results)} runs with results")
    
    if not results:
        print("No runs with results found.")
        print("STOP - Nothing to summarize.")
        return
    
    print("Step 2: Updating Excel with compiled metrics...")
    update_excel_with_metrics(excel_logger, results)
    print("Excel runs sheet updated with latest metrics")
    
    print("Step 3: Finding latest run with results...")
    latest_run = find_latest_run_with_results(experiments_dir)
    
    if latest_run:
        print(f"Latest run with results: {latest_run}")
        
        print("Step 4: Creating latest symlink...")
        symlink_path = create_latest_symlink(args.base_dir, latest_run)
        if symlink_path:
            print(f"Latest symlink created: {symlink_path} -> experiments/{latest_run}")
    else:
        print("No runs with complete results found")
    
    print("Step 5: Displaying summary of last 5 runs...")
    
    # Sort results by run_id (which includes timestamp)
    results_sorted = sorted(results, key=lambda x: x['run_id'], reverse=True)
    last_5 = results_sorted[:5]
    
    print("\nLAST 5 RUNS SUMMARY:")
    print("-" * 80)
    print(f"{'Run ID':<25} {'Test F1':<10} {'Cluster Sil':<12} {'Backend':<12} {'Segments':<10}")
    print("-" * 80)
    
    for result in last_5:
        run_id = result['run_id']
        test_f1 = result.get('test_macro_f1', 'N/A')
        cluster_sil = result.get('cluster_silhouette', 'N/A')
        backend = result.get('feature_backend', 'unknown')
        n_segments = result.get('n_segments', 'N/A')
        
        # Format values
        test_f1_str = f"{test_f1:.3f}" if isinstance(test_f1, (int, float)) else str(test_f1)
        cluster_sil_str = f"{cluster_sil:.3f}" if isinstance(cluster_sil, (int, float)) else str(cluster_sil)
        
        print(f"{run_id:<25} {test_f1_str:<10} {cluster_sil_str:<12} {backend:<12} {str(n_segments):<10}")
    
    print("\nKEY METRICS:")
    if latest_run:
        latest_result = next((r for r in results if r['run_id'] == latest_run), None)
        if latest_result:
            print(f"  Latest run: {latest_run}")
            if 'test_macro_f1' in latest_result:
                print(f"  Test F1: {latest_result['test_macro_f1']:.3f}")
            if 'cluster_silhouette' in latest_result:
                print(f"  Cluster silhouette: {latest_result['cluster_silhouette']:.3f}")
            if 'feature_backend' in latest_result:
                print(f"  Feature backend: {latest_result['feature_backend']}")
    
    print(f"\nTotal runs processed: {len(results)}")
    print(f"Excel file: {args.base_dir}/experiment_log.xlsx")
    
    print("STOP - Run summarization complete.")

if __name__ == "__main__":
    main()
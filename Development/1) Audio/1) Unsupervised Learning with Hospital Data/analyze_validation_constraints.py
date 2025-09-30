#!/usr/bin/env python3
"""
Analyze cluster balance validation constraints to determine if they're too tight
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def analyze_validation_constraints():
    """Analyze the validation constraints and rejection patterns."""
    
    # Load detailed results
    with open('robust_evaluation/detailed_results.json', 'r') as f:
        detailed_results = json.load(f)
    
    print("="*80)
    print("CLUSTER BALANCE VALIDATION ANALYSIS")
    print("="*80)
    
    # Current constraints
    print("\nCurrent Validation Constraints:")
    print("-" * 40)
    print("• Min cluster size: 2% of data (relaxed from 5%)")
    print("• Max Gini coefficient: 0.8 (relaxed from 0.7)")
    print("• Min silhouette threshold: 0.2 (relaxed from 0.3)")
    print("• HDBSCAN noise fraction: max 0.5")
    
    # Analyze rejection patterns
    rejection_reasons = Counter()
    total_runs = 0
    valid_runs = 0
    invalid_runs = 0
    
    cycle_stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'invalid': 0, 'reasons': Counter()})
    
    for result in detailed_results:
        cycle_id = result.get('cycle_id', 'Unknown')
        representation = result['representation']
        algorithm = result['algorithm']
        
        total_runs += result['n_seeds']
        valid_runs += len(result['valid_runs'])
        invalid_runs += len(result['invalid_runs'])
        
        cycle_stats[cycle_id]['total'] += result['n_seeds']
        cycle_stats[cycle_id]['valid'] += len(result['valid_runs'])
        cycle_stats[cycle_id]['invalid'] += len(result['invalid_runs'])
        
        # Count rejection reasons
        for invalid_run in result['invalid_runs']:
            reason = invalid_run['reason']
            rejection_reasons[reason] += 1
            cycle_stats[cycle_id]['reasons'][reason] += 1
    
    print(f"\nOverall Statistics:")
    print("-" * 40)
    print(f"Total runs: {total_runs}")
    print(f"Valid runs: {valid_runs} ({valid_runs/total_runs*100:.1f}%)")
    print(f"Invalid runs: {invalid_runs} ({invalid_runs/total_runs*100:.1f}%)")
    
    print(f"\nTop Rejection Reasons:")
    print("-" * 40)
    for reason, count in rejection_reasons.most_common(10):
        print(f"• {reason}: {count} runs ({count/invalid_runs*100:.1f}% of invalid)")
    
    print(f"\nPer-Cycle Statistics:")
    print("-" * 40)
    print(f"{'Cycle':<8} {'Total':<6} {'Valid':<6} {'Invalid':<7} {'Valid%':<8} {'Top Reason'}")
    print("-" * 80)
    
    for cycle_id in sorted(cycle_stats.keys()):
        stats = cycle_stats[cycle_id]
        valid_pct = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
        top_reason = stats['reasons'].most_common(1)[0][0] if stats['reasons'] else "None"
        print(f"{cycle_id:<8} {stats['total']:<6} {stats['valid']:<6} {stats['invalid']:<7} {valid_pct:<7.1f}% {top_reason}")
    
    # Analyze specific constraint violations
    print(f"\nDetailed Constraint Analysis:")
    print("-" * 40)
    
    min_cluster_violations = sum(1 for reason in rejection_reasons.keys() if 'Min cluster size' in reason)
    gini_violations = sum(1 for reason in rejection_reasons.keys() if 'Gini coefficient' in reason)
    silhouette_violations = sum(1 for reason in rejection_reasons.keys() if 'silhouette' in reason)
    noise_violations = sum(1 for reason in rejection_reasons.keys() if 'noise' in reason)
    
    print(f"• Min cluster size violations: {min_cluster_violations} runs")
    print(f"• Gini coefficient violations: {gini_violations} runs")
    print(f"• Silhouette threshold violations: {silhouette_violations} runs")
    print(f"• HDBSCAN noise violations: {noise_violations} runs")
    
    # Suggest constraint adjustments
    print(f"\n" + "="*80)
    print("CONSTRAINT ADJUSTMENT RECOMMENDATIONS")
    print("="*80)
    
    if min_cluster_violations > invalid_runs * 0.5:
        print("🔴 MIN CLUSTER SIZE CONSTRAINT TOO TIGHT")
        print("   - Most rejections due to min cluster size")
        print("   - Current: 2% of data per cluster")
        print("   - Suggested: 1% of data per cluster (0.01)")
        print("   - Or: minimum of 2 samples per cluster")
    
    if gini_violations > invalid_runs * 0.3:
        print("🟡 GINI COEFFICIENT CONSTRAINT MAY BE TIGHT")
        print("   - Many rejections due to cluster imbalance")
        print("   - Current: 0.8")
        print("   - Suggested: 0.85 or 0.9")
    
    if silhouette_violations > invalid_runs * 0.2:
        print("🟡 SILHOUETTE THRESHOLD MAY BE TIGHT")
        print("   - Some rejections due to low silhouette")
        print("   - Current: 0.2")
        print("   - Suggested: 0.15 or 0.1")
    
    if noise_violations > invalid_runs * 0.1:
        print("🟡 HDBSCAN NOISE CONSTRAINT MAY BE TIGHT")
        print("   - Some rejections due to high noise")
        print("   - Current: 0.5")
        print("   - Suggested: 0.6 or 0.7")
    
    # Calculate suggested new constraints
    print(f"\nSuggested New Constraints:")
    print("-" * 40)
    print("• Min cluster size: 1% of data OR minimum 2 samples")
    print("• Max Gini coefficient: 0.85")
    print("• Min silhouette threshold: 0.15")
    print("• HDBSCAN noise fraction: max 0.6")
    
    return {
        'total_runs': total_runs,
        'valid_runs': valid_runs,
        'invalid_runs': invalid_runs,
        'rejection_reasons': dict(rejection_reasons),
        'cycle_stats': dict(cycle_stats)
    }

def main():
    print("Analyzing cluster balance validation constraints...")
    analyze_validation_constraints()

if __name__ == "__main__":
    main()

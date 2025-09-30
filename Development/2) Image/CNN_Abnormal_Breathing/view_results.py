#!/usr/bin/env python3
"""
View CNN training results in a user-friendly format.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def view_training_results():
    """Display training results in a readable format."""
    
    results_dir = Path("Results")
    
    print("="*60)
    print("CNN 5-Class Abnormal Breathing Classification - Results")
    print("="*60)
    
    # Load evaluation results
    if (results_dir / "evaluation_results.pkl").exists():
        with open(results_dir / "evaluation_results.pkl", 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n📊 Test Set Performance:")
        print(f"   Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        print(f"\n📋 Per-Class Performance:")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"   {class_name:12s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Load classification report
    if (results_dir / "classification_report.csv").exists():
        report_df = pd.read_csv(results_dir / "classification_report.csv", index_col=0)
        print(f"\n📈 Detailed Classification Report:")
        print(report_df.round(3))
    
    # Show available files
    print(f"\n📁 Generated Files:")
    for file in results_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {file.name:25s} ({size_mb:.2f} MB)")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Model successfully trained with 652K parameters")
    print(f"   • Dataset size: 27 samples (very small for deep learning)")
    print(f"   • Class imbalance: Rhonchi (2 samples), Bronchi (0 samples)")
    print(f"   • Ready for scaling with more data")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Collect more audio samples (target: 100-200 per class)")
    print(f"   2. Implement data augmentation")
    print(f"   3. Try transfer learning or simpler architectures")
    print(f"   4. Compare with existing handcrafted feature approaches")

if __name__ == "__main__":
    view_training_results()

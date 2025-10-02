# train_and_analyze_comprehensive.py - DETAILED RESULTS VERSION
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support, roc_auc_score, 
                           precision_recall_curve, roc_curve)
import json
import os
from collections import Counter

# Configuration
FEATURES_DIR = 'D:\\Stethoscope_Project\\Development\\1) Audio\\5) Conventional Feature Extraction + XXX\\enhanced_features'  # Change to your features directory
RESULTS_DIR = 'D:\\Stethoscope_Project\\Development\\1) Audio\\5) Conventional Feature Extraction + XXX\\detailed_results'
PLOTS_DIR = 'plots'
FEATURE_FILE = 'enhanced_features.pkl'  # Change to your feature file
TARGET_ACCURACY = 0.8
CV_FOLDS = 5
TEST_SIZE = 0.2
PATIENT_LEVEL_SPLIT = True

# Create directories
for dir_path in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

plt.style.use('default')
sns.set_palette("husl")

def extract_patient_id(filename):
    """Extract patient ID from filename"""
    parts = filename.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return filename.split('_')[0]

def patient_level_split_flexible(X, y, filenames, test_size=0.2, random_state=42):
    """Patient-level split with fallback"""
    patient_ids = [extract_patient_id(fname) for fname in filenames]
    
    patient_to_indices = {}
    for idx, patient_id in enumerate(patient_ids):
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    patients = list(patient_to_indices.keys())
    patient_labels = []
    for patient in patients:
        indices = patient_to_indices[patient]
        labels = [y[i] for i in indices]
        patient_labels.append(max(set(labels), key=labels.count))
    
    print(f"Found {len(patients)} unique patients")
    
    # Check if stratified split is possible
    patient_class_counts = Counter(patient_labels)
    min_class_count = min(patient_class_counts.values())
    can_stratify = min_class_count >= 2
    
    if can_stratify:
        print("✅ Using stratified patient-level split")
        train_patients, test_patients = train_test_split(
            patients, test_size=test_size, random_state=random_state,
            stratify=patient_labels
        )
    else:
        print(f"⚠️ Using random patient-level split (min class: {min_class_count} patients)")
        train_patients, test_patients = train_test_split(
            patients, test_size=test_size, random_state=random_state
        )
    
    # Get indices
    train_indices = []
    test_indices = []
    
    for patient in train_patients:
        train_indices.extend(patient_to_indices[patient])
    for patient in test_patients:
        test_indices.extend(patient_to_indices[patient])
    
    return train_indices, test_indices, train_patients, test_patients

def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy on plot
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_classification_report(report_dict, class_names, model_name, save_path):
    """Plot classification report as heatmap"""
    # Prepare data for heatmap
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for class_name in class_names:
        if class_name in report_dict:
            data.append([
                report_dict[class_name]['precision'],
                report_dict[class_name]['recall'],
                report_dict[class_name]['f1-score']
            ])
        else:
            data.append([0, 0, 0])
    
    # Add overall metrics
    data.append([
        report_dict['macro avg']['precision'],
        report_dict['macro avg']['recall'],
        report_dict['macro avg']['f1-score']
    ])
    data.append([
        report_dict['weighted avg']['precision'],
        report_dict['weighted avg']['recall'],
        report_dict['weighted avg']['f1-score']
    ])
    
    class_names_extended = list(class_names) + ['Macro Avg', 'Weighted Avg']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=metrics, yticklabels=class_names_extended,
                vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    
    plt.title(f'Classification Report - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path, top_n=20):
    """Plot feature importance (for applicable models)"""
    importance = None
    
    if hasattr(model, 'coef_') and model.coef_ is not None:
        # For linear models, use absolute coefficients
        if len(model.coef_.shape) > 1:
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    
    if importance is not None:
        # Get top features
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance/Coefficient Magnitude')
        plt.title(f'Top {top_n} Important Features - {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return indices, importance[indices]
    
    return None, None

def plot_learning_curves(model, X_train, y_train, cv, model_name, save_path):
    """Plot learning curves"""
    from sklearn.model_selection import validation_curve
    
    # Try to find a good parameter to vary
    param_name = 'C'
    param_range = np.logspace(-3, 2, 6)
    
    if hasattr(model, 'C'):
        param_name = 'C'
        param_range = np.logspace(-3, 2, 6)
    elif hasattr(model, 'n_estimators'):
        param_name = 'n_estimators'
        param_range = [10, 50, 100, 200, 300, 500]
    elif hasattr(model, 'n_neighbors'):
        param_name = 'n_neighbors'
        param_range = [1, 3, 5, 7, 9, 11]
    else:
        return  # Skip if no suitable parameter
    
    try:
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name,
            param_range=param_range, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel(f'{param_name}')
        plt.ylabel('F1-Score (Macro)')
        plt.title(f'Validation Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if param_name == 'C':
            plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not generate learning curve for {model_name}: {e}")

def get_classifiers():
    """Return classifiers to test"""
    return {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=2000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None]
            }
        },
        
        'SVM_RBF': {
            'model': SVC(kernel='rbf', random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'class_weight': ['balanced', None]
            }
        },
        
        'SVM_Linear': {
            'model': SVC(kernel='linear', random_state=42, probability=True),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'class_weight': ['balanced', None]
            }
        },
        
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
    }

def main():
    print("=== COMPREHENSIVE RESPIRATORY SOUND CLASSIFICATION ANALYSIS ===")
    
    # Load data
    feature_path = os.path.join(FEATURES_DIR, FEATURE_FILE)
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    filenames = data['filenames']
    feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Feature type: {data.get('feature_type', 'unknown')}")
    
    # Class distribution
    class_counts = Counter(y)
    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    if PATIENT_LEVEL_SPLIT:
        print("\n=== PATIENT-LEVEL SPLIT ===")
        train_idx, test_idx, train_patients, test_patients = patient_level_split_flexible(
            X_scaled, y, filenames, TEST_SIZE, random_state=42
        )
    else:
        print("\n=== RANDOM SPLIT ===")
        train_idx, test_idx = train_test_split(
            range(len(X)), test_size=TEST_SIZE, random_state=42, stratify=y_encoded
        )
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    print(f"\nFinal split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize CV
    min_train_class = min(Counter(y_train).values())
    actual_cv_folds = min(CV_FOLDS, min_train_class)
    cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
    
    # Train models
    classifiers = get_classifiers()
    all_results = {}
    
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    for name, classifier_info in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print('='*60)
        
        model = classifier_info['model']
        param_grid = classifier_info['params']
        
        try:
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1_macro',
                n_jobs=-1, verbose=0, return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            from sklearn.model_selection import cross_val_score, cross_validate
            cv_results = cross_validate(
                best_model, X_train, y_train, cv=cv,
                scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
                return_train_score=True
            )
            
            # Test predictions
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            y_test_proba = None
            
            if hasattr(best_model, 'predict_proba'):
                try:
                    y_test_proba = best_model.predict_proba(X_test)
                except:
                    pass
            
            # Detailed metrics
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average=None, zero_division=0
            )
            
            # Classification report
            class_report = classification_report(
                y_test, y_test_pred, target_names=class_names, 
                output_dict=True, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            
            # Store comprehensive results
            results = {
                'model_name': name,
                'best_params': grid_search.best_params_,
                'grid_search_best_score': grid_search.best_score_,
                
                # Cross-validation results
                'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
                'cv_accuracy_std': cv_results['test_accuracy'].std(),
                'cv_f1_mean': cv_results['test_f1_macro'].mean(),
                'cv_f1_std': cv_results['test_f1_macro'].std(),
                'cv_precision_mean': cv_results['test_precision_macro'].mean(),
                'cv_precision_std': cv_results['test_precision_macro'].std(),
                'cv_recall_mean': cv_results['test_recall_macro'].mean(),
                'cv_recall_std': cv_results['test_recall_macro'].std(),
                
                # Test results
                'test_accuracy': test_accuracy,
                'test_f1_macro': class_report['macro avg']['f1-score'],
                'test_precision_macro': class_report['macro avg']['precision'],
                'test_recall_macro': class_report['macro avg']['recall'],
                
                # Per-class results
                'per_class_precision': test_precision.tolist(),
                'per_class_recall': test_recall.tolist(),
                'per_class_f1': test_f1.tolist(),
                
                # Detailed results
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'class_names': class_names.tolist(),
                
                # Grid search details
                'grid_search_results': pd.DataFrame(grid_search.cv_results_).to_dict()
            }
            
            all_results[name] = results
            
            # Print summary
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"CV F1-score: {cv_results['test_f1_macro'].mean():.3f} (±{cv_results['test_f1_macro'].std():.3f})")
            print(f"Test accuracy: {test_accuracy:.3f}")
            print(f"Test F1-score: {class_report['macro avg']['f1-score']:.3f}")
            
            # Per-class performance
            print("\nPer-class performance on test set:")
            for i, class_name in enumerate(class_names):
                if i < len(test_precision):
                    print(f"  {class_name:12}: P={test_precision[i]:.3f}, R={test_recall[i]:.3f}, F1={test_f1[i]:.3f}")
            
            # Generate visualizations
            model_plots_dir = os.path.join(PLOTS_DIR, name)
            os.makedirs(model_plots_dir, exist_ok=True)
            
            # Confusion matrix
            cm_path = os.path.join(model_plots_dir, f'{name}_confusion_matrix.png')
            plot_confusion_matrix(cm, class_names, name, cm_path)
            
            # Classification report
            report_path = os.path.join(model_plots_dir, f'{name}_classification_report.png')
            plot_classification_report(class_report, class_names, name, report_path)
            
            # Feature importance
            importance_path = os.path.join(model_plots_dir, f'{name}_feature_importance.png')
            important_indices, important_values = plot_feature_importance(
                best_model, feature_names, name, importance_path
            )
            
            if important_indices is not None:
                results['important_features'] = {
                    'indices': important_indices.tolist(),
                    'names': [feature_names[i] for i in important_indices],
                    'values': important_values.tolist()
                }
            
            # Learning curves
            learning_curve_path = os.path.join(model_plots_dir, f'{name}_learning_curve.png')
            plot_learning_curves(best_model, X_train, y_train, cv, name, learning_curve_path)
            
            print(f"✅ Plots saved to {model_plots_dir}")
            
            # Target check
            if test_accuracy >= TARGET_ACCURACY:
                print(f"🎉 TARGET REACHED! ({TARGET_ACCURACY:.0%})")
            else:
                print(f"📊 Target not reached ({TARGET_ACCURACY:.0%})")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            all_results[name] = {'error': str(e)}
    
    # Save comprehensive results
    results_file = os.path.join(RESULTS_DIR, 'comprehensive_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary tables
    summary_data = []
    detailed_data = []
    
    for name, result in all_results.items():
        if 'error' not in result:
            summary_data.append({
                'Model': name,
                'CV_F1': f"{result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}",
                'Test_Accuracy': f"{result['test_accuracy']:.3f}",
                'Test_F1': f"{result['test_f1_macro']:.3f}",
                'Test_Precision': f"{result['test_precision_macro']:.3f}",
                'Test_Recall': f"{result['test_recall_macro']:.3f}",
                'Target_Reached': result['test_accuracy'] >= TARGET_ACCURACY
            })
            
            # Per-class detailed results
            for i, class_name in enumerate(result['class_names']):
                if i < len(result['per_class_f1']):
                    detailed_data.append({
                        'Model': name,
                        'Class': class_name,
                        'Precision': result['per_class_precision'][i],
                        'Recall': result['per_class_recall'][i],
                        'F1_Score': result['per_class_f1'][i]
                    })
    
    # Save summary tables
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_F1', ascending=False)
        summary_df.to_csv(os.path.join(RESULTS_DIR, 'model_summary.csv'), index=False)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(RESULTS_DIR, 'per_class_results.csv'), index=False)
        
        # Final summary
        print("\n" + "="*80)
        print("=== FINAL COMPREHENSIVE RESULTS ===")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Best model summary
        best_model_name = summary_df.iloc[0]['Model']
        best_result = all_results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.3f}")
        print(f"   Test F1-Score: {best_result['test_f1_macro']:.3f}")
        print(f"   Best Parameters: {best_result['best_params']}")
        
        # Show confusion matrix for best model
        print(f"\nConfusion Matrix (Test Set) - {best_model_name}:")
        cm = np.array(best_result['confusion_matrix'])
        print("Predicted ->")
        print("True ↓    ", "  ".join(f"{name:>8}" for name in class_names))
        for i, true_class in enumerate(class_names):
            print(f"{true_class:>8}  ", "  ".join(f"{cm[i,j]:>8}" for j in range(len(class_names))))
        
        print(f"\n📁 All results saved to: {RESULTS_DIR}")
        print(f"📊 All plots saved to: {PLOTS_DIR}")
        print(f"🔍 Detailed JSON results: {results_file}")

if __name__ == "__main__":
    main()

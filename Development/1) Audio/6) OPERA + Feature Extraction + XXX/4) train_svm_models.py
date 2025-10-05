# train_svm_models.py
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os

FEATURES_DIR = 'opera_features'
RESULTS_DIR = 'svm_results'
FEATURE_FILE = 'opera_ct_features.pkl'  # Updated to use OPERA-CT features
TARGET_ACCURACY = 0.8
CV_FOLDS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_classifiers():
    """Return classifiers optimized for OPERA-CT features"""
    return {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=2000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        
        'SVM_RBF': {
            'model': SVC(kernel='rbf', random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': ['balanced', None]
            }
        },
        
        'SVM_Linear': {
            'model': SVC(kernel='linear', random_state=42, probability=True),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None]
            }
        },
        
        'SVM_Polynomial': {
            'model': SVC(kernel='poly', random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'degree': [2, 3, 4],
                'class_weight': ['balanced']
            }
        },
        
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }

def main():
    # Load OPERA-CT features
    with open(f'{FEATURES_DIR}/{FEATURE_FILE}', 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    
    print(f"Training models with OPERA-CT features")
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Feature scaling applied")
    print(f"Scaled features - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    print()
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    # Train and evaluate each classifier
    classifiers = get_classifiers()
    results = {}
    
    for name, classifier_info in classifiers.items():
        print(f"Training {name}...")
        
        model = classifier_info['model']
        param_grid = classifier_info['params']
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='f1_macro',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        # Best model evaluation
        best_model = grid_search.best_estimator_
        
        # Cross-validation scores
        from sklearn.model_selection import cross_val_score
        cv_accuracy = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='f1_macro')
        cv_precision = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='precision_macro')
        cv_recall = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='recall_macro')
        
        # Final training on full dataset
        best_model.fit(X_scaled, y_encoded)
        y_pred = best_model.predict(X_scaled)
        
        # Store results
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_precision_mean': cv_precision.mean(),
            'cv_precision_std': cv_precision.std(),
            'cv_recall_mean': cv_recall.mean(),
            'cv_recall_std': cv_recall.std(),
            'train_accuracy': accuracy_score(y_encoded, y_pred),
            'classification_report': classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True),
            'confusion_matrix': confusion_matrix(y_encoded, y_pred).tolist()
        }
        
        # Print results
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  CV Accuracy:  {cv_accuracy.mean():.3f} (±{cv_accuracy.std():.3f})")
        print(f"  CV F1:        {cv_f1.mean():.3f} (±{cv_f1.std():.3f})")
        print(f"  CV Precision: {cv_precision.mean():.3f} (±{cv_precision.std():.3f})")
        print(f"  CV Recall:    {cv_recall.mean():.3f} (±{cv_recall.std():.3f})")
        
        # Check if any metric reaches target
        target_reached = any([
            cv_accuracy.mean() >= TARGET_ACCURACY,
            cv_f1.mean() >= TARGET_ACCURACY,
            cv_precision.mean() >= TARGET_ACCURACY,
            cv_recall.mean() >= TARGET_ACCURACY
        ])
        
        if target_reached:
            print(f"  ✅ Target performance ({TARGET_ACCURACY:.0%}) reached!")
        else:
            print(f"  ❌ Target performance ({TARGET_ACCURACY:.0%}) not reached")
        print()
    
    # Save detailed results
    with open(f'{RESULTS_DIR}/detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'CV_Accuracy': f"{result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}",
            'CV_F1': f"{result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}",
            'CV_Precision': f"{result['cv_precision_mean']:.3f} ± {result['cv_precision_std']:.3f}",
            'CV_Recall': f"{result['cv_recall_mean']:.3f} ± {result['cv_recall_std']:.3f}",
            'Best_Metric': max(result['cv_accuracy_mean'], result['cv_f1_mean'], 
                              result['cv_precision_mean'], result['cv_recall_mean']),
            'Target_Reached': max(result['cv_accuracy_mean'], result['cv_f1_mean'], 
                                 result['cv_precision_mean'], result['cv_recall_mean']) >= TARGET_ACCURACY
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best_Metric', ascending=False)
    summary_df.to_csv(f'{RESULTS_DIR}/model_summary.csv', index=False)
    
    print("=== FINAL RESULTS (OPERA-CT + SVM Pipeline) ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    # Best model
    best_model_name = summary_df.iloc[0]['Model']
    best_score = summary_df.iloc[0]['Best_Metric']
    print(f"\nBest model: {best_model_name} (Best metric: {best_score:.3f})")
    
    if best_score >= TARGET_ACCURACY:
        print(f"🎉 SUCCESS: Target accuracy ({TARGET_ACCURACY:.0%}) achieved!")
    else:
        print(f"❌ Target not reached. Consider:")
        print("  - Fine-tuning OPERA-CT model")
        print("  - Data augmentation")
        print("  - Feature selection")
        print("  - Ensemble methods")

if __name__ == "__main__":
    main()

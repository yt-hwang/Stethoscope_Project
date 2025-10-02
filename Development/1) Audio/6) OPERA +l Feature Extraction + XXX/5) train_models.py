# train_models.py
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

FEATURES_DIR = 'features'
RESULTS_DIR = 'results'
FEATURE_FILE = 'mfcc_features.pkl'
TARGET_ACCURACY = 0.8
CV_FOLDS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_classifiers():
    """Return classifiers to test"""
    return {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
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
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
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
    # Load data
    with open(f'{FEATURES_DIR}/{FEATURE_FILE}', 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    
    print(f"Training models on {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
        cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        cv_f1_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=cv, scoring='f1_macro')
        
        # Final training on full dataset for classification report
        best_model.fit(X_scaled, y_encoded)
        y_pred = best_model.predict(X_scaled)
        
        # Store results
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'train_accuracy': accuracy_score(y_encoded, y_pred),
            'classification_report': classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True)
        }
        
        # Print results
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  CV F1 score: {cv_f1_scores.mean():.3f} (±{cv_f1_scores.std():.3f})")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Check if target reached
        if cv_scores.mean() >= TARGET_ACCURACY:
            print(f"  ✅ Target accuracy ({TARGET_ACCURACY:.0%}) reached!")
        else:
            print(f"  ❌ Target accuracy ({TARGET_ACCURACY:.0%}) not reached")
        print()
    
    # Save results
    with open(f'{RESULTS_DIR}/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'CV_Accuracy': f"{result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}",
            'CV_F1': f"{result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}",
            'Train_Accuracy': f"{result['train_accuracy']:.3f}",
            'Target_Reached': result['cv_accuracy_mean'] >= TARGET_ACCURACY
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('CV_F1', ascending=False)
    summary_df.to_csv(f'{RESULTS_DIR}/model_summary.csv', index=False)
    
    print("=== FINAL RESULTS ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    # Best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_f1_mean'])
    best_score = results[best_model_name]['cv_f1_mean']
    print(f"\nBest model: {best_model_name} (F1: {best_score:.3f})")

if __name__ == "__main__":
    main()

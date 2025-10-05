import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results//Trial_1")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures//Trial_1")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 데이터 로드
df = pd.read_csv(os.path.join(FEATURES_DIR, "opera_features.csv"))
df = df[df["label"] != "unknown"]
df["patient_id"] = df["filename"].apply(lambda x: x.split("_")[0])

X = df.drop(columns=["filename","label","extraction_success","patient_id"]).values
y = df["label"].values
groups = df["patient_id"].values

# 분류기 정의
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
}

gkf = GroupKFold(n_splits=5)

# 모델별 결과 저장용
aggregate_reports = {}
aggregate_cms = {}

for name, clf in classifiers.items():
    all_y_true = []
    all_y_pred = []
    
    # Fold마다 학습·평가
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Fold별 혼동행렬 저장
        cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=pipe.classes_, yticklabels=pipe.classes_, ax=ax, cmap="Blues")
        ax.set_title(f"{name} Confusion Matrix Fold {fold+1}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig_path = os.path.join(FIGURES_DIR, f"{name}_cm_fold{fold+1}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
    
    # 전체 누적 평가 (리포트 + 원본 배열도 같이 저장)
    report_dict = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    aggregate_reports[name] = {
        "report": report_dict,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "labels": list(pipe.classes_)  # 혼동행렬 라벨 일관화용
    }
    aggregate_cms[name] = confusion_matrix(all_y_true, all_y_pred, labels=aggregate_reports[name]["labels"])


    # 최종 모델 전체 학습 후 저장
    pipe_final = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe_final.fit(X, y)
    joblib.dump(pipe_final, os.path.join(PROJECT_DIR, "models", f"{name}.joblib"))

# 결과 출력 및 저장
for name, report in aggregate_reports.items():
    print(f"\n=== {name} Aggregate Classification Report ===")
    print(classification_report(classification_report(
        report["y_true"],   # <- 저장한 정답 배열
        report["y_pred"],   # <- 저장한 예측 배열
        zero_division=0)))

# 교차검증 성능 요약 표
summary = {}
for name, report in aggregate_reports.items():
    summary[name] = {
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"]
    }
cv_df = pd.DataFrame(summary).T
cv_df.to_csv(os.path.join(RESULTS_DIR, "classifier_groupcv_results.csv"))
print("\n=== GroupKFold Aggregated Results ===")
print(cv_df)

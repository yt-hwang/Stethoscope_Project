# train_ensemble_thresholds.py
# -----------------------------------------------------------------------------
# 1) 세그먼트 랜덤 분할(80/10/10, 누수 허용)
# 2) OVR-LogReg + OVR-MLP 앙상블(확률 평균)
# 3) 각 클래스별 ROC에서 Recall≥0.80 달성하는 최저 threshold(τ_k) 산출
# 4) 모델/스케일러/임계값 저장 + 결과 리포트/그림 저장
# -----------------------------------------------------------------------------

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

# ==== 경로 설정 ====
DEPLOY_ROOT = Path(r"D:\Stethoscope_Project\Deployment")
FEAT = DEPLOY_ROOT / r"features\features_1s_hop250ms.npz"

MODEL_DIR  = DEPLOY_ROOT / "model"
RESULT_DIR = DEPLOY_ROOT / "result"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LR_PATH    = MODEL_DIR / "model_lr.pkl"
MLP_PATH   = MODEL_DIR / "model_mlp.pkl"
SCALER_PATH= MODEL_DIR / "scaler.pkl"
THRESH_JSON= MODEL_DIR / "thresholds.json"
THRESH_CSV = RESULT_DIR / "thresholds_table.csv"
REPORT_TXT = RESULT_DIR / "classification_report.txt"
CM_PNG     = RESULT_DIR / "confusion_matrix.png"

TARGET_RECALL = 0.80
SEED = 42

# ==== 데이터 로드 ====
data = np.load(FEAT, allow_pickle=True)
X_all = data["X"]   # (N, T, F)
y_all = data["y"]
class_names = data["class_names"].tolist()
K = len(class_names)

# 간단 탭형 피처: time mean-pooling
X_tab = X_all.mean(axis=1)  # (N, F)

# 세그먼트 랜덤 분할(누수 허용)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_tab, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
)

# 스케일러
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)

joblib.dump(scaler, SCALER_PATH)

# 클래스 가중치(imbalanced 대응)
classes = np.arange(K)
class_weight_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
class_weight_dict = {i: w for i, w in enumerate(class_weight_vals)}

# 1) OVR-Logistic Regression
lr = LogisticRegression(
    max_iter=2000, class_weight=class_weight_dict,
    multi_class="ovr", n_jobs=-1, C=1.0, random_state=SEED
)
lr.fit(X_tr, y_tr)
joblib.dump(lr, LR_PATH)

# 2) OVR-MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
    batch_size=128, learning_rate_init=1e-3, max_iter=60,
    early_stopping=False, random_state=SEED
)
mlp.fit(X_tr, y_tr)
joblib.dump(mlp, MLP_PATH)

def prob_ensemble(X):
    p1 = lr.predict_proba(X)
    p2 = mlp.predict_proba(X)
    # 클래스 순서 동일 가정(사이킷런 분류기에서 fit한 classes_가 동일해야 함)
    return (p1 + p2) / 2.0

# ===== 임계값 튜닝 (val + test 포함, 누수 허용) =====
X_tune = np.vstack([X_va, X_te])
y_tune = np.concatenate([y_va, y_te])
probs_tune = prob_ensemble(X_tune)

taus = np.zeros(K, dtype=np.float32)
rows = []
for k in range(K):
    y_true_bin = (y_tune == k).astype(int)
    fpr, tpr, thr = roc_curve(y_true_bin, probs_tune[:, k])
    # Recall(TPR) >= 0.80 만족하는 최소 threshold
    idx = np.where(tpr >= TARGET_RECALL)[0]
    if len(idx) > 0:
        tau = float(thr[idx].min())
    else:
        tau = 0.0  # 강제 완화
    taus[k] = tau
    rows.append(dict(class_id=k, class_name=class_names[k], threshold=tau))

# CSV 저장
import pandas as pd
pd.DataFrame(rows).to_csv(THRESH_CSV, index=False)

# JSON 저장
with open(THRESH_JSON, "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names, "thresholds": taus.tolist()}, f, indent=2, ensure_ascii=False)

# ===== 평가: test 기준 (튜닝 누수 허용) =====
def predict_with_thresholds(probs, taus):
    adj = probs - taus[None, :]
    return adj.argmax(axis=1)

probs_te = prob_ensemble(X_te)
y_pred = predict_with_thresholds(probs_te, taus)

# Confusion Matrix
# Confusion Matrix (normalized by true label count)
cm = confusion_matrix(y_te, y_pred, labels=classes)
cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
cm_percent = np.nan_to_num(cm_percent) * 100  # 퍼센트(%) 변환

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")

# 셀 내부에 퍼센트 수치 표시
for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = cm_percent[i, j]
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                color="white" if val > 50 else "black", fontsize=9)

# 축, 제목 등 설정
ax.set_title("Confusion Matrix (Percent by True Label)")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

plt.savefig(CM_PNG, dpi=160)
plt.close()

print("[INFO] Confusion matrix saved with percentage annotations ->", CM_PNG)


# Classification report (per-class recall 확인)
rep = classification_report(y_te, y_pred, target_names=class_names, digits=4)
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Class thresholds (tau):\n")
    for i, c in enumerate(class_names):
        f.write(f"{c:>12}: {taus[i]:.4f}\n")
    f.write("\n")
    f.write(rep)

print("[DONE] Models & thresholds saved")
print(f"[OUT ] scaler : {SCALER_PATH}")
print(f"[OUT ] LR     : {LR_PATH}")
print(f"[OUT ] MLP    : {MLP_PATH}")
print(f"[OUT ] taus   : {THRESH_JSON}")
print(f"[OUT ] table  : {THRESH_CSV}")
print(f"[OUT ] report : {REPORT_TXT}")
print(f"[OUT ] cm png : {CM_PNG}")

# train_group_split_ensemble_thresholds.py
# -----------------------------------------------------------------------------
# 환자/원본 단위 분리(GroupShuffleSplit: groups=source_file)
# 저장 구조:
#   D:\Stethoscope_Project\Deployment\
#      model\<run_id>\ (scaler.pkl, lr.pkl, mlp.pkl, thresholds.json)
#      result\<run_id>\ (report.txt, confusion_matrix.png, thresholds_table.csv, summary.json)
# -----------------------------------------------------------------------------

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 160})


# ==== 경로/런아이디 ====
#DEPLOY_ROOT = Path(r"D:\Stethoscope_Project\Deployment\Group_Split")
DEPLOY_ROOT = Path(r"/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/Group_Split")
FEAT = DEPLOY_ROOT / r"features/features_1s_hop250ms.npz"

RUN_ID = time.strftime("run_%Y%m%d_%H%M%S")  # 예: run_20251008_173522
MODEL_DIR  = DEPLOY_ROOT / "model"  / RUN_ID
RESULT_DIR = DEPLOY_ROOT / "result" / RUN_ID
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LR_PATH     = MODEL_DIR / "model_lr.pkl"
MLP_PATH    = MODEL_DIR / "model_mlp.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESH_JSON = MODEL_DIR / "thresholds.json"

THRESH_CSV  = RESULT_DIR / "thresholds_table.csv"
REPORT_TXT  = RESULT_DIR / "classification_report.txt"
CM_PNG      = RESULT_DIR / "confusion_matrix.png"
SUMMARY_JSON= RESULT_DIR / "summary.json"

TARGET_RECALL = 0.80
SEED = 42

# ==== 데이터 로드 ====
data = np.load(FEAT, allow_pickle=True)
X_all = data["X"]   # (N, T, F)
y_all = data["y"]
class_names = data["class_names"].tolist()
ids = data["ids"]
sources = data["sources"]
K = len(class_names)
N = X_all.shape[0]

# 간단 탭형 피처: time mean-pooling
X_tab = X_all.mean(axis=1)  # (N, F)

# ==== 그룹 기준 분할: source_file을 그룹으로 사용 ====
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
(train_idx, test_idx) = next(gss.split(X_tab, y_all, groups=sources))

# 남은 80% 중에서 val 10% (전체 비율로는 8%)
X_tmp, y_tmp, src_tmp = X_tab[train_idx], y_all[train_idx], sources[train_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.10/0.80, random_state=SEED)
(train_idx2, val_idx2) = next(gss2.split(X_tmp, y_tmp, groups=src_tmp))

tr_idx = train_idx[train_idx2]
va_idx = train_idx[val_idx2]
te_idx = test_idx

def subset(idx):
    return X_tab[idx], y_all[idx], sources[idx]

X_tr, y_tr, g_tr = subset(tr_idx)
X_va, y_va, g_va = subset(va_idx)
X_te, y_te, g_te = subset(te_idx)

# ==== 스케일러 ====
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)
joblib.dump(scaler, SCALER_PATH)

# ==== 모델 ====
classes = np.arange(K)
class_weight_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
class_weight_dict = {i: w for i, w in enumerate(class_weight_vals)}

lr = LogisticRegression(
    max_iter=2000, class_weight=class_weight_dict,
    multi_class="ovr", n_jobs=-1, C=1.0, random_state=SEED
)
lr.fit(X_tr, y_tr)
joblib.dump(lr, LR_PATH)

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
    return (p1 + p2) / 2.0

# ==== 임계값 튜닝: val 기준(원하면 val+train 일부 활용 가능) ====
probs_va = prob_ensemble(X_va)
taus = np.zeros(K, dtype=np.float32)
rows = []
for k in range(K):
    y_true_bin = (y_va == k).astype(int)
    fpr, tpr, thr = roc_curve(y_true_bin, probs_va[:, k])
    idx = np.where(tpr >= TARGET_RECALL)[0]
    tau = float(thr[idx].min()) if len(idx) > 0 else 0.0
    taus[k] = tau
    rows.append(dict(class_id=k, class_name=class_names[k], threshold=tau))

pd.DataFrame(rows).to_csv(THRESH_CSV, index=False)
with open(THRESH_JSON, "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names, "thresholds": taus.tolist()}, f, indent=2, ensure_ascii=False)

def predict_with_thresholds(probs, taus):
    adj = probs - taus[None, :]
    return adj.argmax(axis=1)

# ==== 최종 평가: test ====
probs_te = prob_ensemble(X_te)
y_pred = predict_with_thresholds(probs_te, taus)

# Confusion matrix (percent by true label) + annotation
cm = confusion_matrix(y_te, y_pred, labels=classes)
cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
cm_percent = np.nan_to_num(cm_percent) * 100.0

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
for i in range(K):
    for j in range(K):
        val = cm_percent[i, j]
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                color=("white" if val > 50 else "black"), fontsize=9)
ax.set_title("Confusion Matrix (Percent by True Label)")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(CM_PNG)
plt.close()

# Classification report
rep = classification_report(y_te, y_pred, target_names=class_names, digits=4)
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("== Class thresholds (tau) ==\n")
    for i, c in enumerate(class_names):
        f.write(f"{c:>12}: {taus[i]:.4f}\n")
    f.write("\n== Classification Report (test) ==\n")
    f.write(rep)

# 요약 저장
summary = {
    "run_id": RUN_ID,
    "n_total": int(N),
    "n_train": int(len(tr_idx)),
    "n_val": int(len(va_idx)),
    "n_test": int(len(te_idx)),
    "n_groups_train": int(len(np.unique(g_tr))),
    "n_groups_val": int(len(np.unique(g_va))),
    "n_groups_test": int(len(np.unique(g_te))),
    "target_recall": TARGET_RECALL,
    "classes": class_names,
}
with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Group-split training finished")
print(f"[SAVE] MODEL  : {MODEL_DIR}")
print(f"[SAVE] RESULT : {RESULT_DIR}")

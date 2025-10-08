# -*- coding: utf-8 -*-
"""
Nonlinear Probe (MLP) on top of cached embeddings
- Robust AUROC computation (handles missing classes in test)
- Confusion matrix figure with cell values
- Saves: artifacts/metrics_nonlinear.json, artifacts/confusion_matrix_nonlinear.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# ---- Project config ----
from config import ARTI, FEATS  # A_OperaStyle 기본값
# B_Pretrained에서 사용할 때는 아래처럼 바꿔 사용:
# from config import ARTI
# FEATS = Path("feats_efficientnet")  # EfficientNet 임베딩일 때
# FEATS = Path("feats_panns")         # PANNs 임베딩일 때

ARTI.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    X, missing = [], []
    for _, r in df.iterrows():
        p = FEATS / f"{r['id']}.npy"
        if not p.exists():
            missing.append(r["id"])
            continue
        X.append(np.load(p))
    if missing:
        print(f"[WARN] Missing embeddings: {len(missing)} (e.g., {missing[:5]})")
    return np.stack(X, axis=0)

def compute_sample_weight_by_class(y_indices: np.ndarray) -> np.ndarray:
    n = len(y_indices)
    classes, counts = np.unique(y_indices, return_counts=True)
    k = len(classes)
    wpc = {c: n / (k * cnt) for c, cnt in zip(classes, counts)}
    return np.array([wpc[c] for c in y_indices], dtype=np.float32)

def plot_confusion_matrix_with_values(cm: np.ndarray, class_names: list, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9, fontweight="bold"
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")

# ---------- main ----------
def main():
    df = pd.read_csv(ARTI / "dataset.csv")
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    X_tr = load_embeddings(tr);  y_tr = tr["label"].to_numpy()
    X_va = load_embeddings(va);  y_va = va["label"].to_numpy()
    X_te = load_embeddings(te);  y_te = te["label"].to_numpy()

    # Label encoding
    le = LabelEncoder()
    y_tr_i = le.fit_transform(y_tr)
    y_va_i = le.transform(y_va)
    y_te_i = le.transform(y_te)
    classes_all = list(le.classes_)

    # Scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # MLP classifier (nonlinear)
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )

    # class imbalance → sample weights
    w_tr = compute_sample_weight_by_class(y_tr_i)
    clf.fit(X_tr, y_tr_i, sample_weight=w_tr)

    # Predict
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)  # [N, K_all]

    # ---- Robust AUROC (subset to present classes + renormalize) ----
    present = np.unique(y_te_i)
    if len(present) >= 2:
        proba_sub = y_proba[:, present]
        row_sums = proba_sub.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze(-1) == 0)
        if np.any(zero_rows):
            proba_sub[zero_rows, :] = 1.0 / proba_sub.shape[1]
            row_sums = proba_sub.sum(axis=1, keepdims=True)
        proba_sub /= row_sums
        y_te_mapped = np.searchsorted(present, y_te_i)

        if len(present) == 2:
            auroc = roc_auc_score(y_te_mapped, proba_sub[:, 1])
        else:
            auroc = roc_auc_score(y_te_mapped, proba_sub, multi_class="ovr", average="macro")
    else:
        auroc = float("nan")
        print("[WARN] Test set has <2 classes; AUROC undefined (NaN).")

    # Report (only present/pred classes to avoid undefined metrics)
    keep_idx = np.union1d(np.unique(y_te_i), np.unique(y_pred))
    rep = classification_report(
        y_te_i,
        y_pred,
        labels=keep_idx,
        target_names=[classes_all[i] for i in keep_idx],
        zero_division=0,
        output_dict=True,
    )

    # Confusion matrix with values
    cm = confusion_matrix(y_te_i, y_pred, labels=keep_idx)
    plot_confusion_matrix_with_values(
        cm,
        [classes_all[i] for i in keep_idx],
        ARTI / "confusion_matrix_nonlinear.png",
        "Confusion Matrix (Nonlinear Probe)",
    )

    # Save metrics
    (ARTI / "metrics_nonlinear.json").write_text(
        json.dumps(
            {
                "auroc_macro_ovr": float(auroc),
                "classes": classes_all,
                "present_labels": [classes_all[i] for i in present],
                "report": rep,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[RESULT] Nonlinear AUROC(macro-ovr): {auroc}")
    print(f"[SAVED] {ARTI / 'metrics_nonlinear.json'}")

if __name__ == "__main__":
    main()

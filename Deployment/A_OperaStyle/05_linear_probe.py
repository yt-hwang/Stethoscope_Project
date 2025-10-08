# -*- coding: utf-8 -*-
"""
Linear Probe (Logistic Regression) with robust AUROC computation
- Train: train+val
- Test: handle missing classes in test by subsetting & renormalizing proba
- Outputs: artifacts/metrics.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ---- project config ----
from config import ARTI, FEATS  # A_OperaStyle: FEATS 는 feats 디렉토리, ARTI 는 artifacts 디렉토리

def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Load per-id .npy embeddings from FEATS directory."""
    X = []
    missing = []
    for _, r in df.iterrows():
        p = FEATS / f"{r['id']}.npy"
        if not p.exists():
            missing.append(r["id"])
            continue
        X.append(np.load(p))
    if missing:
        print(f"[WARN] Missing embeddings for {len(missing)} samples, e.g., {missing[:5]}")
    return np.stack(X, axis=0)

def main():
    ARTI.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ARTI / "dataset.csv")

    # splits
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    # load features
    X_tr = load_embeddings(tr)
    X_va = load_embeddings(va)
    X_te = load_embeddings(te)

    y_tr = tr["label"].to_numpy()
    y_va = va["label"].to_numpy()
    y_te = te["label"].to_numpy()

    # encode labels
    le = LabelEncoder()
    y_tr_i = le.fit_transform(y_tr)
    y_va_i = le.transform(y_va)
    y_te_i = le.transform(y_te)
    classes_all = list(le.classes_)

    # model: scaler + logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=4, class_weight="balanced"))
    ])

    # fit on train+val
    X_train_full = np.vstack([X_tr, X_va])
    y_train_full = np.hstack([y_tr_i, y_va_i])
    pipe.fit(X_train_full, y_train_full)

    # predictions
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)   # shape [N, K_all], K_all = len(classes_all)

    # ---- Robust AUROC computation ----
    # Subset to classes that actually appear in test labels, then renormalize probas
    unique_test_classes = np.unique(y_te_i)  # e.g., [0, 2, 3]
    if len(unique_test_classes) >= 2:
        proba_sub = y_proba[:, unique_test_classes]                   # keep only present classes
        # renormalize row-wise so they sum to 1.0 (avoid division by zero)
        row_sums = proba_sub.sum(axis=1, keepdims=True)
        # if any row_sums == 0 (degenerate), fallback to uniform over present classes
        zero_rows = (row_sums.squeeze(-1) == 0)
        if np.any(zero_rows):
            proba_sub[zero_rows, :] = 1.0 / proba_sub.shape[1]
            row_sums = proba_sub.sum(axis=1, keepdims=True)
        proba_sub = proba_sub / row_sums

        # remap y_true indices from present class-ids to [0..C'-1]
        # since unique_test_classes is sorted, searchsorted gives correct mapping
        y_te_mapped = np.searchsorted(unique_test_classes, y_te_i)

        if len(unique_test_classes) == 2:
            # binary AUROC
            auroc = roc_auc_score(y_te_mapped, proba_sub[:, 1])
        else:
            # multiclass AUROC (macro-OVR)
            auroc = roc_auc_score(
                y_te_mapped, proba_sub, multi_class="ovr", average="macro"
            )
    else:
        # not enough classes in test to define AUROC
        auroc = float("nan")
        print("[WARN] Test set has < 2 classes; AUROC undefined (NaN).")

    # classification report (use full class list for consistent output)
    # 기존:
    # rep = classification_report(y_te_i, y_pred, target_names=classes_all, output_dict=True)

    # 교체:
    present_idx = np.union1d(np.unique(y_te_i), np.unique(y_pred))
    rep = classification_report(
        y_te_i,
        y_pred,
        labels=present_idx,
        target_names=[classes_all[i] for i in present_idx],
        zero_division=0,
        output_dict=True,
    )

    # ---- Confusion Matrix with values ----
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    present_idx = np.union1d(np.unique(y_te_i), np.unique(y_pred))
    cm = confusion_matrix(y_te_i, y_pred, labels=present_idx)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Confusion Matrix (Linear Probe)")
    fig.colorbar(im)

    # tick labels
    classes_present = [classes_all[i] for i in present_idx]
    ax.set_xticks(np.arange(len(classes_present)))
    ax.set_yticks(np.arange(len(classes_present)))
    ax.set_xticklabels(classes_present, rotation=45, ha="right")
    ax.set_yticklabels(classes_present)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 숫자 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9, fontweight="bold"
            )

    fig.tight_layout()
    fig.savefig(ARTI / "confusion_matrix_linear.png", dpi=150)
    plt.close(fig)

    print(f"[SAVED] Confusion matrix → {ARTI / 'confusion_matrix_linear.png'}")

    # (optional) confusion matrix — uncomment to save
    # cm = confusion_matrix(y_te_i, y_pred)
    # (ARTI / "confusion_matrix.png").write_text(json.dumps(cm.tolist(), indent=2), encoding="utf-8")

    # save metrics
    (ARTI / "metrics.json").write_text(
        json.dumps(
            {
                "auroc": float(auroc),
                "classes": classes_all,
                "classification_report": rep
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("[RESULT] AUROC:", auroc, "→", ARTI / "metrics.json")

if __name__ == "__main__":
    main()

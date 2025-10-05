#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step16B (robust):
- Evaluate MLP (+ per-class τ tuning) on OPERA features re-extracted from JSON segments.
- If metadata lacks 'diagnosis' (and/or 'group'), restore them from JSON using segment id.
- Accepts either 'path' or 'filepath' column names.
- Supports 5-class by default (Crackle/Healthy/Non-breathing/Rhonchi/Wheezing) or binary mode.

Author: GPT
"""

import os, json, math, random, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================
# [A] Fixed paths & mode
# ================================
PROJECT_DIR   = Path("//home//un_wang//my_stethoscope_project")  # 요청 반영: 고정 경로 표기
# A환경(OPERA)에서 복사해 온 결과물
FEATURES_CSV  = PROJECT_DIR / "features/Segments (Step 16)/opera_features.csv"     # 필수
META_FROM_A   = PROJECT_DIR / "data/audio/Segments_from_JSON/metadata.csv"         # 선택(있으면 조인)
JSON_PATH     = PROJECT_DIR / "data/audio/Raw/breathing_nonbreathing_intervals.json"  # 진단 복원 원본
RESULTS_DIR   = PROJECT_DIR / "results/Step16B_Eval"

# 작업 모드: "multi5" 또는 "binary"
TASK_MODE     = "multi5"

FIVE_CLASSES  = ["Crackle", "Healthy", "Non-breathing", "Rhonchi", "Wheezing"]

# ================================
# [B] Utils
# ================================
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_device(x): return x.cuda() if torch.cuda.is_available() else x

def macro_recall(pred: np.ndarray, true: np.ndarray, n_classes: int) -> float:
    recs = []
    for c in range(n_classes):
        tp = ((pred == c) & (true == c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        recs.append(tp / (tp + fn + 1e-9))
    return float(np.mean(recs))

def per_class_tau_search(probs: np.ndarray, y_true: np.ndarray, n_classes: int, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    taus = np.array([0.5] * n_classes, dtype=float)
    best_macro = -1.0
    improved = True
    while improved:
        improved = False
        for c in range(n_classes):
            best_tau_c, best_local = taus[c], best_macro
            for t in grid:
                th = taus.copy()
                th[c] = t
                score = probs / (th[None, :] + 1e-9)
                y_hat = np.argmax(score, axis=1)
                mrec = macro_recall(y_hat, y_true, n_classes)
                if mrec > best_local:
                    best_local = mrec
                    best_tau_c = t
            if best_local > best_macro + 1e-12:
                taus[c] = best_tau_c
                best_macro = best_local
                improved = True
    return taus, best_macro

# ================================
# [C] Model
# ================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, pdrop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

def train_one_fold(Xtr, ytr, Xva, yva, in_dim, out_dim, class_weights=None, seed=1337):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim, hidden=512, out_dim=out_dim, pdrop=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    cw = None
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)

    Xtr, ytr = to_device(Xtr), to_device(ytr)
    Xva, yva = to_device(Xva), to_device(yva)

    best_val, best_state = 1e9, None
    patience, bad = 12, 0

    for ep in range(1, 200 + 1):
        model.train()
        idx = torch.randperm(Xtr.size(0), device=Xtr.device)
        logits = model(Xtr[idx])
        loss   = F.cross_entropy(logits, ytr[idx], weight=cw)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(Xva), yva, weight=cw).item()

        if val_loss < best_val - 1e-6:
            best_val, best_state = val_loss, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        P = F.softmax(model(Xva), dim=1).detach().cpu().numpy()
    return P

# ================================
# [D] Load & repair metadata
# ================================
def load_json_map(json_path: Path):
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        J = json.load(f)
    # 원본 파일명(stem 포함) → diagnosis
    diag_map = {}
    for fname, info in J.items():
        base = Path(fname).stem  # ex) ABCD_001
        diag = info.get("diagnosis", None)
        if diag is not None:
            diag_map[base] = diag
    return diag_map

def infer_base_id_from_segment_id(seg_id: str):
    # 생성 규칙: f"{base_id}_{label}_{i:03d}"
    # 끝의 _{i:03d} 제거 + 마지막 '_' 앞의 label 제거
    # 예) "AB12_001_breathing_007" -> "AB12_001"
    parts = seg_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return seg_id  # 안전망

def repair_meta(df_feat: pd.DataFrame, meta: pd.DataFrame, json_path: Path) -> pd.DataFrame:
    df = df_feat.copy()

    # id 필수
    assert "id" in df.columns, "features CSV must contain 'id' column."

    # path/filepath 정규화: 둘 중 하나만 있으면 'path'로 통일
    if "path" not in df.columns and "filepath" in df.columns:
        df["path"] = df["filepath"]

    if meta is not None:
        # 메타 조인할 수 있는 컬럼만 사용
        cols = [c for c in ["id","diagnosis","group","label","path","filepath"] if c in meta.columns]
        meta_slim = meta[cols].drop_duplicates("id")
        df = df.merge(meta_slim, on="id", how="left", suffixes=("", "_meta"))

        # path 보강
        if "path" not in df.columns:
            if "filepath" in df.columns:
                df["path"] = df["filepath"]
            elif "path_meta" in df.columns:
                df["path"] = df["path_meta"]

    # diagnosis/group 누락 시 JSON으로 복원
    need_diag = "diagnosis" not in df.columns or df["diagnosis"].isna().any()
    need_group = "group" not in df.columns or df["group"].isna().any()

    if need_diag or need_group:
        diag_map = load_json_map(json_path)
        if need_diag and "diagnosis" not in df.columns:
            df["diagnosis"] = np.nan
        if need_group and "group" not in df.columns:
            df["group"] = np.nan

        # id→base_id→JSON lookup
        base_ids = df["id"].astype(str).map(infer_base_id_from_segment_id)
        if need_diag:
            df["diagnosis"] = df["diagnosis"].fillna(base_ids.map(diag_map))
        if need_group:
            # group 규칙: base_id의 첫 토큰(예: 'AB12_001' -> 'AB12')
            df["group"] = df["group"].fillna(base_ids.map(lambda s: s.split("_")[0] if isinstance(s,str) and "_" in s else s))

    # label 누락이면 파일명에서 복원 시도 (…_breathing_007.wav 형태)
    if "label" not in df.columns or df["label"].isna().any():
        if "path" in df.columns:
            df["label"] = df.get("label", pd.Series([np.nan]*len(df)))
            df["label"] = df["label"].fillna(
                df["path"].astype(str).map(lambda p: re.search(r"_(breathing|nonbreathing)_\d{3}\.wav$", p))
            )
            df["label"] = df["label"].map(lambda m: m.group(1) if m else np.nan)
        # 그래도 없으면 id에서 유추
        if df["label"].isna().any():
            df["label"] = df["label"].fillna(
                df["id"].astype(str).map(lambda s: "breathing" if s.endswith("_breathing") else ("nonbreathing" if s.endswith("_nonbreathing") else np.nan))
            )

    return df

# ================================
# [E] Main
# ================================
def main():
    print(f"[RUN] Step16B | mode={TASK_MODE}")
    assert FEATURES_CSV.exists(), f"Missing features csv: {FEATURES_CSV}"
    safe_mkdir(RESULTS_DIR)

    df_feat = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    assert len(feat_cols) > 0, "No feat_* columns found in features CSV."

    meta = pd.read_csv(META_FROM_A) if META_FROM_A.exists() else None
    df = repair_meta(df_feat, meta, JSON_PATH)

    # ===== 라벨 공간 정의 =====
    if TASK_MODE == "binary":
        assert "label" in df.columns, "Binary mode requires 'label' (breathing/nonbreathing)."
        df = df[df["label"].isin(["breathing","nonbreathing"])].copy()
        label_names = ["breathing","nonbreathing"]
        y_vec = df["label"].map({n:i for i,n in enumerate(label_names)}).values
    else:
        assert "diagnosis" in df.columns, "multi5 mode requires 'diagnosis'. (Recovered from JSON if missing)"
        df = df[df["diagnosis"].isin(FIVE_CLASSES)].copy()
        label_names = FIVE_CLASSES
        map_to_idx = {n:i for i,n in enumerate(label_names)}
        y_vec = df["diagnosis"].map(map_to_idx).values

    # ===== 그룹 =====
    assert "group" in df.columns, "group is required for GroupKFold. (Recovered from JSON/base_id)"
    groups = df["group"].astype(str).values

    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(y_vec, dtype=torch.long)
    n_classes = len(label_names)

    print(f"[INFO] n={len(df)}, classes={label_names}, dims={X.shape[1]}")
    if TASK_MODE == "multi5":
        print(df["diagnosis"].value_counts(dropna=False))
    else:
        print(df["label"].value_counts(dropna=False))

    # class weights (inverse freq)
    counts = np.bincount(y_vec, minlength=n_classes).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    class_weights = inv / inv.sum() * n_classes

    # ===== CV + τ 튜닝 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gkf = GroupKFold(n_splits=5)
    all_probs, all_true = [], []

    fold = 0
    for tr_idx, va_idx in gkf.split(X, y, groups):
        fold += 1
        print(f"\n[FOLD {fold}] train={len(tr_idx)} val={len(va_idx)}")
        P_val = train_one_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                               in_dim=X.shape[1], out_dim=n_classes,
                               class_weights=class_weights, seed=1337+fold)

        y_true = y[va_idx].cpu().numpy()
        y_base = np.argmax(P_val, axis=1)
        base_macro = macro_recall(y_base, y_true, n_classes)
        print(f"[BASE] macro_recall={base_macro:.3f}")

        taus, tuned_macro = per_class_tau_search(P_val, y_true, n_classes)
        print(f"[TUNE] macro={tuned_macro:.3f} | taus={np.round(taus,2)}")

        score = P_val / (taus[None, :] + 1e-9)
        y_hat = np.argmax(score, axis=1)
        all_probs.append(P_val); all_true.append(y_true)

        print(classification_report(y_true, y_hat, target_names=label_names, digits=2))

    all_probs = np.concatenate(all_probs, axis=0)
    all_true  = np.concatenate(all_true, axis=0)

    y_pred = np.argmax(all_probs, axis=1)
    print("\n--- Aggregate Classification Report (argmax, no global τ) ---")
    print(classification_report(all_true, y_pred, target_names=label_names, digits=2))
    print(f"[MACRO RECALL] {macro_recall(y_pred, all_true, n_classes):.3f}")

    safe_mkdir(RESULTS_DIR)
    out_csv = RESULTS_DIR / f"Step16B_{TASK_MODE}_preds.csv"
    pd.DataFrame({"true": all_true, "pred": y_pred}).to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    print("[DONE]")

if __name__ == "__main__":
    main()

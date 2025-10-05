#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step16A (JSON-direct): Audio segmentation from breathing_nonbreathing_intervals.json
→ (optional) augmentation → OPERA feature re-extract → per-class τ tuning

JSON만 있으면 되고, metadata.csv는 필요 없음.

Usage:
  python //home//un_wang//my_stethoscope_project/scripts/GPT/step16A_audio_from_json_reextract_then_tau.py \
    --project_dir "//home//un_wang//my_stethoscope_project" \
    --interval_json "<ABS_OR_REL_PATH_TO>/breathing_nonbreathing_intervals.json" \
    --audio_root "<원본 wav들이 있는 최상위 폴더>" \
    --out_wav_dir "//home//un_wang//my_stethoscope_project/data/segments_step16A" \
    --do_aug false \
    --extract_cmd "python //home//un_wang//my_stethoscope_project/scripts/opera/extract_opera.py --input_dir {in_dir} --output_csv {out_csv}" \
    --features_csv "//home//un_wang//my_stethoscope_project/features/opera_features_step16A.csv"

설명:
- audio_root 아래에서 JSON 키(원본 wav 파일명)와 같은 파일을 재귀적으로 찾아 세그먼트를 만듭니다.
- group은 파일명에서 첫 '_' 이전 토큰(예: H001_xxx.wav -> H001)으로 추정합니다. 필요하면 --group_from_regex로 변경.
- 증강(do_aug=true) 시 pitch/noise/stretch를 가볍게 수행(기본 off).
- OPERA 임베딩 추출은 --extract_cmd로 외부 추출기를 호출(당신이 쓰던 추출 스크립트를 그대로 연결). {in_dir}, {out_csv} 플레이스홀더를 각각 실제 경로로 치환해서 실행합니다.
- features_csv가 만들어지면, 간단한 MLP+per-class τ 튜닝으로 macro recall을 출력합니다(스텝7/10 계열과 동일한 목표).

작성: 당신의 실험 로그 포맷을 그대로 따르며, PROJECT_DIR 기본값은 요구사항대로 설정.
"""

import os, sys, json, glob, re, argparse, subprocess, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

# ---------------------------
# Defaults (요청한 프로젝트 경로)
# ---------------------------
DEFAULT_PROJECT_DIR = "//home//un_wang//my_stethoscope_project"

# ---------------------------
# Audio helpers
# ---------------------------
def find_audio_path(audio_root, filename):
    """audio_root 아래에서 filename과 동일한 basename을 가진 파일을 재귀 검색"""
    cand = list(Path(audio_root).rglob(filename))
    return str(cand[0]) if len(cand) > 0 else None

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def segment_and_save(wav_path, intervals, label, out_dir, sr_target=None, base_id=""):
    """
    intervals: [[start_sec, end_sec], ...]
    저장 파일명: {base_id}_{label}_{idx:03d}.wav
    """
    if wav_path is None or not os.path.exists(wav_path):
        return []

    audio, sr = sf.read(wav_path)  # shape: [T] or [T, C]
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    out_paths = []
    for idx, (s, e) in enumerate(intervals):
        s = max(0.0, float(s))
        e = max(s, float(e))
        s_i = int(round(s * sr))
        e_i = int(round(e * sr))
        seg = audio[s_i:e_i]
        if seg.size < sr * 0.1:  # 100ms 미만이면 스킵
            continue
        out_name = f"{base_id}_{label}_{idx:03d}.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, seg, sr)
        out_paths.append(out_path)
    return out_paths

def augment_basic(in_path, out_dir, aug_cfg):
    """
    아주 가벼운 증강들: pitch(±1 semitone 근처), gain noise(SNR), time-stretch(±5%)
    파형 레벨에서 numpy로 간단 처리(고급 품질은 과하지 않게).
    """
    audio, sr = sf.read(in_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    stems = []
    base = Path(in_path).stem

    # noise
    if aug_cfg.get("noise", False):
        snr_db = aug_cfg.get("noise_snr_db", 15.0)
        sig_pow = np.mean(audio**2) + 1e-9
        noise_pow = sig_pow / (10**(snr_db/10))
        noise = np.random.randn(len(audio)) * math.sqrt(noise_pow)
        noisy = audio + noise
        outp = os.path.join(out_dir, base + "_augNoise.wav")
        sf.write(outp, noisy, sr)
        stems.append(outp)

    # time-stretch (very simple resample-based; keep length approx)
    if aug_cfg.get("stretch", False):
        rate = 1.0 + random.uniform(-0.05, 0.05)
        idx = np.arange(0, len(audio), rate)
        stretched = np.interp(idx, np.arange(len(audio)), audio)
        outp = os.path.join(out_dir, base + "_augStretch.wav")
        sf.write(outp, stretched, sr)
        stems.append(outp)

    # pitch shift (cheap resample trick: not formant-preserving, but ok as light aug)
    if aug_cfg.get("pitch", False):
        cents = random.uniform(-100, 100)  # ±1 semitone
        ratio = 2 ** (cents / 1200.0)
        idx = np.arange(0, len(audio), 1/ratio)
        pitched = np.interp(idx, np.arange(len(audio)), audio)
        # match length
        if len(pitched) > len(audio):
            pitched = pitched[:len(audio)]
        else:
            pad = np.zeros(len(audio)-len(pitched))
            pitched = np.concatenate([pitched, pad])
        outp = os.path.join(out_dir, base + "_augPitch.wav")
        sf.write(outp, pitched, sr)
        stems.append(outp)

    return stems

# ---------------------------
# Simple MLP for OPERA features
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def per_class_tau_search(probs, y_true, classes, guards=None):
    """
    각 클래스 별 τ 그리드 탐색 (0.05~0.95 step=0.05).
    guards: {class_idx: min_recall} 같은 형태(옵션)
    return best_macro, best_taus
    """
    n_cls = len(classes)
    taus = np.full(n_cls, 0.5, dtype=np.float32)
    best_macro = -1.0
    best_taus = taus.copy()

    grid = np.linspace(0.05, 0.95, 19)
    for _ in range(3):  # 좌표 상승 3 라운드
        improved = False
        for c in range(n_cls):
            cur = best_taus.copy()
            loc_best = best_macro
            loc_tau = cur[c]
            for t in grid:
                cur[c] = t
                # predict
                pred = []
                for i in range(len(y_true)):
                    pc = probs[i]
                    # thresholding: winner-takes-all but require pc[c] >= tau[c]
                    # if none meet threshold → take argmax anyway (fail-safe)
                    meets = (pc >= cur).astype(np.int32)
                    if meets.sum() == 0:
                        yhat = int(np.argmax(pc))
                    else:
                        # among meets, highest prob
                        idxs = np.where(meets)[0]
                        yhat = int(idxs[np.argmax(pc[idxs])])
                    pred.append(yhat)
                pred = np.array(pred)
                macro = macro_recall(pred, y_true, n_cls)

                # guard check
                if guards:
                    ok = True
                    rep = report_counts(pred, y_true, n_cls)
                    for gk, gmin in guards.items():
                        if rep["recall"][gk] < gmin:
                            ok = False
                            break
                    if not ok:
                        continue

                if macro > loc_best:
                    loc_best = macro
                    loc_tau = t
            if loc_best > best_macro:
                improved = True
                best_macro = loc_best
                best_taus[c] = loc_tau
        if not improved:
            break
    return best_macro, best_taus

def macro_recall(y_pred, y_true, n_cls):
    recs = []
    for c in range(n_cls):
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        recs.append(tp / (tp+fn+1e-9))
    return float(np.mean(recs))

def report_counts(y_pred, y_true, n_cls):
    rec = []
    for c in range(n_cls):
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        rec.append(tp/(tp+fn+1e-9))
    return {"recall": rec}

# ---------------------------
# Main
# ---------------------------
def infer_group_from_name(name, regex=None):
    """
    기본: 파일명의 첫 '_' 이전 토큰을 group으로 사용 (예: H001_xxx.wav -> H001).
    regex를 주면 첫 캡쳐 그룹을 group으로 사용.
    """
    stem = Path(name).stem
    if regex:
        m = re.search(regex, stem)
        if m:
            return m.group(1)
    if "_" in stem:
        return stem.split("_")[0]
    return stem  # fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=DEFAULT_PROJECT_DIR, type=str)
    ap.add_argument("--interval_json", required=True, type=str, help="breathing_nonbreathing_intervals.json 경로")
    ap.add_argument("--audio_root", required=True, type=str, help="원본 wav들이 있는 최상위 폴더")
    ap.add_argument("--out_wav_dir", default=None, type=str, help="세그먼트/증강 wav 저장 폴더")
    ap.add_argument("--do_aug", default="false", type=str, help="true/false")
    ap.add_argument("--features_csv", default=None, type=str, help="OPERA 임베딩 csv 저장 경로")
    ap.add_argument("--extract_cmd", default="", type=str,
                    help="외부 OPERA 추출 커맨드. {in_dir}, {out_csv} 플레이스홀더 사용")
    ap.add_argument("--group_from_regex", default=None, type=str, help="그룹 추출용 정규식(옵션)")
    ap.add_argument("--kfold", default=5, type=int)

    # aug config (light)
    ap.add_argument("--aug_noise", default="false", type=str)
    ap.add_argument("--aug_stretch", default="false", type=str)
    ap.add_argument("--aug_pitch", default="false", type=str)
    ap.add_argument("--aug_noise_snr_db", default=15.0, type=float)

    args = ap.parse_args()
    PROJECT_DIR = args.project_dir
    print(f"[RUN] Step16A (JSON-direct)")
    print(f"[PROJECT_DIR] {PROJECT_DIR}")

    # dirs
    out_wav_dir = args.out_wav_dir or os.path.join(PROJECT_DIR, "data/segments_step16A")
    safe_mkdir(out_wav_dir)
    seg_orig_dir = os.path.join(out_wav_dir, "orig")
    seg_aug_dir  = os.path.join(out_wav_dir, "aug")
    safe_mkdir(seg_orig_dir)
    safe_mkdir(seg_aug_dir)

    # load JSON
    assert os.path.exists(args.interval_json), "interval_json 경로가 유효하지 않습니다."
    with open(args.interval_json, "r", encoding="utf-8") as f:
        interval_map = json.load(f)

    # 1) segment
    print("[STEP] segment from JSON …")
    rows = []
    for fname, info in interval_map.items():
        wav_path = find_audio_path(args.audio_root, fname)
        if wav_path is None:
            print(f"[WARN] not found under audio_root: {fname}")
            continue

        base_id = Path(fname).stem
        group = infer_group_from_name(fname, args.group_from_regex)

        # breathing
        br_list = info.get("breathing", []) or []
        nb_list = info.get("nonbreathing", []) or []

        br_outs = segment_and_save(wav_path, br_list, "breathing", seg_orig_dir, base_id=base_id)
        for p in br_outs:
            rows.append({"id": Path(p).stem, "path": p, "label": "breathing", "group": group})

        nb_outs = segment_and_save(wav_path, nb_list, "nonbreathing", seg_orig_dir, base_id=base_id)
        for p in nb_outs:
            rows.append({"id": Path(p).stem, "path": p, "label": "nonbreathing", "group": group})

    meta_df = pd.DataFrame(rows)
    print(f"[SEGMENT] total segments: {len(meta_df)} (breathing={sum(meta_df.label=='breathing')}, nonbreathing={sum(meta_df.label=='nonbreathing')})")

    # 2) augment (optional)
    do_aug = (args.do_aug.lower() == "true")
    if do_aug:
        print("[STEP] augmentation …")
        aug_cfg = {
            "noise": (args.aug_noise.lower() == "true"),
            "stretch": (args.aug_stretch.lower() == "true"),
            "pitch": (args.aug_pitch.lower() == "true"),
            "noise_snr_db": args.aug_noise_snr_db
        }
        aug_rows = []
        for _, r in meta_df.iterrows():
            outs = augment_basic(r["path"], seg_aug_dir, aug_cfg)
            for op in outs:
                aug_rows.append({"id": Path(op).stem, "path": op, "label": r["label"], "group": r["group"]})
        if aug_rows:
            meta_df = pd.concat([meta_df, pd.DataFrame(aug_rows)], ignore_index=True)
        print(f"[AUG] after augmentation: {len(meta_df)} segments")

    # 저장 (참고용)
    meta_csv_path = os.path.join(PROJECT_DIR, "metadata", "metadata_from_json_step16A.csv")
    safe_mkdir(os.path.dirname(meta_csv_path))
    meta_df.to_csv(meta_csv_path, index=False, encoding="utf-8")
    print(f"[META] saved: {meta_csv_path}")

    # 3) OPERA feature extraction
    features_csv = args.features_csv or os.path.join(PROJECT_DIR, "features", "opera_features_step16A.csv")
    safe_mkdir(os.path.dirname(features_csv))
    if args.extract_cmd.strip():
        cmd = args.extract_cmd.format(in_dir=out_wav_dir, out_csv=features_csv)
        print(f"[EXTRACT] run: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print("[ERROR] feature extractor failed. Stop here.")
            sys.exit(1)
    else:
        if not os.path.exists(features_csv):
            print("[WARN] --extract_cmd 미지정 & features_csv 존재하지 않음 → 여기서 종료합니다.")
            print("      (당신이 쓰던 OPERA 추출기를 --extract_cmd로 연결해 주세요.)")
            sys.exit(0)

    # 4) Train (MLP) + per-class τ tuning (breathing vs nonbreathing)
    print("[STEP] Load features & run MLP + per-class τ")
    df = pd.read_csv(features_csv)
    # 기대 컬럼 예: id, group, label, feat_0 ... feat_767
    # 유효성 체크
    req_cols = {"id", "group", "label"}
    assert req_cols.issubset(set(df.columns)), f"features_csv에 {req_cols} 컬럼이 필요합니다."
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert len(feat_cols) > 0, "features_csv에 feat_* 컬럼이 없습니다."

    # breathing/nonbreathing → 나중 단계에서 H/W/R까지 확장할 수도 있지만,
    # 이 스텝은 재추출 검증이 목적이므로 2-class로 먼저 확인.
    label_map = {"breathing": 0, "nonbreathing": 1}
    keep = df["label"].isin(label_map.keys())
    df = df[keep].reset_index(drop=True)

    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor([label_map[s] for s in df["label"].values], dtype=torch.long)
    groups = df["group"].values
    classes = ["breathing", "nonbreathing"]
    n_cls = len(classes)

    # 모델/옵티마
    in_dim = X.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_one_fold(X_tr, y_tr, X_va, y_va):
        model = MLP(in_dim, hidden=512, num_classes=n_cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best = 1e9; best_state=None
        for ep in range(1, 51):
            model.train()
            idx = torch.randperm(len(X_tr))
            xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

            # val
            model.eval()
            with torch.no_grad():
                lv = F.cross_entropy(model(X_va.to(device)), y_va.to(device)).item()
            if lv < best:
                best = lv; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            P = F.softmax(model(X_va.to(device)), dim=1).cpu().numpy()
        return P

    kf = GroupKFold(n_splits=args.kfold)
    all_probs = []
    all_true  = []
    print(f"[INFO] kfold={args.kfold}, device={device}")
    for fold, (tr, va) in enumerate(kf.split(X, y, groups), 1):
        print(f"[FOLD {fold}] train={len(tr)} val={len(va)}")
        P = train_one_fold(X[tr], y[tr], X[va], y[va])
        all_probs.append(P); all_true.append(y[va].numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_true  = np.concatenate(all_true, axis=0)

    # per-class τ (fail-safe: argmax if none meet)
    base_pred = np.argmax(all_probs, axis=1)
    base_macro = macro_recall(base_pred, all_true, n_cls)
    print(f"[BASE macro recall] {base_macro:.3f}")

    best_macro, best_taus = per_class_tau_search(all_probs, all_true, classes, guards=None)
    # 최종 예측
    yhat = []
    for i in range(len(all_true)):
        pc = all_probs[i]
        meets = (pc >= best_taus).astype(np.int32)
        if meets.sum()==0:
            yhat.append(int(np.argmax(pc)))
        else:
            idxs = np.where(meets)[0]
            yhat.append(int(idxs[np.argmax(pc[idxs])]))
    yhat = np.array(yhat)

    print("\n--- Classification Report (2-class) ---")
    print(classification_report(all_true, yhat, target_names=classes, digits=2))
    print(f"[PER-CLASS τ] taus={np.round(best_taus, 2).tolist()}  |  [MACRO RECALL] {macro_recall(yhat, all_true, n_cls):.3f}")
    print("[DONE]")

if __name__ == "__main__":
    main()

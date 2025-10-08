#!/usr/bin/env python3
import os, json, argparse, glob, datetime
import numpy as np
import pandas as pd
import torch

def find_latest_summary_dir(result_root: str, tag_prefix: str):
    pattern = os.path.join(result_root, f"{tag_prefix}_seed*_summary")
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(f"No summary dir found by pattern: {pattern}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def load_final_model(summary_dir: str, device):
    ckpt_path = os.path.join(summary_dir, "final_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"final_model.pt not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj["state_dict"]
    input_dim  = obj["input_dim"]
    num_classes = obj["num_classes"]
    class_names = obj["class_names"]
    scaler     = obj["scaler"]
    best_tau   = obj.get("best_tau", None)  # 있을 수도/없을 수도

    # 모델 아키텍처 (학습 스크립트와 동일)
    import torch.nn as nn
    class OptimizedLinearModel(nn.Module):
        def __init__(self, input_dim, num_classes, dropout=0.15):
            super().__init__()
            self.feature_norm = nn.LayerNorm(input_dim)
            self.feature_dropout = nn.Dropout(dropout)
            self.hidden = nn.Linear(input_dim, input_dim // 2)
            self.hidden_norm = nn.LayerNorm(input_dim // 2)
            self.hidden_dropout = nn.Dropout(dropout * 0.5)
            self.classifier = nn.Linear(input_dim // 2, num_classes)
        def forward(self, x):
            x = self.feature_norm(x); x = self.feature_dropout(x)
            x = torch.relu(self.hidden(x)); x = self.hidden_norm(x); x = self.hidden_dropout(x)
            return self.classifier(x)

    model = OptimizedLinearModel(input_dim, num_classes, dropout=0.15).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, class_names, np.array(best_tau) if best_tau is not None else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv")
    ap.add_argument("--results_dir", default=r"D:\Stethoscope_Project\Deployment\Result")
    ap.add_argument("--tag", default="Step17A_Optimized_80Percent")
    ap.add_argument("--summary_dir", default=None, help="지정 시 이 폴더의 final_model.pt 사용")
    ap.add_argument("--use_tau", action="store_true", help="훈련에서 저장된 per-class tau가 있으면 적용해 argmax")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_root = args.results_dir

    # summary 디렉토리 선택
    if args.summary_dir:
        summary_dir = args.summary_dir
    else:
        summary_dir = find_latest_summary_dir(result_root, args.tag)
    print(f"[INFO] Using summary dir: {summary_dir}")

    # 모델 로드
    model, scaler, class_names, best_tau = load_final_model(summary_dir, device)
    print(f"[INFO] best_tau available: {best_tau is not None}")

    # 데이터 로드 & 전처리
    df = pd.read_csv(args.csv)
    drop_cols = [c for c in ['filename', 'label', 'extraction_success'] if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    Xs = scaler.transform(X)
    batch = torch.tensor(Xs, dtype=torch.float32).to(device)

    # 예측
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    if args.use_tau and best_tau is not None:
        q = probs / best_tau.reshape(1, -1)
        pred_idx = q.argmax(1)
        used_probs = q
    else:
        pred_idx = probs.argmax(1)
        used_probs = probs

    pred_labels = [class_names[i] for i in pred_idx]

    # 결과 저장
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(result_root, f"predictions_{args.tag}_{ts}.csv")
    out_df = pd.DataFrame({
        "pred_label": pred_labels,
        **{f"prob_{cls}": used_probs[:, i] for i, cls in enumerate(class_names)}
    })
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved predictions → {out_csv}")

if __name__ == "__main__":
    main()

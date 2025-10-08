#!/usr/bin/env python3
import os, json, argparse, glob, shutil, numpy as np, pandas as pd

def export_from_one_seed(seed_summary_csv: str):
    summary_csv = seed_summary_csv
    rows = pd.read_csv(summary_csv)
    tag = os.path.basename(summary_csv).replace("_summary.csv","")
    result_dir = os.path.dirname(summary_csv)
    root = os.path.dirname(result_dir)

    # champion fold 찾기
    best_idx = rows['macro_recall_tau'].idxmax()
    champ_fold = int(rows.loc[best_idx, 'fold'])
    champ_metric = float(rows.loc[best_idx, 'macro_recall_tau'])
    seed = rows.loc[best_idx, 'seed']
    tag_seed = f"{tag}"

    fold_dir = os.path.join(root, f"{tag_seed}_fold{champ_fold}")
    summary_dir = os.path.join(root, f"{tag_seed}_summary")
    os.makedirs(summary_dir, exist_ok=True)

    src = os.path.join(fold_dir, "model_best.pt")
    dst = os.path.join(summary_dir, "final_model.pt")
    shutil.copyfile(src, dst)

    # 평균 tau 계산(있으면)
    taus = []
    for f in range(1, 6):
        p = os.path.join(root, f"{tag_seed}_fold{f}", "taus.npy")
        if os.path.exists(p): taus.append(np.load(p))
    avg_tau = np.mean(np.stack(taus, axis=0), axis=0).tolist() if len(taus)==5 else None

    info = {
        "strategy": "champion_fold",
        "champion_fold": champ_fold,
        "champion_metric_macro_recall_tau": champ_metric,
        "final_model_path": dst,
        "avg_tau_over_folds": avg_tau
    }
    with open(os.path.join(summary_dir, "final_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[OK] {dst} (champion fold: {champ_fold}, MR_tau={champ_metric:.3f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True,
        help=r"예) D:\Stethoscope_Project\Deployment\Result\Step17A_Optimized_80Percent_seed268400_summary.csv")
    args = ap.parse_args()
    export_from_one_seed(args.summary_csv)

if __name__ == "__main__":
    main()

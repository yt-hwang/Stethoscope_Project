#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step17A Optimized 80% — Clean Final Version
Fixed all indentation & classification summary aggregation
"""

import os, json, time, math, random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# ======================================================
# Helper + Accumulator
# ======================================================
reports_all = []

def _report_df(y_true, y_pred, class_names, fold_idx, version_tag):
    rep = classification_report(
        y_true, y_pred, target_names=class_names, digits=4,
        output_dict=True, zero_division=0
    )
    df = pd.DataFrame(rep).T.reset_index().rename(columns={'index': 'label'})
    df['fold'] = int(fold_idx)
    df['version'] = str(version_tag)
    for col in ['precision', 'recall', 'f1-score', 'support']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df


# ======================================================
# (중략) — 기존 main() 코드 전부 그대로 유지
# ======================================================
# 👇 아래 부분만 교체 (classification report 저장 구간부터)


        # ==========================================================
        # Save classification report (Fixed indentation)
        # ==========================================================
        print("  📋 Generating classification reports...")

        # 먼저 fold-level report 누적
        try:
            reports_all.append(_report_df(y_true, y_pred_raw, class_names, fold_idx=fold, version_tag="RAW"))
            reports_all.append(_report_df(y_true, y_pred_tau, class_names, fold_idx=fold, version_tag="TAU"))
        except Exception as e:
            print(f"[WARN] failed to append report for fold {fold}: {e}")

        # RAW report 파일
        with open(os.path.join(fold_dir, "classification_report_raw.txt"), 'w') as f:
            f.write(f"CLASSIFICATION REPORT - RAW PREDICTIONS\n")
            f.write(f"Fold {fold} (OPTIMIZED Seed {actual_seed})\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_true, y_pred_raw, target_names=class_names, digits=4))

        # TAU report 파일
        with open(os.path.join(fold_dir, "classification_report_tau.txt"), 'w') as f:
            f.write(f"CLASSIFICATION REPORT - TAU ADJUSTED\n")
            f.write(f"Fold {fold} (OPTIMIZED Seed {actual_seed})\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_true, y_pred_tau, target_names=class_names, digits=4))


# ======================================================
# (중략) — 기존 summary_dir 만들고 confusion matrix 저장 이후
# ======================================================

    # === Aggregated Classification Report Summary ===
    try:
        if len(reports_all) > 0:
            rep_df = pd.concat(reports_all, ignore_index=True)

            # 저장 경로: summary_dir
            by_fold_path = os.path.join(summary_dir, "classification_report_by_fold.csv")
            rep_df.to_csv(by_fold_path, index=False, encoding="utf-8-sig")

            avg_cols = ['precision', 'recall', 'f1-score', 'support']
            summary_df = (
                rep_df
                .groupby(['version', 'label'], as_index=False)[avg_cols]
                .agg({'precision':'mean','recall':'mean','f1-score':'mean','support':'sum'})
            )
            summary_df.to_csv(os.path.join(summary_dir, "classification_report_summary.csv"),
                              index=False, encoding="utf-8-sig")

            macro_weighted_df = summary_df[
                summary_df['label'].isin(['macro avg', 'weighted avg'])
            ].sort_values(['version','label'])
            macro_weighted_df.to_csv(os.path.join(summary_dir, "classification_report_macro_weighted.csv"),
                                     index=False, encoding="utf-8-sig")

            # Best fold selection (macro F1 기준)
            macro_rows = rep_df[rep_df['label'] == 'macro avg'].copy()
            for c in ['precision', 'recall', 'f1-score']:
                macro_rows[c] = pd.to_numeric(macro_rows[c], errors='coerce').fillna(0.0)

            best_overall = macro_rows.sort_values('f1-score', ascending=False).head(1)
            best_by_version = (
                macro_rows.sort_values('f1-score', ascending=False)
                .groupby('version', as_index=False).head(1)
            )

            lines = []
            if not best_overall.empty:
                r = best_overall.iloc[0]
                lines.append(
                    f"[BEST OVERALL] fold={int(r['fold'])}, version={r['version']}, "
                    f"macroF1={r['f1-score']:.4f}, macroP={r['precision']:.4f}, macroR={r['recall']:.4f}"
                )
            for _, r in best_by_version.iterrows():
                lines.append(
                    f"[BEST by VERSION] {r['version']}: fold={int(r['fold'])}, "
                    f"macroF1={r['f1-score']:.4f}, macroP={r['precision']:.4f}, macroR={r['recall']:.4f}"
                )

            with open(os.path.join(summary_dir, "best_fold_selection.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print("✅ Aggregated classification report summary saved.")
    except Exception as e:
        print(f"[WARN] Summary aggregation failed: {e}")

# parse_breathing_intervals.py
# Input: Excel file with breathing intervals
# Output: JSON file with breathing and non-breathing intervals

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# ===== User Settings =====
# Mac
EXCEL_PATH = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx")   # 입력 Excel 파일
OUT_JSON = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/Output/breathing_nonbreathing_intervals.json")      # 출력 JSON 파일
# Windows
#EXCEL_PATH = Path("D:\\Stethoscope_Project\\Audio shared\\ML test sound list\\ML test sound list breathing info.xlsx")   # 입력 Excel 파일
#OUT_JSON = Path("D:\\Stethoscope_Project\\Deployment\\1 Final Pipeline\\2 Model Training with Replayed Sound\\Output\\breathing_nonbreathing_intervals.json")      # 출력 JSON 파일

DURATION_SEC = 30.0                                           # Total audio length (seconds)
STRICT_DROP = False                                            # If dirty data exists, remove the file itself
# ======================

def _as_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        val = float(x)
        # Handle NaN values
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: t[0])
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def _clip_interval(iv: Tuple[float, float], lo: float, hi: float) -> Tuple[float, float] | None:
    s, e = iv
    # Handle None values (from NaN in Excel)
    if s is None or e is None:
        return None
    s = max(lo, s)
    e = min(hi, e)
    if e <= s:
        return None
    return (s, e)


def parse_sheet(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    n_rows, n_cols = df.shape
    i = 0
    while i < n_rows:
        rowA = df.iloc[i]
        file_name = rowA.iloc[1]
        if pd.isna(file_name):
            i += 1
            continue
        file_name = str(file_name).strip()

        if i + 1 >= n_rows:
            i += 1
            continue
        rowB = df.iloc[i + 1]

        diagnosis = rowB.iloc[1]
        diagnosis = "" if pd.isna(diagnosis) else str(diagnosis).strip()

        vals: List[float | None] = []
        for c in range(2, n_cols):
            vals.append(_as_float(rowB.iloc[c]))

        inhale, exhale = [], []
        dirty, dirty_reasons = False, []

        for k in range(0, len(vals), 2):
            s = vals[k]
            e = vals[k + 1] if k + 1 < len(vals) else None
            if s is None and e is None:
                break
            if s is None or e is None:
                dirty, dirty_reasons = True, dirty_reasons + ["partial_phase"]
                continue
            if s < 0 or e < 0:
                dirty, dirty_reasons = True, dirty_reasons + ["negative_time"]
                continue
            if e <= s:
                dirty, dirty_reasons = True, dirty_reasons + ["invalid_order"]
                continue

            phase_is_inhale = ((k // 2) % 2 == 0)
            iv = (s, e)
            (inhale if phase_is_inhale else exhale).append(iv)

        out[file_name] = {
            "diagnosis": diagnosis,
            "inhale": inhale,
            "exhale": exhale,
            "dirty": dirty,
            "dirty_reasons": sorted(set(dirty_reasons)),
        }
        i += 2
    return out


def build_breathing_nonbreathing(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for file_name, rec in parsed.items():
        if STRICT_DROP and rec.get("dirty"):
            continue
        raw = rec.get("inhale", []) + rec.get("exhale", [])
        clipped = [_clip_interval((s, e), 0.0, DURATION_SEC) for s, e in raw]
        clipped = [iv for iv in clipped if iv is not None]
        breathing = _merge_intervals(clipped)
        
        # Skip files with no breathing data
        if not breathing:
            continue
            
        non_breathing: List[Tuple[float, float]] = []
        if breathing[0][0] > 0.0:
            non_breathing.append((0.0, breathing[0][0]))
        for a, b in zip(breathing, breathing[1:]):
            if b[0] > a[1]:
                non_breathing.append((a[1], b[0]))
        if breathing[-1][1] < DURATION_SEC:
            non_breathing.append((breathing[-1][1], DURATION_SEC))
        result[file_name] = {
            "diagnosis": rec.get("diagnosis", ""),
            "breathing": [[float(s), float(e)] for (s, e) in breathing],
            "non_breathing": [[float(s), float(e)] for (s, e) in non_breathing],
        }
    return result


def parse_excel_to_intervals(excel_path: Path) -> Dict[str, Dict[str, Any]]:
    xls = pd.ExcelFile(excel_path)
    all_parsed: Dict[str, Dict[str, Any]] = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        parsed = parse_sheet(df)
        all_parsed.update(parsed)
    return build_breathing_nonbreathing(all_parsed)


if __name__ == "__main__":
    result = parse_excel_to_intervals(EXCEL_PATH)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Done. {len(result)} files parsed.")
    for k in list(result.keys())[:3]:
        print(f"- {k}: breathing={len(result[k]['breathing'])}, non_breathing={len(result[k]['non_breathing'])}")

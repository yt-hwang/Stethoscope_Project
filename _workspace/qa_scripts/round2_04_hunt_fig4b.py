# round2_04: hunt the Fig 4B confusion matrix (diagonal ~89/85/87/94/65).
# Scan all confusion_matrix*.csv and classification_report*.txt/csv, compute per-class
# recall %, and flag any 5-class result (esp. with Non-breathing) whose diagonal recalls
# are near the target set {89,85,87,94,65} in any order.
import os, re, glob
import numpy as np
from pathlib import Path

BASE = Path("/Users/yunhwang/Desktop/Stethoscope_Project")
TARGET = sorted([89,85,87,94,65])

def near_target(vals):
    v = sorted(round(x) for x in vals if not np.isnan(x))
    if len(v)!=5: return False
    return all(abs(a-b)<=3 for a,b in zip(v, TARGET))

print("TARGET Fig4B diagonal (sorted):", TARGET)
print("="*70)

# 1) confusion_matrix*.csv files -> recall diagonal
cm_files = glob.glob(str(BASE/"**/confusion_matrix*.csv"), recursive=True)
cm_files += glob.glob(str(BASE/"**/*cm*.csv"), recursive=True)
cm_files = sorted(set(cm_files))
print(f"\n[CM CSV files: {len(cm_files)}]")
for f in cm_files:
    try:
        M = np.loadtxt(f, delimiter=",")
        if M.ndim!=2 or M.shape[0]!=M.shape[1]: continue
        rec = np.array([M[i,i]/M[i].sum()*100 if M[i].sum()>0 else np.nan for i in range(M.shape[0])])
        flag = " <<< NEAR FIG4B" if M.shape[0]==5 and near_target(rec) else ""
        if M.shape[0]==5:
            print(f"  {os.path.relpath(f,BASE)}: n={M.shape[0]} recall%={np.round(rec,1).tolist()}{flag}")
    except Exception as e:
        pass

# 2) classification_report*.txt -> parse recall column
print("\n[classification_report .txt files]")
rep_files = sorted(set(glob.glob(str(BASE/"**/classification_report*.txt"), recursive=True) +
                       glob.glob(str(BASE/"**/*classification*report*.txt"), recursive=True)))
row_re = re.compile(r"^\s*([A-Za-z\-]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$")
for f in rep_files:
    txt = open(f, errors="ignore").read()
    recalls=[]; names=[]
    for line in txt.splitlines():
        m=row_re.match(line)
        if m and m.group(1) not in ("accuracy","macro","weighted"):
            names.append(m.group(1)); recalls.append(float(m.group(3))*100)
    if len(recalls)==5:
        flag=" <<< NEAR FIG4B" if near_target(recalls) else ""
        print(f"  {os.path.relpath(f,BASE)}: {list(zip(names,[round(r) for r in recalls]))}{flag}")

# 3) classification_report*.csv
print("\n[classification_report .csv files]")
csv_reps = sorted(set(glob.glob(str(BASE/"**/classification_report*.csv"), recursive=True) +
                      glob.glob(str(BASE/"**/*class*report*.csv"), recursive=True)))
for f in csv_reps:
    try:
        import csv as _csv
        rows=list(_csv.reader(open(f,errors="ignore")))
        recalls=[]; names=[]
        for r in rows:
            if len(r)>=4 and r[0] and r[0] not in ("","accuracy","macro avg","weighted avg"):
                try:
                    rec=float(r[2]); recalls.append(rec*100 if rec<=1 else rec); names.append(r[0])
                except: pass
        if len(recalls)==5:
            flag=" <<< NEAR FIG4B" if near_target(recalls) else ""
            print(f"  {os.path.relpath(f,BASE)}: {list(zip(names,[round(x) for x in recalls]))}{flag}")
    except Exception as e:
        pass
print("\nDONE hunt.")

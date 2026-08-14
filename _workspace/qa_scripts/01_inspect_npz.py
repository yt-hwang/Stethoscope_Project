import re
import numpy as np
from pathlib import Path
from collections import Counter

NPZ = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Paper/Final_Code_Manuscript/1_training_pipeline/features_64mel.npz")

d = np.load(NPZ, allow_pickle=True)
print("keys:", list(d.keys()))
X = d["X"]; y = d["y"].astype(str); fn = d["filenames"].astype(str); classes = d["classes"].astype(str).tolist()
print("X.shape:", X.shape)
print("classes (npz order):", classes)
print("y class distribution:", dict(Counter(y.tolist())))

_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

def stem_without_time_range(stem):
    m = _TSPAN_RE.search(stem)
    return stem[:m.start()] if m else stem

def filename_to_group(f):
    return stem_without_time_range(Path(f).stem)

groups = np.array([filename_to_group(f) for f in fn])
uniq = sorted(set(groups.tolist()))
print("\n#unique groups (filename_to_group as in train script):", len(uniq))
print("sample groups (first 40):")
for g in uniq[:40]:
    print("   ", g)

# Try to extract patient prefix (letters+digits before first underscore / channel marker)
def patient_id(g):
    # common patterns: H002, P0123, etc. take leading alnum token
    m = re.match(r"([A-Za-z]+[0-9]+)", g)
    return m.group(1) if m else g

pats = sorted(set(patient_id(g) for g in uniq))
print("\n#unique 'patient-like' prefixes (leading letters+digits):", len(pats))
print(pats)

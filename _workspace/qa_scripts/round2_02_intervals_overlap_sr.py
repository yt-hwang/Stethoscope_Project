# round2_02: breath-phase duration distribution, overlap factor, RAW sample rates,
# RAW wav count vs used, excluded set.
import json, re
import numpy as np
from pathlib import Path
import soundfile as sf

BASE = Path("/Users/yunhwang/Desktop/Stethoscope_Project")
INTERVALS = BASE/"Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/Output/breathing_nonbreathing_intervals.json"
RAW = BASE/"Audio shared/ML test sound list/RAW sound_ML test sound list"

print("="*70)
print("OVERLAP FACTOR: 2s window / 0.5s hop")
print("="*70)
WIN, HOP = 2.0, 0.5
print(f"overlap factor = WIN/HOP = {WIN}/{HOP} = {WIN/HOP:.1f}x")
print("A point in time is covered by up to WIN/HOP windows -> literature 'Nx overlap'.")
print("Note training/export use 0.5s hop; deployed app uses 1.0s hop -> app overlap = 2.0/1.0 = 2x")

print("\n"+"="*70)
print("BREATH-PHASE DURATION DISTRIBUTION (from intervals JSON 'breathing')")
print("="*70)
d = json.load(open(INTERVALS))
durs = []
per_rec = {}
for rec, obj in d.items():
    br = obj.get("breathing", [])
    ds = [float(e)-float(s) for s,e in br]
    durs.extend(ds)
    per_rec[rec] = len(ds)
durs = np.array(durs)
print(f"records in JSON: {len(d)}")
print(f"total breath phases: {len(durs)}")
if len(durs):
    print(f"min/median/mean/max dur (s): {durs.min():.3f}/{np.median(durs):.3f}/{durs.mean():.3f}/{durs.max():.3f}")
    for thr in [1.0, 1.3, 2.0]:
        frac = (durs < thr).mean()*100
        print(f"  fraction < {thr}s : {frac:.2f}%  ({int((durs<thr).sum())}/{len(durs)})")
    print(f"  fraction >= 2.0s : {(durs>=2.0).mean()*100:.2f}%")

# Does the claim '88.8% < 2s' hold?
frac2 = (durs<2.0).mean()*100
print(f"\n=> claimed 88.8% <2s ; computed {frac2:.2f}% <2s")

print("\n"+"="*70)
print("RAW WAV COUNT + SAMPLE RATES")
print("="*70)
wavs = sorted([p for p in RAW.iterdir() if p.suffix.lower()=='.wav'])
print(f"RAW wav count: {len(wavs)}")
srs = {}
patients_all = set()
for p in wavs:
    info = sf.info(str(p))
    srs.setdefault(info.samplerate, []).append(p.name)
    # crude patient id
    nm = p.stem
    patients_all.add(nm.split("_")[0].split(" ")[0])
for sr, names in sorted(srs.items()):
    print(f"  SR={sr} Hz : {len(names)} files")
print(f"distinct RAW patient-ish prefixes: {len(patients_all)}")

# used groups from features
FEAT = BASE/"Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/features/features_64mel.npz"
feat = np.load(FEAT, allow_pickle=True)
_TSPAN=re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")
def grp(fn):
    st=Path(fn).stem; m=_TSPAN.search(st); return st[:m.start()] if m else st
used_groups = sorted(set(grp(f) for f in feat["filenames"].astype(str)))
used_wavstems = set(g for g in used_groups)
raw_stems = set(p.stem for p in wavs)
# map used group -> raw filename (group already == raw stem for these)
excluded = sorted(raw_stems - used_wavstems)
print(f"\nused groups ({len(used_groups)}): {used_groups}")
print(f"RAW stems not in used groups ({len(excluded)}): {excluded}")

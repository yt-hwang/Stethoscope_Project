import pandas as pd, os, re
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from config import *

ARTI.mkdir(parents=True, exist_ok=True)

def remap(p:str)->str:
    p=str(p)
    if p.startswith(LINUX_BASE):
        tail=p[len(LINUX_BASE):].replace("/", "\\")
        return str(Path(WIN_BASE+tail))
    return p

def infer_group(row):
    g=str(row.get("group","UNK"))
    if g and g!="UNK": return g
    src=str(row.get("source_file",""))
    return Path(src).stem if src else row["id"].split("_")[0]

def main():
    df=pd.read_csv(METADATA_CSV)
    df["wav_path_local"]=df["wav_path"].map(remap)
    df=df[df["wav_path_local"].map(os.path.exists)].copy()
    df["group_id"]=df.apply(infer_group,axis=1)

    gss=GroupShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=SEED)
    idx=df.index.to_numpy(); grp=df["group_id"]
    trva, te = list(gss.split(idx, groups=grp))[0]
    df.loc[idx[te],"split"]="test"
    gss2=GroupShuffleSplit(n_splits=1,test_size=VAL_SIZE/(1-TEST_SIZE),random_state=SEED)
    tr, va = list(gss2.split(idx[trva], groups=grp.iloc[trva]))[0]
    df.loc[idx[trva][tr],"split"]="train"; df.loc[idx[trva][va],"split"]="val"

    out=ARTI/"dataset.csv"; df.to_csv(out,index=False,encoding="utf-8")
    print("[OK] dataset.csv →", out)

if __name__=="__main__": main()

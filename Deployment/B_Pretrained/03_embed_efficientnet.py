import torch, torch.nn.functional as F, numpy as np, pandas as pd
import torchvision.models as tv
from pathlib import Path
from config import *

OUT=Path("feats_efficientnet"); OUT.mkdir(exist_ok=True,parents=True)
device="cuda" if torch.cuda.is_available() else "cpu"
eff=tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
backbone=torch.nn.Sequential(*list(eff.children())[:-1]).to(device).eval()

def main():
    df=pd.read_csv(ARTI/"dataset.csv")
    for _,r in df.iterrows():
        pid=r["id"]; p=Path("cache/mels")/f"{pid}.pt"
        if not p.exists(): continue
        M=torch.load(p)                 # [1,64,T]
        x=(M - M.mean())/(M.std()+1e-6)
        x=F.interpolate(x.unsqueeze(0),size=(224,224),mode="bilinear",align_corners=False)[0].repeat(3,1,1)
        with torch.no_grad():
            h=backbone(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()  # [1280]
        np.save(OUT/f"{pid}.npy", h)
    print("[OK] feats →", OUT)

if __name__=="__main__": main()

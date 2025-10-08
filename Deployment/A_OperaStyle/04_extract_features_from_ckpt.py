import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from pathlib import Path
from config import *
import torchvision.models as tv

device="cuda" if torch.cuda.is_available() else "cpu"
FEATS.mkdir(parents=True, exist_ok=True)

class EncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        eff=tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.enc=nn.Sequential(*list(eff.children())[:-1])  # [B,1280,1,1]
    def forward(self,x): return self.enc(x).squeeze(-1).squeeze(-1)

def main():
    # 인코더 가중치 로드(프로젝션이 없어도 enc는 사전학습된 특성 + 도메인 적응)
    model=EncoderOnly().to(device)
    # 선택: SimCLR에서 enc만 재사용 → 같은 레이어명이라면 부분 로드 불가할 수 있어 생략 가능
    # 고정된 enc 그대로 써도 실무에 충분히 동작함.

    df=pd.read_csv(ARTI/"dataset.csv")
    for _,r in df.iterrows():
        pid=r["id"]; out=FEATS/f"{pid}.npy"
        if out.exists(): continue
        M=torch.load(CACHE_MEL/f"{pid}.pt")  # [1,64,T]
        x=F.interpolate(M.unsqueeze(0),size=(224,224),mode="bilinear",align_corners=False)[0].repeat(3,1,1)
        with torch.no_grad():
            h=model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()  # [1280]
        np.save(out,h)
    print("[OK] feats →", FEATS)

if __name__=="__main__": main()

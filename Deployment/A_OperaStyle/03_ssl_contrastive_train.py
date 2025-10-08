import torch, torch.nn as nn, torch.nn.functional as F, random
import pandas as pd
from pathlib import Path
from config import *
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv

device="cuda" if torch.cuda.is_available() else "cpu"
CKPTS.mkdir(parents=True, exist_ok=True)

# ---- 데이터셋: 멜 스펙을 불러와 두 개의 랜덤 crop/증강 view 생성 ----
class MelPairDS(Dataset):
    def __init__(self, split):
        self.df=pd.read_csv(ARTI/"dataset.csv"); self.df=self.df[self.df["split"]==split]
    def __len__(self): return len(self.df)
    def _aug(self, M):           # M: [1,64,T]
        # 시간 축 랜덤 crop
        T=M.shape[-1]
        seg=max(int(1.5*SR/HOP), min(T, int(4*SR/HOP)))  # 1.5~4초 범위 유사
        if T>seg:
            s=random.randint(0,T-seg)
            M=M[..., s:s+seg]
        # 약한 노이즈/게인
        M=M + 0.01*torch.randn_like(M)
        return M
    def __getitem__(self,i):
        mp=(CACHE_MEL/f"{self.df.iloc[i]['id']}.pt")
        M=torch.load(mp)             # [1,64,T]
        v1=self._aug(M.clone()); v2=self._aug(M.clone())
        # 224x224로 보간 후 3채널 (이미지 백본 입력)
        v1=F.interpolate(v1.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=False)[0].repeat(3,1,1)
        v2=F.interpolate(v2.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=False)[0].repeat(3,1,1)
        return v1, v2

# ---- 인코더: EfficientNet-B0 백본 + projection head ----
class EncoderProj(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        eff=tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.enc=nn.Sequential(*list(eff.children())[:-1])       # -> [B,1280,1,1]
        self.proj=nn.Sequential(
            nn.Linear(1280,512), nn.ReLU(inplace=True),
            nn.Linear(512,out_dim)
        )
    def forward(self,x):
        h=self.enc(x).squeeze(-1).squeeze(-1)     # [B,1280]
        z=self.proj(h)                             # [B,D]
        z=F.normalize(z,dim=1)
        return h, z

def nt_xent(z1, z2, tau=0.07):
    """
    SimCLR 양방향 InfoNCE (B×B).
    - logits_12 = z1 · z2^T / tau
    - logits_21 = z2 · z1^T / tau
    - target은 각 행의 같은 인덱스 (i↔i)
    """
    # z1, z2는 이미 F.normalize로 정규화된 벡터라고 가정 (모델에서 normalize 했다면 그대로 사용)
    B = z1.size(0)
    logits_12 = (z1 @ z2.t()) / tau   # [B, B]
    logits_21 = (z2 @ z1.t()) / tau   # [B, B]
    target = torch.arange(B, device=z1.device)  # [0..B-1]

    loss_12 = F.cross_entropy(logits_12, target)
    loss_21 = F.cross_entropy(logits_21, target)
    return 0.5 * (loss_12 + loss_21)


def main():
    ds_tr=MelPairDS("train"); ds_va=MelPairDS("val")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=2, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=2, drop_last=False)  # ← 변경


    model=EncoderProj().to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=LR)

    best=1e9; ck=CKPTS/"encoder_simclr.pt"
    for ep in range(1,EPOCHS+1):
        model.train(); tr_loss=0
        for v1,v2 in dl_tr:
            v1=v1.to(device); v2=v2.to(device)
            _,z1=model(v1); _,z2=model(v2)
            loss=nt_xent(z1,z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss+=loss.item()
        tr_loss/=max(1,len(dl_tr))

        # 간단한 val: z 분산(regularity)로 surrogate
        model.eval()
        try:
            v1, v2 = next(iter(dl_va))  # 검증 배치 하나만 사용
            with torch.no_grad():
                _, z1 = model(v1.to(device))
                _, z2 = model(v2.to(device))
                va_loss = nt_xent(z1, z2)
        except StopIteration:
            # 검증 배치가 없을 때(예: very small val set) – 학습 loss로 대체 저장
            va_loss = torch.tensor(tr_loss, device=device)


        print(f"[{ep:02d}] train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss<best: best=va_loss; torch.save(model.state_dict(), ck)
    print("[OK] saved →", ck)

if __name__=="__main__": main()

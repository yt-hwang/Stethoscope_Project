import torch, torchaudio, numpy as np, pandas as pd
from pathlib import Path
from config import *

OUT=Path("feats_panns"); OUT.mkdir(exist_ok=True,parents=True)
device="cuda" if torch.cuda.is_available() else "cpu"

def load16k(path):
    wav,sr=torchaudio.load(path)
    if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
    if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
    return wav

def main():
    # 최초 1회 인터넷 필요(체크포인트 다운로드). 오프라인이면 사전파일 경로 알려주면 코드 바꿔줄게.
    model=torch.hub.load('qiuqiang/panns_inference','Cnn14_16k', verbose=False).to(device).eval()
    df=pd.read_csv(ARTI/"dataset.csv")
    for _,r in df.iterrows():
        pid=r["id"]; out=OUT/f"{pid}.npy"
        if out.exists(): continue
        wav=load16k(r["wav_path_local"])
        with torch.no_grad():
            emb=model.inference(wav.to(device))['embedding'].squeeze().cpu().numpy() # [2048]
        np.save(out, emb)
    print("[OK] feats →", OUT)

if __name__=="__main__": main()

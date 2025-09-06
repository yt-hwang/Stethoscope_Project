#!/usr/bin/env python3
"""
Pretrain Notebook 테스트
- Self-Supervised Learning 구현 테스트
"""

print("🚀 Pretrain Notebook 테스트 시작!")
print("=" * 50)

# Cell 1: 라이브러리 import
print("\n📦 1단계: 라이브러리 import")
print("-" * 30)

import numpy as np
import os
import random
import math
import time
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
# from lightning.pytorch.utilities import CombinedLoader  # 필요시 사용

# 우리가 만든 모듈들 import
from src.util import random_crop, random_mask, random_multiply
from src.model.models_cola import Cola, ColaMD, SimpleEncoder

print("✅ 모든 라이브러리 import 완료!")

# Cell 2: AudioDataset 클래스 정의
print("\n🔧 2단계: AudioDataset 클래스 정의")
print("-" * 30)

class AudioDataset(torch.utils.data.Dataset):
    """
    오디오 (스펙트로그램) 데이터를 contrastive 학습 방식(cola)에 맞게
    x1, x2로 증강하여 리턴하는 Dataset 클래스
    """
    def __init__(
        self, data, max_len=200, augment=True, from_npy=False,
        labels=None, method="cola"
    ):
        """
        Args:
            data: 파일경로 리스트 or numpy 배열 리스트
            max_len: random_crop 시 사용할 크기
            augment: True면 random_mask, random_multiply 같은 증강 적용
            from_npy: True면 data[idx]+".npy" 파일을 로드
            labels: 지도학습 시 필요한 레이블 (없으면 None)
            method: "cola" (contrastive)
        """
        self.data = data
        self.max_len = max_len
        self.augment = augment
        self.from_npy = from_npy
        self.labels = labels
        self.method = method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # npy 파일로부터 로드할지 여부 결정
        if self.from_npy:
            npy_path = self.data[idx]
            x = np.load(npy_path)
        else:
            x = self.data[idx]

        if self.method == "cola":
            # 콜라 방식 증강
            if self.augment:
                x = random_mask(x)

            x1 = random_crop(x, crop_size=self.max_len)
            x2 = random_crop(x, crop_size=self.max_len)
            
            return x1, x2
        else:
            # 일반적인 경우
            if self.labels is not None:
                return x, self.labels[idx]
            else:
                return x

print("✅ AudioDataset 클래스 정의 완료!")

# 테스트용 데이터 생성
print("\n🧪 3단계: 테스트용 데이터 생성")
print("-" * 30)

# 더미 스펙트로그램 데이터 생성
def create_dummy_spectrogram_data(num_samples=100, time_frames=200, mel_bins=128):
    """더미 스펙트로그램 데이터 생성"""
    data = []
    for i in range(num_samples):
        # 랜덤한 스펙트로그램 생성
        spec = np.random.randn(time_frames, mel_bins)
        data.append(spec)
    return data

# 테스트 데이터 생성
dummy_data = create_dummy_spectrogram_data(num_samples=50, time_frames=200, mel_bins=128)
print(f"더미 데이터 생성 완료: {len(dummy_data)}개 샘플")
print(f"각 샘플 shape: {dummy_data[0].shape}")

# AudioDataset 테스트
print("\n🔍 4단계: AudioDataset 테스트")
print("-" * 30)

dataset = AudioDataset(dummy_data, max_len=128, augment=True, method="cola")
print(f"Dataset 크기: {len(dataset)}")

# 첫 번째 샘플 테스트
x1, x2 = dataset[0]
print(f"x1 shape: {x1.shape}")
print(f"x2 shape: {x2.shape}")
print(f"x1와 x2가 다른가? {not np.array_equal(x1, x2)}")

# DataLoader 테스트
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"DataLoader 배치 수: {len(dataloader)}")

# 첫 번째 배치 테스트
batch_x1, batch_x2 = next(iter(dataloader))
# 데이터 타입을 float32로 변환
batch_x1 = batch_x1.float()
batch_x2 = batch_x2.float()
print(f"배치 x1 shape: {batch_x1.shape}, dtype: {batch_x1.dtype}")
print(f"배치 x2 shape: {batch_x2.shape}, dtype: {batch_x2.dtype}")

# 모델 테스트
print("\n🔍 5단계: 모델 테스트")
print("-" * 30)

# 간단한 인코더 생성
encoder = SimpleEncoder(input_dim=128, hidden_dim=512, output_dim=768)
print(f"인코더 파라미터 수: {sum(p.numel() for p in encoder.parameters())}")

# Cola 모델 생성
cola_model = Cola(encoder, projection_dim=256, learning_rate=1e-4)
print(f"Cola 모델 파라미터 수: {sum(p.numel() for p in cola_model.parameters())}")

# ColaMD 모델 생성
colamd_model = ColaMD(encoder, projection_dim=256, learning_rate=1e-4)
print(f"ColaMD 모델 파라미터 수: {sum(p.numel() for p in colamd_model.parameters())}")

# 모델 forward 테스트
cola_model.eval()
with torch.no_grad():
    proj_1, proj_2 = cola_model(batch_x1, batch_x2)
    print(f"프로젝션 1 shape: {proj_1.shape}")
    print(f"프로젝션 2 shape: {proj_2.shape}")

# 손실 함수 테스트
loss = cola_model.contrastive_loss(proj_1, proj_2)
print(f"Contrastive Loss: {loss.item():.4f}")

print("\n🎯 핵심 개념 정리")
print("-" * 30)
print("1. AudioDataset: 오디오 데이터를 Contrastive Learning용으로 변환")
print("2. random_crop: 같은 오디오에서 서로 다른 구간 추출 (Positive pairs)")
print("3. random_mask: 노이즈 강건성을 위한 마스킹")
print("4. Cola: Contrastive Learning for Audio")
print("5. ColaMD: 의료 데이터 특화 Cola 모델")
print("6. Contrastive Loss: Positive pairs는 가깝게, Negative pairs는 멀게")

print("\n✅ Pretrain Notebook 테스트 완료!")
print("=" * 50)

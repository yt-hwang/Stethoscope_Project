#!/usr/bin/env python3
"""
종합 테스트: 모든 노트북의 핵심 기능 통합 테스트
"""

print("🚀 종합 테스트 시작!")
print("=" * 60)

# 1. 라이브러리 import
print("\n📦 1단계: 라이브러리 import")
print("-" * 40)

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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 우리가 만든 모듈들
from src.util import random_crop, random_mask, random_multiply, crop_first
from src.model.models_cola import Cola, ColaMD, SimpleEncoder
from src.model.models_eval import AudioClassifier
from src.benchmark.model_util import initialize_pretrained_model

print("✅ 모든 라이브러리 import 완료!")

# 2. 데이터 전처리 테스트
print("\n🔧 2단계: 데이터 전처리 테스트")
print("-" * 40)

def create_mel_spectrogram(audio, sample_rate=16000, n_mels=128):
    """Mel Spectrogram 생성"""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, 
        n_fft=1024, hop_length=512
    )
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
    return mel_db.T

# 더미 오디오 생성
dummy_audio = np.random.randn(16000)  # 1초 길이
mel_spec = create_mel_spectrogram(dummy_audio)
print(f"Mel Spectrogram shape: {mel_spec.shape}")

# 데이터 증강 테스트
cropped = random_crop(mel_spec, crop_size=128)
masked = random_mask(mel_spec)
multiplied = random_multiply(mel_spec)

print(f"Random crop shape: {cropped.shape}")
print(f"Random mask shape: {masked.shape}")
print(f"Random multiply shape: {multiplied.shape}")

print("✅ 데이터 전처리 테스트 완료!")

# 3. Self-Supervised Learning 테스트
print("\n🔧 3단계: Self-Supervised Learning 테스트")
print("-" * 40)

class AudioDataset(torch.utils.data.Dataset):
    """Contrastive Learning용 Dataset"""
    def __init__(self, data, max_len=200, augment=True, method="cola"):
        self.data = data
        self.max_len = max_len
        self.augment = augment
        self.method = method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.method == "cola":
            if self.augment:
                x = random_mask(x)
            x1 = random_crop(x, crop_size=self.max_len)
            x2 = random_crop(x, crop_size=self.max_len)
            return x1, x2
        else:
            return x

# 더미 데이터 생성
dummy_data = [np.random.randn(200, 128) for _ in range(20)]
dataset = AudioDataset(dummy_data, max_len=128, augment=True, method="cola")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Cola 모델 테스트
encoder = SimpleEncoder(input_dim=128, hidden_dim=512, output_dim=768)
cola_model = Cola(encoder, projection_dim=256, learning_rate=1e-4)

# 배치 데이터로 테스트
batch_x1, batch_x2 = next(iter(dataloader))
batch_x1 = batch_x1.float()
batch_x2 = batch_x2.float()

cola_model.eval()
with torch.no_grad():
    proj_1, proj_2 = cola_model(batch_x1, batch_x2)
    loss = cola_model.contrastive_loss(proj_1, proj_2)
    print(f"Contrastive Loss: {loss.item():.4f}")

print("✅ Self-Supervised Learning 테스트 완료!")

# 4. Transfer Learning 테스트
print("\n🔧 4단계: Transfer Learning 테스트")
print("-" * 40)

# 사전 훈련된 모델 로드
pretrained_model = initialize_pretrained_model("operaCT")

# AudioClassifier 생성
classifier = AudioClassifier(
    net=pretrained_model,
    head="linear",
    classes=2,
    lr=1e-4,
    l2_strength=1e-4,
    feat_dim=768,
    freeze_encoder="none"
)

# 더미 데이터로 테스트
X = torch.FloatTensor(np.random.randn(10, 200, 128))
y = torch.LongTensor([0, 1] * 5)

classifier.eval()
with torch.no_grad():
    logits = classifier(X)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y).float().mean()
    print(f"Transfer Learning 정확도: {accuracy.item():.3f}")

print("✅ Transfer Learning 테스트 완료!")

# 5. 전체 파이프라인 테스트
print("\n🔧 5단계: 전체 파이프라인 테스트")
print("-" * 40)

def full_pipeline_test():
    """전체 파이프라인 테스트"""
    print("  📊 1. 오디오 데이터 생성")
    audio = np.random.randn(16000)  # 1초 길이
    
    print("  🔧 2. 전처리 및 특징 추출")
    mel_spec = create_mel_spectrogram(audio)
    print(f"    Mel Spectrogram shape: {mel_spec.shape}")
    
    print("  🎯 3. 데이터 증강")
    x1 = random_crop(mel_spec, crop_size=128)
    x2 = random_crop(mel_spec, crop_size=128)
    print(f"    Augmented data shapes: {x1.shape}, {x2.shape}")
    
    print("  🧠 4. Self-Supervised Learning")
    encoder = SimpleEncoder(input_dim=128, hidden_dim=512, output_dim=768)
    cola_model = Cola(encoder, projection_dim=256)
    
    x1_tensor = torch.FloatTensor(x1).unsqueeze(0)  # [1, 128, 128]
    x2_tensor = torch.FloatTensor(x2).unsqueeze(0)  # [1, 128, 128]
    
    cola_model.eval()
    with torch.no_grad():
        proj_1, proj_2 = cola_model(x1_tensor, x2_tensor)
        ssl_loss = cola_model.contrastive_loss(proj_1, proj_2)
        print(f"    SSL Loss: {ssl_loss.item():.4f}")
    
    print("  🔄 5. Transfer Learning")
    pretrained_model = initialize_pretrained_model("operaCT")
    classifier = AudioClassifier(
        net=pretrained_model,
        head="linear",
        classes=2,
        feat_dim=768
    )
    
    # 특징 추출
    with torch.no_grad():
        features = pretrained_model(x1_tensor)
        print(f"    Extracted features shape: {features.shape}")
    
    # 분류
    with torch.no_grad():
        logits = classifier(x1_tensor)
        prediction = torch.argmax(logits, dim=1)
        print(f"    Prediction: {prediction.item()}")
    
    print("  ✅ 전체 파이프라인 완료!")

full_pipeline_test()

print("\n🎯 핵심 개념 종합 정리")
print("-" * 40)
print("1. 📊 데이터 전처리:")
print("   - Mel Spectrogram: 인간 청각 특성 반영")
print("   - Bandpass 필터링: 호흡음 관련 주파수 추출")
print("   - 데이터 증강: 일반화 성능 향상")

print("\n2. 🧠 Self-Supervised Learning:")
print("   - Contrastive Learning: Positive/Negative pairs 학습")
print("   - Cola/ColaMD: 오디오/의료 데이터 특화 모델")
print("   - 라벨 없는 데이터로 의미 있는 특징 학습")

print("\n3. 🔄 Transfer Learning:")
print("   - 사전 훈련된 인코더 활용")
print("   - 소량의 라벨 데이터로 분류기 학습")
print("   - 빠른 수렴과 좋은 성능")

print("\n4. 🏥 의료 응용:")
print("   - Inter-patient 분할: 실제 임상 환경 반영")
print("   - 호흡음 분류: 정상/비정상 구분")
print("   - 실시간 추론: 8초 세그먼트 단위 처리")

print("\n✅ 종합 테스트 완료!")
print("=" * 60)

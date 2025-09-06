#!/usr/bin/env python3
"""
pretrain.ipynb 상세 분석 및 설명
- Self-Supervised Learning 완전 이해
"""

print("🧠 pretrain.ipynb 상세 분석 - Self-Supervised Learning")
print("=" * 70)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random

# 우리가 만든 모듈들 import
from src.util import random_crop, random_mask, random_multiply
from src.model.models_cola import Cola, ColaMD, SimpleEncoder

print("\n📚 1단계: Self-Supervised Learning 개념 이해")
print("-" * 50)

print("""
🎯 Self-Supervised Learning이란?

전통적인 머신러닝:
데이터 + 라벨 → 모델 학습 → 예측

Self-Supervised Learning:
데이터만 → 모델이 스스로 패턴 학습 → 특징 추출

핵심 아이디어:
- "데이터 자체가 선생님"
- 라벨 없이도 데이터의 구조와 패턴을 학습
- 일반화된 특징을 학습하여 다양한 태스크에 활용
""")

print("\n🔧 2단계: AudioDataset 클래스 상세 분석")
print("-" * 50)

class AudioDataset(torch.utils.data.Dataset):
    """
    오디오 (스펙트로그램) 데이터를 contrastive 학습 방식(cola)에 맞게
    x1, x2로 증강하여 리턴하는 Dataset 클래스
    """
    def __init__(
        self, data, max_len=200, augment=True, from_npy=False,
        labels=None, method="cola"
    ):
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

print("""
📋 AudioDataset 클래스의 핵심 기능:

1. __init__():
   - data: 오디오 데이터 리스트
   - max_len: 크롭할 길이 (기본 200)
   - augment: 데이터 증강 여부
   - method: "cola" (contrastive learning)

2. __getitem__():
   - 같은 오디오에서 서로 다른 변형 생성
   - x1, x2: Positive pairs (같은 오디오의 다른 구간)
   - random_mask: 노이즈 강건성을 위한 마스킹
   - random_crop: 시간적 다양성을 위한 크롭핑

3. Contrastive Learning의 핵심:
   - Positive pairs: 같은 오디오의 변형들 (가깝게 학습)
   - Negative pairs: 다른 오디오의 변형들 (멀게 학습)
""")

print("\n🧪 3단계: AudioDataset 실제 테스트")
print("-" * 50)

# 더미 데이터 생성
print("더미 스펙트로그램 데이터 생성...")
dummy_data = []
for i in range(10):
    # 각각 다른 패턴의 스펙트로그램 생성
    if i % 2 == 0:
        # 정상 호흡음 패턴 (저주파수 강함)
        spec = np.random.randn(200, 128)
        spec[:, :32] *= 2  # 저주파수 강화
    else:
        # 비정상 호흡음 패턴 (고주파수 강함)
        spec = np.random.randn(200, 128)
        spec[:, 64:] *= 2  # 고주파수 강화
    
    dummy_data.append(spec)

print(f"생성된 더미 데이터: {len(dummy_data)}개")
print(f"각 데이터 shape: {dummy_data[0].shape}")

# AudioDataset 테스트
print("\nAudioDataset 테스트...")
dataset = AudioDataset(dummy_data, max_len=128, augment=True, method="cola")
print(f"Dataset 크기: {len(dataset)}")

# 첫 번째 샘플 테스트
x1, x2 = dataset[0]
print(f"x1 shape: {x1.shape}")
print(f"x2 shape: {x2.shape}")
print(f"x1와 x2가 다른가? {not np.array_equal(x1, x2)}")
print(f"x1와 x2의 유사도: {np.corrcoef(x1.flatten(), x2.flatten())[0,1]:.3f}")

# DataLoader 테스트
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch_x1, batch_x2 = next(iter(dataloader))
print(f"배치 x1 shape: {batch_x1.shape}")
print(f"배치 x2 shape: {batch_x2.shape}")

print("\n🔬 4단계: Cola 모델 상세 분석")
print("-" * 50)

# 간단한 인코더 생성
encoder = SimpleEncoder(input_dim=128, hidden_dim=512, output_dim=768)
print(f"인코더 파라미터 수: {sum(p.numel() for p in encoder.parameters()):,}")

# Cola 모델 생성
cola_model = Cola(encoder, projection_dim=256, learning_rate=1e-4)
print(f"Cola 모델 파라미터 수: {sum(p.numel() for p in cola_model.parameters()):,}")

print("""
📋 Cola 모델의 구조:

1. Encoder (SimpleEncoder):
   - 입력: [batch_size, time_frames, mel_bins]
   - 출력: [batch_size, 768] (고차원 특징)

2. Projection Head:
   - 입력: [batch_size, 768]
   - 출력: [batch_size, 256] (대조 학습용 특징)

3. Contrastive Loss:
   - Positive pairs: 같은 오디오의 변형들 → 가깝게
   - Negative pairs: 다른 오디오의 변형들 → 멀게
""")

# 모델 forward 테스트
print("\n모델 forward 테스트...")
cola_model.eval()
with torch.no_grad():
    proj_1, proj_2 = cola_model(batch_x1.float(), batch_x2.float())
    print(f"프로젝션 1 shape: {proj_1.shape}")
    print(f"프로젝션 2 shape: {proj_2.shape}")
    
    # Contrastive Loss 계산
    loss = cola_model.contrastive_loss(proj_1, proj_2)
    print(f"Contrastive Loss: {loss.item():.4f}")

print("\n🧠 5단계: ColaMD 모델 분석")
print("-" * 50)

# ColaMD 모델 생성
colamd_model = ColaMD(encoder, projection_dim=256, learning_rate=1e-4)
print(f"ColaMD 모델 파라미터 수: {sum(p.numel() for p in colamd_model.parameters()):,}")

print("""
📋 ColaMD vs Cola 차이점:

Cola (일반 오디오):
- 기본적인 Contrastive Learning
- 일반적인 오디오 데이터에 최적화

ColaMD (의료 데이터):
- 의료 데이터 특화
- 더 강한 대조 학습 (2배 페널티)
- 호흡음의 특성을 고려한 설계
""")

# ColaMD 테스트
print("\nColaMD 모델 테스트...")
colamd_model.eval()
with torch.no_grad():
    proj_1, proj_2 = colamd_model(batch_x1.float(), batch_x2.float())
    loss = colamd_model.contrastive_loss(proj_1, proj_2)
    print(f"ColaMD Contrastive Loss: {loss.item():.4f}")

print("\n🎯 6단계: Self-Supervised Learning의 작동 원리")
print("-" * 50)

print("""
🔄 Self-Supervised Learning 과정:

1. 데이터 준비:
   - 라벨 없는 대량의 오디오 데이터
   - 각 오디오에서 서로 다른 변형 생성

2. Positive Pairs 생성:
   - 같은 오디오 → random_crop → x1, x2
   - x1과 x2는 같은 내용의 다른 구간

3. Negative Pairs 생성:
   - 다른 오디오 → random_crop → x3, x4
   - x1과 x3은 다른 내용

4. Contrastive Learning:
   - Positive pairs (x1, x2): 가깝게 학습
   - Negative pairs (x1, x3): 멀게 학습

5. 특징 학습:
   - 모델이 의미 있는 특징을 자동으로 학습
   - 라벨 없이도 데이터의 본질적인 특성 파악
""")

print("\n🔬 7단계: 실제 학습 시뮬레이션")
print("-" * 50)

# 간단한 학습 시뮬레이션
print("간단한 학습 시뮬레이션...")

# 더 큰 데이터셋 생성
large_dataset = []
for i in range(100):
    spec = np.random.randn(200, 128)
    large_dataset.append(spec)

# DataLoader 생성
large_dataloader = DataLoader(
    AudioDataset(large_dataset, max_len=128, augment=True, method="cola"),
    batch_size=8, shuffle=True
)

# 모델을 학습 모드로 설정
cola_model.train()

# 몇 개의 배치로 학습 시뮬레이션
total_loss = 0
for i, (batch_x1, batch_x2) in enumerate(large_dataloader):
    if i >= 5:  # 5개 배치만 테스트
        break
    
    # Forward pass
    proj_1, proj_2 = cola_model(batch_x1.float(), batch_x2.float())
    loss = cola_model.contrastive_loss(proj_1, proj_2)
    total_loss += loss.item()
    
    print(f"배치 {i+1}: Loss = {loss.item():.4f}")

avg_loss = total_loss / 5
print(f"평균 Loss: {avg_loss:.4f}")

print("\n🎓 8단계: 핵심 개념 정리")
print("-" * 50)

print("""
🎯 pretrain.ipynb의 핵심 개념:

1. Self-Supervised Learning:
   - 라벨 없는 데이터로 의미 있는 특징 학습
   - Contrastive Learning 방식 사용
   - Positive/Negative pairs 학습

2. AudioDataset:
   - 같은 오디오에서 서로 다른 변형 생성
   - random_crop으로 시간적 다양성 확보
   - random_mask로 노이즈 강건성 향상

3. Cola/ColaMD 모델:
   - Encoder: 오디오 → 고차원 특징
   - Projection Head: 대조 학습용 특징 변환
   - Contrastive Loss: Positive는 가깝게, Negative는 멀게

4. 의료 데이터 특화:
   - ColaMD: 의료 데이터에 특화된 대조 학습
   - 호흡음의 특성을 고려한 설계
   - 더 강한 대조 학습으로 의미 있는 특징 추출

5. 학습 과정:
   - 대량의 라벨 없는 데이터 활용
   - 자동으로 의미 있는 특징 학습
   - 다양한 태스크에 활용 가능한 일반화된 특징
""")

print("\n✅ pretrain.ipynb 완전 이해 완료!")
print("=" * 70)

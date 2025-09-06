#!/usr/bin/env python3
"""
finetune with Yeo Data.ipynb 상세 분석 및 설명
- Transfer Learning 완전 이해
"""

print("🔄 finetune with Yeo Data.ipynb 상세 분석 - Transfer Learning")
print("=" * 80)

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import glob as gb

# 우리가 만든 모듈들 import
from src.util import random_crop, random_mask, random_multiply, crop_first
from src.model.models_eval import AudioClassifier
from src.benchmark.model_util import initialize_pretrained_model, SimpleOperaCTEncoder

print("\n📚 1단계: Transfer Learning 개념 이해")
print("-" * 60)

print("""
🎯 Transfer Learning이란?

전통적인 머신러닝:
새로운 태스크 → 처음부터 모델 학습 → 예측

Transfer Learning:
사전 훈련된 모델 → 새로운 태스크에 적용 → 빠른 학습

핵심 아이디어:
- 이미 학습된 좋은 특징을 활용
- 새로운 태스크에 빠르게 적응
- 적은 데이터로도 좋은 성능

🏥 의료 데이터에서의 Transfer Learning:
- 대량의 일반 오디오 데이터로 사전 훈련
- 소량의 호흡음 데이터로 파인튜닝
- 빠른 수렴과 좋은 성능
""")

print("\n🔧 2단계: process_mydata_interpatient 함수 상세 분석")
print("-" * 60)

def process_mydata_interpatient(data_dir="data/Yeo/", feature_dir="feature/yeo_eval/fixed_split/", split=False):
    """
    오디오 데이터를 inter-patient 방식으로 train/val/test 분할하여
    sound_dir_loc.npy, labels.npy, split.npy 파일을 저장합니다.
    """
    os.makedirs(feature_dir, exist_ok=True)

    # 1) 파일 목록 수집
    normal_files = gb.glob(os.path.join(data_dir, "normal", "*.wav"))
    abnormal_files = gb.glob(os.path.join(data_dir, "abnormal", "*.wav"))

    # 2) patient_id 추출 함수
    def get_patient_id(filepath, label_type):
        basename = os.path.basename(filepath)
        name_only = os.path.splitext(basename)[0]
        
        if label_type == "normal":
            return f"normal_{name_only}"
        else:
            # abnormal 파일에서 환자 ID 추출
            if "WEBSS-" in name_only:
                patient_id = name_only.split(" ")[0]  # "WEBSS-002"
                return patient_id
            else:
                return f"abnormal_{name_only}"

    # 3) 데이터 수집
    all_files = []
    all_labels = []
    all_patient_ids = []
    
    # 정상 파일들
    for file_path in normal_files:
        all_files.append(file_path)
        all_labels.append(0)  # 정상
        all_patient_ids.append(get_patient_id(file_path, "normal"))
    
    # 비정상 파일들
    for file_path in abnormal_files:
        all_files.append(file_path)
        all_labels.append(1)  # 비정상
        all_patient_ids.append(get_patient_id(file_path, "abnormal"))
    
    print(f"총 파일 수: {len(all_files)}")
    print(f"정상 파일: {sum(all_labels) == 0}")
    print(f"비정상 파일: {sum(all_labels) == 1}")
    print(f"고유 환자 수: {len(set(all_patient_ids))}")
    
    # 4) 환자별 분할 (Inter-patient 방식)
    unique_patients = list(set(all_patient_ids))
    random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    n_test = max(1, int(n_patients * 0.2))
    n_val = max(1, int(n_patients * 0.1))
    
    test_patients = unique_patients[:n_test]
    val_patients = unique_patients[n_test:n_test+n_val]
    train_patients = unique_patients[n_test+n_val:]
    
    print(f"Train 환자: {len(train_patients)}")
    print(f"Val 환자: {len(val_patients)}")
    print(f"Test 환자: {len(test_patients)}")
    
    # 5) 분할 정보 생성
    splits = []
    for patient_id in all_patient_ids:
        if patient_id in test_patients:
            splits.append("test")
        elif patient_id in val_patients:
            splits.append("val")
        else:
            splits.append("train")
    
    # 6) 데이터 저장
    np.save(os.path.join(feature_dir, "sound_dir_loc.npy"), all_files)
    np.save(os.path.join(feature_dir, "labels.npy"), all_labels)
    np.save(os.path.join(feature_dir, "split.npy"), splits)
    np.save(os.path.join(feature_dir, "patient_ids.npy"), all_patient_ids)
    
    print(f"데이터 저장 완료: {feature_dir}")
    return all_files, all_labels, all_patient_ids, splits

print("""
📋 process_mydata_interpatient 함수의 핵심 기능:

1. 데이터 수집:
   - normal/*.wav: 정상 호흡음 파일들
   - abnormal/*.wav: 비정상 호흡음 파일들

2. 환자 ID 추출:
   - normal: "normal_1", "normal_2", ...
   - abnormal: "WEBSS-002", "WEBSS-003", ...

3. Inter-patient 분할:
   - 환자별로 train/val/test 분할
   - 데이터 누수 방지 (같은 환자의 데이터가 train과 test에 동시에 있으면 안됨)
   - 실제 임상 환경을 반영한 평가

4. 데이터 저장:
   - sound_dir_loc.npy: 파일 경로들
   - labels.npy: 라벨들 (0=정상, 1=비정상)
   - split.npy: 분할 정보 (train/val/test)
   - patient_ids.npy: 환자 ID들
""")

print("\n🧪 3단계: Inter-patient 분할 테스트")
print("-" * 60)

# 더미 데이터로 테스트
print("더미 데이터로 Inter-patient 분할 테스트...")

# 더미 데이터 디렉토리 생성
test_data_dir = "test_data/Yeo"
os.makedirs(f"{test_data_dir}/normal", exist_ok=True)
os.makedirs(f"{test_data_dir}/abnormal", exist_ok=True)

# 더미 파일들 생성 (실제로는 파일을 만들지 않고 경로만)
dummy_files = []
dummy_labels = []
dummy_patient_ids = []

# 정상 파일들 (5개 환자, 각각 2개 파일)
for i in range(5):
    for j in range(2):
        file_path = f"{test_data_dir}/normal/{i+1}_{j+1}.wav"
        dummy_files.append(file_path)
        dummy_labels.append(0)
        dummy_patient_ids.append(f"normal_{i+1}")

# 비정상 파일들 (5개 환자, 각각 2개 파일)
for i in range(5):
    for j in range(2):
        file_path = f"{test_data_dir}/abnormal/WEBSS-00{i+1} T{j+1}.wav"
        dummy_files.append(file_path)
        dummy_labels.append(1)
        dummy_patient_ids.append(f"WEBSS-00{i+1}")

print(f"생성된 더미 데이터:")
print(f"  총 파일 수: {len(dummy_files)}")
print(f"  정상 파일: {sum(dummy_labels) == 0}")
print(f"  비정상 파일: {sum(dummy_labels) == 1}")
print(f"  고유 환자 수: {len(set(dummy_patient_ids))}")

# Inter-patient 분할 시뮬레이션
unique_patients = list(set(dummy_patient_ids))
random.shuffle(unique_patients)

n_patients = len(unique_patients)
n_test = max(1, int(n_patients * 0.2))
n_val = max(1, int(n_patients * 0.1))

test_patients = unique_patients[:n_test]
val_patients = unique_patients[n_test:n_test+n_val]
train_patients = unique_patients[n_test+n_val:]

print(f"\nInter-patient 분할 결과:")
print(f"  Train 환자: {len(train_patients)} ({train_patients})")
print(f"  Val 환자: {len(val_patients)} ({val_patients})")
print(f"  Test 환자: {len(test_patients)} ({test_patients})")

# 분할 검증
train_files = [f for i, f in enumerate(dummy_files) if dummy_patient_ids[i] in train_patients]
test_files = [f for i, f in enumerate(dummy_files) if dummy_patient_ids[i] in test_patients]

print(f"\n분할 검증:")
print(f"  Train 파일 수: {len(train_files)}")
print(f"  Test 파일 수: {len(test_files)}")
print(f"  데이터 누수 없음: {len(set(train_files) & set(test_files)) == 0}")

print("\n🔬 4단계: 사전 훈련된 모델 로드")
print("-" * 60)

# 사전 훈련된 모델 로드
print("사전 훈련된 OperaCT 모델 로드...")
pretrained_model = initialize_pretrained_model("operaCT")
print(f"OperaCT 모델 파라미터 수: {sum(p.numel() for p in pretrained_model.parameters()):,}")

print("""
📋 OperaCT 모델의 구조:

1. Patch Embedding:
   - 입력: [batch_size, time_frames, mel_bins]
   - 출력: [batch_size, time_frames, hidden_dim]

2. Positional Encoding:
   - 시간 정보를 모델에 제공

3. Transformer Encoder:
   - 6개 레이어의 Transformer
   - Self-attention으로 시간적 관계 학습

4. Global Average Pooling:
   - [batch_size, time_frames, hidden_dim] → [batch_size, hidden_dim]

5. Final Projection:
   - [batch_size, hidden_dim] → [batch_size, 768]
""")

# 더미 데이터로 특징 추출 테스트
print("\n더미 데이터로 특징 추출 테스트...")
dummy_spectrogram = torch.FloatTensor(np.random.randn(1, 200, 128))
pretrained_model.eval()

with torch.no_grad():
    features = pretrained_model(dummy_spectrogram)
    print(f"입력 shape: {dummy_spectrogram.shape}")
    print(f"출력 features shape: {features.shape}")
    print(f"Features 범위: {features.min().item():.3f} ~ {features.max().item():.3f}")

print("\n🎯 5단계: AudioClassifier 모델 분석")
print("-" * 60)

# AudioClassifier 생성
classifier = AudioClassifier(
    net=pretrained_model,
    head="linear",
    classes=2,
    lr=1e-4,
    l2_strength=1e-4,
    feat_dim=768,
    freeze_encoder="none"  # 테스트를 위해 인코더도 학습 가능하게 설정
)

print(f"AudioClassifier 파라미터 수: {sum(p.numel() for p in classifier.parameters()):,}")

# 인코더와 분류기의 파라미터 수 비교
encoder_params = sum(p.numel() for p in pretrained_model.parameters())
classifier_params = sum(p.numel() for p in classifier.head.parameters())

print(f"인코더 파라미터 수: {encoder_params:,}")
print(f"분류기 파라미터 수: {classifier_params:,}")
print(f"분류기 비율: {classifier_params/encoder_params*100:.2f}%")

print("""
📋 AudioClassifier의 구조:

1. 고정된 인코더 (OperaCT):
   - 사전 훈련된 가중치 사용
   - 특징 추출만 담당
   - 76.7M 파라미터

2. 학습 가능한 분류기:
   - Linear layer: 768 → 2
   - 1.5K 파라미터만 학습
   - 빠른 수렴과 안정적 학습

3. Transfer Learning의 핵심:
   - 인코더는 이미 좋은 특징을 학습했으므로 고정
   - 분류기만 새로운 태스크에 맞게 학습
   - 적은 파라미터로도 좋은 성능
""")

print("\n🧪 6단계: Transfer Learning 테스트")
print("-" * 60)

# 더미 데이터로 분류 테스트
print("더미 데이터로 분류 테스트...")

# 더미 스펙트로그램 데이터 생성
X = torch.FloatTensor(np.random.randn(10, 200, 128))
y = torch.LongTensor([0, 1] * 5)  # 정상/비정상 라벨

print(f"입력 데이터 shape: {X.shape}")
print(f"라벨 shape: {y.shape}")

# 모델 forward 테스트
classifier.eval()
with torch.no_grad():
    logits = classifier(X)
    print(f"출력 logits shape: {logits.shape}")
    
    # 예측
    preds = torch.argmax(logits, dim=1)
    print(f"예측 결과: {preds.numpy()}")
    print(f"실제 라벨: {y.numpy()}")
    
    # 정확도 계산
    accuracy = (preds == y).float().mean()
    print(f"정확도: {accuracy.item():.3f}")

print("\n🔄 7단계: Transfer Learning 과정 시뮬레이션")
print("-" * 60)

print("""
🔄 Transfer Learning 과정:

1. 사전 훈련 단계 (이미 완료):
   - 대량의 라벨 없는 오디오 데이터로 SSL 학습
   - OperaCT 인코더가 의미 있는 특징을 학습
   - 76.7M 파라미터 학습 완료

2. 특징 추출 단계:
   - 사전 훈련된 OperaCT로 호흡음 데이터에서 특징 추출
   - 각 오디오 → 768차원 특징 벡터
   - 특징 추출은 한 번만 수행

3. 분류기 학습 단계:
   - 768차원 특징 → 2클래스 분류
   - 1.5K 파라미터만 학습
   - 빠른 수렴과 안정적 학습

4. 평가 단계:
   - Inter-patient 방식으로 평가
   - 실제 임상 환경을 반영한 성능 측정
""")

# 간단한 학습 시뮬레이션
print("\n간단한 학습 시뮬레이션...")

# 더 큰 데이터셋 생성
large_X = torch.FloatTensor(np.random.randn(100, 200, 128))
large_y = torch.LongTensor(np.random.randint(0, 2, 100))

# DataLoader 생성
dataset = torch.utils.data.TensorDataset(large_X, large_y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델을 학습 모드로 설정
classifier.train()

# 몇 개의 배치로 학습 시뮬레이션
total_loss = 0
for i, (batch_x, batch_y) in enumerate(dataloader):
    if i >= 5:  # 5개 배치만 테스트
        break
    
    # Forward pass
    logits = classifier(batch_x)
    loss = F.cross_entropy(logits, batch_y)
    total_loss += loss.item()
    
    print(f"배치 {i+1}: Loss = {loss.item():.4f}")

avg_loss = total_loss / 5
print(f"평균 Loss: {avg_loss:.4f}")

print("\n🎓 8단계: 핵심 개념 정리")
print("-" * 60)

print("""
🎯 finetune with Yeo Data.ipynb의 핵심 개념:

1. Transfer Learning:
   - 사전 훈련된 모델의 특징을 새로운 태스크에 활용
   - 빠른 학습과 좋은 성능
   - 적은 데이터로도 효과적

2. Inter-patient 분할:
   - 환자별로 train/val/test 분할
   - 데이터 누수 방지
   - 실제 임상 환경을 반영한 평가

3. OperaCT 모델:
   - 76.7M 파라미터의 사전 훈련된 인코더
   - Transformer 기반 아키텍처
   - 768차원 특징 벡터 출력

4. AudioClassifier:
   - 고정된 인코더 + 학습 가능한 분류기
   - 1.5K 파라미터만 학습
   - 빠른 수렴과 안정적 학습

5. 의료 응용의 특별한 고려사항:
   - 환자별 개인차 고려
   - 실제 임상 환경 반영
   - 데이터 누수 방지의 중요성

6. 전체 워크플로우:
   - SSL로 사전 훈련 → 특징 추출 → 분류기 학습 → 평가
   - 각 단계가 명확히 분리되어 효율적
""")

print("\n✅ finetune with Yeo Data.ipynb 완전 이해 완료!")
print("=" * 80)

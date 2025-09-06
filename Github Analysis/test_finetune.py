#!/usr/bin/env python3
"""
Finetune Notebook 테스트
- Transfer Learning 구현 테스트
"""

print("🎯 Finetune Notebook 테스트 시작!")
print("=" * 50)

# 라이브러리 import
import numpy as np
import os
import random
import math
import time
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# 우리가 만든 모듈들 import
from src.util import random_crop, random_mask, random_multiply, crop_first
from src.model.models_eval import AudioClassifier, AudioClassifierCLAP, AudioClassifierAudioMAE
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model

print("✅ 모든 라이브러리 import 완료!")

# process_mydata_interpatient 함수 정의
print("\n🔧 1단계: 데이터 처리 함수 정의")
print("-" * 40)

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
    
    # 4) 환자별 분할
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

print("✅ 데이터 처리 함수 정의 완료!")

# 테스트용 데이터 생성
print("\n🧪 2단계: 테스트용 데이터 생성")
print("-" * 40)

# 더미 데이터 디렉토리 생성
test_data_dir = "test_data/Yeo"
os.makedirs(f"{test_data_dir}/normal", exist_ok=True)
os.makedirs(f"{test_data_dir}/abnormal", exist_ok=True)

# 더미 오디오 파일 생성
def create_dummy_audio_file(filepath, duration=1.0, sample_rate=16000):
    """더미 오디오 파일 생성"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 정상: 저주파수 신호
    if "normal" in filepath:
        audio = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    else:
        # 비정상: 고주파수 신호
        audio = np.sin(2 * np.pi * 500 * t) + 0.1 * np.random.randn(len(t))
    
    # WAV 파일로 저장
    import soundfile as sf
    sf.write(filepath, audio, sample_rate)

# 더미 파일들 생성
for i in range(5):
    create_dummy_audio_file(f"{test_data_dir}/normal/{i+1}.wav")
    create_dummy_audio_file(f"{test_data_dir}/abnormal/WEBSS-00{i+1} T1.wav")

print("✅ 더미 데이터 생성 완료!")

# 데이터 처리 함수 테스트
print("\n🔍 3단계: 데이터 처리 함수 테스트")
print("-" * 40)

try:
    files, labels, patient_ids, splits = process_mydata_interpatient(
        data_dir=test_data_dir, 
        feature_dir="test_feature/yeo_eval/"
    )
    print("✅ 데이터 처리 성공!")
except Exception as e:
    print(f"❌ 데이터 처리 실패: {e}")
    print("더미 데이터로 계속 진행...")
    
    # 더미 데이터 생성
    files = [f"dummy_{i}.wav" for i in range(10)]
    labels = [0, 1] * 5
    patient_ids = [f"patient_{i}" for i in range(10)]
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2

# 모델 테스트
print("\n🔍 4단계: Transfer Learning 모델 테스트")
print("-" * 40)

# 사전 훈련된 모델 로드
try:
    pretrained_model = initialize_pretrained_model("operaCT")
    print("✅ 사전 훈련된 모델 로드 성공!")
    print(f"모델 파라미터 수: {sum(p.numel() for p in pretrained_model.parameters())}")
except Exception as e:
    print(f"❌ 사전 훈련된 모델 로드 실패: {e}")
    print("간단한 인코더로 대체...")
    
    # 간단한 인코더 생성
    from src.model.models_cola import SimpleEncoder
    pretrained_model = SimpleEncoder(input_dim=128, hidden_dim=512, output_dim=768)

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

print(f"AudioClassifier 파라미터 수: {sum(p.numel() for p in classifier.parameters())}")

# 더미 데이터로 테스트
print("\n🔍 5단계: 더미 데이터로 모델 테스트")
print("-" * 40)

# 더미 스펙트로그램 데이터 생성
dummy_spectrograms = []
dummy_labels = []

for i in range(20):
    # 더미 스펙트로그램 생성
    spec = np.random.randn(200, 128)  # [time_frames, mel_bins]
    dummy_spectrograms.append(spec)
    dummy_labels.append(i % 2)  # 0 또는 1

# PyTorch 텐서로 변환
X = torch.FloatTensor(np.array(dummy_spectrograms))
y = torch.LongTensor(dummy_labels)

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

print("\n🎯 핵심 개념 정리")
print("-" * 40)
print("1. process_mydata_interpatient: 환자별로 데이터 분할 (데이터 누수 방지)")
print("2. initialize_pretrained_model: 사전 훈련된 모델 로드")
print("3. AudioClassifier: 고정된 인코더 + 학습 가능한 분류기")
print("4. Transfer Learning: 사전 훈련된 특징을 새로운 태스크에 활용")
print("5. Inter-patient 분할: 실제 임상 환경을 반영한 평가 방식")

print("\n✅ Finetune Notebook 테스트 완료!")
print("=" * 50)

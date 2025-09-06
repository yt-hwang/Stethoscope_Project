#!/usr/bin/env python3
"""
Audio Preprocessing Notebook 실행 테스트
- preprocessing_ssl.ipynb의 내용을 단계별로 실행하며 이해
"""

print("🎵 Audio Preprocessing Notebook 실행 시작!")
print("=" * 50)

# Cell 2: 라이브러리 import
print("\n📦 1단계: 라이브러리 import")
print("-" * 30)

import os
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter
import pandas as pd
import librosa
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import time

print("✅ 모든 라이브러리 import 완료!")
print(f"PyTorch 버전: {torch.__version__}")
print(f"Librosa 버전: {librosa.__version__}")

# Cell 3: 데이터 증강 함수들 정의
print("\n🔧 2단계: 데이터 증강 함수들 정의")
print("-" * 30)

def crop_first(data, crop_size=128):
    """첫 번째 부분을 크롭"""
    return data[0: crop_size, :]

def random_crop(data, crop_size=128):
    """랜덤 위치에서 크롭"""
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start: (start + crop_size), :]

def random_mask(data, rate_start=0.1, rate_seq=0.2):
    """랜덤하게 일부 구간을 마스킹"""
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if random.random() < rate_start or (prev_zero and random.random() < rate_seq):
            prev_zero = True
            new_data[i, :] = mean
        else:
            prev_zero = False
    return new_data

def random_multiply(data):
    """랜덤한 배율로 곱하기"""
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)

print("✅ 데이터 증강 함수들 정의 완료!")
print("- crop_first: 첫 번째 부분 크롭")
print("- random_crop: 랜덤 위치 크롭")
print("- random_mask: 랜덤 마스킹")
print("- random_multiply: 랜덤 곱셈")

# 테스트용 더미 데이터 생성
print("\n🧪 3단계: 테스트용 더미 데이터 생성")
print("-" * 30)

# 더미 오디오 데이터 생성 (200 프레임, 128 멜 빈)
dummy_audio = np.random.randn(200, 128)
print(f"더미 오디오 데이터 shape: {dummy_audio.shape}")

# 각 함수 테스트
print("\n🔍 4단계: 각 함수 테스트")
print("-" * 30)

# 1. crop_first 테스트
cropped_first = crop_first(dummy_audio, crop_size=128)
print(f"crop_first 결과 shape: {cropped_first.shape}")

# 2. random_crop 테스트
cropped_random = random_crop(dummy_audio, crop_size=128)
print(f"random_crop 결과 shape: {cropped_random.shape}")

# 3. random_mask 테스트
masked_audio = random_mask(dummy_audio)
print(f"random_mask 결과 shape: {masked_audio.shape}")
print(f"마스킹된 픽셀 수: {np.sum(masked_audio == masked_audio.mean())}")

# 4. random_multiply 테스트
multiplied_audio = random_multiply(dummy_audio)
print(f"random_multiply 결과 shape: {multiplied_audio.shape}")
print(f"곱셈 배율 범위: {multiplied_audio.min():.3f} ~ {multiplied_audio.max():.3f}")

print("\n✅ 모든 함수 테스트 완료!")
print("=" * 50)

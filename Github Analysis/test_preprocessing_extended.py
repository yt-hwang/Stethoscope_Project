#!/usr/bin/env python3
"""
Audio Preprocessing Notebook 확장 테스트
- 더 많은 함수들을 테스트하고 이해
"""

print("🎵 Audio Preprocessing 확장 테스트 시작!")
print("=" * 60)

# 라이브러리 import
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

# 이전 함수들 재정의
def crop_first(data, crop_size=128):
    return data[0: crop_size, :]

def random_crop(data, crop_size=128):
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start: (start + crop_size), :]

def random_mask(data, rate_start=0.1, rate_seq=0.2):
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
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)

# 새로운 함수들 추가
print("\n🔧 1단계: Bandpass 필터 함수들 정의")
print("-" * 40)

def _butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass 필터 계수 계산"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Butterworth bandpass 필터 적용"""
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def _slice_data_librosa(start, end, data, sample_rate):
    """시간 구간으로 데이터 슬라이싱"""
    max_ind = len(data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return data[start_ind: end_ind]

print("✅ Bandpass 필터 함수들 정의 완료!")

print("\n🔧 2단계: Mel Spectrogram 함수 정의")
print("-" * 40)

def pre_process_audio_mel_t(audio, sample_rate=16000, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512):
    """
    librosa의 melspectrogram을 구한 뒤 dB scale로 변환하고, [Time x Mel-bin] 형태로 리턴.
    """
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
        print("warning in producing spectrogram! (constant array)")
    return mel_db.T

print("✅ Mel Spectrogram 함수 정의 완료!")

print("\n🧪 3단계: 실제 오디오 데이터로 테스트")
print("-" * 40)

# 실제 오디오 파일이 있는지 확인
audio_files = []
for root, dirs, files in os.walk(".."):
    for file in files:
        if file.endswith('.wav'):
            audio_files.append(os.path.join(root, file))

if audio_files:
    print(f"발견된 오디오 파일 수: {len(audio_files)}")
    test_audio_file = audio_files[0]
    print(f"테스트용 오디오 파일: {test_audio_file}")
    
    try:
        # 오디오 로드
        audio, sr = librosa.load(test_audio_file, sr=16000)
        print(f"오디오 길이: {len(audio)} samples")
        print(f"샘플링 레이트: {sr} Hz")
        print(f"오디오 길이: {len(audio)/sr:.2f} 초")
        
        # Bandpass 필터 적용
        print("\n🔍 Bandpass 필터 테스트")
        filtered_audio = _butter_bandpass_filter(audio, 50, 2000, sr)
        print(f"필터링 후 오디오 길이: {len(filtered_audio)} samples")
        
        # Mel Spectrogram 생성
        print("\n🔍 Mel Spectrogram 생성 테스트")
        mel_spec = pre_process_audio_mel_t(filtered_audio, sample_rate=sr)
        print(f"Mel Spectrogram shape: {mel_spec.shape}")
        print(f"Mel Spectrogram 범위: {mel_spec.min():.3f} ~ {mel_spec.max():.3f}")
        
        # 데이터 증강 테스트
        print("\n🔍 데이터 증강 테스트")
        cropped = random_crop(mel_spec, crop_size=128)
        print(f"Random crop 결과 shape: {cropped.shape}")
        
        masked = random_mask(mel_spec)
        print(f"Random mask 결과 shape: {masked.shape}")
        
        multiplied = random_multiply(mel_spec)
        print(f"Random multiply 결과 shape: {multiplied.shape}")
        
        print("\n✅ 실제 오디오 데이터 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오디오 파일 처리 중 오류: {e}")
        print("더미 데이터로 계속 진행...")
        
        # 더미 데이터로 테스트
        dummy_audio = np.random.randn(16000)  # 1초 길이
        print(f"더미 오디오 길이: {len(dummy_audio)} samples")
        
        # Bandpass 필터 적용
        filtered_audio = _butter_bandpass_filter(dummy_audio, 50, 2000, 16000)
        print(f"필터링 후 오디오 길이: {len(filtered_audio)} samples")
        
        # Mel Spectrogram 생성
        mel_spec = pre_process_audio_mel_t(filtered_audio, sample_rate=16000)
        print(f"Mel Spectrogram shape: {mel_spec.shape}")
        
        # 데이터 증강 테스트
        cropped = random_crop(mel_spec, crop_size=128)
        print(f"Random crop 결과 shape: {cropped.shape}")
        
        masked = random_mask(mel_spec)
        print(f"Random mask 결과 shape: {masked.shape}")
        
        multiplied = random_multiply(mel_spec)
        print(f"Random multiply 결과 shape: {multiplied.shape}")
        
        print("\n✅ 더미 데이터 테스트 완료!")

else:
    print("❌ 오디오 파일을 찾을 수 없습니다.")
    print("더미 데이터로 테스트를 진행합니다...")
    
    # 더미 데이터로 테스트
    dummy_audio = np.random.randn(16000)  # 1초 길이
    print(f"더미 오디오 길이: {len(dummy_audio)} samples")
    
    # Bandpass 필터 적용
    filtered_audio = _butter_bandpass_filter(dummy_audio, 50, 2000, 16000)
    print(f"필터링 후 오디오 길이: {len(filtered_audio)} samples")
    
    # Mel Spectrogram 생성
    mel_spec = pre_process_audio_mel_t(filtered_audio, sample_rate=16000)
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    
    # 데이터 증강 테스트
    cropped = random_crop(mel_spec, crop_size=128)
    print(f"Random crop 결과 shape: {cropped.shape}")
    
    masked = random_mask(mel_spec)
    print(f"Random mask 결과 shape: {masked.shape}")
    
    multiplied = random_multiply(mel_spec)
    print(f"Random multiply 결과 shape: {multiplied.shape}")
    
    print("\n✅ 더미 데이터 테스트 완료!")

print("\n🎯 함수별 목적 정리")
print("-" * 40)
print("1. crop_first: 첫 번째 부분을 크롭 (일관된 시작점)")
print("2. random_crop: 랜덤 위치에서 크롭 (시간적 다양성)")
print("3. random_mask: 랜덤하게 일부 구간을 마스킹 (노이즈 강건성)")
print("4. random_multiply: 랜덤한 배율로 곱하기 (볼륨 변화 강건성)")
print("5. _butter_bandpass_filter: 특정 주파수 대역만 통과 (호흡음 관련 주파수 추출)")
print("6. pre_process_audio_mel_t: Mel Spectrogram 생성 (인간 청각 특성 반영)")

print("\n✅ Audio Preprocessing 확장 테스트 완료!")
print("=" * 60)

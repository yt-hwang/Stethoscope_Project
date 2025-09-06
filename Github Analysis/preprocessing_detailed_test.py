#!/usr/bin/env python3
"""
preprocessing_ssl.ipynb 상세 테스트 및 설명
"""

print("🔍 preprocessing_ssl.ipynb 상세 분석 및 테스트")
print("=" * 60)

import numpy as np
import matplotlib.pyplot as plt
import random

# 함수들 정의
def crop_first(data, crop_size=128):
    """첫 번째 부분을 크롭"""
    return data[0: crop_size, :]

def random_crop(data, crop_size=128):
    """랜덤 위치에서 크롭"""
    if data.shape[0] <= crop_size:
        return data
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

print("\n📊 1단계: 테스트용 데이터 생성")
print("-" * 40)

# 더미 스펙트로그램 데이터 생성 (200 프레임, 128 멜 빈)
np.random.seed(42)  # 재현 가능한 결과를 위해
original_data = np.random.randn(200, 128)
print(f"원본 데이터 shape: {original_data.shape}")
print(f"원본 데이터 범위: {original_data.min():.3f} ~ {original_data.max():.3f}")

print("\n🔧 2단계: 각 함수별 상세 테스트")
print("-" * 40)

# 1. crop_first 테스트
print("\n1️⃣ crop_first 함수 테스트:")
cropped_first = crop_first(original_data, crop_size=128)
print(f"   원본: {original_data.shape} → 크롭 후: {cropped_first.shape}")
print(f"   첫 5개 프레임이 같은가? {np.array_equal(original_data[:5], cropped_first[:5])}")

# 2. random_crop 테스트
print("\n2️⃣ random_crop 함수 테스트:")
cropped_random = random_crop(original_data, crop_size=128)
print(f"   원본: {original_data.shape} → 크롭 후: {cropped_random.shape}")
print(f"   랜덤 크롭이므로 매번 다른 결과가 나옵니다")

# 3. random_mask 테스트
print("\n3️⃣ random_mask 함수 테스트:")
masked_data = random_mask(original_data)
print(f"   원본: {original_data.shape} → 마스킹 후: {masked_data.shape}")
print(f"   마스킹된 픽셀 수: {np.sum(masked_data == masked_data.mean())}")
print(f"   마스킹 비율: {np.sum(masked_data == masked_data.mean()) / masked_data.size * 100:.1f}%")

# 4. random_multiply 테스트
print("\n4️⃣ random_multiply 함수 테스트:")
multiplied_data = random_multiply(original_data)
print(f"   원본: {original_data.shape} → 곱셈 후: {multiplied_data.shape}")
print(f"   곱셈 배율: {multiplied_data.mean() / original_data.mean():.3f}")

print("\n🎯 3단계: 함수들의 목적과 효과")
print("-" * 40)

print("""
📋 각 함수의 목적과 효과:

1. crop_first():
   - 목적: 일관된 시작점에서 데이터 추출
   - 효과: 모델이 항상 같은 위치에서 시작하도록 함
   - 사용 시기: 호흡 주기의 시작 부분이 중요할 때

2. random_crop():
   - 목적: 시간적 다양성 확보
   - 효과: 모델이 다양한 시간 구간에서 특징을 학습
   - 사용 시기: 호흡 주기의 다양한 구간을 학습하고 싶을 때

3. random_mask():
   - 목적: 노이즈에 강한 모델 학습
   - 효과: 부분적 정보 손실에도 강건한 특징 학습
   - 사용 시기: 실제 환경의 노이즈를 모사하고 싶을 때

4. random_multiply():
   - 목적: 볼륨 변화에 강한 모델 학습
   - 효과: 다양한 녹음 볼륨에서도 일관된 성능
   - 사용 시기: 녹음 조건이 다양할 때
""")

print("\n🔬 4단계: 실제 호흡음 데이터로 테스트")
print("-" * 40)

# 실제 오디오 파일이 있는지 확인
import os
import librosa

audio_files = []
for root, dirs, files in os.walk(".."):
    for file in files:
        if file.endswith('.wav'):
            audio_files.append(os.path.join(root, file))

if audio_files:
    print(f"발견된 오디오 파일: {len(audio_files)}개")
    test_file = audio_files[0]
    print(f"테스트 파일: {test_file}")
    
    try:
        # 오디오 로드
        audio, sr = librosa.load(test_file, sr=16000)
        print(f"오디오 길이: {len(audio)} samples ({len(audio)/sr:.2f}초)")
        
        # Mel Spectrogram 생성
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, n_fft=1024, hop_length=512
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = mel_spec.T  # [time, mel_bins]
        
        print(f"Mel Spectrogram shape: {mel_spec.shape}")
        
        # 데이터 증강 테스트
        print("\n실제 데이터로 증강 테스트:")
        
        # random_crop 테스트
        cropped = random_crop(mel_spec, crop_size=128)
        print(f"  random_crop: {mel_spec.shape} → {cropped.shape}")
        
        # random_mask 테스트
        masked = random_mask(mel_spec)
        mask_ratio = np.sum(masked == masked.mean()) / masked.size * 100
        print(f"  random_mask: 마스킹 비율 {mask_ratio:.1f}%")
        
        # random_multiply 테스트
        multiplied = random_multiply(mel_spec)
        multiply_ratio = multiplied.mean() / mel_spec.mean()
        print(f"  random_multiply: 배율 {multiply_ratio:.3f}")
        
        print("✅ 실제 데이터 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오디오 파일 처리 실패: {e}")
        print("더미 데이터로 계속 진행...")
else:
    print("❌ 오디오 파일을 찾을 수 없습니다.")
    print("더미 데이터로 계속 진행...")

print("\n🎓 5단계: 핵심 개념 정리")
print("-" * 40)

print("""
🎯 preprocessing_ssl.ipynb의 핵심 개념:

1. 데이터 증강의 중요성:
   - 적은 데이터로도 다양한 변형 생성
   - 모델의 일반화 성능 향상
   - 실제 환경의 다양한 조건 모사

2. 호흡음 분석에서의 특별한 고려사항:
   - 시간적 변형: 호흡 주기의 다양한 구간
   - 노이즈 강건성: 의료 환경의 다양한 노이즈
   - 볼륨 변화: 환자별, 장비별 녹음 조건 차이

3. 각 함수의 전략적 사용:
   - crop_first: 일관된 시작점이 중요할 때
   - random_crop: 시간적 다양성이 필요할 때
   - random_mask: 노이즈 강건성이 필요할 때
   - random_multiply: 볼륨 변화에 강해야 할 때
""")

print("\n✅ preprocessing_ssl.ipynb 완전 이해 완료!")
print("=" * 60)

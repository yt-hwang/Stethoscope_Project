#!/usr/bin/env python3
"""
호흡음 분석 데모 - 이론과 실습을 함께하는 튜토리얼
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SR, N_FFT, HOP_LEN, N_MELS, FMIN, FMAX
from src.audio_io import load_audio, pre_emphasis
from src.features import stft_mag_db, logmel, mfcc, wheeze_indicators

def analyze_respiratory_sound(audio_path, title="호흡음 분석"):
    """
    호흡음 파일을 분석하고 시각화하는 함수
    """
    print(f"\n🎵 {title} 분석 시작...")
    print(f"파일: {audio_path}")
    
    # 1. 오디오 로딩
    print("\n1️⃣ 오디오 로딩 및 전처리...")
    y = load_audio(audio_path)
    y_pre = pre_emphasis(y)
    
    print(f"   - 샘플링 레이트: {SR} Hz")
    print(f"   - 길이: {len(y)/SR:.2f}초")
    print(f"   - 샘플 수: {len(y)}")
    
    # 2. STFT 분석
    print("\n2️⃣ STFT (Short-Time Fourier Transform) 분석...")
    S_db = stft_mag_db(y_pre)
    print(f"   - STFT 크기: {S_db.shape}")
    print(f"   - 시간 프레임: {S_db.shape[1]}개")
    print(f"   - 주파수 빈: {S_db.shape[0]}개")
    
    # 3. Mel-spectrogram
    print("\n3️⃣ Mel-spectrogram 분석...")
    M_db = logmel(y_pre)
    print(f"   - Mel-spectrogram 크기: {M_db.shape}")
    
    # 4. MFCC
    print("\n4️⃣ MFCC (Mel-Frequency Cepstral Coefficients) 분석...")
    mfcc_features = mfcc(y_pre, n_mfcc=20)
    print(f"   - MFCC 크기: {mfcc_features.shape}")
    
    # 5. Wheezing 지표
    print("\n5️⃣ Wheezing 지표 계산...")
    indicators = wheeze_indicators(S_db)
    for key, value in indicators.items():
        print(f"   - {key}: 평균 {np.mean(value):.4f}, 표준편차 {np.std(value):.4f}")
    
    # 6. 시각화
    print("\n6️⃣ 시각화 생성...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'{title} - 종합 분석', fontsize=16, fontweight='bold')
    
    # 시간 도메인
    time_axis = np.linspace(0, len(y)/SR, len(y))
    axes[0].plot(time_axis, y)
    axes[0].set_title('시간 도메인 신호')
    axes[0].set_xlabel('시간 (초)')
    axes[0].set_ylabel('진폭')
    axes[0].grid(True)
    
    # STFT Spectrogram
    time_stft = np.linspace(0, len(y)/SR, S_db.shape[1])
    freq_stft = np.linspace(0, SR/2, S_db.shape[0])
    im1 = axes[1].pcolormesh(time_stft, freq_stft, S_db, shading='gouraud', cmap='viridis')
    axes[1].set_title('STFT Spectrogram (dB)')
    axes[1].set_xlabel('시간 (초)')
    axes[1].set_ylabel('주파수 (Hz)')
    axes[1].set_ylim([0, 4000])  # 호흡음에 중요한 0-4kHz 범위
    plt.colorbar(im1, ax=axes[1], label='dB')
    
    # Mel-spectrogram
    time_mel = np.linspace(0, len(y)/SR, M_db.shape[1])
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    im2 = axes[2].pcolormesh(time_mel, mel_freqs, M_db, shading='gouraud', cmap='viridis')
    axes[2].set_title('Mel-spectrogram (dB)')
    axes[2].set_xlabel('시간 (초)')
    axes[2].set_ylabel('Mel 주파수 (Hz)')
    plt.colorbar(im2, ax=axes[2], label='dB')
    
    # Wheezing 지표들
    time_indicators = np.linspace(0, len(y)/SR, len(indicators['flatness']))
    axes[3].plot(time_indicators, indicators['flatness'], label='Spectral Flatness', alpha=0.7)
    axes[3].plot(time_indicators, indicators['centroid']/1000, label='Spectral Centroid (kHz)', alpha=0.7)
    axes[3].plot(time_indicators, indicators['e_ratio_100_1k__1k_2_5k'], label='Energy Ratio (100-1k)/(1k-2.5k)', alpha=0.7)
    axes[3].set_title('Wheezing 지표들')
    axes[3].set_xlabel('시간 (초)')
    axes[3].set_ylabel('정규화된 값')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    
    # 결과 저장
    output_path = f"analysis_{Path(audio_path).stem}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   - 분석 결과 저장: {output_path}")
    
    return {
        'audio': y,
        'stft': S_db,
        'mel': M_db,
        'mfcc': mfcc_features,
        'indicators': indicators
    }

def compare_sounds(sound_files, labels):
    """
    여러 호흡음 파일을 비교 분석
    """
    print(f"\n🔍 {len(sound_files)}개 호흡음 비교 분석...")
    
    fig, axes = plt.subplots(len(sound_files), 3, figsize=(18, 4*len(sound_files)))
    if len(sound_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (file_path, label) in enumerate(zip(sound_files, labels)):
        print(f"\n분석 중: {label} ({file_path})")
        
        # 오디오 로딩
        y = load_audio(file_path)
        y_pre = pre_emphasis(y)
        
        # 특징 추출
        S_db = stft_mag_db(y_pre)
        M_db = logmel(y_pre)
        indicators = wheeze_indicators(S_db)
        
        # 시간축
        time_axis = np.linspace(0, len(y)/SR, len(y))
        time_stft = np.linspace(0, len(y)/SR, S_db.shape[1])
        time_indicators = np.linspace(0, len(y)/SR, len(indicators['flatness']))
        
        # 시간 도메인
        axes[i, 0].plot(time_axis, y)
        axes[i, 0].set_title(f'{label} - 시간 도메인')
        axes[i, 0].set_xlabel('시간 (초)')
        axes[i, 0].set_ylabel('진폭')
        axes[i, 0].grid(True)
        
        # STFT
        freq_stft = np.linspace(0, SR/2, S_db.shape[0])
        im = axes[i, 1].pcolormesh(time_stft, freq_stft, S_db, shading='gouraud', cmap='viridis')
        axes[i, 1].set_title(f'{label} - STFT Spectrogram')
        axes[i, 1].set_xlabel('시간 (초)')
        axes[i, 1].set_ylabel('주파수 (Hz)')
        axes[i, 1].set_ylim([0, 4000])
        plt.colorbar(im, ax=axes[i, 1], label='dB')
        
        # Wheezing 지표
        axes[i, 2].plot(time_indicators, indicators['flatness'], label='Spectral Flatness', alpha=0.7)
        axes[i, 2].plot(time_indicators, indicators['centroid']/1000, label='Spectral Centroid (kHz)', alpha=0.7)
        axes[i, 2].plot(time_indicators, indicators['e_ratio_100_1k__1k_2_5k'], label='Energy Ratio', alpha=0.7)
        axes[i, 2].set_title(f'{label} - Wheezing 지표')
        axes[i, 2].set_xlabel('시간 (초)')
        axes[i, 2].set_ylabel('정규화된 값')
        axes[i, 2].legend()
        axes[i, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("   - 비교 분석 결과 저장: comparative_analysis.png")

if __name__ == "__main__":
    # 실제 호흡음 파일들 분석
    audio_base_path = Path("../../Audio shared/Hospital sound")
    
    # 분석할 파일들 선택 (다양한 환자에서)
    sound_files = [
        audio_base_path / "WEBSS002_3" / "WEBSS-002 TP 2_60sec.wav",
        audio_base_path / "WEBSS003_6" / "WEBSS-003 TP1_60sec.wav", 
        audio_base_path / "WEBSS004_5" / "WEBSS-004 TP1_60sec.wav"
    ]
    
    labels = ["WEBSS-002 (천식)", "WEBSS-003 (천식)", "WEBSS-004 (천식)"]
    
    # 개별 분석
    for file_path, label in zip(sound_files, labels):
        if file_path.exists():
            analyze_respiratory_sound(str(file_path), label)
        else:
            print(f"⚠️ 파일을 찾을 수 없습니다: {file_path}")
    
    # 비교 분석
    existing_files = [(str(f), l) for f, l in zip(sound_files, labels) if f.exists()]
    if existing_files:
        files, labels = zip(*existing_files)
        compare_sounds(files, labels)
    
    print("\n✅ 분석 완료! 생성된 이미지 파일들을 확인해보세요.")

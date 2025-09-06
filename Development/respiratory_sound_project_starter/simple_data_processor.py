#!/usr/bin/env python3
"""
간단한 호흡음 데이터 처리기 - 실제 파일 구조에 맞춰 수정
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 설정
from src.config import SR, N_FFT, HOP_LEN, N_MELS, FMIN, FMAX
from src.audio_io import load_audio, pre_emphasis
from src.features import stft_mag_db, logmel, mfcc, wheeze_indicators

def process_audio_file(audio_path, label='unknown'):
    """
    단일 오디오 파일 처리
    """
    print(f"Processing: {audio_path}")
    
    try:
        # 1. 오디오 로딩 및 전처리
        y = load_audio(str(audio_path))
        y = pre_emphasis(y)
        
        # 2. 정규화
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y / rms * 0.1
        
        # 3. 특징 추출
        features = {}
        
        # STFT 및 Mel-spectrogram
        S_db = stft_mag_db(y)
        M_db = logmel(y)
        mfcc_features = mfcc(y, n_mfcc=20)
        
        # Wheezing 지표
        indicators = wheeze_indicators(S_db)
        
        # 시간 도메인 특징
        rms_feature = librosa.feature.rms(y=y, hop_length=HOP_LEN)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LEN)[0]
        
        # 특징 벡터 생성
        feature_vector = []
        
        # MFCC 통계 (평균, 표준편차)
        feature_vector.extend(np.mean(mfcc_features, axis=1))  # 20개
        feature_vector.extend(np.std(mfcc_features, axis=1))   # 20개
        
        # Wheezing 지표 통계
        feature_vector.append(np.mean(indicators['flatness']))
        feature_vector.append(np.std(indicators['flatness']))
        feature_vector.append(np.mean(indicators['centroid']))
        feature_vector.append(np.std(indicators['centroid']))
        feature_vector.append(np.mean(indicators['e_ratio_100_1k__1k_2_5k']))
        feature_vector.append(np.std(indicators['e_ratio_100_1k__1k_2_5k']))
        
        # 시간 도메인 특징 통계
        feature_vector.append(np.mean(rms_feature))
        feature_vector.append(np.std(rms_feature))
        feature_vector.append(np.mean(zcr))
        feature_vector.append(np.std(zcr))
        feature_vector.append(np.mean(spectral_centroid))
        feature_vector.append(np.std(spectral_centroid))
        
        return np.array(feature_vector), label
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def create_simple_dataset():
    """
    실제 파일 구조를 기반으로 간단한 데이터셋 생성
    """
    print("🚀 간단한 호흡음 데이터셋 생성 시작")
    
    # 오디오 파일 경로
    audio_base_path = Path("../../Audio shared/Hospital sound")
    
    # 환자별 라벨링 (메타데이터 기반)
    patient_labels = {
        'WEBSS002': 'asthma',  # 천식
        'WEBSS003': 'asthma',  # 천식
        'WEBSS004': 'asthma',  # 천식
        'WEBSS005': 'normal',  # No pull diagnosis
        'WEBSS006': 'asthma',  # 천식
        'WEBSS007': 'asthma'   # 천식
    }
    
    dataset = []
    labels = []
    
    # 각 환자 폴더 처리
    for folder_name in audio_base_path.iterdir():
        if folder_name.is_dir():
            patient_id = folder_name.name.split('_')[0]  # WEBSS002_3 -> WEBSS002
            label = patient_labels.get(patient_id, 'unknown')
            
            print(f"\n📁 Processing {folder_name.name} (Label: {label})")
            
            # 폴더 내 모든 wav 파일 처리
            for audio_file in folder_name.glob("*.wav"):
                feature_vector, file_label = process_audio_file(audio_file, label)
                
                if feature_vector is not None:
                    dataset.append(feature_vector)
                    labels.append(file_label)
                    print(f"  ✅ {audio_file.name}: {len(feature_vector)} features")
                else:
                    print(f"  ❌ Failed: {audio_file.name}")
    
    if dataset:
        X = np.array(dataset)
        y = np.array(labels)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        
        # 클래스 분포
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} samples")
        
        return X, y
    else:
        print("❌ No data processed successfully")
        return None, None

def analyze_features(X, y):
    """
    특징 분석 및 시각화
    """
    print("\n🔍 Feature Analysis")
    
    # 특징 이름들
    feature_names = [
        'mfcc_mean_0', 'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4',
        'mfcc_mean_5', 'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9',
        'mfcc_mean_10', 'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13', 'mfcc_mean_14',
        'mfcc_mean_15', 'mfcc_mean_16', 'mfcc_mean_17', 'mfcc_mean_18', 'mfcc_mean_19',
        'mfcc_std_0', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4',
        'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9',
        'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12', 'mfcc_std_13', 'mfcc_std_14',
        'mfcc_std_15', 'mfcc_std_16', 'mfcc_std_17', 'mfcc_std_18', 'mfcc_std_19',
        'flatness_mean', 'flatness_std', 'centroid_mean', 'centroid_std',
        'energy_ratio_mean', 'energy_ratio_std', 'rms_mean', 'rms_std',
        'zcr_mean', 'zcr_std', 'spectral_centroid_mean', 'spectral_centroid_std'
    ]
    
    # 클래스별 특징 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Feature Distribution by Class', fontsize=16)
    
    # 중요한 특징들 선택
    important_features = [
        ('flatness_mean', 40),  # Wheezing 지표
        ('centroid_mean', 42),  # 주파수 중심
        ('energy_ratio_mean', 44),  # 에너지 비율
        ('spectral_centroid_mean', 50)  # 스펙트럼 중심
    ]
    
    for idx, (feature_name, feature_idx) in enumerate(important_features):
        ax = axes[idx // 2, idx % 2]
        
        # 클래스별 박스플롯
        class_data = []
        class_labels = []
        
        for cls in np.unique(y):
            class_mask = y == cls
            class_data.append(X[class_mask, feature_idx])
            class_labels.append(cls)
        
        ax.boxplot(class_data, labels=class_labels)
        ax.set_title(f'{feature_name}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Feature analysis saved: feature_analysis.png")

def prepare_training_data(X, y, test_size=0.2, random_state=42):
    """
    훈련 데이터 준비
    """
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # 특징 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Training data prepared:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Classes: {label_encoder.classes_}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def main():
    """
    메인 실행 함수
    """
    # 데이터셋 생성
    X, y = create_simple_dataset()
    
    if X is not None and y is not None:
        # 특징 분석
        analyze_features(X, y)
        
        # 훈련 데이터 준비
        X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(X, y)
        
        # 결과 저장
        np.savez('simple_dataset.npz',
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test)
        
        print("💾 Dataset saved: simple_dataset.npz")
        
        # 간단한 분류기 테스트
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n🤖 Testing with Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        print("\n📊 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    else:
        print("❌ Dataset creation failed")

if __name__ == "__main__":
    main()

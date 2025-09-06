#!/usr/bin/env python3
"""
호흡음 데이터 전처리 파이프라인
- 오디오 로딩, 정규화, 노이즈 제거
- 특징 추출 및 라벨링
- 데이터셋 생성
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
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

class RespiratoryDataProcessor:
    """
    호흡음 데이터 전처리 클래스
    """
    
    def __init__(self, sample_rate=SR, frame_length=25, hop_length=10):
        self.sample_rate = sample_rate
        self.frame_length = frame_length  # ms
        self.hop_length = hop_length      # ms
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_audio(self, audio_path, apply_preemphasis=True):
        """
        오디오 파일 로딩 및 기본 전처리
        """
        print(f"Loading: {audio_path}")
        
        # 1. 오디오 로딩
        y = load_audio(audio_path, sr=self.sample_rate)
        
        # 2. Pre-emphasis (고주파수 강조)
        if apply_preemphasis:
            y = pre_emphasis(y)
        
        # 3. 정규화 (RMS 기반)
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y / rms * 0.1  # RMS를 0.1로 정규화
        
        # 4. 짧은 무음 구간 제거 (에너지 기반)
        y = self._remove_silence(y)
        
        return y
    
    def _remove_silence(self, y, frame_length=1024, hop_length=512, threshold=0.01):
        """
        무음 구간 제거
        """
        # 에너지 계산
        energy = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            energy.append(np.mean(frame**2))
        
        energy = np.array(energy)
        
        # 임계값 이하 구간 찾기
        silent_frames = energy < threshold
        
        if not np.any(silent_frames):
            return y
        
        # 연속된 무음 구간의 시작/끝 찾기
        silent_regions = []
        start = None
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start is None:
                start = i
            elif not is_silent and start is not None:
                silent_regions.append((start * hop_length, i * hop_length))
                start = None
        
        # 마지막 무음 구간 처리
        if start is not None:
            silent_regions.append((start * hop_length, len(y)))
        
        # 무음 구간 제거
        if silent_regions:
            # 무음이 아닌 구간들만 추출
            non_silent_parts = []
            last_end = 0
            
            for start, end in silent_regions:
                if start > last_end:
                    non_silent_parts.append(y[last_end:start])
                last_end = end
            
            if last_end < len(y):
                non_silent_parts.append(y[last_end:])
            
            if non_silent_parts:
                y = np.concatenate(non_silent_parts)
        
        return y
    
    def extract_features(self, y, feature_type='all'):
        """
        다양한 특징 추출
        """
        features = {}
        
        if feature_type in ['all', 'spectral']:
            # STFT 기반 특징
            S_db = stft_mag_db(y)
            features['stft_db'] = S_db
            
            # Mel-spectrogram
            M_db = logmel(y)
            features['mel_db'] = M_db
            
            # MFCC
            mfcc_features = mfcc(y, n_mfcc=20)
            features['mfcc'] = mfcc_features
            
            # Wheezing 지표
            indicators = wheeze_indicators(S_db)
            features.update(indicators)
        
        if feature_type in ['all', 'temporal']:
            # 시간 도메인 특징
            features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LEN)[0]
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)[0]
            
            # 스펙트럼 특징
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate, hop_length=HOP_LEN)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate, hop_length=HOP_LEN)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate, hop_length=HOP_LEN)[0]
            
            features['spectral_centroid'] = spectral_centroids
            features['spectral_rolloff'] = spectral_rolloff
            features['spectral_bandwidth'] = spectral_bandwidth
        
        return features
    
    def create_segments(self, y, segment_length_sec=2.0, overlap_ratio=0.5):
        """
        오디오를 고정 길이 세그먼트로 분할
        """
        segment_length = int(segment_length_sec * self.sample_rate)
        hop_length = int(segment_length * (1 - overlap_ratio))
        
        segments = []
        for start in range(0, len(y) - segment_length, hop_length):
            segment = y[start:start + segment_length]
            segments.append(segment)
        
        return segments
    
    def create_dataset_from_metadata(self, metadata_path, audio_base_path):
        """
        메타데이터를 기반으로 데이터셋 생성
        """
        print("Creating dataset from metadata...")
        
        # 메타데이터 로딩
        metadata = pd.read_csv(metadata_path)
        
        dataset = []
        labels = []
        
        for idx, row in metadata.iterrows():
            audio_file = row['Audio File']
            diagnosis = row['Diagnosis']
            
            # NaN 값 체크
            if pd.isna(audio_file):
                continue
                
            # 라벨 결정 (간단한 규칙 기반)
            if pd.isna(diagnosis) or diagnosis == 'No pull diagnosis':
                label = 'normal'
            elif 'Asthma' in str(diagnosis):
                label = 'asthma'
            else:
                label = 'unknown'
            
            # 실제 파일 구조에 맞게 경로 수정
            # 예: WEBSS-002-01.wav -> WEBSS002_3/WEBSS-002 TP 2_60sec.wav
            patient_id = str(audio_file).split('-')[0] + str(audio_file).split('-')[1]
            
            # 환자별 폴더 찾기
            patient_folders = {
                'WEBSS002': 'WEBSS002_3',
                'WEBSS003': 'WEBSS003_6', 
                'WEBSS004': 'WEBSS004_5',
                'WEBSS005': 'WEBSS005_6',
                'WEBSS006': 'WEBSS006_12',
                'WEBSS007': 'WEBSS007_7'
            }
            
            if patient_id in patient_folders:
                folder_name = patient_folders[patient_id]
                # 실제 파일명 패턴에 맞게 변환
                file_num = str(audio_file).split('-')[2].split('.')[0]
                actual_filename = f"WEBSS-{patient_id[4:]} TP{file_num}_60sec.wav"
                audio_path = Path(audio_base_path) / folder_name / actual_filename
            else:
                continue
            
            if audio_path.exists():
                try:
                    # 오디오 전처리
                    y = self.load_and_preprocess_audio(str(audio_path))
                    
                    # 세그먼트 생성
                    segments = self.create_segments(y, segment_length_sec=2.0)
                    
                    for segment in segments:
                        # 특징 추출
                        features = self.extract_features(segment)
                        
                        # 특징 벡터화
                        feature_vector = self._vectorize_features(features)
                        
                        dataset.append(feature_vector)
                        labels.append(label)
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        return np.array(dataset), np.array(labels)
    
    def _vectorize_features(self, features):
        """
        특징들을 하나의 벡터로 변환
        """
        vector = []
        
        # MFCC (20개 계수)
        if 'mfcc' in features:
            mfcc_mean = np.mean(features['mfcc'], axis=1)
            mfcc_std = np.std(features['mfcc'], axis=1)
            vector.extend(mfcc_mean)
            vector.extend(mfcc_std)
        
        # Wheezing 지표들
        for key in ['flatness', 'centroid', 'e_ratio_100_1k__1k_2_5k']:
            if key in features:
                vector.append(np.mean(features[key]))
                vector.append(np.std(features[key]))
        
        # 시간 도메인 특징들
        for key in ['rms', 'zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']:
            if key in features:
                vector.append(np.mean(features[key]))
                vector.append(np.std(features[key]))
        
        return np.array(vector)
    
    def prepare_training_data(self, X, y, test_size=0.2, random_state=42):
        """
        훈련 데이터 준비
        """
        # 라벨 인코딩
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # 특징 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def analyze_dataset_statistics(X, y, feature_names=None):
    """
    데이터셋 통계 분석
    """
    print("\n📊 Dataset Statistics")
    print("=" * 50)
    
    # 클래스 분포
    unique, counts = np.unique(y, return_counts=True)
    print(f"Classes: {unique}")
    print(f"Counts: {counts}")
    print(f"Total samples: {len(y)}")
    
    # 특징 통계
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Mean values: {np.mean(X, axis=0)[:5]}...")  # 처음 5개만
    print(f"Std values: {np.std(X, axis=0)[:5]}...")
    
    # 클래스별 특징 분포
    for class_label in unique:
        class_mask = y == class_label
        class_features = X[class_mask]
        print(f"\nClass {class_label}:")
        print(f"  Samples: {np.sum(class_mask)}")
        print(f"  Mean feature values: {np.mean(class_features, axis=0)[:3]}...")

def main():
    """
    메인 실행 함수
    """
    print("🚀 호흡음 데이터 전처리 파이프라인 시작")
    
    # 데이터 프로세서 초기화
    processor = RespiratoryDataProcessor()
    
    # 메타데이터 경로
    metadata_path = "../../Audio shared/Sheet 1-Tabular_asthma_data.csv"
    audio_base_path = "../../Audio shared/Hospital sound"
    
    # 데이터셋 생성
    X, y = processor.create_dataset_from_metadata(metadata_path, audio_base_path)
    
    if len(X) > 0:
        # 데이터셋 통계
        analyze_dataset_statistics(X, y)
        
        # 훈련 데이터 준비
        X_train, X_test, y_train, y_test = processor.prepare_training_data(X, y)
        
        print(f"\n✅ 전처리 완료!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Classes: {processor.label_encoder.classes_}")
        
        # 결과 저장
        np.savez('preprocessed_dataset.npz',
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                feature_names=None)  # TODO: 실제 특징 이름들 추가
        
        print("💾 전처리된 데이터셋 저장: preprocessed_dataset.npz")
        
    else:
        print("❌ 데이터셋 생성 실패")

if __name__ == "__main__":
    main()

"""
OPERA 피처 추출 with 레이블 매핑
기존 3-extract_features.py의 구조를 활용하되 피처 추출은 OPERA 사용
"""
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import logging

# OPERA 모듈 임포트
sys.path.append(os.path.expanduser("~/OPERA"))
from src.benchmark.model_util import extract_opera_feature

# 로깅 설정 (기존 코드 스타일 유지)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OPERAFeatureExtractor:
    def __init__(self, segment_dir, output_dir="features", batch_size=16):
        self.segment_dir = segment_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """디렉토리 구조 생성 (기존 코드 스타일)"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.output_dir}")
    
    def get_segments_and_labels(self):
        """세그먼트와 레이블 쌍 추출 (.wav.lab 또는 .lab)"""
        wav_files = sorted(glob.glob(os.path.join(self.segment_dir, "*.wav")))
        segments_data = []
        
        for wav_path in wav_files:
            base = os.path.splitext(wav_path)[0]  # .../segment_0001.wav → .../segment_0001
            # 우선순위: .wav.lab → .lab
            lab_path = base + ".wav.lab"
            if not os.path.exists(lab_path):
                lab_path = base + ".lab"
            
            try:
                with open(lab_path, 'r') as f:
                    label = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Label file not found: {lab_path}")
                label = 'unknown'
            except Exception as e:
                logger.error(f"Error reading label file {lab_path}: {e}")
                label = 'unknown'
            
            segments_data.append({
                'wav_path': wav_path,
                'filename': os.path.basename(wav_path),
                'label': label
            })
        
        logger.info(f"Found {len(segments_data)} segment-label pairs")
        return segments_data

    
    def process_files_in_batches(self, segments_data):
        """배치 단위로 OPERA 피처 추출 (기존 구조 활용)"""
        all_features = []
        failed_files = []
        
        # 배치로 처리
        for i in tqdm(range(0, len(segments_data), self.batch_size), 
                     desc="Extracting OPERA features"):
            batch_data = segments_data[i:i+self.batch_size]
            batch_paths = [item['wav_path'] for item in batch_data]
            
            try:
                # OPERA 피처 추출
                batch_features = extract_opera_feature(
                    batch_paths,
                    pretrain="operaCT",
                    input_sec=2,
                    dim=768
                )
                all_features.extend(batch_features)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # 개별 파일로 재시도
                for item in batch_data:
                    try:
                        feature = extract_opera_feature(
                            [item['wav_path']],
                            pretrain="operaCT",
                            input_sec=2,
                            dim=768
                        )
                        all_features.extend(feature)
                    except Exception as e2:
                        logger.error(f"Individual file failed {item['filename']}: {e2}")
                        # 실패한 파일은 NaN으로 처리
                        all_features.append([np.nan] * 768)
                        failed_files.append(item['filename'])
        
        logger.info(f"Feature extraction completed. Failed files: {len(failed_files)}")
        return np.array(all_features), failed_files
    
    def save_features(self, features, segments_data, failed_files):
        """피처와 메타데이터 저장 (기존 구조 활용)"""
        # NumPy 배열로 저장
        features_file = os.path.join(self.output_dir, "opera_features.npy")
        np.save(features_file, features)
        
        # DataFrame으로 저장
        df = pd.DataFrame(features)
        df['filename'] = [item['filename'] for item in segments_data]
        df['label'] = [item['label'] for item in segments_data]
        df['extraction_success'] = [fname not in failed_files for fname in df['filename']]
        
        csv_file = os.path.join(self.output_dir, "opera_features.csv")
        df.to_csv(csv_file, index=False)
        
        # 요약 정보 저장
        summary = {
            'total_files': len(segments_data),
            'successful_extractions': len(segments_data) - len(failed_files),
            'failed_extractions': len(failed_files),
            'feature_dimension': 768,
            'unique_labels': df['label'].unique().tolist()
        }
        
        summary_file = os.path.join(self.output_dir, "extraction_summary.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Features saved: {features_file}, {csv_file}")
        logger.info(f"Summary: {summary}")
        
        return df

def main():
    """메인 실행 함수"""
    # 설정 - 실제 데이터 경로에 맞게 수정
    segment_dir = "data/audio/segments_2000ms"  # 변경된 경로
    output_dir = "features"
    batch_size = 8
    
    # 나머지는 동일...
  # 세그먼트 오디오 폴더
    output_dir = "features"
    batch_size = 8  # GPU 메모리에 따라 조정
    
    # 피처 추출기 초기화
    extractor = OPERAFeatureExtractor(segment_dir, output_dir, batch_size)
    
    # 1. 세그먼트-레이블 쌍 수집
    segments_data = extractor.get_segments_and_labels()
    
    if not segments_data:
        logger.error("No segment files found!")
        return
    
    # 2. OPERA 피처 추출
    features, failed_files = extractor.process_files_in_batches(segments_data)
    
    # 3. 결과 저장
    df = extractor.save_features(features, segments_data, failed_files)
    
    print("=== OPERA 피처 추출 완료 ===")
    print(f"총 파일: {len(segments_data)}")
    print(f"성공: {len(segments_data) - len(failed_files)}")
    print(f"실패: {len(failed_files)}")
    print(f"피처 shape: {features.shape}")
    print(f"결과 저장: {output_dir}/")

if __name__ == "__main__":
    main()

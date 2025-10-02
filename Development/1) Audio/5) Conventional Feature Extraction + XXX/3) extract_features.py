# extract_features_enhanced_fixed.py - FIXED VERSION
import os
import numpy as np
import librosa
from glob import glob
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

SEGMENTS_DIR = 'D:\\Stethoscope_Project\\Development\\1) Audio\\0) Data\\segments_2000ms'
FEATURES_DIR = 'D:\\Stethoscope_Project\\Development\\1) Audio\\5) Conventional Feature Extraction + XXX\\enhanced_features'
TARGET_SR = 16000

os.makedirs(FEATURES_DIR, exist_ok=True)

def extract_enhanced_features(audio_path, debug=False):
    """Extract comprehensive audio features with fixed dimensions"""
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        if len(y) == 0:
            return None
        
        features = []
        
        # MFCC features (26: 13 mean + 13 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))  # 13 features
        features.extend(np.std(mfcc, axis=1))   # 13 features
        
        # Mel spectrogram features (13)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))  # 13 features
        
        # Spectral features (6)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        features.append(np.mean(spectral_bandwidth))
        features.append(np.mean(spectral_rolloff))
        features.append(np.mean(zero_crossing))
        features.append(np.std(zero_crossing))
        
        # Chroma features (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))  # 12 features
        
        # Additional spectral features (4)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1))  # 7 features
        
        # RMS Energy (1)
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        
        # Convert to numpy array and handle any remaining issues
        features = np.array(features, dtype=np.float32)
        
        # Replace any NaN or inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if debug:
            print(f"Feature vector length: {len(features)}")
            print(f"Feature statistics: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def test_feature_consistency(segments_dir, num_test=5):
    """Test if feature extraction produces consistent dimensions"""
    wav_files = glob(os.path.join(segments_dir, "*.wav"))[:num_test]
    
    print(f"Testing feature consistency on {len(wav_files)} files...")
    
    feature_lengths = []
    for wav_path in wav_files:
        features = extract_enhanced_features(wav_path, debug=True)
        if features is not None:
            feature_lengths.append(len(features))
            print(f"File: {os.path.basename(wav_path)}, Features: {len(features)}")
    
    if feature_lengths:
        unique_lengths = set(feature_lengths)
        if len(unique_lengths) == 1:
            print(f"✅ All features have consistent length: {feature_lengths[0]}")
            return feature_lengths[0]
        else:
            print(f"❌ Inconsistent feature lengths: {unique_lengths}")
            return None
    else:
        print("❌ No features extracted")
        return None

def main():
    # First, test feature consistency
    expected_length = test_feature_consistency(SEGMENTS_DIR)
    if expected_length is None:
        print("❌ Feature extraction test failed. Please check audio files.")
        return
    
    print(f"\n✅ Expected feature length: {expected_length}")
    print("Proceeding with full feature extraction...\n")
    
    # Get all WAV files
    wav_files = glob(os.path.join(SEGMENTS_DIR, "*.wav"))
    
    features_list = []
    labels_list = []
    filenames_list = []
    errors = 0
    
    print(f"Extracting enhanced features from {len(wav_files)} files...")
    
    for i, wav_path in enumerate(wav_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(wav_files)}... (Errors: {errors})")
        
        # Get label
        lab_path = wav_path + '.lab'
        if os.path.exists(lab_path):
            try:
                with open(lab_path, 'r') as f:
                    label = f.read().strip()
            except:
                errors += 1
                continue
        else:
            errors += 1
            continue
        
        # Extract enhanced features
        features = extract_enhanced_features(wav_path)
        if features is not None and len(features) == expected_length:
            features_list.append(features)
            labels_list.append(label)
            filenames_list.append(os.path.basename(wav_path))
        else:
            errors += 1
            if features is not None:
                print(f"❌ Wrong feature length for {os.path.basename(wav_path)}: {len(features)} != {expected_length}")
    
    print(f"\nExtraction complete!")
    print(f"Successfully processed: {len(features_list)}")
    print(f"Errors: {errors}")
    
    if len(features_list) == 0:
        print("❌ No features extracted successfully!")
        return
    
    # Convert to numpy array
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    filenames_array = np.array(filenames_list)
    
    print(f"Final feature array shape: {features_array.shape}")
    
    # Save enhanced features
    data = {
        'features': features_array,
        'labels': labels_array,
        'filenames': filenames_array,
        'feature_type': 'enhanced_conventional',
        'n_features': features_array.shape[1],
        'feature_names': [
            'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5', 
            'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10',
            'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13',
            'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5',
            'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10',
            'mfcc_std_11', 'mfcc_std_12', 'mfcc_std_13',
            'mel_1', 'mel_2', 'mel_3', 'mel_4', 'mel_5', 'mel_6', 'mel_7',
            'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12', 'mel_13',
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_bandwidth_mean',
            'spectral_rolloff_mean', 'zero_crossing_mean', 'zero_crossing_std',
            'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6',
            'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12',
            'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3', 
            'spectral_contrast_4', 'spectral_contrast_5', 'spectral_contrast_6', 'spectral_contrast_7',
            'rms_energy'
        ]
    }
    
    output_path = os.path.join(FEATURES_DIR, 'enhanced_features.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved {len(features_list)} enhanced feature vectors")
    print(f"Feature dimensions: {features_array.shape[1]}")
    
    # Print class distribution
    class_counts = Counter(labels_list)
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {features_array.mean():.4f}")
    print(f"  Std: {features_array.std():.4f}")
    print(f"  Min: {features_array.min():.4f}")
    print(f"  Max: {features_array.max():.4f}")
    
    print("\n🎉 Enhanced feature extraction complete!")
    print(f"Expected improvement: ~50-60% test accuracy (up from ~37%)")

if __name__ == "__main__":
    main()

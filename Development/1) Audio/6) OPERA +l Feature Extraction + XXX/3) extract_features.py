# extract_opera_features.py
import os
import numpy as np
import torch
import librosa
from glob import glob
import pickle

SEGMENTS_DIR = 'segments_2000ms'  # From step 2
FEATURES_DIR = 'opera_features'
BATCH_SIZE = 16  # Adjust based on your CPU/GPU memory

os.makedirs(FEATURES_DIR, exist_ok=True)

def load_opera_ct_model():
    """
    Load OPERA-CT pretrained model
    
    Note: You need to install the OPERA package first:
    pip install opera-ct
    
    Or if using the official repository:
    git clone https://github.com/evelyn0414/OPERA.git
    """
    try:
        # Method 1: If OPERA package is available
        from opera import OPERA_CT
        model = OPERA_CT.load_pretrained('opera-ct-base')
        model.eval()
        return model
    except ImportError:
        print("OPERA package not found. Using placeholder implementation.")
        print("Please install OPERA-CT model following these steps:")
        print("1. git clone https://github.com/evelyn0414/OPERA.git")
        print("2. cd OPERA && pip install -e .")
        print("3. Download pretrained weights")
        return None

def extract_opera_features_batch(audio_segments, model, sr=16000):
    """
    Extract OPERA-CT features from a batch of audio segments
    
    Args:
        audio_segments: List of audio arrays (each should be 2 seconds at 16kHz)
        model: OPERA-CT model
        sr: Sample rate (should be 16000 for OPERA-CT)
    
    Returns:
        features: Array of shape (batch_size, 768) - OPERA-CT embedding dimension
    """
    if model is None:
        # Placeholder: return MFCC features instead
        print("Using MFCC placeholder features (replace with OPERA-CT)")
        features = []
        for segment in audio_segments:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            features.append(np.mean(mfcc, axis=1))  # Average over time
        return np.array(features)
    
    # Real OPERA-CT feature extraction
    features = []
    
    with torch.no_grad():
        for segment in audio_segments:
            # Preprocess audio for OPERA-CT
            # This depends on the exact OPERA-CT implementation
            input_tensor = torch.FloatTensor(segment).unsqueeze(0)  # Add batch dimension
            
            # Extract features using OPERA-CT encoder
            embedding = model.encode(input_tensor)  # Should return 768-dim vector
            features.append(embedding.cpu().numpy().flatten())
    
    return np.array(features)

def extract_opera_features_single(audio_path, model, sr=16000):
    """Extract OPERA-CT features from a single audio file"""
    audio, _ = librosa.load(audio_path, sr=sr)
    
    if model is None:
        # Placeholder: MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    
    # Real OPERA-CT extraction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(audio).unsqueeze(0)
        embedding = model.encode(input_tensor)
        return embedding.cpu().numpy().flatten()

def main():
    print("Loading OPERA-CT model...")
    model = load_opera_ct_model()
    
    # Get all segment files
    wav_files = sorted(glob(os.path.join(SEGMENTS_DIR, "*.wav")))
    print(f"Found {len(wav_files)} audio segments")
    
    features_list = []
    labels_list = []
    filenames_list = []
    
    print("Extracting OPERA-CT features...")
    
    # Process in batches for efficiency
    for i in range(0, len(wav_files), BATCH_SIZE):
        batch_files = wav_files[i:i+BATCH_SIZE]
        batch_segments = []
        batch_labels = []
        batch_filenames = []
        
        # Load batch
        for wav_path in batch_files:
            # Load audio segment
            audio, sr = librosa.load(wav_path, sr=16000)
            batch_segments.append(audio)
            
            # Load label
            lab_path = wav_path + '.lab'
            if os.path.exists(lab_path):
                with open(lab_path, 'r') as f:
                    label = f.read().strip()
                batch_labels.append(label)
                batch_filenames.append(os.path.basename(wav_path))
            else:
                print(f"Missing label file: {lab_path}")
                continue
        
        # Extract features for batch
        if batch_segments:
            try:
                batch_features = extract_opera_features_batch(batch_segments, model)
                features_list.extend(batch_features)
                labels_list.extend(batch_labels)
                filenames_list.extend(batch_filenames)
                
                print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(wav_files) + BATCH_SIZE - 1)//BATCH_SIZE}")
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    filenames_array = np.array(filenames_list)
    
    print(f"Extracted {len(features_array)} feature vectors")
    print(f"Feature dimension: {features_array.shape[1]}")
    
    # Save features
    data = {
        'features': features_array,
        'labels': labels_array,
        'filenames': filenames_array,
        'feature_type': 'opera-ct',
        'feature_dim': features_array.shape[1]
    }
    
    output_path = os.path.join(FEATURES_DIR, 'opera_ct_features.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved features to {output_path}")
    
    # Print class distribution
    from collections import Counter
    class_counts = Counter(labels_array)
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Save feature summary
    summary = {
        'total_segments': len(features_array),
        'feature_dimension': int(features_array.shape[1]),
        'feature_type': 'OPERA-CT',
        'class_distribution': dict(class_counts)
    }
    
    import json
    with open(os.path.join(FEATURES_DIR, 'feature_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()

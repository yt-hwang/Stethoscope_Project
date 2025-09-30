# OPERA-CT Input Requirements & Pipeline Compatibility

## Discovered Technical Specifications

### Core Requirements
- **Input Processing**: Audio → Spectrograms → 4×4 patches
- **Patch Size**: 4×4 pixels from spectrogram representations
- **Feature Output**: 768-dimensional embeddings
- **Architecture**: Hierarchical Token-Semantic Audio Transformer
- **Training Method**: Random cropping of spectrograms for variable length audio

### Critical Pipeline Implications

## 1. **Spectrogram Preprocessing Requirements**

OPERA-CT expects spectrograms as input, not raw audio. Our preprocessing pipeline needs to:

### Before OPERA-CT Processing:
```
Raw Audio → Our Preprocessing → Spectrogram → OPERA-CT (4×4 patches) → 768-dim features
```

### Current Pipeline Issue:
```python
# Current (INCORRECT):
audio = apply_preprocessing(audio, sr, config)
features = opera_ct_model(audio)  # ❌ Raw audio won't work

# Correct Approach:
audio = apply_preprocessing(audio, sr, config) 
spectrogram = audio_to_spectrogram(audio, sr)  # ✅ Convert to spectrogram
features = opera_ct_model(spectrogram)  # ✅ Feed spectrogram
```

## 2. **Preprocessing Order Considerations**

### Option A: Preprocess Audio → Spectrogram
```python
# Our current approach
audio = peak_normalize(audio)
audio = apply_bandpass_filter(audio, sr)
spectrogram = librosa.stft(audio)  # Convert to spectrogram for OPERA-CT
```

### Option B: Preprocess Spectrogram Directly  
```python
# Alternative approach
spectrogram = librosa.stft(audio)
spectrogram = preprocess_spectrogram(spectrogram)  # Apply filtering in frequency domain
```

## 3. **Required Pipeline Updates**

### Update FeatureExtractor Class:
```python
class FeatureExtractor:
    def _extract_opera_ct_features(self, audio, sr):
        # Convert audio to spectrogram format expected by OPERA-CT
        spectrogram = self._audio_to_opera_spectrogram(audio, sr)
        
        # Apply OPERA-CT model to spectrogram
        features = self.model(spectrogram)  # Should return 768-dim vector
        return features
    
    def _audio_to_opera_spectrogram(self, audio, sr):
        # Convert to spectrogram format compatible with OPERA-CT
        # This needs to match OPERA-CT's expected input format
        pass
```

## 4. **Compatibility Questions to Resolve**

### A. Spectrogram Format
- **What type of spectrogram?** (STFT, Mel-spectrogram, Log-mel?)
- **What frequency resolution?** (n_fft, hop_length, window)
- **What frequency range?** (0-8kHz for 16kHz audio?)
- **What time resolution?** (How many time frames?)

### B. Preprocessing Compatibility
- **When to apply our preprocessing?** (Before or after spectrogram conversion?)
- **Does OPERA-CT expect normalized spectrograms?**
- **Are there specific amplitude ranges expected?**

### C. Segmentation Interaction
- **How does segmentation work with spectrogram patches?**
- **Does OPERA-CT handle variable-length spectrograms?**
- **Should we segment audio or spectrograms?**

## 5. **Updated Pipeline Architecture**

### Revised Feature Extraction Flow:
```
1. Load Raw Audio (16kHz, mono)
2. Apply Audio-Domain Preprocessing (our methods: A0-D2)
3. Convert to OPERA-CT Compatible Spectrogram
4. Extract 768-dim Features via OPERA-CT
5. Use Features for Classification/Clustering
```

### Implementation Priority:
1. **Investigate OPERA-CT spectrogram requirements** (critical)
2. **Update FeatureExtractor to handle spectrogram conversion**
3. **Test preprocessing compatibility** (audio vs spectrogram domain)
4. **Validate feature extraction pipeline**
5. **Run systematic validation**

## 6. **Fallback Strategy**

If OPERA-CT requirements are incompatible with our preprocessing:

### Option 1: Dual Pipeline
```python
# Extract both for comparison
opera_features = extract_opera_ct_features(raw_audio)  # Minimal preprocessing
handcrafted_features = extract_our_features(preprocessed_audio)  # Our methods
```

### Option 2: Preprocessing Adaptation
```python
# Adapt our methods to work in spectrogram domain
spectrogram = audio_to_spectrogram(raw_audio)
processed_spectrogram = apply_spectrogram_preprocessing(spectrogram, method)
features = opera_ct_model(processed_spectrogram)
```

## 7. **Action Items**

### Immediate (Phase 0):
- [ ] Install OPERA-CT and examine input/output formats
- [ ] Test with sample audio to understand spectrogram requirements
- [ ] Document exact input specifications
- [ ] Update FeatureExtractor class accordingly

### Before Full Validation:
- [ ] Test preprocessing compatibility with OPERA-CT
- [ ] Validate that our 16 methods work with spectrogram input
- [ ] Ensure feature extraction produces consistent 768-dim outputs
- [ ] Verify segmentation works with OPERA-CT requirements

## 8. **Risk Mitigation**

### High Risk: Incompatible Preprocessing
- **Impact**: Our preprocessing methods might not work with OPERA-CT
- **Mitigation**: Implement dual extraction (OPERA-CT + handcrafted features)

### Medium Risk: Performance Degradation  
- **Impact**: Preprocessing might hurt OPERA-CT performance
- **Mitigation**: Test with minimal preprocessing baseline

### Low Risk: Technical Integration Issues
- **Impact**: Implementation challenges with spectrogram conversion
- **Mitigation**: Use mel-spectrogram fallback approach

---

**CONCLUSION**: We need to understand OPERA-CT's exact input requirements before running the comprehensive validation. The spectrogram requirement is a critical architectural decision that affects our entire preprocessing pipeline.

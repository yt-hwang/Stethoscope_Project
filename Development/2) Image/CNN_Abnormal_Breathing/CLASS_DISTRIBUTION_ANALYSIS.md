# Class Distribution Analysis - 5-Class Abnormal Breathing Classification

## 📊 **Class Mapping Strategy**

Based on the breathing_nonbreathing_intervals.json data, here's how we map various diagnosis strings to our 5 classes:

### **Class Definitions**
1. **Wheezing** (0) - High-pitched whistling sounds during breathing
2. **Crackle** (1) - Discontinuous, explosive sounds (like Velcro being pulled apart)
3. **Rhonchi** (2) - Low-pitched, continuous sounds (like snoring)
4. **Bronchi** (3) - Bronchial breath sounds (louder, harsher than normal)
5. **Healthy** (4) - Normal, clear breath sounds

### **Diagnosis String Mapping**

| Original Diagnosis | Mapped Class | Notes |
|-------------------|--------------|-------|
| "Wheezing, Brhonchi" | Bronchi | Contains "bronchi" keyword |
| "Wheezing" | Wheezing | Direct match |
| "Wheezing (잘들림)" | Wheezing | Wheezing with Korean annotation |
| "Wheezing(잘들림 - 호흡은 정상처럼 들림)" | Wheezing | Wheezing with detailed Korean annotation |
| "Wheezing(잘들림 - 호흡은 Bronchial asthma wheezing 저음 wheezing)" | Wheezing | Complex wheezing description |
| "Wheezing (잘들림 - 25초 이후 말이랑 섞임)" | Wheezing | Wheezing with timing annotation |
| "Crackle" | Crackle | Direct match |
| "Crackle (잘들림)" | Crackle | Crackle with Korean annotation |
| "Crackle (Exhale이 잘 안들림)" | Crackle | Crackle with breathing phase annotation |
| "Crackle (Exhale 구간 애매함)" | Crackle | Crackle with ambiguous phase |
| "Crackle (전체적으로 약하게 들림, 정상호흡과 유사하게)" | Crackle | Weak crackle description |
| "Rhonchi" | Rhonchi | Direct match |
| "Healthy" | Healthy | Direct match |

## 🔍 **Expected Class Distribution**

Based on the diagnosis strings found in the JSON file:

### **Sample Distribution (from available data)**
- **Wheezing**: ~12 files (including variations)
- **Crackle**: ~8 files (including variations)  
- **Rhonchi**: ~2 files
- **Bronchi**: ~2 files (from "Wheezing, Bronchi" entries)
- **Healthy**: ~4 files

### **Class Balance Considerations**
⚠️ **Imbalanced Dataset**: The current distribution shows significant class imbalance:
- Wheezing: Most common (~43%)
- Crackle: Second most common (~29%)
- Healthy: Moderate (~14%)
- Rhonchi: Less common (~7%)
- Bronchi: Least common (~7%)

## 🎯 **Training Strategy for Imbalanced Data**

### **Data Augmentation (Recommended)**
1. **Time Stretching**: Vary playback speed slightly
2. **Pitch Shifting**: Small frequency shifts
3. **Noise Addition**: Add background noise
4. **Mixup**: Blend samples from different classes
5. **Oversampling**: Generate more samples for minority classes

### **Loss Function Adjustments**
1. **Class Weights**: Weight loss inversely proportional to class frequency
2. **Focal Loss**: Focus on hard-to-classify examples
3. **Balanced Sampling**: Sample equally from each class during training

### **Evaluation Metrics**
1. **Per-Class Metrics**: Precision, Recall, F1-score for each class
2. **Macro/Micro Averages**: Account for class imbalance
3. **Confusion Matrix**: Visualize class-wise performance
4. **ROC-AUC**: For each class individually

## 📈 **Expected Performance**

### **Baseline Expectations**
- **Overall Accuracy**: 50-75% (due to 5 classes + imbalance)
- **Healthy Class**: Likely highest performance (clear audio patterns)
- **Wheezing Class**: Good performance (distinctive high-frequency patterns)
- **Crackle Class**: Moderate performance (complex discontinuous patterns)
- **Rhonchi Class**: Lower performance (limited training data)
- **Bronchi Class**: Lower performance (limited training data)

### **Improvement Strategies**
1. **Collect More Data**: Especially for Rhonchi and Bronchi classes
2. **Data Augmentation**: Increase effective dataset size
3. **Transfer Learning**: Use pre-trained audio models
4. **Ensemble Methods**: Combine multiple models
5. **Active Learning**: Focus on uncertain predictions

## 🔧 **Implementation Notes**

### **Class Weight Calculation**
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights for imbalanced dataset
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

### **Balanced Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# Create weighted sampler for balanced training
sample_weights = [class_weights[y] for y in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

### **Evaluation Focus**
- Monitor per-class performance closely
- Use macro-averaged metrics for overall assessment
- Pay attention to minority class performance
- Consider clinical significance of misclassifications

## 📋 **Action Items**

1. **Verify Class Mapping**: Double-check diagnosis string mappings
2. **Implement Class Weights**: Add weighted loss function
3. **Add Data Augmentation**: Implement time stretching, pitch shifting
4. **Monitor Training**: Watch for overfitting on majority classes
5. **Evaluate Carefully**: Use appropriate metrics for imbalanced data

---

*This analysis helps guide the training strategy for the 5-class CNN model, ensuring proper handling of the imbalanced dataset and realistic performance expectations.*

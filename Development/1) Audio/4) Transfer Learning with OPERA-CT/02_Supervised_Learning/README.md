# OPERA-CT Supervised Learning Experiments

This folder contains all supervised learning experiments using OPERA-CT for classification tasks.

## 📁 Structure

### 01_Breathing_Classification/
- **Purpose**: Classify breathing vs non-breathing segments
- **Method**: OPERA-CT frozen + custom classification head
- **Content**: Individual models (RF, SVM, LR) and ensemble methods
- **Results**: Timeline visualizations, confusion matrices, accuracy scores
- **Key Question**: "Can OPERA-CT features classify breathing patterns?"

### 02_Fine_Tuning/
- **Purpose**: Fine-tune OPERA-CT on respiratory data
- **Status**: Future experiments
- **Key Question**: "Can fine-tuning improve OPERA-CT performance?"

### 03_Custom_Heads/
- **Purpose**: Test different classification head architectures
- **Status**: Future experiments
- **Key Question**: "What's the optimal head architecture?"

## 🏆 Key Results

- **Best Individual Model**: Random Forest (varies by segment size)
- **Best Ensemble Method**: Varies by segment size
- **Timeline Visualizations**: All 12 files with Excel vs Model comparison
- **Consistent Format**: Matches handcrafted features visualization

## 📊 Performance by Segment Size

| Segment Size | Best Individual | Best Ensemble | Accuracy |
|--------------|----------------|---------------|----------|
| 0.25s | Random Forest | RF Heavy | ~75% |
| 0.5s | Random Forest | Best 2 | ~76% |
| 1.0s | Random Forest | Hard Voting | ~69% |

## 🔬 Key Insights

1. **OPERA-CT Features Work**: Successfully classify breathing vs non-breathing
2. **Segment Size Matters**: 0.5s segments often perform best
3. **Ensemble Methods Help**: Often outperform individual models
4. **Visualization Consistency**: Same format as handcrafted features

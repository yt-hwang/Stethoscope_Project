#!/usr/bin/env python3
"""
호흡음 분석을 위한 딥러닝 모델들
- CNN for Spectrogram
- LSTM for Sequential Features  
- Hybrid CNN-LSTM
- Transfer Learning with Pretrained Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 설정
from src.config import SR, N_FFT, HOP_LEN, N_MELS, FMIN, FMAX
from src.audio_io import load_audio, pre_emphasis
from src.features import stft_mag_db, logmel, mfcc, wheeze_indicators

class RespiratoryDataset(Dataset):
    """
    호흡음 데이터셋 클래스
    """
    def __init__(self, audio_paths, labels, feature_type='spectrogram', segment_length=2.0):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_type = feature_type
        self.segment_length = segment_length
        self.sample_rate = SR
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # 오디오 로딩 및 전처리
        y = load_audio(str(audio_path))
        y = pre_emphasis(y)
        
        # 정규화
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y / rms * 0.1
        
        # 특징 추출
        if self.feature_type == 'spectrogram':
            # Mel-spectrogram
            mel_spec = logmel(y)
            # 고정 크기로 패딩/크롭
            target_time_frames = 3000  # 고정 시간 프레임 수
            if mel_spec.shape[1] > target_time_frames:
                mel_spec = mel_spec[:, :target_time_frames]
            else:
                # 패딩
                pad_width = target_time_frames - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            # 정규화
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            return torch.FloatTensor(mel_spec), torch.LongTensor([label])
        
        elif self.feature_type == 'mfcc':
            # MFCC
            mfcc_features = mfcc(y, n_mfcc=20)
            # 고정 크기로 패딩/크롭
            target_time_frames = 3000
            if mfcc_features.shape[1] > target_time_frames:
                mfcc_features = mfcc_features[:, :target_time_frames]
            else:
                pad_width = target_time_frames - mfcc_features.shape[1]
                mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            # 정규화
            mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
            return torch.FloatTensor(mfcc_features), torch.LongTensor([label])
        
        elif self.feature_type == 'stft':
            # STFT
            stft_spec = stft_mag_db(y)
            # 고정 크기로 패딩/크롭
            target_time_frames = 3000
            if stft_spec.shape[1] > target_time_frames:
                stft_spec = stft_spec[:, :target_time_frames]
            else:
                pad_width = target_time_frames - stft_spec.shape[1]
                stft_spec = np.pad(stft_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            # 정규화
            stft_spec = (stft_spec - stft_spec.mean()) / (stft_spec.std() + 1e-8)
            return torch.FloatTensor(stft_spec), torch.LongTensor([label])

class CNNRespiratoryClassifier(nn.Module):
    """
    호흡음 분류를 위한 CNN 모델
    """
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.5):
        super(CNNRespiratoryClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, height, width)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTMRespiratoryClassifier(nn.Module):
    """
    호흡음 분류를 위한 LSTM 모델
    """
    def __init__(self, input_size=20, hidden_size=64, num_layers=2, num_classes=2, dropout_rate=0.5):
        super(LSTMRespiratoryClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Reshape for LSTM: (batch, time, features)
        if x.dim() == 3:  # (batch, height, width)
            x = x.transpose(1, 2)  # (batch, width, height)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        
        # Classifier
        x = F.relu(self.fc1(output))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class HybridCNNLSTM(nn.Module):
    """
    CNN + LSTM 하이브리드 모델
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # LSTM
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Add channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.view(batch_size, x.size(2), -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        x = lstm_out[:, -1, :]
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelTrainer:
    """
    모델 훈련 클래스
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target.squeeze())
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_data_loaders(audio_base_path, batch_size=8, test_size=0.2):
    """
    데이터 로더 생성
    """
    # 오디오 파일 경로 수집
    audio_paths = []
    labels = []
    
    patient_labels = {
        'WEBSS002': 1,  # asthma
        'WEBSS003': 1,  # asthma
        'WEBSS004': 1,  # asthma
        'WEBSS005': 0,  # normal
        'WEBSS006': 1,  # asthma
        'WEBSS007': 1   # asthma
    }
    
    audio_base = Path(audio_base_path)
    
    for folder in audio_base.iterdir():
        if folder.is_dir():
            patient_id = folder.name.split('_')[0]
            label = patient_labels.get(patient_id, 0)
            
            for audio_file in folder.glob("*.wav"):
                audio_paths.append(audio_file)
                labels.append(label)
    
    # Train/Test split
    from sklearn.model_selection import train_test_split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        audio_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Datasets
    train_dataset = RespiratoryDataset(train_paths, train_labels, feature_type='spectrogram')
    test_dataset = RespiratoryDataset(test_paths, test_labels, feature_type='spectrogram')
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device='cpu', class_names=['Normal', 'Asthma']):
    """
    모델 평가
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.squeeze().cpu().numpy())
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_preds, all_targets

def main():
    """
    메인 실행 함수
    """
    print("🚀 딥러닝 모델 훈련 시작")
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로더 생성
    audio_base_path = "../../Audio shared/Hospital sound"
    train_loader, test_loader = create_data_loaders(audio_base_path, batch_size=4)
    
    # 모델들 테스트
    models = {
        'CNN': CNNRespiratoryClassifier(),
        'LSTM': LSTMRespiratoryClassifier(input_size=3000),  # Fixed time frames
        'Hybrid': HybridCNNLSTM()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🤖 Training {model_name} model...")
        
        trainer = ModelTrainer(model, device)
        best_acc = trainer.train(train_loader, test_loader, epochs=30, lr=0.001)
        
        # 평가
        model.load_state_dict(torch.load('best_model.pth'))
        preds, targets = evaluate_model(model, test_loader, device)
        
        results[model_name] = best_acc
        
        # 훈련 히스토리 저장
        trainer.plot_training_history()
        plt.savefig(f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 결과 비교
    print("\n📊 Model Comparison:")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.2f}%")

if __name__ == "__main__":
    main()

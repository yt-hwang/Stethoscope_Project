#!/usr/bin/env python3
"""
나머지 노트북들 상세 분석 및 설명
- fine tuning simulation.ipynb
- RNN experiment.ipynb
"""

print("🔬 나머지 노트북들 상세 분석")
print("=" * 60)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# 우리가 만든 모듈들 import
from src.util import random_crop, random_mask, random_multiply, crop_first
from src.model.models_eval import AudioClassifier
from src.benchmark.model_util import initialize_pretrained_model

print("\n📚 1단계: fine tuning simulation.ipynb 분석")
print("-" * 50)

print("""
🎯 fine tuning simulation.ipynb의 목적:

이 노트북은 Transfer Learning의 파인튜닝 과정을 시뮬레이션합니다.

주요 기능:
1. 하이퍼파라미터 튜닝
2. 학습 과정 모니터링
3. 성능 평가 및 비교
4. 다양한 설정으로 실험

핵심 개념:
- Learning Rate 스케줄링
- Early Stopping
- Model Checkpointing
- 성능 메트릭 추적
""")

print("\n🔧 2단계: 파인튜닝 시뮬레이션 구현")
print("-" * 50)

class FineTuningSimulator:
    """파인튜닝 시뮬레이션 클래스"""
    
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            # Forward pass
            logits = self.model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= 0.001 * param.grad.data  # 간단한 SGD
                    param.grad.zero_()
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                logits = self.model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def simulate_training(self, epochs=10):
        """학습 시뮬레이션"""
        print(f"파인튜닝 시뮬레이션 시작 ({epochs} 에포크)")
        print("-" * 40)
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate_epoch()
            
            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:2d}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        print("-" * 40)
        print("파인튜닝 시뮬레이션 완료!")
        
        return self.history

print("✅ FineTuningSimulator 클래스 정의 완료!")

print("\n🧪 3단계: 파인튜닝 시뮬레이션 테스트")
print("-" * 50)

# 더미 데이터 생성
X = torch.FloatTensor(np.random.randn(100, 200, 128))
y = torch.LongTensor(np.random.randint(0, 2, 100))

# 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# DataLoader 생성
train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=8, shuffle=False)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=8, shuffle=False)

print(f"데이터 분할 완료:")
print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# 모델 생성
pretrained_model = initialize_pretrained_model("operaCT")
classifier = AudioClassifier(
    net=pretrained_model,
    head="linear",
    classes=2,
    feat_dim=768
)

# 파인튜닝 시뮬레이션 실행
simulator = FineTuningSimulator(classifier, train_loader, val_loader, test_loader)
history = simulator.simulate_training(epochs=5)

print("\n📊 4단계: RNN experiment.ipynb 분석")
print("-" * 50)

print("""
🎯 RNN experiment.ipynb의 목적:

이 노트북은 RNN 모델을 실험하는 노트북입니다.

주요 기능:
1. LSTM/GRU 모델 구현
2. 시계열 데이터 처리
3. RNN vs CNN 성능 비교
4. 다양한 RNN 아키텍처 실험

핵심 개념:
- 시계열 데이터의 순차적 처리
- LSTM의 장기 의존성 학습
- GRU의 간소화된 구조
- Bidirectional RNN
""")

print("\n🔧 5단계: RNN 모델 구현")
print("-" * 50)

class LSTMRespiratoryClassifier(nn.Module):
    """LSTM 기반 호흡음 분류기"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # bidirectional이므로 *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x: [batch_size, time_steps, features]
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # 분류
        logits = self.classifier(last_output)
        
        return logits

class GRURespiratoryClassifier(nn.Module):
    """GRU 기반 호흡음 분류기"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x: [batch_size, time_steps, features]
        gru_out, hidden = self.gru(x)
        
        # 마지막 타임스텝의 출력 사용
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # 분류
        logits = self.classifier(last_output)
        
        return logits

print("✅ LSTM/GRU 모델 정의 완료!")

print("\n🧪 6단계: RNN 모델 테스트")
print("-" * 50)

# 더미 시계열 데이터 생성
time_series_data = torch.FloatTensor(np.random.randn(20, 200, 128))  # [batch, time, features]
labels = torch.LongTensor(np.random.randint(0, 2, 20))

print(f"시계열 데이터 shape: {time_series_data.shape}")
print(f"라벨 shape: {labels.shape}")

# LSTM 모델 테스트
print("\nLSTM 모델 테스트:")
lstm_model = LSTMRespiratoryClassifier(input_size=128, hidden_size=256, num_layers=2)
print(f"LSTM 파라미터 수: {sum(p.numel() for p in lstm_model.parameters()):,}")

lstm_model.eval()
with torch.no_grad():
    lstm_output = lstm_model(time_series_data)
    print(f"LSTM 출력 shape: {lstm_output.shape}")

# GRU 모델 테스트
print("\nGRU 모델 테스트:")
gru_model = GRURespiratoryClassifier(input_size=128, hidden_size=256, num_layers=2)
print(f"GRU 파라미터 수: {sum(p.numel() for p in gru_model.parameters()):,}")

gru_model.eval()
with torch.no_grad():
    gru_output = gru_model(time_series_data)
    print(f"GRU 출력 shape: {gru_output.shape}")

print("\n🔬 7단계: RNN vs CNN 성능 비교")
print("-" * 50)

# 간단한 CNN 모델과 비교
class SimpleCNN(nn.Module):
    """간단한 CNN 모델"""
    
    def __init__(self, input_channels=128, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: [batch, time, features] -> [batch, 1, time, features]
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# CNN 모델 테스트
print("CNN 모델 테스트:")
cnn_model = SimpleCNN(input_channels=128)
print(f"CNN 파라미터 수: {sum(p.numel() for p in cnn_model.parameters()):,}")

cnn_model.eval()
with torch.no_grad():
    cnn_output = cnn_model(time_series_data)
    print(f"CNN 출력 shape: {cnn_output.shape}")

print("\n📊 모델별 파라미터 수 비교:")
print(f"  LSTM: {sum(p.numel() for p in lstm_model.parameters()):,}")
print(f"  GRU:  {sum(p.numel() for p in gru_model.parameters()):,}")
print(f"  CNN:  {sum(p.numel() for p in cnn_model.parameters()):,}")

print("\n🎓 8단계: 핵심 개념 정리")
print("-" * 50)

print("""
🎯 나머지 노트북들의 핵심 개념:

1. fine tuning simulation.ipynb:
   - 파인튜닝 과정의 시뮬레이션
   - 하이퍼파라미터 튜닝
   - 학습 과정 모니터링
   - 성능 평가 및 비교

2. RNN experiment.ipynb:
   - 시계열 데이터의 순차적 처리
   - LSTM: 장기 의존성 학습
   - GRU: 간소화된 구조
   - RNN vs CNN 성능 비교

3. 각 모델의 특징:
   - LSTM: 복잡하지만 강력한 장기 의존성 학습
   - GRU: LSTM보다 간단하지만 비슷한 성능
   - CNN: 공간적 특징 학습에 강함
   - RNN: 시간적 특징 학습에 강함

4. 호흡음 분석에서의 적용:
   - 시계열 데이터: 호흡 주기의 시간적 변화
   - 순차적 처리: 호흡음의 연속성 고려
   - 장기 의존성: 호흡 주기 전체의 패턴 학습
""")

print("\n✅ 나머지 노트북들 완전 이해 완료!")
print("=" * 60)

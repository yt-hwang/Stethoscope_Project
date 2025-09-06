"""
Cola and ColaMD models for Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Cola(pl.LightningModule):
    """
    Contrastive Learning for Audio (Cola) model
    """
    def __init__(self, encoder, projection_dim=256, learning_rate=1e-4):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.learning_rate = learning_rate
        
    def forward(self, x1, x2):
        # 특징 추출
        features_1 = self.encoder(x1)
        features_2 = self.encoder(x2)
        
        # 프로젝션
        proj_1 = self.projection_head(features_1)
        proj_2 = self.projection_head(features_2)
        
        return proj_1, proj_2
    
    def contrastive_loss(self, proj_1, proj_2):
        """Contrastive loss 계산"""
        # Positive pairs는 가깝게 (작은 거리)
        positive_loss = F.mse_loss(proj_1, proj_2)
        
        # Negative pairs는 멀게 (큰 거리) - 배치 내 다른 샘플들과
        batch_size = proj_1.size(0)
        negative_loss = 0
        
        if batch_size > 1:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        negative_loss += F.mse_loss(proj_1[i], proj_2[j])
            
            # 정규화
            negative_loss = negative_loss / (batch_size * (batch_size - 1))
        else:
            # 배치 크기가 1인 경우, 랜덤한 negative sample 생성
            negative_sample = torch.randn_like(proj_1)
            negative_loss = F.mse_loss(proj_1, negative_sample)
        
        return positive_loss - negative_loss
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        proj_1, proj_2 = self.forward(x1, x2)
        loss = self.contrastive_loss(proj_1, proj_2)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class ColaMD(Cola):
    """
    Contrastive Learning for Medical Data (ColaMD) model
    의료용 데이터에 특화된 Cola 모델
    """
    def __init__(self, encoder, projection_dim=256, learning_rate=1e-4):
        super().__init__(encoder, projection_dim, learning_rate)
        
    def contrastive_loss(self, proj_1, proj_2):
        """의료 데이터에 특화된 Contrastive loss"""
        # Positive pairs는 가깝게
        positive_loss = F.mse_loss(proj_1, proj_2)
        
        # Negative pairs는 멀게 - 더 강한 페널티
        batch_size = proj_1.size(0)
        negative_loss = 0
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # 의료 데이터의 경우 더 강한 대조 학습
                    negative_loss += F.mse_loss(proj_1[i], proj_2[j]) * 2.0
        
        negative_loss = negative_loss / (batch_size * (batch_size - 1))
        
        return positive_loss - negative_loss

class SimpleEncoder(nn.Module):
    """
    간단한 인코더 모델 (테스트용)
    """
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # [batch_size, time, features] -> [batch_size, features]
        x = x.mean(dim=1)  # 시간 차원 평균
        return self.encoder(x)

"""
Model utilities for pretrained models
"""

import torch
import torch.nn as nn
import os

def initialize_pretrained_model(model_name="operaCT"):
    """
    사전 훈련된 모델 초기화
    """
    if model_name == "operaCT":
        # OperaCT 모델 초기화 (간단한 버전)
        encoder = SimpleOperaCTEncoder()
        return OperaCTModel(encoder)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_encoder_path(model_name="operaCT"):
    """
    인코더 경로 반환
    """
    if model_name == "operaCT":
        return "pretrained_models/operaCT_encoder.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

class SimpleOperaCTEncoder(nn.Module):
    """
    간단한 OperaCT 인코더 (테스트용)
    """
    def __init__(self, input_dim=128, hidden_dim=1024, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Transformer-like 구조
        self.patch_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim))
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Global Average Pooling + Final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch_size, time_frames, mel_bins]
        batch_size, time_frames, mel_bins = x.shape
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch_size, time_frames, hidden_dim]
        
        # Positional encoding
        x = x + self.positional_encoding[:, :time_frames, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, time_frames, hidden_dim]
        
        # Global average pooling
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, time_frames]
        x = self.global_pool(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_dim]
        
        # Final projection
        x = self.final_projection(x)  # [batch_size, output_dim]
        
        return x

class OperaCTModel(nn.Module):
    """
    OperaCT 전체 모델
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x):
        return self.encoder(x)

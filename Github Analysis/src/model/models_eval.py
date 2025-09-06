"""
Evaluation models for Transfer Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AudioClassifier(pl.LightningModule):
    """
    Audio Classifier for Transfer Learning
    고정된 인코더 + 학습 가능한 분류기
    """
    def __init__(self, net, head="linear", classes=2, lr=1e-4, l2_strength=1e-4, 
                 feat_dim=768, freeze_encoder="none"):
        super().__init__()
        self.net = net  # 고정된 인코더
        self.classes = classes
        self.lr = lr
        self.l2_strength = l2_strength
        self.feat_dim = feat_dim
        
        # 인코더 고정 설정
        if freeze_encoder == "all":
            for param in self.net.parameters():
                param.requires_grad = False
        elif freeze_encoder == "none":
            for param in self.net.parameters():
                param.requires_grad = True
        
        # 분류기 헤드 생성
        self.head = self._create_head(head, feat_dim, classes)
        
    def _create_head(self, head_type, feat_dim, classes):
        if head_type == "linear":
            return nn.Linear(feat_dim, classes)
        elif head_type == "mlp":
            return nn.Sequential(
                nn.Linear(feat_dim, feat_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim//2, classes)
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, x):
        # 고정된 인코더로 특징 추출
        with torch.no_grad() if self.net.training == False else torch.enable_grad():
            features = self.net(x)  # [batch_size, feat_dim]
        
        # 분류기로 예측
        logits = self.head(features)  # [batch_size, classes]
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        # L2 정규화
        l2_reg = 0
        for param in self.head.parameters():
            l2_reg += torch.norm(param)
        loss += self.l2_strength * l2_reg
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class AudioClassifierCLAP(AudioClassifier):
    """
    CLAP 기반 Audio Classifier
    """
    def __init__(self, net, head="linear", classes=2, lr=1e-4, l2_strength=1e-4, 
                 feat_dim=512, freeze_encoder="none"):
        super().__init__(net, head, classes, lr, l2_strength, feat_dim, freeze_encoder)

class AudioClassifierAudioMAE(AudioClassifier):
    """
    AudioMAE 기반 Audio Classifier
    """
    def __init__(self, net, head="linear", classes=2, lr=1e-4, l2_strength=1e-4, 
                 feat_dim=768, freeze_encoder="none"):
        super().__init__(net, head, classes, lr, l2_strength, feat_dim, freeze_encoder)

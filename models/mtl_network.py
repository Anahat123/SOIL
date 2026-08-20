import torch
import torch.nn as nn
from models.backbones import Spectral1DCNN, Spatial2DResNet

class SoilMTLNetwork(nn.Module):
    """
    Multi-Task Learning Network for Soil Moisture (Regression) and Soil Texture (Classification).
    """
    def __init__(self, input_mode='1d', in_channels=10, num_texture_classes=4, shared_dim=128):
        super(SoilMTLNetwork, self).__init__()
        
        # 1. Shared Backbone
        if input_mode == '1d':
            self.backbone = Spectral1DCNN(in_channels=in_channels, out_features=shared_dim)
        elif input_mode == '2d':
            self.backbone = Spatial2DResNet(in_channels=in_channels, out_features=shared_dim)
        else:
            raise ValueError("input_mode must be '1d' or '2d'")
            
        # 2. Moisture Head (Regression)
        self.moisture_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Single continuous output
        )
        
        # 3. Texture Head (Classification)
        self.texture_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_texture_classes)  # Logits for N classes
        )

    def forward(self, x):
        shared_features = self.backbone(x)
        
        moisture_pred = self.moisture_head(shared_features)
        texture_logits = self.texture_head(shared_features)
        
        return moisture_pred, texture_logits

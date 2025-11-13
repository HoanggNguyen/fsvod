import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2  # Compatible with PyTorch 1.12.1
import torchvision.transforms as T

class CNNBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = mobilenet_v2(pretrained=config['model']['pretrained']).features  # Lighter for FPS >=15
        self.feature_dim = config['model']['general']['feature_dim']  # 512 for efficiency
        
        # Freeze early layers for transfer to small object drone detection
        for param in list(self.backbone.parameters())[:-10]:  # Fine-tune last layers
            param.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W] or [T, B, C, H, W] for videos (flatten T*B)
        if len(x.shape) == 5:  # Video: [T, B, C, H, W]
            t, b, c, h, w = x.shape
            x = x.view(t * b, c, h, w)
        feats = self.backbone(x)  # [B or T*B, 1280, H/32, W/32] -> adapt to 512
        feats = nn.AdaptiveAvgPool2d((1, 1))(feats)  # Global pool for small objects
        feats = feats.view(-1, self.feature_dim)  # Flatten to [N, D]
        return feats
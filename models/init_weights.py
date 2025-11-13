import torch.nn as nn

def init_weights(model):
    # Custom init for FSVOD: Xavier for linear, normal for conv
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    # Special for small objects: Bias init for detection heads
    if hasattr(model, 'tpn'):
        nn.init.constant_(model.tpn.mlp[-1].bias, -10)  # Low initial obj scores
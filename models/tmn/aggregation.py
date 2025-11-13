import torch

def aggregate_features(tube_feats, method='avg'):
    # tube_feats: [T, D] per tube
    if method == 'avg':
        return torch.mean(tube_feats, dim=0)  # Efficient for FPS
    elif method == 'max':
        return torch.max(tube_feats, dim=0)[0]
    else:
        raise ValueError("Unknown aggregation method")
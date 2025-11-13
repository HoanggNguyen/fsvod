# tests/test_tmn.py
import pytest
import torch
from models.tmn.tmn import TemporalMatchingNetwork
from models.tmn.aggregation import aggregate_features
from utils.losses import MatchingLoss, SupportClsLoss

@pytest.fixture
def config():
    return {
        'model': {
            'general': {'feature_dim': 512},
            'tmn': {
                'aggregation': 'avg',
                'multi_relation': True,
                'temporal_align': True,
                'contrastive': True
            }
        },
        'dataset': {'constraints': {'total_classes': 500}},
        'losses': {'label_smoothing': 0.1}
    }

def test_tmn_forward(config):
    tmn = TemporalMatchingNetwork(config)
    query_tube_feats = torch.rand(2, 512)  # [T=2, D]
    support_feats = torch.rand(3, 512)  # [S=3, D]
    
    match_scores, s_preds = tmn(query_tube_feats, support_feats)
    assert match_scores.shape == (1, 3)  # [N=1, S]
    assert s_preds.shape == (3, 500)  # [S, classes]

def test_aggregation():
    feats = torch.rand(5, 512)  # [T, D]
    agg = aggregate_features(feats, 'avg')
    assert agg.shape == (512,)
    
    with pytest.raises(ValueError):
        aggregate_features(feats, 'invalid')

def test_tmn_losses(config):
    match_loss = MatchingLoss(config)
    scls_loss = SupportClsLoss(config)
    
    query_feats = torch.rand(1, 512)
    support_feats = torch.rand(3, 512)
    labels = torch.tensor([1, 0, 0])  # Positive/negative
    
    lmatch = match_loss(query_feats, support_feats, labels)
    assert lmatch.item() >= 0
    
    s_preds = torch.rand(3, 500)
    s_labels = torch.tensor([0, 1, 2])
    lscls = scls_loss(s_preds, s_labels)
    assert lscls.item() > 0
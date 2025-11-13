# tests/test_tpn.py
import pytest
import torch
import torch.nn as nn
from models.tpn.tpn import TubeProposalNetwork
from utils.losses import TPNLoss

@pytest.fixture
def config():
    return {
        'model': {
            'general': {'feature_dim': 512},
            'tpn': {
                'proposal_num': 500,
                'inter_frame': True,
                'deformable_roi': True,
                'mlp_layers': 2,
                'id_verification': True
            }
        },
        'losses': {'label_smoothing': 0.1}
    }

def test_tpn_forward(config):
    tpn = TubeProposalNetwork(config)
    feats1 = torch.rand(1, 512, 32, 32)  # Mock backbone feats [B, C, H, W]
    feats2 = torch.rand(1, 512, 32, 32)
    proposals = torch.tensor([[0, 0, 10, 10]])  # Mock [N, 4]
    
    out = tpn(feats1, feats2, proposals)
    assert 'obj_scores1' in out
    assert out['boxes1'].shape == (1, 4)  # For N=1
    assert out['id_scores'].shape == (1,)

def test_tpn_loss(config):
    tpn_loss = TPNLoss(config)
    preds = {
        'obj_scores1': torch.tensor([0.5]),
        'obj_scores2': torch.tensor([0.6]),
        'boxes1': torch.tensor([[0,0,10,10]]),
        'boxes2': torch.tensor([[1,1,11,11]]),
        'id_scores': torch.tensor([0.7])
    }
    targets = {
        'gt_obj1': torch.tensor([1]),
        'gt_obj2': torch.tensor([1]),
        'gt_boxes1': torch.tensor([[0,0,10,10]]),
        'gt_boxes2': torch.tensor([[1,1,11,11]]),
        'gt_id': torch.tensor([1])
    }
    
    loss = tpn_loss(preds, targets)
    assert loss.item() > 0  # Positive loss
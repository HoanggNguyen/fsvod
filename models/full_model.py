import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # For FP16 in forward
from utils.losses import TPNLoss, MatchingLoss, SupportClsLoss
from utils.metrics import spatio_temporal_iou  # For eval with ST-IoU
from models.backbone.cnn_backbone import CNNBackbone
from models.tpn.tpn import TubeProposalNetwork
from models.tmn.tmn import TemporalMatchingNetwork

class FSVODModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CNNBackbone(config)
        self.tpn = TubeProposalNetwork(config)
        self.tmn = TemporalMatchingNetwork(config)
        
        self.tpn_loss = TPNLoss(config)
        self.match_loss = MatchingLoss(config)
        self.scls_loss = SupportClsLoss(config)

    def forward(self, query_frames, supports, targets=None):
        # query_frames: [T, C, H, W]
        # supports: [S, C, H, W]
        
        # Use autocast for forward (consistent with scripts)
        with autocast(enabled=self.config['project'].get('fp16', False) and torch.cuda.is_available()):
            # Extract features
            query_feats = [self.backbone(q) for q in query_frames]  # List [T, D]
            support_feats = self.backbone(supports)  # [S, D]
            
            # TPN: Generate tubes (pairwise frames)
            tubes = []
            for i in range(len(query_feats) - 1):
                tpn_out = self.tpn(query_feats[i], query_feats[i+1])
                tubes.append(tpn_out['boxes1'])  # Collect tubes
            
            # Aggregate tube feats
            tube_feats = [aggregate_features(qf) for qf in zip(*query_feats)]  # Placeholder
            
            # TMN+: Match
            match_scores, s_preds = self.tmn(tube_feats, support_feats)
        
        if self.training:
            # Losses
            ltpn = self.tpn_loss(tpn_out, targets['tpn'])
            lmatch = self.match_loss(tube_feats, support_feats, targets['labels'])
            lscls = self.scls_loss(s_preds, targets['support_labels'])
            loss = ltpn + lmatch + lscls
            return loss
        
        # Inference: Distribute matches to frames, eval ST-IoU
        preds = {'tubes': tubes, 'scores': match_scores}
        if targets:
            st_iou = spatio_temporal_iou(preds['tubes'], targets['gt_tubes'])
            print(f"ST-IoU: {st_iou}")  # For benchmark
        
        return preds
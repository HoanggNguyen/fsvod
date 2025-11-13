import torch
import torch.nn as nn
import torch.nn.functional as F

class TPNLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_loss = nn.CrossEntropyLoss()  # Lcls for objectness
        self.id_loss = nn.CrossEntropyLoss()   # Lid for identity verification
        self.reg_loss = nn.SmoothL1Loss()      # Lreg for box regression
        self.label_smoothing = config['losses'].get('label_smoothing', 0.1)

    def forward(self, preds, targets):
        # preds: dict with 'obj_scores1', 'obj_scores2', 'boxes1', 'boxes2', 'id_scores'
        # targets: dict with 'gt_obj1', 'gt_obj2', 'gt_boxes1', 'gt_boxes2', 'gt_id'
        
        # Apply label smoothing to classification targets
        def smooth_labels(labels, epsilon=0.1):
            return (1 - epsilon) * labels + (epsilon / labels.size(1))
        
        # Lcls: average for two frames
        lcls1 = self.cls_loss(preds['obj_scores1'], targets['gt_obj1'])
        lcls2 = self.cls_loss(preds['obj_scores2'], targets['gt_obj2'])
        lcls = (lcls1 + lcls2) / 2
        
        # Lreg: smooth L1, only for foreground
        mask_reg = (targets['gt_obj1'][:, 0] > 0)  # Foreground mask (assuming binary obj)
        lreg1 = self.reg_loss(preds['boxes1'][mask_reg], targets['gt_boxes1'][mask_reg])
        lreg2 = self.reg_loss(preds['boxes2'][mask_reg], targets['gt_boxes2'][mask_reg])
        lreg = (lreg1 + lreg2) / 2
        
        # Lid: identity verification
        lid = self.id_loss(preds['id_scores'], smooth_labels(targets['gt_id'], self.label_smoothing))
        
        ltpn = lcls + lreg + lid
        return ltpn

class MatchingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Assume multi-relation head from FSOD [32], use cosine distance or contrastive
        self.contrastive = nn.TripletMarginLoss(margin=1.0)  # For TMN+

    def forward(self, query_feats, support_feats, labels):
        # Lmatch: matching loss between aggregated tube feats and supports
        # Placeholder: contrastive for diversity
        anchor = query_feats  # Tube aggregated
        positive = support_feats[labels == 1]
        negative = support_feats[labels == 0]
        lmatch = self.contrastive(anchor, positive, negative)
        return lmatch

class SupportClsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls = nn.CrossEntropyLoss(label_smoothing=config['losses']['label_smoothing'])

    def forward(self, support_preds, support_labels):
        # Lscls: support classification for discriminative features
        return self.cls(support_preds, support_labels)
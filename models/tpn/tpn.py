import torch
import torch.nn as nn
from torchvision.ops import RoIAlign  # Deformable not native in 1.12.1; use approx or external if needed
# Note: Deformable RoIAlign requires DCNv2; assume installed or approx with standard RoIAlign for simplicity
from .proposal_pool import generate_proposals, create_proposal_pool

class TubeProposalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proposal_num = config['model']['tpn']['proposal_num']  # 500 for speed
        self.mlp = nn.Sequential(
            nn.Linear(2 * config['model']['general']['feature_dim'], 512),  # Concat 2 frames
            nn.ReLU(),
            nn.Linear(512, 2 + 4 * 2 + 1)  # obj1/obj2 (2), boxes1/2 (8), id (1) -> sigmoid for scores
        )
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/32, sampling_ratio=2)  # Approx deformable
        # If DCN needed: from deform_roi_align import DeformRoIAlign (assume external)

    def forward(self, feats1, feats2, proposals=None):
        # feats1/2: Backbone features for frame1/2 [B, C, H, W]
        if proposals is None:
            proposals = generate_proposals(feats1, self.proposal_num)  # RPN-like
        pool = create_proposal_pool(proposals)  # Union from both frames
        
        # Extract features for each proposal on both frames
        feats_cat = []
        for p in pool:
            f1 = self.roi_align(feats1, p.unsqueeze(0))  # [1, C, 7, 7]
            f2 = self.roi_align(feats2, p.unsqueeze(0))
            f_cat = torch.cat([f1.flatten(), f2.flatten()])  # [2*C]
            feats_cat.append(f_cat)
        feats_cat = torch.stack(feats_cat)  # [N, 2*C]
        
        # MLP: obj scores, boxes, id
        out = self.mlp(feats_cat)  # [N, 2+8+1]
        obj_scores = torch.sigmoid(out[:, :2])  # si1, si2
        boxes = out[:, 2:10].view(-1, 2, 4)  # bi1, bi2
        id_scores = torch.sigmoid(out[:, 10])  # vi
        
        return {'obj_scores1': obj_scores[:,0], 'obj_scores2': obj_scores[:,1],
                'boxes1': boxes[:,0], 'boxes2': boxes[:,1], 'id_scores': id_scores}
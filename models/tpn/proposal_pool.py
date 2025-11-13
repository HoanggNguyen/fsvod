import torch

def generate_proposals(feats, num_proposals=500):
    # Simple RPN placeholder: generate anchors (efficient for drone small objects)
    # Assume grid anchors; in practice, use torchvision RPN
    from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
    rpn = RegionProposalNetwork(...)  # Config-based; placeholder
    proposals = rpn(feats)  # [N, 4]
    return proposals[:num_proposals]

def create_proposal_pool(proposals1, proposals2=None):
    # Union of proposals from frame1 and frame2 (or inter-frame)
    if proposals2 is None:
        return proposals1
    return torch.unique(torch.cat([proposals1, proposals2]), dim=0)
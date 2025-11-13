import torch
import torch.nn as nn
from .aggregation import aggregate_features

class TemporalMatchingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multi_relation_head = nn.ModuleList([nn.Linear(config['model']['general']['feature_dim'], 256) for _ in range(3)])  # From FSOD [32]
        self.temporal_align = nn.Linear(2 * config['model']['general']['feature_dim'], config['model']['general']['feature_dim'])  # TAB
        self.support_cls = nn.Linear(config['model']['general']['feature_dim'], config['dataset']['constraints']['total_classes'])  # For Lscls

    def forward(self, query_tube_feats, support_feats, mode='match'):
        # query_tube_feats: Aggregated from TPN tubes [N, D]
        # support_feats: [S, D]
        
        if mode == 'align':  # Temporal alignment branch
            # Concat 2-frame query for alignment
            aligned = self.temporal_align(torch.cat(query_tube_feats[:2], dim=-1))  # Placeholder for 2 frames
            return aligned
        
        # Aggregate tube
        agg_query = aggregate_features(query_tube_feats, self.config['model']['tmn']['aggregation'])
        
        # Matching: Cosine or multi-relation
        dists = []
        for head in self.multi_relation_head:
            q = head(agg_query)
            s = head(support_feats)
            dist = torch.cosine_similarity(q.unsqueeze(1), s.unsqueeze(0), dim=-1)  # [N, S]
            dists.append(dist)
        match_scores = torch.mean(torch.stack(dists), dim=0)  # Average relations
        
        # Support cls for training
        support_preds = self.support_cls(support_feats)
        
        return match_scores, support_preds
import torch
import numpy as np

def link_tubes(frames_preds, config):
    """Link 2-frame tubes to full video tubes (sequential as in paper)"""
    # From page 7: Link {b1,b2} and {b2,b3} via b2
    tubes = []
    for i in range(len(frames_preds) - 1):
        tube = [frames_preds[i]['boxes'], frames_preds[i+1]['boxes']]  # Aggregate
        tubes.append(np.mean(tube, axis=0))  # Or concat for full tube
    return tubes

def filter_ghost_objects(tubes, thresh=0.5):
    """Filter false positives (ghosts) using tube consistency"""
    return [t for t in tubes if np.std(t) < thresh]  # Low variance for real objects

def tube_aggregation(feats, method='avg'):
    """Aggregate tube features for TMN+"""
    if method == 'avg':
        return torch.mean(feats, dim=0)  # Efficient for FPS
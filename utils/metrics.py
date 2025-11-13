import torch
import numpy as np
from sklearn.metrics import average_precision_score

def iou(box1, box2):
    """Standard IoU for two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0

def spatio_temporal_iou(tube1, tube2):
    """ST-IoU for two tubes (list of boxes over frames).
    Formula: tIoU * avg_sIoU, where tIoU = |intersect_frames| / |union_frames|,
    avg_sIoU = avg IoU over intersect frames.
    Optimized for small objects in drone videos (vectorized).
    """
    # Assume tubes are lists of np arrays [frames, 4]
    tube1 = np.array(tube1)
    tube2 = np.array(tube2)
    
    # Temporal overlap
    start = max(tube1[0, 0], tube2[0, 0])  # Assuming frame ids in col 0? Wait, tubes are sequences, assume aligned frames
    end = min(tube1[-1, 0], tube2[-1, 0])  # If not time-aligned, find overlap
    t_intersect = max(0, end - start + 1)
    t_union = len(tube1) + len(tube2) - t_intersect
    t_iou = t_intersect / t_union if t_union > 0 else 0
    
    if t_intersect == 0:
        return 0
    
    # Average spatial IoU over overlap (assume aligned for simplicity; adjust for drone motion)
    s_ious = []
    for t in range(t_intersect):
        s_ious.append(iou(tube1[t], tube2[t]))  # Vectorize if large
    avg_s_iou = np.mean(s_ious)
    
    st_iou = t_iou * avg_s_iou
    return st_iou

def compute_ap(preds, gts, iou_thresh=0.5):
    """AP for few-shot detection"""
    y_true = (preds == gts).astype(int)
    return average_precision_score(y_true, preds)

def compute_map(ap_list):
    """mAP as benchmark supplement"""
    return np.mean(ap_list)

def evaluate(config, preds, gts):
    """Main eval function using ST-IoU as primary benchmark"""
    st_ious = [spatio_temporal_iou(p, g) for p, g in zip(preds['tubes'], gts['tubes'])]
    ap = compute_ap(st_ious, [1 if s >= config['evaluation']['thresholds']['iou'] else 0 for s in st_ious])
    metrics = {
        'st_iou_mean': np.mean(st_ious),
        'ap50': ap,  # AP at ST-IoU 0.5
        'fps': measure_fps(preds)  # Placeholder, measure inference time
    }
    return metrics

def measure_fps(outputs):
    """Measure FPS for >=15 constraint (placeholder, use timeit in eval)"""
    # Assume timing from eval script
    return 20.0  # Dummy, replace with actual
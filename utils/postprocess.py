import numpy as np

def group_to_intervals(tubes, gap_threshold=5):
    """Group model tubes (list of per-frame bboxes) into detection intervals.
    tubes: list of dicts {'bboxes': [{'frame': int, 'x1':, 'y1':, 'x2':, 'y2':, 'score': float?}]}
    Merge if gap < threshold (for drone motion robustness).
    Return: list of {'bboxes': [...]} for detections.
    """
    if not tubes:
        return []
    
    # Assume one main tube per video (single target); sort by frame
    all_bboxes = []
    for tube in tubes:
        all_bboxes.extend(tube.get('bboxes', []))
    all_bboxes.sort(key=lambda b: b['frame'])
    
    detections = []
    current_interval = []
    prev_frame = -1
    for bbox in all_bboxes:
        if prev_frame == -1 or bbox['frame'] - prev_frame <= gap_threshold:
            current_interval.append(bbox)
        else:
            if current_interval:
                detections.append({'bboxes': current_interval})
            current_interval = [bbox]
        prev_frame = bbox['frame']
    
    if current_interval:
        detections.append({'bboxes': current_interval})
    
    return detections
import json
import os
import random
import cv2
import torch

def load_annotations(ann_path):
    with open(ann_path, 'r') as f:
        data = json.load(f)  # Assume list of dicts: [{'video_id': str, 'annotations': [{'bboxes': [{'frame': int, 'x1':, 'y1':, 'x2':, 'y2':}]}]}]
    return {entry['video_id']: entry['annotations'] for entry in data}  # Dict by video_id

def get_splits(config):
    # No splits in new dataset; return all as 'test' or empty
    return [], [], []  # Base, val_novel, test_novel not used

def sample_episodes(classes, k_shot, supports_dir, is_support=True):
    samples = []
    for cls in classes:
        cls_dir = os.path.join(supports_dir, str(cls))
        images = random.sample(os.listdir(cls_dir), k_shot)
        for img in images:
            # Load image, GT box (assume no box for supports in new dataset; full image as support)
            img_path = os.path.join(cls_dir, img)
            image = torch.from_numpy(cv2.imread(img_path))  # Actual loading
            box = None  # No GT box; use full image or crop if needed
            samples.append((image, box, cls))
    return samples

def list_videos(config):
    anns = load_annotations(os.path.join(config['data']['root'], config['dataset']['paths']['annotations']))
    return list(anns.keys())  # All video_ids
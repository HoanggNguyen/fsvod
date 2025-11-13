import torch
from torch.utils.data import DataLoader, Dataset
import random
import json
import os
import cv2
from .preprocess import preprocess_video, augment_frame
from .dataset_utils import load_annotations, get_splits, sample_episodes, list_videos

class FSVODDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.annotations = load_annotations(os.path.join(config['data']['root'], config['dataset']['paths']['annotations']))
        self.samples_root = os.path.join(config['data']['root'], config['dataset']['paths']['samples'])
        self.video_ids = list_videos(config)  # All video_ids
        # No classes; treat each video as 1-way 3-shot
        self.classes = []  # Not used
        
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_dir = os.path.join(self.samples_root, video_id)
        video_path = os.path.join(video_dir, self.config['dataset']['paths']['videos'])  # drone_video.mp4
        supports_dir = os.path.join(video_dir, self.config['dataset']['paths']['supports'])  # object_images/
        
        # Supports: fixed 3 images (1-way 3-shot)
        support_files = sorted(os.listdir(supports_dir))  # img_1.jpg, etc.
        supports = []
        for sf in support_files:
            img_path = os.path.join(supports_dir, sf)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            supports.append((image, None, video_id))  # No box, class=video_id
        
        # Query video: preprocess at 25 FPS
        frames = preprocess_video(
            video_path, 
            fps=self.config['dataset']['preprocessing']['fps_sample'], 
            size=self.config['dataset']['preprocessing']['frame_size']
        )
        if self.mode == 'train':
            augmented_frames = [augment_frame(f, self.config['preprocessing']['augmentations']) for f in frames]
        else:
            augmented_frames = [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(f.float() / 255.0) for f in frames]
        
        # Anns: intervals for targets (if train/eval)
        anns = self.annotations.get(video_id, [])  # List of {'bboxes': [{'frame':, 'x1':,...}]}
        
        return {
            'video_id': video_id,
            'supports': supports,  # List of (image_tensor, box=None, class=video_id)
            'query_frames': torch.stack(augmented_frames),  # [T, C, H, W]
            'query_anns': anns  # For train/eval
        }

def get_dataloader(config, mode='train', batch_size=None):
    dataset = FSVODDataset(config, mode)
    batch_size = batch_size or config['training']['batch_size'] if mode == 'train' else config['evaluation']['batch_size']
    return DataLoader(dataset, batch_size=batch_size, num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'], drop_last=True if mode=='train' else False)
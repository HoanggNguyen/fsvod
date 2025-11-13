import torch
import yaml
import json
import os
import torch.nn as nn  # For DataParallel
from torch.cuda.amp import autocast  # For FP16
from models.full_model import FSVODModel
from data.dataloader import get_dataloader
from utils.postprocess import group_to_intervals
from utils.logging import setup_logging
from utils.helpers import link_tubes, filter_ghost_objects

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(output_path='submission.json'):
    # Load configs
    base_config = load_config('configs/base_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    dataset_config = load_config('configs/dataset_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    config = {**base_config, **eval_config, **dataset_config, **model_config}

    logger, _ = setup_logging(config)
    device = torch.device(config['project']['device'])
    
    # Multi-GPU setup
    n_gpus = torch.cuda.device_count()
    if config['project']['multi_gpu'] and n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs with DataParallel for prediction")
    else:
        logger.info("Using single GPU or CPU for prediction")
    
    model = FSVODModel(config)
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    if config['project']['multi_gpu'] and n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # FP16 setup
    use_fp16 = config['project']['fp16'] and torch.cuda.is_available()
    if use_fp16:
        logger.info("Using FP16 for prediction inference")
    
    # Use dataloader in 'predict' mode (no anns needed)
    predict_loader = get_dataloader(config, mode='predict', batch_size=1)  # bs=1 for long videos
    
    predictions = []
    with torch.no_grad():
        for batch in predict_loader:
            video_id = batch['video_id'][0]  # str
            query_frames = batch['query_frames'].to(device)  # [1, T, C, H, W] -> squeeze
            supports = torch.stack([s[0] for s in batch['supports']]).to(device).squeeze(0)  # [3, C, H, W]
            
            with autocast(enabled=use_fp16):
                preds = model(query_frames.squeeze(0), supports)
            
            # Post-process: link/filter tubes, group to intervals
            tubes = link_tubes(preds['tubes'], config)
            filtered_tubes = filter_ghost_objects(tubes)
            detections = group_to_intervals(filtered_tubes)
            
            predictions.append({
                'video_id': video_id,
                'detections': detections
            })
    
    # Save as JSON (include all videos, even empty)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
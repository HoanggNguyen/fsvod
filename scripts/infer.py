import torch
import yaml
import cv2
import torch.nn as nn  # For DataParallel
from torch.cuda.amp import autocast  # For FP16
from models.full_model import FSVODModel
from data.preprocess import preprocess_video
from utils.helpers import link_tubes, filter_ghost_objects
from utils.logging import setup_logging  # Add for logging multi-GPU info

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(video_path, support_paths, output_path='output.json'):
    # Load configs
    base_config = load_config('configs/base_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')  # Reuse for infer
    dataset_config = load_config('configs/dataset_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    config = {**base_config, **eval_config, **dataset_config, **model_config}

    logger, _ = setup_logging(config)  # Use logger for multi-GPU info
    device = torch.device(config['project']['device'])
    
    # Multi-GPU setup
    n_gpus = torch.cuda.device_count()
    if config['project']['multi_gpu'] and n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs with DataParallel for inference")
    else:
        logger.info("Using single GPU or CPU for inference")
    
    model = FSVODModel(config)
    model.load_state_dict(torch.load('checkpoints/model.pth'))  # Load to base
    if config['project']['multi_gpu'] and n_gpus > 1:
        model = nn.DataParallel(model)  # Wrap
    model = model.to(device)
    model.eval()
    
    # FP16 setup
    use_fp16 = config['project']['fp16'] and torch.cuda.is_available()
    if use_fp16:
        logger.info("Using FP16 for inference")
    
    # Preprocess query video (drone-like, 15 FPS sample)
    query_frames = preprocess_video(video_path, fps=config['preprocessing']['fps_sample'], size=config['preprocessing']['frame_size'])
    query_frames = torch.stack(query_frames).to(device)  # [T, C, H, W]
    
    # Load supports (few-shot images)
    supports = []
    for sp in support_paths:
        img = cv2.imread(sp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        supports.append(img)
    supports = torch.stack(supports).to(device)
    
    with torch.no_grad():
        with autocast(enabled=use_fp16):
            preds = model(query_frames, supports)
    
    # Post-process: Link tubes, filter ghosts
    tubes = link_tubes(preds['tubes'], config)
    filtered_tubes = filter_ghost_objects(tubes)
    
    # Save/output (e.g., JSON with tubes/boxes)
    import json
    with open(output_path, 'w') as f:
        json.dump({'tubes': [t.tolist() for t in filtered_tubes], 'scores': preds['scores'].tolist()}, f)
    print(f"Inference completed. Output saved to {output_path}")

if __name__ == "__main__":
    # Example call: python infer.py path/to/drone_video.mp4 ['support1.jpg', 'support2.jpg']
    import sys
    main(sys.argv[1], sys.argv[2:])
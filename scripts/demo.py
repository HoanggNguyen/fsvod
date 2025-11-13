import torch
import yaml
import cv2
import torch.nn as nn  # For DataParallel
from torch.cuda.amp import autocast  # For FP16
from models.full_model import FSVODModel
from data.preprocess import preprocess_video
from utils.visualization import visualize_tube
from utils.logging import setup_logging  # Add for logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(video_path, support_paths, output_video='demo_output.mp4'):
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
        logger.info(f"Using {n_gpus} GPUs with DataParallel for demo")
    else:
        logger.info("Using single GPU or CPU for demo")
    
    model = FSVODModel(config)
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    if config['project']['multi_gpu'] and n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # FP16 setup
    use_fp16 = config['project']['fp16'] and torch.cuda.is_available()
    if use_fp16:
        logger.info("Using FP16 for demo inference")
    
    # Preprocess
    query_frames = preprocess_video(video_path, fps=15, size=config['preprocessing']['frame_size'])  # 15 FPS for demo
    query_tensors = torch.stack(query_frames).to(device)
    
    supports = []
    for sp in support_paths:
        img = cv2.imread(sp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        supports.append(img)
    supports = torch.stack(supports).to(device)
    
    with torch.no_grad():
        with autocast(enabled=use_fp16):
            preds = model(query_tensors, supports)
    
    # Visualize (tubes to per-frame boxes)
    tube_boxes = [preds['tubes'][i] for i in range(len(query_frames))]  # Simplify
    visualize_tube(query_frames, tube_boxes, output_path=output_video)
    print(f"Demo video saved to {output_video}")

if __name__ == "__main__":
    # Example: python demo.py path/to/video.mp4 ['support1.jpg']
    import sys
    main(sys.argv[1], sys.argv[2:])
import torch
import time
import yaml
import torch.nn as nn  # For DataParallel
from torch.cuda.amp import autocast  # For FP16
from data.dataloader import get_dataloader
from models.full_model import FSVODModel
from utils.metrics import evaluate, spatio_temporal_iou  # ST-IoU as primary benchmark
from utils.logging import setup_logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs
    base_config = load_config('configs/base_config.yaml')
    eval_config = load_config('configs/eval_config.yaml')
    dataset_config = load_config('configs/dataset_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    config = {**base_config, **eval_config, **dataset_config, **model_config}

    logger, writer = setup_logging(config)
    device = torch.device(config['project']['device'])
    
    # Multi-GPU setup
    n_gpus = torch.cuda.device_count()
    if config['project']['multi_gpu'] and n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs with DataParallel for evaluation")
    else:
        logger.info("Using single GPU or CPU for evaluation")
    
    model = FSVODModel(config)
    model.load_state_dict(torch.load('checkpoints/model.pth'))  # Load to base model
    if config['project']['multi_gpu'] and n_gpus > 1:
        model = nn.DataParallel(model)  # Wrap after loading
    model = model.to(device)
    model.eval()
    
    # FP16 setup for inference
    use_fp16 = config['project']['fp16'] and torch.cuda.is_available()
    if use_fp16:
        logger.info("Using FP16 for evaluation inference")
    
    eval_loader = get_dataloader(config, mode='val', batch_size=config['evaluation']['batch_size'])
    
    all_metrics = []
    total_time = 0
    num_frames = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            start_time = time.time()
            query_frames = batch['query_frames'].to(device)
            supports = torch.stack([s[0] for s in batch['supports']]).to(device)
            gts = batch['query_anns']  # Ground truths for ST-IoU
            
            with autocast(enabled=use_fp16):
                preds = model(query_frames, supports)
            end_time = time.time()
            total_time += end_time - start_time
            num_frames += len(query_frames)
            
            metrics = evaluate(config, preds, gts)  # Includes ST-IoU mean, AP, FPS
            all_metrics.append(metrics)
    
    avg_metrics = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    fps = num_frames / total_time  # Measured FPS (>=15 check)
    avg_metrics['measured_fps'] = fps
    logger.info(f"Evaluation Metrics: {avg_metrics}")
    if fps < config['evaluation']['inference_fps_target']:
        logger.warning(f"FPS {fps} below target {config['evaluation']['inference_fps_target']}")
    
    if writer:
        for k, v in avg_metrics.items():
            writer.add_scalar(f"Metrics/{k}", v)

if __name__ == "__main__":
    main()
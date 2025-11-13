import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn  # For DataParallel
import yaml
from torch.cuda.amp import autocast, GradScaler  # For FP16

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataloader import get_dataloader
from models.full_model import FSVODModel
from utils.logging import setup_logging
from utils.helpers import link_tubes

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs (merge inherits manually for simplicity)
    base_config = load_config('configs/base_config.yaml')
    train_config = load_config('configs/train_config.yaml')
    dataset_config = load_config('configs/dataset_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    config = {**base_config, **train_config, **dataset_config, **model_config}  # Merge

    logger, writer = setup_logging(config)
    device = torch.device(config['project']['device'])
    
    # Multi-GPU setup
    n_gpus = torch.cuda.device_count()
    if config['project']['multi_gpu'] and n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs with DataParallel")
    else:
        logger.info("Using single GPU or CPU")
    
    model = FSVODModel(config)
    if config['project']['multi_gpu'] and n_gpus > 1:
        model = nn.DataParallel(model)  # Wrap for multi-GPU
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=0)
    
    # FP16 setup (AMP)
    use_fp16 = config['project']['fp16'] and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_fp16)
    if use_fp16:
        logger.info("Using FP16 mixed precision for training")
    
    train_loader = get_dataloader(config, mode='train')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            query_frames = batch['query_frames'].to(device)  # [B, T, C, H, W]
            supports = torch.stack([s[0] for s in batch['supports']]).to(device)  # [B*S, C, H, W]
            targets = {
                'tpn': batch['query_anns'],  # Adapt to dict with gt_obj, gt_boxes, gt_id
                'labels': ...,  # From anns
                'support_labels': [s[2] for s in batch['supports']]
            }
            
            optimizer.zero_grad()
            with autocast(enabled=use_fp16):
                loss = model(query_frames, supports, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # For clip_grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {avg_loss}")
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
        
        scheduler.step()
    
    # Save model (unwrap if DataParallel)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), 'checkpoints/model.pth')
    else:
        torch.save(model.state_dict(), 'checkpoints/model.pth')
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
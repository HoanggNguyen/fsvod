import logging
import os
import torch.utils.tensorboard as tb

def setup_logging(config):
    logger = logging.getLogger(config['project']['name'])
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # Create log directory if it doesn't exist
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    logger.addHandler(fh)
    
    if config['logging']['tensorboard']:
        writer = tb.SummaryWriter(config['logging']['log_dir'])
    else:
        writer = None
    
    return logger, writer

def log_metrics(logger, metrics, epoch):
    logger.info(f"Epoch {epoch}: {metrics}")
    # Add to tensorboard if enabled
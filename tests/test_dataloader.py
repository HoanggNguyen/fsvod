# tests/test_dataloader.py
import pytest
import torch
from data.dataloader import FSVODDataset, get_dataloader
from data.preprocess import preprocess_video
from data.dataset_utils import load_annotations

@pytest.fixture
def config():
    # Mock config for drone dataset
    return {
        'data': {'root': 'dataset/'},
        'dataset': {
            'paths': {
                'samples': 'samples/',
                'videos': 'drone_video.mp4',
                'supports': 'object_images/',
                'annotations': 'annotations/annotations.json'
            },
            'preprocessing': {
                'frame_size': [1024, 1024],
                'fps_sample': 25,
                'augmentations': {}
            }
        },
        'few_shot': {'n_way': 1, 'k_shot': 3},  # Adapted for 3 supports per video
        'training': {'batch_size': 1},
        'evaluation': {'batch_size': 1},
        'data': {'num_workers': 0, 'pin_memory': False}  # For testing
    }

def test_dataset_loading(config, tmp_path):
    # Create mock dataset structure
    annotations = [{'video_id': 'drone_video_001', 'annotations': [{'bboxes': [{'frame': 1, 'x1':0, 'y1':0, 'x2':10, 'y2':10}]}]}]
    ann_path = tmp_path / 'annotations' / 'annotations.json'
    ann_path.parent.mkdir()
    with open(ann_path, 'w') as f:
        json.dump(annotations, f)
    
    video_dir = tmp_path / 'samples' / 'drone_video_001'
    video_dir.mkdir(parents=True)
    support_dir = video_dir / 'object_images'
    support_dir.mkdir()
    for i in range(3):
        (support_dir / f'img_{i+1}.jpg').touch()  # Mock images
    
    # Mock video (not actually loading cv2)
    config['data']['root'] = str(tmp_path)
    
    dataset = FSVODDataset(config, mode='val')
    assert len(dataset) == 1
    item = dataset[0]
    assert item['video_id'] == 'drone_video_001'
    assert len(item['supports']) == 3  # 3 supports
    assert isinstance(item['query_frames'], torch.Tensor)  # Preprocessed frames
    assert len(item['query_anns']) == 1  # Annotations

def test_dataloader(config, tmp_path):
    # Similar mock as above
    # ... (setup mock)
    
    dataloader = get_dataloader(config, mode='val')
    assert len(dataloader) == 1
    batch = next(iter(dataloader))
    assert 'video_id' in batch
    assert batch['supports'][0][0].shape[0] == 3  # Channel first
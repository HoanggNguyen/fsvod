# FSVOD Project

## Directory Structure

```
data/
├── __init__.py
├── dataloader.py
├── dataset_utils.py
├── preprocess.py
└── dataset/
    ├── annotations/
    │   └── annotations.json
    └── samples/
        └── [video files]
```

thêm dataset ban tổ chức vào mục dataset.

## Usage

Chạy pipeline: run python scripts/train.py

## Dataset Configuration

Check thêm phần dataloader và dataset_config, đoạn:

```yaml
dataset:
  name: drone_object_search
  constraints:
    total_classes: 10  # Default number of classes for model initialization
    base_classes: 10
    val_novel_classes: 0
    test_novel_classes: 0
    videos_per_class: none  # Per-sample videos
    total_videos: dynamic  # Determined from annotations.json
    annotation_fps: full  # Per-frame when object appears (25 FPS videos)
    format: custom-json  # Single annotations.json with video_id and intervals
    specialization: small_objects_drone 
```
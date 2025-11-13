import cv2
import torch
import torchvision.transforms as T
import random
import torchvision.transforms.functional as TF  # For affine (drone perspective)

def preprocess_video(video_path, fps=15, size=(1024, 1024)):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = max(1, int(frame_rate / fps))  # Sample to meet >=15 FPS potential
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)  # Higher res for small objects
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # [C, H, W]
        frame_idx += 1
    cap.release()
    return frames

def augment_frame(frame, augmentations):
    transforms = []
    if random.random() < augmentations.get('random_flip', 0):
        transforms.append(T.RandomHorizontalFlip(1.0))
    if augmentations.get('random_crop', False):
        transforms.append(T.RandomResizedCrop(size=frame.shape[1:], scale=(0.5, 1.0)))  # Bias to small crops for small objects
    if augmentations.get('color_jitter', False):
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # Drone-specific: Simulate bird's-eye view with affine rotation/shear
    if 'random_rotate' in augmentations:
        angle = random.uniform(augmentations['random_rotate'][0], augmentations['random_rotate'][1])
        frame = TF.affine(frame, angle=angle, translate=(0,0), scale=1.0, shear=0)
    # Small object focus: Random scale
    if 'random_scale' in augmentations:
        scale = random.uniform(augmentations['random_scale'][0], augmentations['random_scale'][1])
        frame = TF.affine(frame, angle=0, translate=(0,0), scale=scale, shear=0)
    transform = T.Compose(transforms + [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(frame.float() / 255.0)
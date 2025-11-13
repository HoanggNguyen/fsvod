import cv2
import numpy as np
import torch

def draw_boxes(frame, boxes, labels, colors=(0, 255, 0)):
    """Draw bounding boxes on frame for visualization"""
    frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame
    for box, label in zip(boxes, labels):
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors, 2)
        cv2.putText(frame, str(label), (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors, 2)
    return frame

def visualize_tube(video_frames, tube_boxes, output_path='output.mp4'):
    """Render tube on video for drone small object inspection"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (video_frames[0].shape[1], video_frames[0].shape[0]))  # 15 FPS
    for frame, boxes in zip(video_frames, tube_boxes):
        vis_frame = draw_boxes(frame, boxes, ['target'])
        out.write(vis_frame)
    out.release()
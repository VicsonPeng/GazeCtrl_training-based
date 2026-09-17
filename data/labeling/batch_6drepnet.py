"""
Batch 6DRepNet Head Pose Estimation Worker
==========================================
Reads a manifest JSON of person crop paths, runs 6DRepNet on each,
and outputs head-pose-derived gaze vectors.

Usage (from the 6DRepNet venv):
    python gaze_workers/batch_6drepnet.py --manifest <path> --output <path>
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# Add sixdrepnet to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SIXDREPNET_DIR = os.path.join(PROJECT_ROOT, "6DRepNet", "sixdrepnet")
sys.path.insert(0, SIXDREPNET_DIR)

from torchvision.models.resnet import Bottleneck
from model import SixDRepNet2
import utils


def load_model(weights_path, device):
    """Load SixDRepNet2 model (ResNet50 backbone)."""
    model = SixDRepNet2(Bottleneck, [3, 4, 6, 3])
    saved = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = saved['model_state_dict'] if 'model_state_dict' in saved else saved
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def headpose_to_gaze_vector(pitch_deg, yaw_deg):
    """
    Convert head pose (pitch, yaw) in degrees to a 3D facing-direction unit vector.
    Convention: dx=right, dy=down, dz=forward (into screen).
    """
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    
    # 6DRepNet native convention: yaw > 0 means head turned LEFT.
    # To match image-space (dx > 0 = right), we must negate the X direction.
    # IMPORTANT: 6DRepNet uses extreme pitches (e.g. 180 or -180) to represent "facing away".
    # In standard math, cos(180) is negative, which would FLIP the left/right direction.
    # To preserve the correct image-space left/right direction (which strictly follows yaw),
    # we use abs(cos(pitch)) for dx. This preserves magnitude without flipping the axis.
    dx = -(abs(math.cos(pitch_rad)) * math.sin(yaw_rad))
    
    # Native pitch > 0 means UP. So -sin(pitch) < 0 (pointing up in image).
    dy = -math.sin(pitch_rad)
    
    # For dz, we WANT the extreme pitch to flip the sign to negative (looking away).
    dz = math.cos(pitch_rad) * math.cos(yaw_rad)
    
    # Normalize
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    if norm > 1e-8:
        dx /= norm
        dy /= norm
        dz /= norm
    
    return dx, dy, dz


def main():
    parser = argparse.ArgumentParser(description="Batch 6DRepNet head pose estimation")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--head_crop_dir", default=None, help="Directory to save head crop images (optional)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[6DRepNet] Device: {device}")

    # Load model
    weights_path = os.path.join(
        SIXDREPNET_DIR, "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
    )
    if not os.path.isfile(weights_path):
        print(f"[6DRepNet] ERROR: Weights not found at {weights_path}")
        sys.exit(1)

    model = load_model(weights_path, device)
    print(f"[6DRepNet] Model loaded from {weights_path}")

    # Load YOLO head model
    yolo_head_path = os.path.join(PROJECT_ROOT, "yolo_seg", "yolov8_head.pt")
    if not os.path.isfile(yolo_head_path):
        print(f"[6DRepNet] ERROR: YOLO head model not found at {yolo_head_path}")
        sys.exit(1)
    
    head_yolo = YOLO(yolo_head_path)
    print(f"[6DRepNet] YOLO Head model loaded from {yolo_head_path}")

    # Read manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    crop_paths = manifest.get("crop_paths", [])
    print(f"[6DRepNet] Processing {len(crop_paths)} crops...")

    # Head crop output dir
    head_crop_dir = None
    if args.head_crop_dir:
        head_crop_dir = args.head_crop_dir
        os.makedirs(head_crop_dir, exist_ok=True)
        print(f"[6DRepNet] Head crops will be saved to: {head_crop_dir}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}
    failed = []

    for crop_path in crop_paths:
        try:
            img = cv2.imread(crop_path)
            if img is None:
                failed.append(crop_path)
                continue

            # 1. Run YOLO Head detection
            head_results = head_yolo(img, verbose=False)
            head_boxes = head_results[0].boxes

            if head_boxes is None or len(head_boxes) == 0:
                print(f"[6DRepNet] No head detected: {crop_path}")
                failed.append(crop_path)
                continue

            # 2. Get highest confidence head box & scale by 1.4x
            best_idx = int(torch.argmax(head_boxes.conf).item())
            bx = head_boxes.xyxy[best_idx].cpu().numpy()
            
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            w = (bx[2] - bx[0]) * 1.4
            h = (bx[3] - bx[1]) * 1.4
            
            x1 = max(0, int(cx - w / 2.0))
            y1 = max(0, int(cy - h / 2.0))
            x2 = min(img.shape[1], int(cx + w / 2.0))
            y2 = min(img.shape[0], int(cy + h / 2.0))
            
            head_crop = img[y1:y2, x1:x2]
            if head_crop.shape[0] == 0 or head_crop.shape[1] == 0:
                print(f"[6DRepNet] Invalid head crop shape: {crop_path}")
                failed.append(crop_path)
                continue

            # 3. Save head crop image (if requested)
            head_crop_path = None
            if head_crop_dir is not None:
                crop_name = os.path.splitext(os.path.basename(crop_path))[0]
                head_crop_save_path = os.path.join(head_crop_dir, crop_name + ".jpg")
                cv2.imwrite(head_crop_save_path, head_crop)
                head_crop_path = head_crop_save_path

            img_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                R_pred = model(img_tensor)

            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            pitch_deg = float(euler[:, 0].cpu().numpy()[0])
            yaw_deg = float(euler[:, 1].cpu().numpy()[0])
            roll_deg = float(euler[:, 2].cpu().numpy()[0])

            dx, dy, dz = headpose_to_gaze_vector(pitch_deg, yaw_deg)

            results[crop_path] = {
                "gaze_source": "6drepnet_headpose",
                "gaze_dx": round(dx, 6),
                "gaze_dy": round(dy, 6),
                "gaze_dz": round(dz, 6),
                "pitch_deg": round(pitch_deg, 4),
                "yaw_deg": round(yaw_deg, 4),
                "roll_deg": round(roll_deg, 4),
                "head_crop_path": head_crop_path,
                "head_bbox": [int(x1), int(y1), int(x2), int(y2)],
            }

        except Exception as e:
            print(f"[6DRepNet] Error on {crop_path}: {e}")
            failed.append(crop_path)

    output_data = {
        "results": results,
        "failed": failed,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[6DRepNet] Done: {len(results)} success, {len(failed)} failed")
    print(f"[6DRepNet] Results saved to {args.output}")


if __name__ == "__main__":
    main()

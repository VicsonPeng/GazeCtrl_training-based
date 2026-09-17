"""
Batch 3DGazeNet Gaze Estimation Worker (WSL)
=============================================
Uses the official 3DGazeNet demo's GazeNetInference class to ensure
preprocessing (affine transform cropping, undo_roll) matches training.

Outputs gaze vectors in IMAGE-SPACE convention:
  dx > 0 = looking right
  dy > 0 = looking down
  dz > 0 = looking forward (into screen)

Usage (from the WSL 3DGazeNet conda env):
    conda activate 3DGazeNet
    python gaze_workers/batch_3dgazenet.py --manifest <path> --output <path>
"""

import argparse
import json
import os
import sys
import traceback

import cv2
import numpy as np
import torch

# ─── Setup paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEMO_DIR = os.path.join(PROJECT_ROOT, "3DGazeNet", "demo")

# Add demo dir to sys.path so its `models` and `utils` packages are importable
sys.path.insert(0, DEMO_DIR)

from inference import GazeNetInference


def main():
    parser = argparse.ArgumentParser(description="Batch 3DGazeNet gaze estimation")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--det_thresh", type=float, default=0.5,
                        help="InsightFace face detection threshold")
    parser.add_argument("--det_size", type=int, default=640,
                        help="InsightFace detection input size")
    args = parser.parse_args()

    # Change to demo dir so relative paths (eyes3d.pkl, checkpoints) resolve correctly
    original_dir = os.getcwd()
    os.chdir(DEMO_DIR)

    print(f"[3DGazeNet] Device: {args.device}")
    print(f"[3DGazeNet] Using official demo inference pipeline")
    print(f"[3DGazeNet] det_thresh={args.det_thresh}, det_size={args.det_size}")

    # Initialize the official demo's inference handler
    # This loads: InsightFace face detector + GazePredictorHandler (with correct preprocessing)
    gazenet = GazeNetInference(det_thresh=args.det_thresh, det_size=args.det_size)
    print("[3DGazeNet] GazeNetInference initialized")

    os.chdir(original_dir)

    # Read manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    crop_paths = manifest.get("crop_paths", [])
    print(f"[3DGazeNet] Processing {len(crop_paths)} crops...")

    results = {}
    failed = []

    for i, crop_path in enumerate(crop_paths):
        try:
            img = cv2.imread(crop_path)
            if img is None:
                failed.append(crop_path)
                continue

            # Step 1: Face detection — get det_score + kps in one call
            faces_raw = gazenet.face_detector.model.get(img)
            if len(faces_raw) == 0:
                failed.append(crop_path)
                continue

            face = faces_raw[0]  # Use the first (highest confidence) face
            det_score = float(face.det_score)
            kps = face.kps.astype(int)

            # Check for NaN in landmarks
            if np.any(np.isnan(face.kps)):
                print(f"[3DGazeNet] NaN landmarks: {crop_path}")
                failed.append(crop_path)
                continue

            # Step 2: Gaze prediction using the official demo's handler
            with torch.no_grad():
                gaze_result = gazenet.gaze_predictor(img, kps, undo_roll=True)

            if gaze_result is None:
                failed.append(crop_path)
                continue

            gaze_vec = gaze_result.get('gaze_combined')
            if gaze_vec is None:
                gaze_vec = gaze_result.get('gaze_out')
            if gaze_vec is None:
                failed.append(crop_path)
                continue

            # Check for NaN in gaze vector
            if np.any(np.isnan(gaze_vec)):
                print(f"[3DGazeNet] NaN gaze vector: {crop_path}")
                failed.append(crop_path)
                continue

            # Convert to image-space convention (matching 6DRepNet: dx>0=right, dy>0=down, dz>0=forward)
            # The official demo's draw_gaze applies -vector for both X and Y to draw correctly on the image.
            # This proves that 3DGazeNet's raw X and Y axes are inverted relative to image-space.
            # Raw dz is mostly positive for front-facing people, so it remains unchanged.
            gaze_dx = round(float(gaze_vec[0]), 6)
            gaze_dy = round(float(-gaze_vec[1]), 6)
            gaze_dz = round(float(gaze_vec[2]), 6)

            results[crop_path] = {
                "gaze_source": "3dgazenet",
                "gaze_dx": gaze_dx,
                "gaze_dy": gaze_dy,
                "gaze_dz": gaze_dz,
                "facial_landmarks": face.kps.tolist(),
                "pitch_deg": None,
                "yaw_deg": None,
                "roll_deg": None,
                "det_score": det_score,
            }

            if (i + 1) % 500 == 0:
                print(f"[3DGazeNet] Processed {i+1}/{len(crop_paths)}, "
                      f"{len(results)} success, {len(failed)} failed")

        except Exception as e:
            print(f"[3DGazeNet] Error on {crop_path}: {e}")
            traceback.print_exc()
            failed.append(crop_path)

        # Periodic save every 5000 samples for crash recovery
        if (i + 1) % 5000 == 0:
            tmp_data = {"results": results, "failed": failed}
            with open(args.output + ".tmp", 'w', encoding='utf-8') as f:
                json.dump(tmp_data, f, ensure_ascii=False)
            print(f"[3DGazeNet] Checkpoint saved at {i+1}/{len(crop_paths)}")

    # Save final output
    output_data = {
        "results": results,
        "failed": failed,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[3DGazeNet] Done: {len(results)} success, {len(failed)} failed")
    print(f"[3DGazeNet] Results saved to {args.output}")


if __name__ == "__main__":
    main()

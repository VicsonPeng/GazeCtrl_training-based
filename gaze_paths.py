"""Every path the pipeline needs, resolved from environment variables.

Nothing here is required: the defaults reproduce the layout this project was
developed on, so scripts run unchanged in that environment. To run elsewhere,
export the roots you actually have -- the rest derive from them:

    export GAZE_ROOT=/your/workspace          # everything defaults under here
    export GAZE_DATASET=$GAZE_ROOT/gaze_dataset      # Wan turnaround frames + labels
    export GAZE_RAW=$GAZE_ROOT/condition_dataset_v2  # source crops + phase3 HITL output
    export GAZE_MODELS=$GAZE_ROOT/models             # diffusion / vae / controlnet weights
    export GAZE_WORK=$GAZE_ROOT/runs                 # checkpoints, per-run csvs, logs
    export GAZE_OUT=$GAZE_ROOT/outputs               # figures, demo payloads, html
    export GAZE_THIRD=$GAZE_ROOT/third_party         # 6DRepNet, yolo_seg, Wan2.2
"""
import os

def _env(name, default):
    return os.environ.get(name, default)

ROOT    = _env("GAZE_ROOT",    "/project/vicson_gaze")
DATASET = _env("GAZE_DATASET", ROOT + "/gaze_dataset")
RAW     = _env("GAZE_RAW",     ROOT + "/condition_dataset_v2")
MODELS  = _env("GAZE_MODELS",  ROOT + "/gaze_controlnet/models")
WORK    = _env("GAZE_WORK",    "/home/vicson/gaze_diag")
OUT     = _env("GAZE_OUT",     ROOT + "/gaze_controlnet/eval_samples")
THIRD   = _env("GAZE_THIRD",   ROOT)

# third-party checkpoints used as measurement instruments, never trained here
SIXDREP_DIR  = _env("GAZE_SIXDREP_DIR",  THIRD + "/6DRepNet/sixdrepnet")
SIXDREP_CKPT = _env("GAZE_SIXDREP_CKPT", SIXDREP_DIR + "/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")
YOLO_HEAD    = _env("GAZE_YOLO_HEAD",    THIRD + "/yolo_seg/yolov8_head.pt")
WAN_DIR      = _env("GAZE_WAN_DIR",      THIRD + "/Wan2.2")

# weights inside GAZE_MODELS (layout as produced by setup/download_models.py)
BACKBONE = MODELS + "/diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
VAE      = MODELS + "/vae/split_files/vae/qwen_image_vae.safetensors"
CONTROLNET_INIT = MODELS + "/controlnet/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors"

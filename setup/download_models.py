"""
download_models.py
Fetch the frozen weights the gaze ControlNet pipeline runs on.

Required (nothing works without these three):
  1. qwen_image_edit_2509_fp8_e4m3fn.safetensors   Qwen backbone, frozen   (~19 GB)
  2. qwen_image_vae.safetensors                    Qwen 3D VAE             (~250 MB)
  3. Qwen-Image-InstantX-ControlNet-Union          ControlNet init         (~3.3 GB)

Optional (--optional), not loaded by any current code path:
  4. qwen_2.5_vl_7b_fp8_scaled.safetensors         Qwen VL text encoder    (~8.7 GB)
     The pipeline feeds a zero tensor as txt_emb -- every bit of control arrives through
     the ControlNet -- so this is only needed if you re-enable text conditioning.
  5. Qwen-Image-Lightning-4steps-V1.0.safetensors  Lightning LoRA          (~1.6 GB)
     For few-step sampling experiments; the training and eval scripts do not use it.

Files land in the layout gaze_paths.py expects, so the two stay in sync.

Usage:
  python download_models.py             # the three required weights
  python download_models.py --optional  # those plus the text encoder and Lightning LoRA
  python download_models.py --check     # report what is present, download nothing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze_paths import MODELS as DEFAULT_MODELS_DIR  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
#   dir  : subdirectory of the models root that the file is downloaded into
#   path : path inside the HuggingFace repo
#   The final location is <models_dir>/<dir>/<path> -- which is exactly what
#   gaze_paths.BACKBONE / VAE / CONTROLNET_INIT point at.
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_MODELS = {
    "backbone": {
        "dir":   "diffusion_models",
        "repo":  "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "path":  "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "size":  "~19 GB",
    },
    "vae": {
        "dir":   "vae",
        "repo":  "Comfy-Org/Qwen-Image_ComfyUI",
        "path":  "split_files/vae/qwen_image_vae.safetensors",
        "size":  "~250 MB",
    },
    "controlnet_union": {
        "dir":   "controlnet",
        "repo":  "Comfy-Org/Qwen-Image-InstantX-ControlNets",
        "path":  "split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors",
        "size":  "~3.3 GB",
    },
}

OPTIONAL_MODELS = {
    "text_encoder": {
        "dir":   "text_encoders",
        "repo":  "Comfy-Org/Qwen-Image_ComfyUI",
        "path":  "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "size":  "~8.7 GB",
    },
    "lora_lightning": {
        "dir":   "loras",
        "repo":  "lightx2v/Qwen-Image-Lightning",
        "path":  "Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        "size":  "~1.6 GB",
    },
}


def _local_path(spec: dict, models_dir: str) -> str:
    return os.path.join(models_dir, spec["dir"], spec["path"])


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(key: str, spec: dict, models_dir: str, check_only: bool = False) -> bool:
    local_path = _local_path(spec, models_dir)
    exists = os.path.exists(local_path)

    status = "✓" if exists else "✗"
    note   = f"({_file_mb(local_path) / 1024:.1f} GB)" if exists else f"({spec['size']})"
    print(f"  [{status}] {key:20s}  {os.path.basename(local_path)}  {note}")

    if check_only or exists:
        return exists

    target_dir = os.path.join(models_dir, spec["dir"])
    os.makedirs(target_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        print(f"        Downloading from {spec['repo']} ...")
        got = hf_hub_download(repo_id=spec["repo"], filename=spec["path"], local_dir=target_dir)
        print(f"        Saved → {got}")
        return os.path.exists(local_path)
    except Exception as e:
        print(f"        ERROR: {e}")
        print(f"        Manual download:")
        print(f"          https://huggingface.co/{spec['repo']}/resolve/main/{spec['path']}")
        print(f"          → {local_path}")
        return False


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default=DEFAULT_MODELS_DIR,
                    help="weights root (default: gaze_paths.MODELS, i.e. $GAZE_MODELS)")
    ap.add_argument("--optional",   action="store_true",
                    help="also fetch the text encoder and Lightning LoRA (no current code loads them)")
    ap.add_argument("--check",      action="store_true",
                    help="report which weights are present without downloading")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    to_download = dict(REQUIRED_MODELS)
    if args.optional:
        to_download.update(OPTIONAL_MODELS)

    print(f"Models directory: {args.models_dir}\n")
    action = "Checking" if args.check else "Downloading"
    print(f"{action} {len(to_download)} model(s):\n")

    results = {}
    for key, spec in to_download.items():
        results[key] = _download_one(key, spec, args.models_dir, check_only=args.check)

    ok  = sum(v for v in results.values())
    tot = len(results)
    print(f"\n{'Present' if args.check else 'Done'}: {ok}/{tot}")

    if ok < tot:
        missing = [k for k, v in results.items() if not v]
        print(f"Missing: {missing}")
        if args.check:
            sys.exit(1)


if __name__ == "__main__":
    main()

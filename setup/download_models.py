"""
download_models.py
Downloads all models required by the Qwen Image Edit 2509 + ControlNet Union pipeline.

Models (ComfyUI format from Comfy-Org HuggingFace repos):
  1. qwen_image_edit_2509_fp8_e4m3fn.safetensors  (FLUX diffusion model, ~12 GB)
  2. qwen_image_vae.safetensors                    (VAE, ~335 MB)
  3. qwen_2.5_vl_7b_fp8_scaled.safetensors        (Qwen VL text encoder, ~8 GB)
  4. Qwen-Image-Lightning-4steps-V1.0.safetensors  (Lightning LoRA, ~800 MB)
  5. Qwen-Image-InstantX-ControlNet-Union          (ControlNet, ~3 GB)

Additionally for FLUX.1-dev training (sd-scripts):
  6. flux1-dev.safetensors          (FLUX.1-dev diffusion model, ~24 GB)
  7. ae.safetensors                 (FLUX VAE)
  8. clip_l.safetensors             (CLIP-L text encoder)
  9. t5xxl_fp16.safetensors         (T5-XXL text encoder, ~9 GB)

Usage:
  python download_models.py                  # download all
  python download_models.py --inference_only # Qwen models only (skip FLUX training models)
  python download_models.py --training_only  # FLUX training models only
  python download_models.py --check          # verify files exist without downloading
"""

import argparse
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_MODELS = {
    "unet": {
        "local": "diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "repo":  "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "path":  "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "size":  "~12 GB",
    },
    "vae_qwen": {
        "local": "vae/qwen_image_vae.safetensors",
        "repo":  "Comfy-Org/Qwen-Image_ComfyUI",
        "path":  "split_files/vae/qwen_image_vae.safetensors",
        "size":  "~335 MB",
    },
    "clip_qwen": {
        "local": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "repo":  "Comfy-Org/Qwen-Image_ComfyUI",
        "path":  "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "size":  "~8 GB",
    },
    "lora_lightning": {
        "local": "loras/Qwen-Image-Lightning-4steps-V1.0.safetensors",
        "repo":  "lightx2v/Qwen-Image-Lightning",
        "path":  "Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        "size":  "~800 MB",
    },
    "controlnet_union": {
        "local": "controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors",
        "repo":  "Comfy-Org/Qwen-Image-InstantX-ControlNets",
        "path":  "split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors",
        "size":  "~3 GB",
    },
}

TRAINING_MODELS = {
    "flux_unet": {
        "local": "diffusion_models/flux1-dev.safetensors",
        "repo":  "black-forest-labs/FLUX.1-dev",
        "path":  "flux1-dev.safetensors",
        "size":  "~24 GB",
    },
    "flux_vae": {
        "local": "vae/ae.safetensors",
        "repo":  "black-forest-labs/FLUX.1-dev",
        "path":  "ae.safetensors",
        "size":  "~335 MB",
    },
    "clip_l": {
        "local": "text_encoders/clip_l.safetensors",
        "repo":  "comfyanonymous/flux_text_encoders",
        "path":  "clip_l.safetensors",
        "size":  "~250 MB",
    },
    "t5xxl": {
        "local": "text_encoders/t5xxl_fp16.safetensors",
        "repo":  "comfyanonymous/flux_text_encoders",
        "path":  "t5xxl_fp16.safetensors",
        "size":  "~9 GB",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(key: str, spec: dict, models_dir: str, check_only: bool = False) -> bool:
    local_path = os.path.join(models_dir, spec["local"])
    exists = os.path.exists(local_path)

    status = "✓" if exists else "✗"
    note   = f"({spec['size']})" if not exists else f"({_file_mb(local_path):.0f} MB)"
    print(f"  [{status}] {key:20s}  {os.path.basename(local_path)}  {note}")

    if check_only or exists:
        return exists

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        print(f"        Downloading from {spec['repo']} ...")
        hf_hub_download(
            repo_id=spec["repo"],
            filename=spec["path"],
            local_dir=os.path.join(models_dir, os.path.dirname(spec["local"])),
            local_dir_use_symlinks=False,
        )
        # huggingface_hub saves as the basename in local_dir; rename if needed
        downloaded = os.path.join(
            models_dir, os.path.dirname(spec["local"]),
            os.path.basename(spec["path"])
        )
        if downloaded != local_path and os.path.exists(downloaded):
            os.rename(downloaded, local_path)
        print(f"        Saved → {local_path}")
        return True
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
    ap.add_argument("--models_dir",     default=MODELS_DIR)
    ap.add_argument("--inference_only", action="store_true",
                    help="Only download Qwen inference models (skip FLUX training models)")
    ap.add_argument("--training_only",  action="store_true",
                    help="Only download FLUX training models")
    ap.add_argument("--check",          action="store_true",
                    help="Check which models are present without downloading")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    to_download = {}
    if not args.training_only:
        to_download.update(INFERENCE_MODELS)
    if not args.inference_only:
        to_download.update(TRAINING_MODELS)

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

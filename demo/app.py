"""Interactive gaze-conditioned reorientation — upload a photo, dial a gaze, get the
same person turned that way.

Runs three ways, all from this one file:
  * a Hugging Face Space on ZeroGPU  (`import spaces` succeeds -> the worker is decorated)
  * a Space or server with a dedicated GPU (`PERSISTENT_GPU=1`, models stay resident)
  * locally, `python demo/app.py`, for checking a checkpoint before publishing it

Weights are resolved through gaze_paths / env vars, so the same file works in the lab and
on the Space. On a Space, set GAZE_MODELS to the snapshot directory and GAZE_CKPT to the
ControlNet checkpoint.
"""
import os, sys, math, io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
import gradio as gr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "src"))
from gaze_paths import BACKBONE, VAE, WORK                      # noqa: E402
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode,  # noqa: E402
                         vae_decode, patchify, unpatchify,
                         compute_rope_freqs_3d, compute_text_rope_freqs)

CKPT = os.environ.get("GAZE_CKPT", f"{WORK}/_cn_hitl/cn_step085000.safetensors")
IMSZ, GRID = 512, 32
PERSISTENT = os.environ.get("PERSISTENT_GPU", "0") == "1"

try:                      # ZeroGPU allocates a GPU per call; absent everywhere else
    import spaces
    ZERO = True
except Exception:
    spaces = None
    ZERO = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS = {}


def _load():
    """Load once per process. On ZeroGPU the weights land on CPU and are moved inside the
    GPU-decorated worker; anywhere else they go straight to the device and stay there."""
    if _MODELS:
        return _MODELS
    dev = "cpu" if (ZERO and not PERSISTENT) else DEVICE
    vae = load_qwen_vae(str(VAE), device=dev)
    bb = QwenBackbone(str(BACKBONE), device=dev)
    cn = QwenControlNet()
    cn.load_state_dict(load_file(CKPT))
    cn.to(dev, torch.bfloat16).eval()
    _MODELS.update(vae=vae, bb=bb, cn=cn, dev=dev)
    return _MODELS


def _square(img: Image.Image, size: int = IMSZ) -> Image.Image:
    """Pad to square then resize — the exact framing the model was trained on."""
    w, h = img.size
    m = max(w, h)
    pad = ((m - w) // 2, (m - h) // 2, m - w - (m - w) // 2, m - h - (m - h) // 2)
    return ImageOps.expand(img.convert("RGB"), pad).resize((size, size), Image.BILINEAR)


def _to_latent_input(img: Image.Image, dev) -> torch.Tensor:
    a = np.array(_square(img)).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev, torch.bfloat16)


def gaze_vector(yaw_deg: float, pitch_deg: float):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p))


def condition_image(gz) -> Image.Image:
    """The control signal itself: a flat RGB field, R=dx G=dy B=dz, each mapped linearly."""
    dx, dy, dz = gz
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    col = np.clip(np.array([(dx / n + 1) / 2, (dy / n + 1) / 2, (dz / n + 1) / 2]) * 255, 0, 255)
    return Image.fromarray(np.full((IMSZ, IMSZ, 3), col, dtype=np.uint8))


def _schedule(n, shift=3.0, start=0.999):
    ts = start / (shift * (1 - start) + start + 1e-9)
    t = torch.linspace(ts, 1e-5, n + 1)[:-1]
    return shift / (shift + (1.0 / t.clamp(1e-6, 1 - 1e-6) - 1.0))


@torch.no_grad()
def _run(image: Image.Image, yaw: float, pitch: float, steps: int, cn_scale: float, seed: int):
    M = _load()
    vae, bb, cn = M["vae"], M["bb"], M["cn"]
    dev = DEVICE
    if M["dev"] != dev:                      # ZeroGPU: bring the weights in for this call
        vae.to(dev); bb.to(dev); cn.to(dev)
        M["dev"] = dev

    gz = gaze_vector(yaw, pitch)
    cond = _to_latent_input(condition_image(gz), dev)
    ref = patchify(vae_encode(vae, _to_latent_input(image, dev))).to(torch.bfloat16)
    cp = patchify(vae_encode(vae, cond)).to(torch.bfloat16)

    te = torch.zeros((1, 7, 3584), device=dev, dtype=torch.bfloat16)
    imf = compute_rope_freqs_3d(GRID, GRID, t_patches=2, device=dev)
    imf_cn = compute_rope_freqs_3d(GRID, GRID, t_patches=1, device=dev)
    txf = compute_text_rope_freqs(7, device=dev)

    torch.manual_seed(int(seed))
    x = torch.randn(1, 16, 1, 64, 64, device=dev, dtype=torch.bfloat16)
    S = _schedule(int(steps))
    for i, s in enumerate(S):
        si = s.unsqueeze(0).to(dev, x.dtype)
        sn = S[i + 1].unsqueeze(0).to(dev, x.dtype) if i + 1 < len(S) else torch.zeros(1, device=dev, dtype=x.dtype)
        res = [r * cn_scale for r in cn.forward(patchify(x), cp, te, si, img_freqs=imf_cn, txt_freqs=txf)]
        v = bb.forward(patchify(x), te, si, res, img_freqs=imf, txt_freqs=txf, ref_packed=ref)
        x = x - (si - sn).view(1, 1, 1, 1, 1) * unpatchify(v, GRID, GRID)

    out = vae_decode(vae, x)[0]
    arr = ((out.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


_worker = spaces.GPU(duration=120)(_run) if (ZERO and not PERSISTENT) else _run


def reorient(image, yaw, pitch, steps, cn_scale, seed):
    if image is None:
        raise gr.Error("Upload a photo of a person first.")
    gz = gaze_vector(yaw, pitch)
    out = _worker(image, yaw, pitch, steps, cn_scale, seed)
    facing = "toward camera" if gz[2] > 0.25 else ("facing away" if gz[2] < -0.25 else "profile")
    readout = (f"gaze (dx, dy, dz) = ({gz[0]:+.3f}, {gz[1]:+.3f}, {gz[2]:+.3f})   ·   {facing}\n"
               f"yaw {yaw:+.0f}°   pitch {pitch:+.0f}°   ·   {int(steps)} steps   "
               f"cn_scale {cn_scale}   seed {int(seed)}\n"
               f"checkpoint {Path(CKPT).stem}")
    return out, condition_image(gz), readout


DESCRIPTION = """
# Gaze-Conditioned Body Reorientation

Upload a photo of a person and dial a target gaze. The model turns the **same identity** —
head and body — to face that direction, over the full 360° including facing away.

The gaze vector drives generation directly: it is rendered as a flat RGB field
(**R** = dx, **G** = dy, **B** = dz) that a trained ControlNet reads. There is no skeleton
or pose-prediction step in between, and no text prompt — the colour patch on the right *is*
the entire control signal. Identity comes from the source photo, which enters the frozen
Qwen-Image-Edit backbone as an in-context reference.

**Convention** — dx > 0 looks image-right, dy > 0 looks up, dz > 0 looks toward the camera.
Yaw ±180° means facing away.
"""

with gr.Blocks(title="Gaze-Conditioned Reorientation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            src = gr.Image(label="Source photo", type="pil", height=380)
            yaw = gr.Slider(-180, 180, value=0, step=5, label="Yaw  (− left · + right · ±180 away)")
            pitch = gr.Slider(-45, 45, value=0, step=5, label="Pitch  (− down · + up)")
            with gr.Accordion("Sampling", open=False):
                steps = gr.Slider(8, 32, value=16, step=1, label="Denoising steps")
                scale = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                  label="ControlNet scale (1.0 matches training)")
                seed = gr.Number(value=0, precision=0, label="Seed")
            go = gr.Button("Reorient", variant="primary")
        with gr.Column(scale=1):
            out = gr.Image(label="Output", height=380)
            cond = gr.Image(label="Condition fed to the ControlNet", height=130)
            info = gr.Textbox(label="What was run", lines=3, show_copy_button=True)

    gr.Markdown(
        "Works best on a single person, roughly upright, face visible in the source. "
        "Identity preservation is marginal in the current checkpoint (ArcFace similarity "
        "≈ 0.28 on held-out identities) and degrades most at large rotations — that "
        "limitation is real and documented in the repository."
    )
    go.click(reorient, [src, yaw, pitch, steps, scale, seed], [out, cond, info])

if __name__ == "__main__":
    demo.queue(max_size=12).launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")

---
title: Gaze-Conditioned Body Reorientation
emoji: 👁️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Turn a person in a photo to face any gaze direction
---

# Gaze-Conditioned Body Reorientation

Upload a photo, dial a target gaze `(dx, dy, dz)`, and the same identity turns — head and
body — to face that direction, over the full 360° including facing away.

A frozen **Qwen-Image-Edit-2509** backbone is steered by a trained **InstantX ControlNet
Union**. The gaze vector is the control signal itself, rendered as a flat RGB field
(R = dx, G = dy, B = dz); there is no skeleton stage and no text prompt.

Code, training recipe, dataset pipeline and honest evaluation:
**https://github.com/VicsonPeng/GazeCtrl_training-based**

## Deploying this Space

This directory is what gets pushed to the Space. `app.py` and `gaze_paths.py` are copied
from the repository root, and `qwen_models.py` from `src/`:

```bash
huggingface-cli repo create gaze-reorientation --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<user>/gaze-reorientation && cd gaze-reorientation
cp <repo>/demo/space/README.md  <repo>/demo/space/requirements.txt .
cp <repo>/demo/app.py <repo>/gaze_paths.py <repo>/src/qwen_models.py .
git add -A && git commit -m "gaze reorientation demo" && git push
```

Then upload the weights (they are far too large for the Space repo itself) to a model repo
and point the Space at it:

```bash
huggingface-cli upload baki0115/gaze-controlnet-qwen-image-edit cn_step085000.safetensors
```

and set these as Space **variables and secrets**:

| variable | value |
|---|---|
| `GAZE_MODELS` | local snapshot dir for the backbone + VAE (see `app.py`) |
| `GAZE_CKPT` | path to `cn_step085000.safetensors` after download |
| `PERSISTENT_GPU` | `1` only on a dedicated-GPU Space; leave unset on ZeroGPU |

### A caveat worth reading before you pay for hardware

The fp8 backbone is **20.4 GB** and the ControlNet another **3.5 GB**. On ZeroGPU the
process keeps the weights on CPU between calls and moves them to the GPU per request, so the
first seconds of every generation are spent on that transfer. It fits in 40 GB, but it is
not fast. A dedicated-GPU Space with `PERSISTENT_GPU=1` keeps them resident and is far more
responsive; ZeroGPU is the free option, not the good one.

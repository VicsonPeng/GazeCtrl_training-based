# Gaze-Conditioned Body Reorientation — training-based

Edit a single photo of a person so the **same identity** turns to face a target gaze
direction `(dx, dy, dz)` — head *and* body — by conditioning a frozen
Qwen-Image-Edit-2509 through a trained InstantX ControlNet.

The gaze vector drives generation **directly**. There is no intermediate skeleton or
pose-prediction stage: the condition is the gaze itself, rendered as a dense map the
ControlNet reads. That directness is the point of the method.

> This repository is the **training-based** half of the project. The training-free half
> lives in its own repository.

| | |
|---|---|
| **Live demo** | [https://vicsonpeng.github.io/GazeCtrl_training-based/](https://vicsonpeng.github.io/GazeCtrl_training-based/) — browse 1500 generations: 25 sources × 12 yaw × 5 pitch, click any cell |
| **Weights** | [baki0115/gaze-controlnet-qwen-image-edit](https://huggingface.co/baki0115/gaze-controlnet-qwen-image-edit) — trained ControlNet checkpoints |
| **Labels** | [baki0115/GazeCtrl_dataset](https://huggingface.co/datasets/baki0115/GazeCtrl_dataset) — the human-curated gaze labels |
| **Interactive results** | [baki0115/gaze-ctrl-360](https://huggingface.co/spaces/baki0115/gaze-ctrl-360) — 18 sources x 24 gaze targets, with ground-truth reference frames |

> **Released model: `cn_step040000` from curriculum stage 2.** Selected over the perceptual-loss
> variant on 3000 generations across 25 sources at 1024². It follows the requested direction more
> accurately (median |Δyaw| 19.2° vs 25.2°, within-30° 65.5% vs 55.6%, better on 22 of 25 sources);
> the variant preserves identity slightly better (`id_sim` +0.301 vs +0.264) and ships alongside as
> a reference checkpoint. See [Model selection](#model-selection).


---

## Method

```
              target gaze (dx,dy,dz)
                        │
                        ▼
              flat RGB map  R=(dx+1)/2  G=(dy+1)/2  B=(dz+1)/2      512×512
                        │
                     VAE encode
                        │
   noisy latent ──► InstantX ControlNet Union  (5 double-stream blocks, TRAINED)
                        │
                        │ 5 residuals
                        ▼
   source image ──► Qwen-Image-Edit-2509 fp8  (FROZEN)  ──► velocity ──► output
     (VAE-encoded, in-context reference token: this is what carries identity)
```

With more than one person in the frame a flat whole-frame condition is ambiguous, so the
colour is painted inside a box instead — see
[Multi-person](#multi-person-saying-which-person-the-condition-is-about).

- **Backbone**: `qwen_image_edit_2509_fp8_e4m3fn`, frozen (1933 tensors, `requires_grad=False`).
- **ControlNet**: InstantX ControlNet Union, initialised from its pretrained weights
  (181/181 keys load) — 1768 M trainable parameters.
- **Condition**: the gaze vector as a whole-image solid RGB colour. Per-axis linear, no clamp.
- **Text**: none. `txt_emb` is a zero tensor, so *all* control comes through the ControlNet.
- **Identity**: the source image enters the backbone as an in-context reference token
  (`ref_packed`), not through the ControlNet.
- **Objective**: flow-matching velocity MSE, head-weighted:
  `w = 1 + head_boost · head_mask + eye_boost · eye_mask`.

**Gaze convention** (used everywhere): `dx > 0` right, `dy > 0` **up**, `dz > 0` toward the
camera. See [`docs/conventions.md`](docs/conventions.md) — the label sources disagree about
this and the disagreement is a real source of error.

---

## Results

Trained on 183 identities / 1864 human-curated frames (COCO + WiderFace driven through
Wan2.2 turnaround generation), 90 000 steps.

| | training identities | held-out 70k | held-out 90k |
|---|---|---|---|
| `gaze_err` ↓ | **19.11°** | 55.97° | 57.28° |
| `ctrl_slope` (1.0 ideal) | +0.874 | +0.642 | +0.667 |
| `id_sim` (ArcFace) ↑ | — | +0.277 | +0.282 |
| `face_det` | — | 70 % | 75 % |

Read these honestly:

- **The generalisation gap is large** — 19° on training identities against ~56° on held-out
  ones. Held-out is only 8 identities, and *all* of their labels are automatic
  (6DRepNet/3DGazeNet) while 48 % of the training labels are human, so part of that gap is
  label noise rather than generalisation. It is not all label noise.
- **70k → 90k did not improve held-out performance.** The last 20 000 steps only moved the
  training-set metric.
- **`id_sim ≈ 0.28` sits on ArcFace's own same-person threshold.** Identity preservation is
  marginal, which is what motivates the curriculum work below.
- **`cross_gaze_lpips` is not an identity metric.** It measures how much the output changes
  across gazes, mixing wanted pose change with unwanted appearance drift; it cannot separate
  them, and a low value means collapse, not fidelity.
- **`gaze_err` is a proxy.** It measures *head pose* (6DRepNet) against the requested gaze
  vector. Head pose and gaze differ by construction — median 26.7° between 3DGazeNet and
  6DRepNet on the same frame — so a floor of that order is expected.

### A metric defect that invalidated four runs

`ctrl_slope` polyfits raw `atan2` degrees with no unwrapping, so a target near ±180° can be
measured as −178° for a head physically 7° away — enough to drag a perfect +1.000 fit to
−0.349. 92 % of the SHHQ identities carry such a pick (SHHQ is 30 % back-facing) against
31 % of the HITL ones. **Four SHHQ runs spanning 60 000 steps all logged `ctrl_slope ≈ 0`
while the checkpoints in fact track the condition at +1.25.** The wrap-safe fit is in
[`eval/eval_true_v3.py`](eval/eval_true_v3.py) and
[`train/train_stage2.py`](train/train_stage2.py); see
[`docs/conventions.md`](docs/conventions.md). `gaze_err` was never affected.

### Curriculum

**Stage 1** ([`train/train_stage1.py`](train/train_stage1.py)) pretrains on SHHQ full-body
turnaround pairs with a head-preserving random crop, mixed 80/20 with the close-up HITL
pairs, to learn the head-pose ↔ full-body relation. 35 000 steps from the InstantX weights.

**Stage 2** ([`train/train_stage2.py`](train/train_stage2.py)) inverts the mix — the full
HITL pool at 70 %, SHHQ kept at 30 % as replay so full-body reorientation does not decay —
and turns on `eye_boost` (eye boxes exist only for HITL), which is what targets "head pose
right, pupils wrong". 50 000 steps from stage 1, so that 35 000 + 50 000 = 85 000 matches
the HITL-only checkpoint it is compared against. Every eval scores **both** pools, so SHHQ
decay is visible as it happens.

Stage-1 checkpoint at 35 000 steps, all 50 SHHQ identities × 5 targets, `cn_scale = 1.0`,
16 steps, seed 0 ([`eval/eval_true_v3.py`](eval/eval_true_v3.py)):

| | stage 1, full frame | stage 1, head crop | HITL 85k, full frame |
|---|---|---|---|
| `ctrl_slope` wrap-safe | +1.254 | **+1.098** | +1.143 |
| `ctrl_slope` raw (broken) | +0.236 | +0.598 | +0.454 |
| `gaze_err` ↓ | 46.59° | **27.29°** | 35.91° |
| median \|yaw error\| ↓ | 28.3° | **14.4°** | 20.0° |
| `id_sim` ↑ | +0.186 | **+0.265** | +0.115 |

The head-crop column is measured over 159/250 generations against 243/250 for the full
frame — YOLO head detection fails more often on tight crops — so it is the optimistic end of
the range, not a like-for-like number.

---

## Multi-person: saying *which* person the condition is about

A flat gaze map fills the whole frame, so it says "look this way" about the *picture*. With
one person in frame that is the same thing as saying it about the person. With two it is not,
and the second person turns as well.

The condition is therefore painted **inside a box** and left neutral grey outside. A box and
not a mask: a silhouette encodes the pose the person is *currently* in, which is exactly the
thing the edit has to change, so a mask-shaped condition tells the model the answer it is
supposed to produce and leaks the source pose into the target.

### The box rule

Three boxes, and a priority between them:

1. **box1(target)** — the target's head box. The final box **must contain it whole**.
2. **box1(other)** — every other person's head box. The final box **must contain none** of them.
3. **box2** — the target's own mask bounding box. The final box stays **as close to it as it can**.

Rule 1 wins any conflict with rule 2: a box that has cut off the head of the person it is
meant to be about cannot be repaired by excluding somebody else.
[`src/target_head.py`](src/target_head.py) implements it; `headbox_box()` starts from box2
plus a 5 % pad and cuts one side at a time, ranking candidates by how much of the other
heads survive first and target coverage second.

Measured over the 186 benchmark pairs
([`eval/make_evalboxes.py`](eval/make_evalboxes.py)), against the plain mask bounding box:

| | mask bbox | head rule |
|---|---|---|
| bystander **body** pixels inside the box | 44.7 % | **10.4 %** |
| pairs that fully exclude every other head | — | 73.1 % (136/186) |
| target's head kept whole (of the 177 where a head was detected) | — | **100 %** |
| target mask covered | — | 98.6 % median |
| box area, as a share of the frame | 57.4 % | 46.6 % |

The honest caveat: 23.1 % of boxes cover less than 90 % of the target's mask, because
excluding a neighbour's head sometimes costs a shoulder or an arm. And on 9 of 186 pairs no
head was detected at all, so rule 1 could not be applied.

### What the box bought

`cn_step006000` of the post-trained run, 166 pairs that both settings share, seed 0,
4-step Lightning LoRA at scale 1.0, `cn_scale` 1.0, shift 3.0, 768 px, real human gaze
labels. The two runs differ in the geometry of the condition box and in nothing else.

| | flat whole-frame fill | head-rule box |
|---|---|---|
| target gaze error (median) | 30.9° | 30.9° |
| bystander drift (median) | 13.4° | **8.8°** |
| bystander moved ≥ 30° | 34.9 % | **30.7 %** |

The bystander improvement is real (the box is better on 103 of 162 pairs, sign test
p = 6.8 × 10⁻⁴) and **the target's own control is unchanged** (p = 0.64) — the box narrows
what the condition speaks about without weakening what it says. Counting the pairs that
cross the 30° line in either direction: 13 fixed, 6 broken, net 7.

### The four-way outcome split, and what it is worth

Two people, two questions, four outcomes: did the **target** turn, and did the **bystander**
stay put. [`eval/quadeval.py`](eval/quadeval.py) reports the split.

The criteria are not obvious and the first version of them was wrong. All 166 pairs were
labelled by hand so that the automatic split could be checked against a person:

| | human labels | automatic |
|---|---|---|
| ① target turned / bystander stayed | **83 (50.0 %)** | 109 (65.7 %) |
| ② target turned / bystander turned too | **56 (33.7 %)** | 40 (24.1 %) |
| ③ neither moved | **25 (15.1 %)** | 13 (7.8 %) |
| ④ target still / bystander turned | **2 (1.2 %)** | 4 (2.4 %) |

The first criteria agreed with the human labels on **41 %** of pairs — 55 % on the "did the
target turn" axis, which is barely better than a coin. Three defects caused it:

- `base_err − gaze_err > 20°` can never fire when `base_err` is already small, so any pair
  whose source happened to face near the requested direction was scored "did not turn"
  however well the output matched. Replaced by an absolute criterion OR'd with a relative one.
- A 15° threshold for "the bystander moved" is far tighter than what a person calls moving.
- An out-of-focus passer-by in the background counted as a bystander. The same head measured
  0.0° of drift in the real target frame, which is self-contradictory. Every genuine bystander
  in the benchmark has a head at least 0.478× the target's width; the background heads sit at
  0.33–0.38×, so half the target's head width separates them cleanly.

Corrected, agreement is **66.3 %** overall (77 % on the target axis, 88 % on the bystander
axis). What remains is a one-sided bias: the automatic split puts 65.7 % of pairs in ① where
the human labels put 50.0 %, about 16 points optimistic. **It is sound for ranking
checkpoints of the same lineage and it is not a fact about the world.** Quote it with the
human-agreement figure attached, or not at all.

### Three hypotheses about bystander drift that the data refused

Worth recording, because each was plausible and each is now closed:

- *The composites teach "everybody turns".* No. In the multi-person training images the
  bystander is pixel-identical between input and ground truth — median absolute difference
  **1.37** over the bystander's own mask, against **33.4** over the target's. The supervision
  already says the bystander must not move.
- *The model never saw two equally prominent people.* No. The bystander-to-target head-height
  ratio in the corpus has median **1.04**, and 87 % of pairs are above 0.9.
- *The box is what drags the bystander along.* No. Among pairs where the bystander moved, the
  box covers **7.5 %** of its body; among pairs where it stayed, **11.3 %** — the wrong way
  round, and not significant. On 12 of the 58 drifting pairs the box does not touch the
  bystander at all. Drift is scene-determined, not box-determined.

---

## Repository layout

```
gaze_paths.py              every path, resolved from environment variables
setup/download_models.py   fetch backbone / VAE / ControlNet weights

src/
  qwen_models.py           Qwen backbone, VAE, QwenControlNet, patchify/RoPE helpers
  gaze_hsv_renderer.py     legacy HSV condition renderer (the RGB path superseded it)
  target_head.py           which person the condition is about, and the box it is painted in
  gaze3d_runner.py         the gaze3d estimator, used for measurement only (CC-BY-NC-4.0)

data/wan/                  1. generate turnaround videos with Wan2.2
  gen_dataset.sh             batch driver across free GPUs
  run_one_video.sh           one (sample, mode): generate → 6DRepNet → sample → extract
  sample_frames.py           farthest-point sampling in gaze space
  finalize_dataset.py        extract chosen frames + write labels

data/labeling/             2. label the frames, then cull them by hand
  batch_3dgazenet.py         eye-gaze labels   (fails on profile/back)
  batch_6drepnet.py          head-pose labels  (covers every direction)
  headpose_images.py         6DRepNet second opinion over the whole set
  select_top20.py            fix label convention, filter, 20 diverse frames/identity
  relabel_pitch.py           insightface pitch re-labelling for profile frames
  phase1b_filter.py          pick source photos: single person, eyes visible, sharp
  phase3_select.py           identity-flag + blur filter + frontal quota
  phase3_dedup_id.py         per-identity cross-video farthest-point dedup
  phase4_build.py            builds "Gaze Label Studio", the human review tool

data/compose/              3. compose multi-person training images from single-person pairs
  make_multiperson.py        paste a second person into a turnaround pair, at matched scale

data/preprocess/           4. precompute what training needs
  precompute_head_boxes.py       head boxes in the 512 pad-to-square space
  precompute_head_boxes_dyaug.py the same for the dy-augmented frames
  eye_precompute.py              eye boxes for eye_boost
  build_heldout.py               held-out csv (identities never trained on)
  build_headbox_verify.py        payload for the head-box audit page
  make_trainboxes.py             condition boxes for a corpus, by the head-box rule

train/
  train_gaze_controlnet.py   baseline trainer (HITL-only and SHHQ-only runs)
  train_stage1.py            curriculum stage 1: crop augmentation + mixed pools
  train_stage2.py            curriculum stage 2: HITL-major + SHHQ replay, wrap-safe eval
  train_stage3.py            refinement attempt: loss rebalance + sigma shift (NEGATIVE result)
  train_stage4.py            refinement attempt: LPIPS on the decoded one-step estimate
  train_stage5.py            mask channel + box-restricted condition (multi-person)
  run_train_hitl.sh          waits for a free GPU, auto-resumes after a crash
  run_train_stage1.sh
  run_train_stage2.sh
  run_train_stage3.sh
  run_train_stage4.sh

eval/
  eval_true_v2.py            gaze_err, ctrl_slope, cross_gaze_lpips, id_sim, face_det
  eval_true_v3.py            wrap-safe ctrl_slope + head-crop mode; dumps picks.csv + images
  demo_sweep.py              final demo grid: many sources x 60 gaze targets, any resolution
  run_demo_sweep.sh          sharded runner with GPU claim-locking and resume
  sweep_grid.py              dense 12 yaw x 5 pitch sweep with per-cell measurement
  sweep_ckpt_scale.py        checkpoint x cn_scale grid on one photo
  res_test.py                framing/resolution comparison (512/768, full vs head crop)
  res_probe.py               512/768/1024 incl. extreme angles; VRAM and timing
  res_probe_ar.py            square vs native aspect ratio at matched token budgets
  full_rotation_v2.py        full 360° sweep → demo payload
  build_srcs_v2.py           pick front/side/back source frames + GT galleries
  crop_preview2.py           sanity-check the crop augmentation visually
  eval_gt.py                 score a run against the human-labelled photograph benchmark
  make_evalboxes.py          condition boxes for the benchmark, by the head-box rule
  other_gaze.py              measure every NON-target person's gaze, source vs output
  quadeval.py                the four-way outcome split, with its human-calibrated criteria

demo/
  app.py                     Gradio app (Hugging Face Space)
  build_demo_v2.py           bundle sweeps into the interactive page
  templates/                 page templates
```

---

## Setup

### 1. Environment

```bash
conda create -n gaze python=3.10 -y && conda activate gaze
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Paths

Nothing is hardcoded. Defaults reproduce the layout this was developed on; export what
differs for you:

```bash
export GAZE_ROOT=/your/workspace
export GAZE_MODELS=$GAZE_ROOT/models            # diffusion / vae / controlnet weights
export GAZE_DATASET=$GAZE_ROOT/gaze_dataset     # turnaround frames + label csvs
export GAZE_RAW=$GAZE_ROOT/condition_dataset_v2 # source crops + phase3 HITL output
export GAZE_WORK=$GAZE_ROOT/runs                # checkpoints and per-run csvs
export GAZE_OUT=$GAZE_ROOT/outputs              # figures and demo payloads
export GAZE_THIRD=$GAZE_ROOT/third_party        # 6DRepNet, yolo_seg, Wan2.2
```

### 3. Weights

```bash
python setup/download_models.py        # Qwen-Image-Edit-2509 fp8, VAE, InstantX ControlNet Union
```

Third-party measurement models, placed under `$GAZE_THIRD`:

| what | where it goes | used for |
|---|---|---|
| [6DRepNet360](https://github.com/thohemp/6DRepNet) + `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` | `6DRepNet/sixdrepnet/` | head-pose labels and `gaze_err` |
| [3DGazeNet](https://github.com/Vagver/3DGazeNet) | `3DGazeNet/` | eye-gaze labels (frontal only) |
| YOLOv8 head detector (`yolov8_head.pt`) | `yolo_seg/` | head boxes |
| [Wan2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2) | `Wan2.2/` | turnaround video generation |

These are instruments and data generators. None of them is trained here.

---

## Reproducing

### Build the dataset

```bash
bash data/wan/gen_dataset.sh                     # turnaround videos, distributed over free GPUs
python data/labeling/batch_3dgazenet.py          # eye-gaze labels
python data/labeling/batch_6drepnet.py           # head-pose labels
python data/labeling/select_top20.py             # fix convention, filter, 20 frames/identity
python data/labeling/phase3_select.py            # identity flag + blur filter + frontal quota
python data/labeling/phase3_dedup_id.py          # cross-video dedup per identity
python data/labeling/phase4_build.py             # → Gaze Label Studio (open the HTML, label by hand)
```

`phase4_build.py` emits a self-contained HTML tool. A human goes through every frame and
picks *use 3D / use 6D / set own dy / discard*; its JSON export becomes the training csv.
**This human pass is the dataset's main contribution** — the automatic labels alone are not
good enough, and the audit pages under `eval/` exist to show why.

### Compose multi-person training images

Every turnaround pair has exactly one person in it, so a model trained on them alone never
learns that the condition is about *somebody* rather than about the picture.
`data/compose/make_multiperson.py` builds two-person images out of one-person pairs: person A
turns (both frames come from A's own video), and person B is cut from a single frame of a
different video and pasted into **both** frames at the same place, at a scale matched by head
height, with one colour transform fitted to the scene. B is therefore bit-identical between
input and ground truth, which is the supervision "this person must not move".

```bash
python data/compose/make_multiperson.py --n 10000 --out $GAZE_CORPORA/mp_v5 --dev cuda
python data/preprocess/make_trainboxes.py --data $GAZE_CORPORA/mp_v5
```

The second command writes `boxes_headrule.json` next to the corpus. Those boxes already carry
the 5 % pad, so training reads them with `--box_pad 0`.

### Precompute

```bash
python data/preprocess/precompute_head_boxes.py
python data/preprocess/eye_precompute.py
python data/preprocess/build_heldout.py          # identities held out of training
```

### Train

Baseline (the 90k run the results table reports):

```bash
bash train/run_train_hitl.sh
```

Curriculum stage 1 (crop augmentation, mixed pools, from the InstantX weights):

```bash
bash train/run_train_stage1.sh
```

Then stage 2 — HITL at 70 %, SHHQ kept at 30 % as replay, initialised from the
stage-1 checkpoint (50 000 steps, so the total matches the 85 000-step HITL-only run it is
compared against):

```bash
bash train/run_train_stage2.sh
```

Both runners wait for a genuinely free GPU (<1000 MiB, confirmed on two polls 8 s apart),
auto-resume from the newest checkpoint after a crash, and refuse to start when the disk
cannot hold another checkpoint.

Key flags (`train/train_stage1.py --help` for all):

| flag | meaning |
|---|---|
| `--csv` / `--csv2`, `--mix2` | the two pools and the probability of drawing from the second |
| `--crop_p`, `--crop_k`, `--crop_minfrac` | crop probability; crop side ≥ `k` × head size and ≥ `minfrac` × short side |
| `--head_boost`, `--eye_boost` | loss weight multipliers on the head and eye regions |
| `--cn_scale` | scale applied to the ControlNet residuals |
| `--fixed_src` | 1 pins the source to the most frontal frame; 0 draws a random pair |

The crop is taken in **original image coordinates** (not in the pad-to-square space, which
would drag letterbox bars in), must contain the head boxes of **both** source and target,
and the **same rectangle is applied to both** — a different framing per side makes the pair
unlearnable. Head and eye boxes are re-mapped through the crop, or the loss would weight the
wrong patches. `eval/crop_preview2.py` renders what the augmentation actually does.

#### Stage 5 — the condition becomes about a person

```bash
python train/train_stage5.py --data $GAZE_CORPORA/mp_v5 --out $GAZE_WORK/_cn_pt2B \
       --resume $GAZE_WORK/_cn_stage2/cn_step040000.safetensors --start_step 0 \
       --boxes boxes_headrule.json --cond_box 1 --box_pad 0 \
       --steps 6000 --lr 5e-5 --accum 8 --mask_drop 0.1
```

Two things differ from stage 2: a zero-initialised mask channel summed into the ControlNet's
image embedding, so the model is told *which* person; and `--cond_box 1`, which paints the
gaze colour only inside that person's box. Because the mask embedder starts at zero, step 0
reproduces stage 2 exactly — anything that changes afterwards was bought by this run.

### Evaluate

```bash
python eval/eval_true_v2.py --ckpt $GAZE_WORK/_cn_hitl/cn_step090000.safetensors \
                            --csv  $GAZE_WORK/_cn_hitl/heldout.csv \
                            --cn_scale 1.0 --n_ids 0
```

`--n_ids 0` evaluates every identity in the csv. Pass `--cn_scale 1.0` to match training —
the value used at inference changes the numbers.

Metric definitions, and what each one cannot tell you, are in
[`docs/conventions.md`](docs/conventions.md).

#### Against the photograph benchmark

`eval_true_v*` compare a run against frames this project's own data engine generated, which
makes them an in-distribution diagnostic. `eval/eval_gt.py` compares against **real
photographs with human gaze labels** — 50 scenes, 360 views, and for a pair (view A → view B)
the frame in which that person really is facing the requested direction.

```bash
# condition boxes for the benchmark, by the head-box rule (once)
python eval/make_evalboxes.py --all 1

# score a checkpoint on the multi-person split
NAME=myrun python eval/eval_gt.py --name myrun --size 768 --steps 4 --seed 0        --pairs $GAZE_GT_EVAL/pairs_multi200.json        --cn $GAZE_WORK/_cn_pt2B/cn_step006000.safetensors        --cond_box 1 --box_json $GAZE_EVAL/boxes_headrule.json

# every non-target person's gaze, source vs output
python eval/other_gaze.py --runs myrun --out $GAZE_GT_EVAL/og_myrun.csv

# the four-way split
python eval/quadeval.py --name myrun --ogaze $GAZE_GT_EVAL/og_myrun.csv
```

`quadeval.py` drops any pair missing from the cached head-width table, so a run is scored on
the same pairs the criteria were calibrated on. Pass `--require_heads 0` to keep them.

### Interactive demo

The published page is at
**[huggingface.co/spaces/baki0115/gaze-ctrl-360](https://huggingface.co/spaces/baki0115/gaze-ctrl-360)**
— a static viewer over generations produced offline, not a live model. To rebuild it:

```bash
python eval/build_srcs_v2.py                                   # choose front/side/back sources
python eval/full_rotation_v2.py --ckpt <ckpt> --srcs <json> --out <json> --cn_scale 1.0
python demo/build_demo_v2.py                                   # → self-contained HTML
```

`demo/app.py` is the Gradio version that takes an uploaded photo.

---

## Model selection

Two candidates, 3000 generations: 25 sources (SHHQ full-body, SHHQ head-crop, HITL close-up,
and ten photographs from outside the project) × 12 yaw × 5 pitch × 2 checkpoints, at 1024²,
`cn_scale=1.0`, 16 steps, one fixed seed.

| source group | checkpoint | median \|Δyaw\| ↓ | within 30° ↑ | `id_sim` ↑ | face det |
|---|---|---|---|---|---|
| SHHQ full-body | **stage2-40k** | **23.9°** | **56.1 %** | +0.163 | 127/180 |
| | stage4-8k | 34.2° | 45.3 % | **+0.219** | **146/180** |
| SHHQ head-crop | **stage2-40k** | **18.1°** | **66.5 %** | +0.212 | 361/420 |
| | stage4-8k | 22.6° | 61.0 % | **+0.255** | 361/420 |
| HITL close-up | **stage2-40k** | **17.7°** | **70.5 %** | +0.278 | 255/300 |
| | stage4-8k | 25.4° | 54.9 % | **+0.303** | **268/300** |
| zero-shot | **stage2-40k** | **19.9°** | **65.3 %** | +0.318 | 510/600 |
| | stage4-8k | 25.6° | 55.6 % | **+0.356** | 511/600 |
| **all 1500 each** | **stage2-40k** | **19.2°** | **65.5 %** | +0.264 | 1253/1500 |
| | stage4-8k | 25.2° | 55.6 % | **+0.301** | **1286/1500** |

The trade is consistent — the direction of every comparison is the same in all four regimes.
`stage2-40k` is released because the project's claim is that the gaze vector *controls* the
figure, and 6° of median error plus ten points of hit rate are visible where 0.037 of cosine
similarity is not. Note that **both** sit near ArcFace's own same-person threshold (≈0.28), so
this is a choice between two marginal models on identity, not between a good one and a better one.

Reproduce with [`eval/demo_sweep.py`](eval/demo_sweep.py) / [`eval/run_demo_sweep.sh`](eval/run_demo_sweep.sh).

## Resolution: the finding that reframed the facial artifacts

Outputs carried oversized flat-black pupils. Two rounds of loss engineering were spent on it —
rebalancing the head/eye weighting ([`train/train_stage3.py`](train/train_stage3.py), which made
everything worse) and adding a perceptual LPIPS term on the decoded one-step estimate
([`train/train_stage4.py`](train/train_stage4.py), a clean trade). The actual cause was
resolution. Same weights, same seed, same target, changing only framing and generation size:

| mode | eye span | relative | what the eye shows |
|---|---|---|---|
| full frame, 512² *(as shipped)* | 25 px | 1.0× | a dark smudge |
| full frame, 768² | 41 px | 1.6× | iris discernible |
| head crop, 512² | 68 px | 2.7× | lashes, lid, eyeliner |
| head crop, 768² | **106 px** | **4.2×** | full eye anatomy |

At 25 px across both eyes a pupil is about five pixels, and a dark smudge is close to the correct
answer at that scale. **Check that a quantity has room to move before optimising it.**

**Generating above the training resolution is safe here.** The ControlNet saw only a 32×32 patch
grid, yet control does not degrade at 64×64 (1024²). The reason is attributable to the condition
design: the gaze map is a *spatially uniform* colour, so the adapter's function is
position-independent and nothing misaligns when the grid changes. A spatially structured
condition (depth, skeleton) would not be expected to transfer this way. Non-square grids also run
correctly — `compute_rope_freqs_3d` and `unpatchify` take separate `h` and `w` — which would remove
the letterbox waste, but the measured benefit was within noise and it is left unproven.
Probes: [`eval/res_test.py`](eval/res_test.py), [`eval/res_probe.py`](eval/res_probe.py),
[`eval/res_probe_ar.py`](eval/res_probe_ar.py).

## Known limitations

- **Held-out evaluation is thin**: 8 identities, automatic labels only.
- **Label semantics are not uniform.** 3DGazeNet labels only frontal frames (it needs visible
  eyes); 6DRepNet carries every side and back frame; the human labels concentrate on
  profiles. So the meaning of a condition colour shifts systematically with the direction
  being asked for. This is the single biggest known defect in the data.
- **Identity preservation is marginal** (`id_sim ≈ 0.28`) and no loss term currently targets it.
- **The head boxes driving `head_boost` were not all correct.** An audit of 1268 frames
  removed 121 (9.5 %), of which 43 came from the originally precomputed set — earlier
  training weighted the wrong region on those frames.
- **Back-facing outputs cannot be scored for identity**: ArcFace needs a face, so `id_sim`
  is averaged only over generations where one was detected, and `face_det` is reported
  alongside it.
- **The four-way split is optimistic by about 16 points** on class ①, measured against human
  labels on all 166 pairs. Use it to rank checkpoints, not to state a result.
- **The photograph benchmark's multi-person split is small and correlated**: 186 pairs over
  only 21 scenes, 9 pairs per scene, so the effective sample size is well below 186.
- **9 of the 186 benchmark pairs have merged masks** (two people in one mask) and 9 have no
  detected head, so the box rule falls back to the plain mask bounding box on those.
- **The multi-person training images are composites.** Person B is pasted in, so lighting and
  contact shadows are approximate; a model could in principle key on that rather than on the
  condition. The bystander-drift measurements above argue it does not, but the composites are
  not photographs and that bound stands.
- **Pixel-identical is not reachable at bf16.** Re-running the same checkpoint, seed and
  prompt gives a median absolute difference of 0–0.92 within one session and about 1.0 across
  sessions or GPUs, so no pixel comparison below that scale means anything.

# Conventions, label provenance, and what each metric can and cannot tell you

## 1. The gaze convention

One convention is used everywhere in this repository:

```
dx > 0  →  subject looks to the IMAGE right
dy > 0  →  subject looks UP
dz > 0  →  subject looks TOWARD the camera     (dz < 0 = facing away)
```

Vectors are unit length. The condition map encodes them per-axis and linearly, with no clamp:

```python
R = (dx + 1) / 2,  G = (dy + 1) / 2,  B = (dz + 1) / 2      # scaled to 0..255
```

Derived angles used in evaluation and in the demo pages:

```python
yaw   = degrees(atan2(dx, dz))     # 0 = toward camera, +90 = image right, ±180 = facing away
pitch = degrees(asin(dy))          # + = up
```

## 2. The two label sources disagree, and one of them was sign-flipped

Labels come from two estimators that do not measure the same thing and did not agree on
sign:

- **3DGazeNet** estimates **eye gaze**. Its `dx > 0` already means image-right. It requires
  visible eyes, so it produces nothing usable in profile or from behind.
- **6DRepNet** estimates **head pose**. Its raw `dx` and `dy` are **sign-flipped** relative
  to 3DGazeNet and to the image convention, and it flips `dx` again once the subject turns
  past profile.

Mixing the two raw outputs put both left/right conventions into one training set: the same
`dx` sign appeared for subjects looking left and looking right, so the ControlNet could not
learn left/right at all and `gaze_err` sat around 47°.

**The fix, applied to every 6DRepNet-sourced frame** (3DGazeNet frames are left alone):

```python
dx = -hp_dx
dy = -hp_dy
dz =  hp_dz
if dz < 0:        # 6DRepNet flips dx once the subject faces away; undo that
    dx = -dx      # so dx stays in one world frame across the full 360°
```

### The measurement side has the opposite sign, and must not be double-corrected

`_head_dir` (6DRepNet run on a *generated* image during evaluation) computes

```python
dx = -sin(yaw) * cos(pitch);  dy = sin(pitch);  dz = cos(yaw) * cos(pitch)
```

which is *already* `(-hp_dx, -hp_dy, hp_dz)` — the corrected convention. Evaluation must
therefore use `_head_dir`'s output as-is and apply only the `if dz < 0: dx = -dx` flip.
Negating again inverts the measurement: `ctrl_slope` goes negative and `gaze_err` inflates
to ~67°, which looks exactly like a training failure but is purely a metric artefact. The
loss is unaffected — `gaze_err` and `ctrl_slope` are monitoring only.

## 3. Label provenance is tied to direction — the biggest known defect

Counting the human-culled SHHQ set (1268 frames) by label source against gaze direction:

| source | front | 3/4 | side | back-3/4 | back |
|---|---|---|---|---|---|
| 3DGazeNet (eye gaze) | 174 | 146 | 36 | 0 | 0 |
| 6DRepNet (head pose) | 103 | 132 | 239 | 63 | 119 |
| manual (human) | 0 | 0 | 242 | 0 | 0 |

The sources are almost perfectly confounded with direction: frontal frames are mostly eye
gaze, side and back frames can only be head pose, and the human labels sit entirely on
profiles. **The meaning of a condition colour therefore shifts systematically with the
direction being requested** — which is worse than random noise, because it correlates with
the variable being controlled.

On frames where both estimators fired, they disagree by a **median of 26.7°** (mean 38.1°;
44.5 % of frames disagree by more than 30°). That sets a floor under `gaze_err` that no
amount of training removes.

## 4. Metrics

All are computed in `eval/eval_true_v2.py`. For each identity it sorts frames by `dz`,
takes 5 evenly spaced ones as targets, and uses the most frontal frame as the source.

### `gaze_err` (degrees, lower better) — a proxy

```
generated image → YOLOv8 head box → 6DRepNet → (dx,dy,dz)
gaze_err = angle between that vector and the requested condition vector
```

It measures **head pose**, not gaze. Head pose and gaze differ by construction (§3), so a
systematic offset is expected and is not a defect of the model.

### `ctrl_slope` (unitless, 1.0 ideal) — control responsiveness

A linear fit over the 5 targets of *requested* yaw against *measured* yaw.

| value | meaning |
|---|---|
| ≈ 1.0 | ask for 30° of turn, get 30° |
| ≈ 0 | output ignores the condition |
| < 0 | output turns the wrong way |

This separates "does it obey" from "is the measurement accurate" better than `gaze_err` does.

#### The raw formula is broken near +/-180 deg, and it cost us four training runs

Both yaws come from `atan2`, so they live on a circle, but the fit was linear and did no
unwrapping. A target at +175 deg whose generated head lands 7 deg away can be measured as
-178 deg -- 7 deg physically, 353 deg numerically. One such point drags a perfect +1.000 fit
down to **-0.349**:

```
req  = [-90, -45, 0, 45, 175]
meas = [-90, -45, 0, 45, 175]   -> slope = +1.000
meas = [-90, -45, 0, 45, -178]  -> slope = -0.349     (physical error: 7 deg)
```

92 % of the SHHQ identities carry at least one pick with `|yaw| > 150 deg` (SHHQ is 30 %
back-facing frames), against 31 % of the HITL identities. That asymmetry, not a difference
in what the model learned, is why **four separate SHHQ runs spanning 60 000 steps all logged
`ctrl_slope ~ 0`** while the same checkpoints measure **+1.25** once the wrap is handled.

The fix, used by `eval/eval_true_v3.py` and `train/train_stage2.py`, fits the measurement
against its own unwrapped representative:

```python
def wrap180(a): return (a + 180.0) % 360.0 - 180.0
slope = np.polyfit(req, req + wrap180(meas - req), 1)[0]
```

1.0 still means ideal and 0.0 still means "ignores the condition", but a near-180 deg target
no longer detonates the fit. `eval_true_v3.py` reports `ctrl_slope_RAW` alongside it so the
new numbers stay comparable to the old training logs, and dumps every
`(requested_yaw, measured_yaw)` pair to `picks.csv` so any future metric can be recomputed
offline without a GPU.

**`gaze_err` was never affected** -- it is the 3-D angle between two unit vectors and never
passes through `atan2`.

### `cross_gaze_lpips` — **not** an identity metric

Mean pairwise LPIPS between the 5 generations of one identity at different gazes. It
measures how much the output *changes* across gazes, which mixes the pose change you want
with the appearance drift you do not. It cannot separate them. A low value means the model
collapsed to one output, not that identity was preserved. Do not read a rise as drift or a
fall as fidelity.

### `id_sim` (ArcFace cosine, higher better) — identity preservation

```
cosine( ArcFace_buffalo_l(generated), ArcFace_buffalo_l(source) )
```

Averaged only over generations where a face was detected in **both** images, because ArcFace
needs a visible face and fails on profile and back-facing outputs. `face_det` is reported
next to it so the denominator is visible. Around 0.28 is ArcFace's own same-person
threshold — a score at that level means identity preservation is marginal, not comfortable.

### `face_det` — detection rate

The fraction of generations where ArcFace found a face. Partly a legitimate consequence of
back-facing targets, partly a signal that faces are degrading. Read it together with
`id_sim`, never alone.

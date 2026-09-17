#!/usr/bin/env python
"""
Fix label convention + filter inaccurate gaze + select 20 diverse frames/identity.

1. Fix labels.csv: flip dx,dy for 6DRepNet-source rows so the WHOLE dataset uses
   dx>0=right, dy>0=up, dz>0=toward-camera (was inconsistent: 6DRepNet rows flipped).
2. Drop likely gaze errors: 3DGazeNet frames whose gaze disagrees with the
   independent head-pose (6DRepNet on the frame) by > ANGLE_THRESH degrees, plus
   frames with no head detection.
3. Per identity, farthest-point-sample 20 gaze-diverse frames from the survivors.

Outputs: gaze_dataset/labels.csv (fixed in place, +angle col),
         gaze_dataset/labels_top20.csv (20/identity).
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import csv, numpy as np
from collections import defaultdict

DST = DATASET
ANGLE_THRESH = 60.0
N_SELECT = 20


def norm(a):
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a * 0


def fps(vecs, k):
    if len(vecs) <= k:
        return list(range(len(vecs)))
    start = int(np.argmax(np.linalg.norm(vecs - vecs.mean(0), axis=1)))
    ch = [start]; d = np.linalg.norm(vecs - vecs[start], axis=1)
    while len(ch) < k:
        nx = int(np.argmax(d)); ch.append(nx)
        d = np.minimum(d, np.linalg.norm(vecs - vecs[nx], axis=1))
    return sorted(ch)


rows = list(csv.DictReader(open(f"{DST}/labels.csv")))
H = {r["id"]: r for r in csv.DictReader(open(f"{DST}/headpose.csv"))}

for r in rows:
    dx, dy, dz = float(r["dx"]), float(r["dy"]), float(r["dz"])
    if r["source"] == "6DRepNet":          # fix flipped convention
        dx, dy = -dx, -dy
    r["dx"], r["dy"], r["dz"] = f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}"
    g = norm(np.array([dx, dy, dz]))
    h = H.get(r["id"])
    ang = ""
    if h and float(h["conf"]) >= 0.2:
        hv = norm(np.array([-float(h["hp_dx"]), -float(h["hp_dy"]), float(h["hp_dz"])]))
        if np.linalg.norm(hv) > 0.5:
            ang = f"{np.degrees(np.arccos(np.clip(np.dot(g, hv), -1, 1))):.1f}"
    r["head_angle"] = ang

# write fixed master labels (with angle col)
cols = ["id", "sample", "mode", "frame", "dx", "dy", "dz", "source", "head_angle", "image", "viz"]
with open(f"{DST}/labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in cols})

# filter + select
def good(r):
    if r["head_angle"] == "":          # no head detection
        return False
    a = float(r["head_angle"])
    # 3DGazeNet: eyes vs head — allow up to 60 deg deviation, drop bigger (gaze error)
    # 6DRepNet: stored vs fresh head-pose should agree closely — drop >45 (unreliable pose)
    if r["source"] == "3DGazeNet" and a > ANGLE_THRESH:
        return False
    if r["source"] == "6DRepNet" and a > 45.0:
        return False
    return True

byid = defaultdict(list)
for r in rows:
    if good(r):
        byid[r["sample"]].append(r)

sel = []
for sample, rs in byid.items():
    vecs = np.array([[float(x["dx"]), float(x["dy"]), float(x["dz"])] for x in rs])
    for j in fps(vecs, N_SELECT):
        sel.append(rs[j])

with open(f"{DST}/labels_top20.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in sel:
        w.writerow({k: r[k] for k in cols})

print(f"dropped (gaze error / no head): {sum(1 for r in rows)-sum(len(v) for v in byid.values())}")
print(f"survivors: {sum(len(v) for v in byid.values())}  | selected: {len(sel)} ({len(byid)} identities)")
percounts = [len(v) for v in byid.values()]
print(f"per-identity survivors min {min(percounts)} max {max(percounts)}")

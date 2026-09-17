#!/usr/bin/env python
"""
Extract the selected diverse+accurate frames (from sample_frames.py _cands.txt)
out of the clean turnaround videos, and write a labelled gaze dataset.

For each video: read _sample/<tag>_cands.txt (frame idx + dx dy dz), pull those
frames from the CLEAN video, save as images, and append to labels.csv.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, csv, glob
import cv2

ROOT = ROOT
O = f"{ROOT}/Wan2.2/turnaround_out"
DST = f"{ROOT}/gaze_dataset"
os.makedirs(f"{DST}/frames", exist_ok=True)
os.makedirs(f"{DST}/frames_viz", exist_ok=True)

rows = []
for cands in sorted(glob.glob(f"{O}/_sample/*_cands.txt")):
    tag = os.path.basename(cands).replace("_cands.txt", "")          # e.g. 035313_back
    sample, mode = tag.rsplit("_", 1)
    clean = f"{O}/shhq_image_{sample}__p0__{mode}.mp4"
    viz = f"{O}/viz_hc_{tag}.mp4"
    want = {}
    for ln in open(cands):
        p = ln.split()
        if len(p) >= 4:
            want[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))

    for src, outdir in [(clean, f"{DST}/frames"), (viz, f"{DST}/frames_viz")]:
        cap = cv2.VideoCapture(src); fi = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if fi in want:
                cv2.imwrite(f"{outdir}/{tag}_f{fi:03d}.jpg", f)
            fi += 1
        cap.release()

    for fi, (dx, dy, dz) in sorted(want.items()):
        src_model = "3DGazeNet" if dz >= 0.3 else "6DRepNet"
        rows.append([f"{tag}_f{fi:03d}", sample, mode, fi, dx, dy, dz, src_model,
                     f"frames/{tag}_f{fi:03d}.jpg", f"frames_viz/{tag}_f{fi:03d}.jpg"])

with open(f"{DST}/labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "sample", "mode", "frame", "dx", "dy", "dz", "source", "image", "viz"])
    w.writerows(rows)

print(f"wrote {len(rows)} labelled frames from {len(set(r[1]+r[2] for r in rows))} videos")
print(f"  -> {DST}/frames/ , frames_viz/ , labels.csv")

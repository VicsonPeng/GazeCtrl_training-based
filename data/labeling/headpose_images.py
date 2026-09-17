#!/usr/bin/env python
"""
Run 6DRepNet head pose on every dataset frame (image), to use head direction as
an independent second opinion for gaze-accuracy filtering.
Output: gaze_dataset/headpose.csv  (id, hp_dx, hp_dy, hp_dz, conf)
Head-pose dir convention here: dx=sin(yaw), dy=-sin(pitch), dz=cos(yaw)cos(pitch).
(yaw sign convention vs the gaze labels is resolved downstream by correlation.)
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import csv, math, sys, os
sys.path.insert(0, SIXDREP_DIR)
import cv2, numpy as np, torch
from torchvision import transforms
from ultralytics import YOLO
from model import SixDRepNet2
from torchvision.models.resnet import Bottleneck
import utils

DST = DATASET
SNAP = SIXDREP_CKPT
YOLO_HEAD = YOLO_HEAD
dev = "cuda:0"


def main():
    model = SixDRepNet2(Bottleneck, [3, 4, 6, 3])
    sd = torch.load(SNAP, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=False); model.to(dev).eval()
    det = YOLO(YOLO_HEAD)
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
                             transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    rows = list(csv.DictReader(open(f"{DST}/labels.csv")))
    out = open(f"{DST}/headpose.csv", "w"); out.write("id,hp_dx,hp_dy,hp_dz,conf\n")
    n = 0
    for r in rows:
        img = cv2.imread(f"{DST}/{r['image']}")
        if img is None:
            out.write(f"{r['id']},0,0,0,0\n"); continue
        res = det(img, verbose=False)
        b = res[0].boxes
        if b is None or len(b) == 0:
            out.write(f"{r['id']},0,0,0,0\n"); continue
        xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy()
        k = int(np.argmax((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])))
        x1, y1, x2, y2 = xy[k]; pad = 0.4 * max(x2 - x1, y2 - y1)
        H, W = img.shape[:2]
        cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        cx2, cy2 = min(W, int(x2 + pad)), min(H, int(y2 + pad))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            out.write(f"{r['id']},0,0,0,0\n"); continue
        t = tf(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(dev)
        with torch.no_grad():
            R = model(t)
        pitch, yaw, roll = (utils.compute_euler_angles_from_rotation_matrices(R)[0] * 180 / np.pi).cpu().tolist()
        yr, pr = math.radians(yaw), math.radians(pitch)
        dx, dy, dz = math.sin(yr) * math.cos(pr), -math.sin(pr), math.cos(yr) * math.cos(pr)
        out.write(f"{r['id']},{dx:.4f},{dy:.4f},{dz:.4f},{float(cf[k]):.3f}\n")
        n += 1
        if n % 1000 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    out.close(); print(f"done {n}/{len(rows)} -> headpose.csv")


if __name__ == "__main__":
    main()

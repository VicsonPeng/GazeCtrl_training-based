#!/usr/bin/env python
"""Phase3b: per-ID cross-video dedup. Each id has 4 videos (la/tr/tl/ud), each
contributing ~10-12 frames -> ~40/id with heavy redundancy (esp. the frontal
start of every clip). This gathers all of an id's selected frames, recomputes a
head-crop descriptor from the saved letterboxed jpgs, and farthest-point-dedups
across the 4 videos down to ~K per id (keeping ~front_ratio frontal by 6D dz>0).
Only finalizes ids whose all-4 modes are annotated. Output: id_final/<id>.json."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, json, glob
import numpy as np, cv2
from collections import defaultdict
import argparse

P3 = RAW + "/phase3"
MODES = {'la', 'tr', 'tl', 'ud'}


def desc_from(imgpath, head):
    im = cv2.imread(imgpath)
    if im is None or not head:
        return None
    h, w = im.shape[:2]; cx, cy, hh = head
    s = max(24.0, hh * 1.3)
    x1, y1 = int(cx - s / 2), int(cy - s / 2); x2, y2 = int(cx + s / 2), int(cy + s / 2)
    c = im[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if c.size == 0:
        return None
    g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32).ravel()
    g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n > 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--front_ratio', type=float, default=0.65)
    ap.add_argument('--min_dist', type=float, default=0.14)
    args = ap.parse_args()

    byid = defaultdict(list)   # src_id -> list of (tag, frame_dict)
    modes_seen = defaultdict(set)
    for m in sorted(glob.glob(f'{P3}/manifest/*.json')):
        if 'DEMO' in m:
            continue
        d = json.load(open(m))
        modes_seen[d['src_id']].add(d['mode'])
        for fr in d['frames']:
            byid[d['src_id']].append((d['tag'], fr))

    os.makedirs(f'{P3}/id_final', exist_ok=True)
    done = part = 0
    for sid, items in byid.items():
        if not MODES.issubset(modes_seen[sid]):   # wait for all 4 videos
            part += 1; continue
        # build descriptors
        F = []
        for tag, fr in items:
            ip = f'{P3}/{fr["img"]}'
            dsc = desc_from(ip, fr.get('head'))
            if dsc is None:
                continue
            F.append((tag, fr, dsc))
        if not F:
            continue
        D = np.stack([x[2] for x in F])

        def is_front(i):
            d6 = F[i][1].get('d6')
            return d6 is not None and d6[2] is not None and d6[2] > 0
        front = [i for i in range(len(F)) if is_front(i)]
        nonf = [i for i in range(len(F)) if not is_front(i)]

        def fps(pool, cnt, enforce_min):
            if not pool or cnt <= 0:
                return []
            sel = [pool[0]]
            while len(sel) < min(cnt, len(pool)):
                dmin = np.min([1 - D @ D[i] for i in sel], axis=0)
                cand = [j for j in pool if j not in sel]
                j = max(cand, key=lambda c: dmin[c])
                if enforce_min and dmin[j] < args.min_dist:
                    break
                sel.append(j)
            return sel

        nf = min(len(front), round(args.front_ratio * args.k))
        nn = min(len(nonf), args.k - nf)
        if nf + nn < args.k:
            if len(front) > nf:
                nf = min(len(front), args.k - nn)
            elif len(nonf) > nn:
                nn = min(len(nonf), args.k - nf)
        sel = sorted(set(fps(front, nf, False) + fps(nonf, nn, True)))
        frames = []
        for i in sel:
            tag, fr, _ = F[i]
            e = dict(fr); e['tag'] = tag
            frames.append(e)
        nfront = sum(1 for i in sel if is_front(i))
        json.dump(dict(src_id=sid, n_all=len(items), n_final=len(frames),
                       n_front=nfront, frames=frames),
                  open(f'{P3}/id_final/{sid}.json', 'w'), indent=1)
        done += 1
    print(f'dedup done: {done} ids finalized (-> id_final/), {part} ids still partial (waiting for all 4 videos)')


if __name__ == '__main__':
    main()

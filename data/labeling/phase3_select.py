#!/usr/bin/env python
"""Phase3 select (identity-flagged + visual dedup, NOT gaze-estimate based).
Per Vicson's spec:
  - Keep ALL frames of the same take; drop only clear multi-person hallucination.
    Flag each frame's identity: 'ok' (frontal face matches src), 'low' (a face,
    but ArcFace sim<thresh -- usually a profile of the same person), 'noface'
    (turned away). 'low'/'noface' are kept but flagged for careful review.
  - Pick K frames whose HEAD crops are visually distinct (farthest-point on a
    brightness-normalized head-region descriptor). Head region = own face box, or
    the clip-median face box when the face is hidden. No 6DRep/3DGaze used here.
6DRep(d6)/3DGaze(g3) values are attached to chosen frames (from _both.csv) only
for Phase4, never for selection. insightface runs on GPU (CUDAExecutionProvider)."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, csv, os, json
import cv2, numpy as np
from insightface.app import FaceAnalysis

W, H = 512, 768
P3 = RAW + "/phase3"
SRC_DIR = RAW + "/selected_wan_src"


def make_app():
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    a = FaceAnalysis(name='buffalo_l', providers=prov)
    a.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
    return a


def letterbox(img):
    h, w = img.shape[:2]; r = min(W / w, H / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    c = np.zeros((H, W, 3), np.uint8); x, y = (W - nw) // 2, (H - nh) // 2
    c[y:y+nh, x:x+nw] = cv2.resize(img, (nw, nh)); return c, (r, x, y)


def head_crop(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    pw = int(0.3 * max(1, x2 - x1)); ph = int(0.3 * max(1, y2 - y1))
    h, w = frame.shape[:2]
    return frame[max(0, y1-ph):min(h, y2+ph), max(0, x1-pw):min(w, x2+pw)]


def desc(crop):
    if crop.size == 0:
        return None
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32).ravel()
    g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n > 1e-6 else None


def sharpness(crop):
    # Laplacian variance on the head crop (higher = sharper; motion blur -> low)
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def fnum(s):
    s = (s or '').strip(); return float(s) if s not in ('', 'None') else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--min_dist', type=float, default=0.16)
    ap.add_argument('--id_thresh', type=float, default=0.35)
    ap.add_argument('--blur_frac', type=float, default=0.7)   # drop sharp < frac*median
    ap.add_argument('--blur_floor', type=float, default=60.0) # absolute lap-var floor
    ap.add_argument('--swap_win', type=int, default=5)        # snap pick to sharpest +-win frames
    ap.add_argument('--front_ratio', type=float, default=0.65) # target share of 6D dz>0 (frontal)
    args = ap.parse_args()

    src_id, mode = args.tag.rsplit('__', 1)
    src_path = f'{SRC_DIR}/{src_id}.jpg'
    if not os.path.exists(src_path):
        print(f'NO src image {src_path}'); return
    app = make_app()
    sf = app.get(cv2.imread(src_path))
    if not sf:
        print(f'{args.tag}: no face in src -> skip'); return
    src_emb = max(sf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding

    # both-estimator values (attached, not used for selection)
    both = {}
    bp = f'{P3}/viz/{args.tag}_both.csv'
    if os.path.exists(bp):
        for r in csv.DictReader(open(bp)):
            g3 = [fnum(r['g3_dx']), fnum(r['g3_dy']), fnum(r['g3_dz'])]
            d6 = [fnum(r['d6_dx']), fnum(r['d6_dy']), fnum(r['d6_dz'])]
            both[int(r['frame'])] = dict(used=r['used'],
                g3=g3 if g3[0] is not None else None,
                d6=d6 if d6[0] is not None else None,
                chosen=[float(r['dx']), float(r['dy']), float(r['dz'])])

    # single read: detect on original, store letterboxed frame + mapped box + sim
    cap = cv2.VideoCapture(args.video); fi = 0
    F = []  # per frame: dict(idx, lb(frame), box|None, sim|None, flag)
    dropped_multi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        faces = [x for x in app.get(fr) if float(x.det_score) > 0.5]
        lb, (r, ox, oy) = letterbox(fr)
        box = None; sim = None; flag = 'noface'
        if faces:
            faces.sort(key=lambda x: -(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            main = faces[0]; A = (main.bbox[2]-main.bbox[0])*(main.bbox[3]-main.bbox[1])
            if any((o.bbox[2]-o.bbox[0])*(o.bbox[3]-o.bbox[1]) > 0.4*A for o in faces[1:]):
                dropped_multi += 1; fi += 1; continue  # multi-person hallucination
            x1, y1, x2, y2 = main.bbox
            box = [x1*r+ox, y1*r+oy, x2*r+ox, y2*r+oy]  # letterbox coords
            sim = float(np.dot(main.normed_embedding, src_emb))
            flag = 'ok' if sim >= args.id_thresh else 'low'
        F.append(dict(idx=fi, lb=lb, box=box, sim=sim, flag=flag))
        fi += 1
    cap.release()
    N = fi
    if not F:
        print(f'{args.tag}: nothing kept -> skip'); return

    # clip-median face box -> head anchor for no-face frames
    boxes = [f['box'] for f in F if f['box'] is not None]
    if boxes:
        arr = np.array(boxes); anchor = np.median(arr, axis=0).tolist()
    else:
        anchor = [W*0.30, H*0.06, W*0.70, H*0.34]  # top-center fallback

    # descriptors + sharpness (head region), for every kept frame
    for f in F:
        f['hbox'] = f['box'] if f['box'] is not None else anchor
        crop = head_crop(f['lb'], f['hbox'])
        f['d'] = desc(crop); f['sharp'] = sharpness(crop)
    F = [f for f in F if f['d'] is not None]

    # blur filter: drop motion-blurred / ghosted frames (keep sharp facial detail)
    med = float(np.median([f['sharp'] for f in F])) if F else 0.0
    thr = max(args.blur_floor, args.blur_frac * med)
    sharpF = [f for f in F if f['sharp'] >= thr]
    n_blur = len(F) - len(sharpF)
    if len(sharpF) >= 4:            # keep enough; else fall back to sharpest
        F = sharpF
    else:
        F = sorted(F, key=lambda f: -f['sharp'])[:max(args.k, 8)]
        F.sort(key=lambda f: f['idx'])

    # frontal quota: split F into frontal (6D dz>0) / non-frontal pools
    D = np.stack([f['d'] for f in F])

    def is_front(f):
        b = both.get(f['idx'])
        return b is not None and b['d6'] is not None and b['d6'][2] > 0
    front = [i for i, f in enumerate(F) if is_front(f)]
    nonf = [i for i, f in enumerate(F) if not is_front(f)]

    def fps(pool, cnt, enforce_min=True):
        # farthest-point over `pool`; enforce_min=False fills to cnt regardless of
        # min_dist (used for the frontal pool so the quota is met even when frontal
        # frames are similar); True stops early on near-dupes (non-frontal pool).
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

    def swap(sel, pool):  # snap each pick to sharpest same-pool frame within +-win
        out = []
        for i in sel:
            near = [j for j in pool if abs(F[j]['idx'] - F[i]['idx']) <= args.swap_win]
            out.append(max(near, key=lambda j: F[j]['sharp']))
        seen = set(); return [x for x in out if not (x in seen or seen.add(x))]

    nf = min(len(front), round(args.front_ratio * args.k))
    nn = min(len(nonf), args.k - nf)
    if nf + nn < args.k:  # top up from whichever pool has spare
        if len(front) > nf:
            nf = min(len(front), args.k - nn)
        elif len(nonf) > nn:
            nn = min(len(nonf), args.k - nf)
    selF = swap(fps(front, nf, enforce_min=False), front) + swap(fps(nonf, nn), nonf)
    sel = sorted(set(selF), key=lambda i: F[i]['idx'])

    # write selected frames + manifest
    fdir = f'{P3}/frames/{args.tag}'; os.makedirs(fdir, exist_ok=True)
    frames = []
    flags = {'ok': 0, 'low': 0, 'noface': 0}
    for i in sel:
        f = F[i]; fr = f['idx']
        cv2.imwrite(f'{fdir}/f{fr:03d}.jpg', f['lb'])
        x1, y1, x2, y2 = [float(v) for v in f['hbox']]
        head = [round((x1+x2)/2, 1), round((y1+y2)/2, 1), round(y2-y1, 1)]
        b = both.get(fr, dict(used='', g3=None, d6=None, chosen=[0, 0, 1]))
        flags[f['flag']] += 1
        frames.append(dict(frame=fr, img=f'frames/{args.tag}/f{fr:03d}.jpg',
                           id_flag=f['flag'], id_sim=round(f['sim'], 3) if f['sim'] is not None else None,
                           sharp=round(f['sharp'], 1), head=head,
                           used=b['used'], g3=b['g3'], d6=b['d6'], chosen=b['chosen']))
    n_front = sum(1 for f in frames if f['d6'] and f['d6'][2] > 0)
    man = dict(tag=args.tag, src_id=src_id, mode=mode, video=args.video,
               n_frames=N, n_kept=len(F), dropped_multi=dropped_multi, dropped_blur=n_blur,
               blur_thr=round(thr, 1), n_sel=len(frames), n_front=n_front,
               sel_flags=flags, frames=frames)
    os.makedirs(f'{P3}/manifest', exist_ok=True)
    json.dump(man, open(f'{P3}/manifest/{args.tag}.json', 'w'), indent=1)
    fpct = 100 * n_front / len(frames) if frames else 0
    print(f'{args.tag}: kept {len(F)} sharp/{N} (blur-drop {n_blur}) -> selected {len(frames)}  frontal(6D dz>0) {n_front} ({fpct:.0f}%) {flags}')


if __name__ == '__main__':
    main()

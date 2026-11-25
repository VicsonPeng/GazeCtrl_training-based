"""Rebuild mp_v5/boxes.json with the head-box rule (Vicson, 2026-11-22).

The box starts from the target mask's bounding box, must hold the target's head box whole,
must hold no other person's head box, and otherwise stays as close to the mask bbox as it can.
See target_head.headbox_box.

The composites are synthetic, so the two people are known exactly: `mask` is the target and
`bmask` is the pasted bystander. Heads come from yolov8_head and are assigned to one or the
other by mask overlap, which is why the wrong person can never be framed.

The 5% pad is BAKED INTO the stored box, so training must run with --box_pad 0: padding again
after the cut would push the box back over the bystander's head and undo the whole point.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, json, os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target_head as TH

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=CORPORA + '/mp_v5')
ap.add_argument('--out', default='')
ap.add_argument('--stats', default='/tmp/trainboxes.csv')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--device', default='cpu')
ap.add_argument('--pad', type=float, default=0.05)
a = ap.parse_args()
R = a.data
OUT = a.out or f'{R}/boxes_headrule.json'

df = pd.read_csv(f'{R}/index.csv')
KINDS = list(df['kind'].unique())
if a.limit:
    per = max(1, a.limit // len(KINDS))
    df = pd.concat([df[df['kind'] == k].head(per) for k in KINDS])
_op = f'{R}/boxes.json'
OLD = json.load(open(_op)) if os.path.exists(_op) else {}
print(f'old boxes: {len(OLD)} from {_op}' if OLD else
      'no boxes.json -- comparing against the plain mask bbox + pad', flush=True)

from ultralytics import YOLO
hdet = YOLO(YOLO_HEAD)


def frac_in(m, b):
    q = [int(round(v)) for v in b]
    t = float(m.sum())
    if t < 1 or q[2] <= q[0] or q[3] <= q[1]:
        return 0.0
    return float(m[max(0, q[1]):q[3], max(0, q[0]):q[2]].sum()) / t


def _bb(i, o):
    """how much of box i sits inside box o"""
    ix = max(0., min(i[2], o[2]) - max(i[0], o[0]))
    iy = max(0., min(i[3], o[3]) - max(i[1], o[1]))
    return ix * iy / max(1e-6, (i[2] - i[0]) * (i[3] - i[1]))


def padded(b, pad, W, H):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    return (max(0., x0 - pad * w), max(0., y0 - pad * h),
            min(float(W), x1 + pad * w), min(float(H), y1 + pad * h))


out, rows = {}, []
for i, r in enumerate(df.itertuples()):
    n = r.name
    tm = np.array(Image.open(f'{R}/mask/{n}.png').convert('L')) > 127
    if not tm.any():
        continue
    H, W = tm.shape
    bmp = f'{R}/bmask/{n}.png'
    bm = np.array(Image.open(bmp).convert('L')) > 127 if os.path.exists(bmp) else np.zeros_like(tm)
    hb = hdet.predict(f'{R}/input/{n}.png', verbose=False, device=a.device, conf=0.25)[0].boxes
    hs = [tuple(map(float, x)) for x in hb.xyxy.cpu().numpy()] if hb is not None and len(hb) else []
    ti, ov, how = TH.pick_target(hs, tm)
    head = hs[ti] if ti is not None and how != 'weak-mask' else None
    oh = TH.other_heads(hs, ti, [bm] if bm.any() else [], head)
    box, cov, left, ok = TH.headbox_box(tm, head, oh, pad=a.pad, size=(W, H))

    e = OLD.get(n, {})
    sx = W / float(e.get('size', [W, W])[0])
    if e.get('a'):
        old = padded([v * sx for v in e['a']], a.pad, W, H)
    else:
        ys, xs = np.where(tm)
        old = padded((float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)), a.pad, W, H)
    out[n] = dict(a=[round(v, 1) for v in box], b=e.get('b'), size=[W, H],
                  head=[round(v, 1) for v in head] if head else None,
                  oheads=[[round(v, 1) for v in h] for h in oh], how=how, padded=True)
    rows.append(dict(name=n, kind=r.kind, nh=len(hs), noh=len(oh), how=how, head_ok=bool(ok),
                     b_new=round(frac_in(bm, box), 4) if bm.any() else 0.0,
                     b_old=round(frac_in(bm, old), 4) if (bm.any() and old) else 0.0,
                     cov=round(cov, 3),
                     area_new=round((box[2]-box[0])*(box[3]-box[1])/(W*H), 3),
                     area_old=round((old[2]-old[0])*(old[3]-old[1])/(W*H), 3) if old else None,
                     ohead_in=round(max([_bb(h, box) for h in oh], default=0.0), 3)))
    if (i + 1) % 250 == 0:
        print(f'{i+1}/{len(df)}', flush=True)

json.dump(out, open(OUT, 'w'))
D = pd.DataFrame(rows); D.to_csv(a.stats, index=False)
print(f'\nwrote {OUT}  n={len(out)}   stats {a.stats}')
M = D[D.kind.str.startswith('multi') & (D.b_old > 0)]
print(f'\n=== multi(兩人) {len(M)} 對：旁人的 mask 落在框內 ===')
for c, nm in [('b_old', '舊框'), ('b_new', '新框')]:
    q = M[c]
    print(f'  {nm}  中位 {q.median()*100:5.1f}%   >1% {100*(q>.01).mean():5.1f}%   '
          f'>10% {100*(q>.10).mean():5.1f}%   >30% {100*(q>.30).mean():5.1f}%   >50% {100*(q>.50).mean():5.1f}%')
print(f'  框佔畫面 中位  舊 {M.area_old.median()*100:.0f}%  新 {M.area_new.median()*100:.0f}%')
_cov = M['cov']
print(f'  蓋住目標 mask 中位 {_cov.median()*100:.0f}%   覆蓋<50% 的 {100*(_cov<0.5).mean():.1f}%')
print(f'  目標的頭完整保住 {100*D.head_ok.mean():.1f}%   偵測不到頭 {int((D.how=="weak-mask").sum()+(D.how=="none").sum())}')
print(f'  框內還殘留旁人的頭 中位 {M.ohead_in.median()*100:.1f}%   >20% 的 {100*(M.ohead_in>.2).mean():.1f}%')
print(f'\n  per-kind:\n' + D.groupby('kind').agg(n=('cov','size'), b_new=('b_new','median'),
      b_old=('b_old','median'), cov=('cov','median'), area=('area_new','median')).round(3).to_string())
print(f'\n  single/identity 框佔畫面 中位 新 {D[~D.kind.str.startswith("multi")].area_new.median()*100:.0f}% '
      f'(舊 {D[~D.kind.str.startswith("multi")].area_old.median()*100:.0f}%)')

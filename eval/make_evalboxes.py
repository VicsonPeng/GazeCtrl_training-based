"""Build the gaze box for every eval pair, and the other people's boxes beside it.

The box rule (Vicson's): start from the bounding box of the person the TARGET MASK covers,
so the box can never be on the wrong person, then take the boundary in until no other
person is inside it. See target_head.repel_box. No head-width or body-proportion guesses.

Other people come from yolo11m-seg person instances whose IoU with the target mask is low.
The target's head (yolov8_head, picked by mask overlap) is kept as a hard constraint so the
pull-back can never cut the head out.

Writes boxes.json in the same shape as the training set's, plus a stats CSV.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, ast, glob, json, os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target_head as TH

EV = EVAL
GT = GT_EVAL
SZ = 768

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--device', default='cpu')
ap.add_argument('--out', default=f'{EV}/boxes.json')
ap.add_argument('--stats', default='/tmp/evalboxes.csv')
ap.add_argument('--tol', type=float, default=0.02)
ap.add_argument('--head_keep', type=float, default=0.75)
ap.add_argument('--sweep', default='')
ap.add_argument('--slacks', default='0.0,0.3')
ap.add_argument('--all', type=int, default=0, help='every pair in the CSV, not just the scored 166')
a = ap.parse_args()

V = {}
for mf in sorted(glob.glob(os.path.join(EV, 'manifest.json.bak_*'))) + [os.path.join(EV, 'manifest.json')]:
    for sc in json.load(open(mf))['scenes']:
        for v in sc['views']:
            V.setdefault((sc['image'], v['view_id']), v)

d = pd.read_csv(f'{GT}/mp_pt2B_shard0.csv').copy()
d['pid'] = d.scene + '__' + d.view_a + '__to__' + d.view_b
if not a.all:
    d = d[d.det > 0]
    d = d[d.pid.isin(set(pd.read_csv(f'{GT}/other_gaze_pt2.csv').pid))]
if a.limit: d = d.head(a.limit)

from ultralytics import YOLO
seg = YOLO(YOLO_SEG)
hdet = YOLO(YOLO_HEAD)


def _pxin(om, b):
    q = [int(round(v)) for v in b]
    if q[2] <= q[0] or q[3] <= q[1]:
        return 0.0
    return float(om[max(0, q[1]):q[3], max(0, q[0]):q[2]].sum()) / max(1.0, float(om.sum()))


def fin(i, o):
    ix = max(0., min(i[2], o[2]) - max(i[0], o[0]))
    iy = max(0., min(i[3], o[3]) - max(i[1], o[1]))
    return ix * iy / max(1e-6, (i[2] - i[0]) * (i[3] - i[1]))


out, rows = {}, []
tmp = f'/tmp/_mb_{os.getpid()}.jpg'
for i, r in enumerate(d.itertuples()):
    cb = tuple(ast.literal_eval(r.box))
    A, B = V[(r.scene, r.view_a)], V[(r.scene, r.view_b)]
    src = Image.open(os.path.join(EV, A['file'])).convert('RGB').crop(cb).resize((SZ, SZ), Image.LANCZOS)
    m = np.array(Image.open(os.path.join(EV, A['mask'])).convert('L').crop(cb)
                 .resize((SZ, SZ), Image.NEAREST)) > 127
    if not m.any():
        continue
    src.save(tmp, quality=95)

    # other people: person instances that are not the target
    res = seg.predict(tmp, verbose=False, device=a.device, conf=0.25, classes=[0], retina_masks=True)[0]
    others, omasks = [], []
    if res.masks is not None:
        for pm, bx in zip(res.masks.data.cpu().numpy() > 0.5, res.boxes.xyxy.cpu().numpy()):
            if pm.shape != m.shape:
                pm = np.array(Image.fromarray(pm.astype(np.uint8) * 255)
                              .resize((SZ, SZ), Image.NEAREST)) > 127
            iou = float((pm & m).sum()) / max(1, int((pm | m).sum()))
            if iou < 0.3:
                others.append(tuple(map(float, bx))); omasks.append(pm)

    # the target's head, picked by mask overlap, as a hard constraint on the pull-back
    hb = hdet.predict(tmp, verbose=False, device=a.device, conf=0.25)[0].boxes
    hs = [tuple(map(float, x)) for x in hb.xyxy.cpu().numpy()] if hb is not None and len(hb) else []
    sx, sy = SZ / (cb[2] - cb[0]), SZ / (cb[3] - cb[1])
    anc = ((B['face_px'][0] - cb[0]) * sx, (B['face_px'][1] - cb[1]) * sy)
    ti, ov, how = TH.pick_target(hs, m, anc)
    head = hs[ti] if ti is not None and how != 'weak-mask' else None

    oheads = TH.other_heads(hs, ti, omasks, head)
    A_ = TH.gaze_box(m, None, [], mode='full')
    slacks = [float(x) for x in a.slacks.split(',')]
    F_, cov, left, ok = TH.headbox_box(m, head, oheads, slack=slacks[0])
    sw = {}
    for sl in slacks:
        bx_, cv_, lf_, ok_ = TH.headbox_box(m, head, oheads, slack=sl)
        sw[f's{sl}'] = dict(area=round((bx_[2]-bx_[0])*(bx_[3]-bx_[1])/(SZ*SZ), 3),
                            cov=round(cv_, 3), left=lf_, ok=bool(ok_),
                            opx=round(max([_pxin(om, bx_) for om in omasks], default=0.), 3),
                            ohead=round(max([fin(o, bx_) for o in oheads], default=0.), 3),
                            head=round(fin(head, bx_), 3) if head else None,
                            box=[round(v, 1) for v in bx_])

    out[r.pid] = dict(target=[round(v, 1) for v in F_],
                      target_full=[round(v, 1) for v in A_],
                      others=[[round(v, 1) for v in o] for o in others],
                      head=[round(v, 1) for v in head] if head else None,
                      size=[SZ, SZ], crop=list(cb), how=how)
    rows.append(dict(pid=r.pid, scene=r.scene, n_others=len(others), how=how,
                     A_area=(A_[2]-A_[0])*(A_[3]-A_[1])/(SZ*SZ),
                     F_area=(F_[2]-F_[0])*(F_[3]-F_[1])/(SZ*SZ),
                     F_cov=round(cov, 3), others_left=left, n_oheads=len(oheads),
                     A_ohead=round(max([fin(o, A_) for o in oheads], default=0.), 3),
                     F_ohead=round(max([fin(o, F_) for o in oheads], default=0.), 3),
                     A_other=round(max([fin(o, A_) for o in others], default=0.), 3),
                     F_other=round(max([fin(o, F_) for o in others], default=0.), 3),
                     A_opx=round(max([_pxin(om, A_) for om in omasks], default=0.), 3),
                     F_opx=round(max([_pxin(om, F_) for om in omasks], default=0.), 3),


                     F_head=round(fin(head, F_), 3) if head else None,
                     **{f'sw_{k}': json.dumps(v) for k, v in sw.items()}))
    if (i + 1) % 20 == 0: print(f'{i+1}/{len(d)}', flush=True)

json.dump(out, open(a.out, 'w'))
D = pd.DataFrame(rows); D.to_csv(a.stats, index=False)
print(f'\nwrote {a.out}  n={len(out)}')
print()
print(f"{'slack':>8} {'別人的頭全在框外':>16} {'框內別人頭最多':>14} {'蓋住目標mask中位':>16} "
      f"{'頭100%保住':>11} {'框內別人身體':>12} {'佔畫面':>7}")
for sl in slacks:
    S = pd.DataFrame([json.loads(x) for x in D[f'sw_s{sl}']])
    print(f"{sl:>8} {(S['left']==0).mean()*100:15.0f}% {S['ohead'].median()*100:13.1f}% "
          f"{S['cov'].median()*100:15.0f}% {(S['head']>=0.999).mean()*100:10.0f}% "
          f"{S['opx'].median()*100:11.1f}% {S['area'].median()*100:6.0f}%")
print(f"{'A 現行':>8} {(D.A_ohead<=0.01).mean()*100:15.0f}% {D.A_ohead.median()*100:13.1f}% "
      f"{100:15.0f}% {100:10.0f}% {D.A_opx.median()*100:11.1f}% {D.A_area.median()*100:6.0f}%")
raise SystemExit
print(f'  tol={a.tol}')
print(f'  別人的像素在框內 >10%   A {(D.A_opx>0.10).mean()*100:5.0f}%   F {(D.F_opx>0.10).mean()*100:5.0f}%')
print(f'  別人的像素在框內 中位   A {D.A_opx.median()*100:5.1f}%   F {D.F_opx.median()*100:5.1f}%')
print(f'  完全排除別人 (<=tol)    A {(D.A_opx<=a.tol).mean()*100:5.0f}%   F {(D.others_left==0).mean()*100:5.0f}%')
print(f'  佔畫面 中位             A {D.A_area.median()*100:5.0f}%   F {D.F_area.median()*100:5.0f}%')
print(f'  蓋住 target mask 中位   A   100%   F {D.F_cov.median()*100:5.0f}%')
print(f'  蓋住 target mask <50%   F {(D.F_cov<0.5).mean()*100:5.0f}%')
print(f'  保住目標的頭 >90%       F {(D.F_head>0.9).mean()*100:5.0f}%')

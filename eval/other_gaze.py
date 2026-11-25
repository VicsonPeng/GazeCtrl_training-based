#!/usr/bin/env python
"""Does the gaze condition move EVERY person in the frame, not just the masked one?

Vicson's reading of the multi-person results: the model never learned that the mask marks
who to edit, so the other person's gaze follows the requested direction too. That fits the
conditioning: the gaze map is a UNIFORM colour fill over the whole image (train_stage5.py
gaze_cond), so it says "look this way" about the picture, not about a person.

Test: in each output, measure the gaze of the head that is NOT the target's, and compare it
to three references:

  fed      the gaze the model was asked for
  src      the same person's gaze in the source crop
  tgtframe the same person's gaze in the real target frame (what SHOULD have happened:
           nothing, since only the masked person was supposed to move)

If the other person's gaze lands on `fed` and away from `src`, the condition is being
applied to everyone.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, ast, json, math, os, sys
import numpy as np, pandas as pd
from PIL import Image
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import target_head as TH
sys.path.insert(0, _os.path.join(_p, "src"))
from gaze3d_runner import load_model, predict

EV = EVAL
GT = GT_EVAL
SZ = 768


def to_ours(g):
    return np.stack([-g[:, 0], g[:, 1], -g[:, 2]], 1)


def ang(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='mp_s5_mask,mp_s5_nomask,mp_new,mp_old')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=f'{GT}/other_gaze.csv')
    a = ap.parse_args()

    V = {}
    import glob
    for mf in sorted(glob.glob(os.path.join(EV, 'manifest.json.bak_*'))) + [os.path.join(EV, 'manifest.json')]:
        for sc in json.load(open(mf))['scenes']:
            for v in sc['views']:
                V.setdefault((sc['image'], v['view_id']), v)

    base = pd.read_csv(f'{GT}/mp_s5_mask_shard0.csv')
    if a.limit:
        base = base.head(a.limit)
    runs = a.runs.split(',')

    from ultralytics import YOLO
    det = YOLO(YOLO_HEAD)
    model = load_model('gf', a.device)

    def crop(path, box):
        return Image.open(path).convert('RGB').crop(box).resize((SZ, SZ), Image.LANCZOS)

    def heads(im, tmp='/tmp/_og.jpg'):
        im.save(tmp, quality=95)
        b = det.predict(tmp, verbose=False, device=a.device, conf=0.25)[0].boxes
        if b is None or len(b) == 0:
            return []
        return [tuple(float(x) for x in q) for q in b.xyxy.cpu().numpy()]

    def split(bs, anchor, mask=None):
        """the target's head, and the other people's.

        The target is the head sitting on the target MASK, not the one nearest the labelled
        face point: when two people stand shoulder to shoulder the nearest-point rule picks
        the wrong one, and then 'the other person' is measured on the person we were asked to
        turn. See target_head.pick_target; the face point only breaks ties.
        """
        if not bs:
            return None, []
        if mask is None:
            c = np.array([[(x[0] + x[2]) / 2, (x[1] + x[3]) / 2] for x in bs])
            i = int(np.argmin(((c - np.array(anchor)) ** 2).sum(1)))
        else:
            i, _ov, _how = TH.pick_target(bs, mask, anchor)
        return bs[i], [b for j, b in enumerate(bs) if j != i]

    def measure(im, box, tmp):
        im.save(tmp, quality=95)
        return tmp

    rows = []
    for k, r in enumerate(base.itertuples()):
        box = tuple(ast.literal_eval(r.box))
        A, B = V[(r.scene, r.view_a)], V[(r.scene, r.view_b)]
        sx = SZ / (box[2] - box[0])
        anc = ((B['face_px'][0] - box[0]) * sx, (B['face_px'][1] - box[1]) * sx)
        fed = (B['dx'], B['dy'], B['dz'])
        pid = f'{r.scene}__{r.view_a}__to__{r.view_b}'

        imgs = {'SRC': crop(os.path.join(EV, A['file']), box),
                'TGTFRAME': crop(os.path.join(EV, B['file']), box)}
        for rn in runs:
            p = f'{GT}/{rn}/img/{pid}.jpg'
            if os.path.exists(p):
                imgs[rn] = Image.open(p).convert('RGB').resize((SZ, SZ), Image.LANCZOS)

        # the other person's box is taken from the SOURCE, then reused on every image, so the
        # same region is measured everywhere and a moved head cannot swap the two people
        bs = heads(imgs['SRC'])
        msrc = np.array(Image.open(os.path.join(EV, A['mask'])).convert('L')
                        .crop(box).resize((SZ, SZ), Image.NEAREST)) > 127
        tb, others = split(bs, anc, msrc if msrc.any() else None)
        if not others:
            continue
        ob = max(others, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        items, keys = [], []
        for nm, im in imgs.items():
            f = f'/tmp/_og_{nm}.jpg'; im.save(f, quality=95)
            items.append((f, ob)); keys.append(nm)
        g = to_ours(predict(model, items, a.device))
        G = dict(zip(keys, g))
        rec = dict(pid=pid, scene=r.scene, n_heads=len(bs),
                   fed_vs_src=round(ang(fed, G['SRC']), 1),
                   tgtframe_vs_src=round(ang(G['TGTFRAME'], G['SRC']), 1))
        for rn in runs:
            if rn in G:
                rec[f'{rn}__vs_fed'] = round(ang(G[rn], fed), 1)
                rec[f'{rn}__vs_src'] = round(ang(G[rn], G['SRC']), 1)
        rows.append(rec)
        if k % 25 == 0:
            print(f'{k}/{len(base)}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f'\n{len(d)} pairs with a second head\n')
    print('關鍵問題：沒被 mask 的那個人，輸出裡的視線比較接近「餵進去的 gaze」還是「他原本的視線」？\n')
    print(f"{'':16s} {'vs 餵進去的':>12s} {'vs 他原本的':>12s}  {'判讀'}")
    print(f"{'真實目標影格':16s} {d.tgtframe_vs_src.median():11.1f}°  <- 他本來就該不動" if False else '')
    print(f"{'(他原本視線 vs 餵進去的 gaze)':16s} {d.fed_vs_src.median():11.1f}°")
    print(f"{'真實目標影格':16s} {'—':>12s} {d.tgtframe_vs_src.median():11.1f}°   GT 裡他動了多少")
    for rn in runs:
        if f'{rn}__vs_fed' in d:
            a1 = d[f'{rn}__vs_fed'].median(); a2 = d[f'{rn}__vs_src'].median()
            verdict = '跟著餵進去的 gaze 走' if a1 < a2 - 5 else ('保持原本視線' if a2 < a1 - 5 else '兩者之間')
            print(f'{rn:16s} {a1:11.1f}° {a2:11.1f}°   {verdict}')
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()

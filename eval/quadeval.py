#!/usr/bin/env python
"""Four-quadrant classification of a GazeCtrl run, the way Vicson asked for it (2026-11-23).

  (1) the target turned      AND the bystander stayed put    <- the one we want
  (2) the target turned      AND the bystander also turned
  (3) neither moved
  (4) the target did not turn AND the bystander turned

The criteria below are the CORRECTED ones. The first version agreed with Vicson's manual
labels on only 41% of 166 eval pairs (55% on the target axis, i.e. barely better than
guessing); these agree on 66% overall, 77% on the target axis and 88% on the bystander axis.
Three bugs were fixed to get there:

  * `base_err - gaze_err > 20deg` can never fire when base_err is already small -- a pair
    whose source person happens to face near the requested direction was always scored
    "did not turn" however well the output matched.  Replaced by an ABSOLUTE criterion
    (did the output land near the request) OR'd with a relative one.
  * the bystander threshold of 15deg was far tighter than a human calls "moved".
  * an out-of-focus passer-by in the background counted as a bystander.  Every real
    bystander in the eval set has a head at least 0.478x the target's width; the background
    heads sit at 0.33-0.38x, so half the target's width separates them cleanly.

KNOWN BIAS, measured: on the mp_hbox run (checkpoint _cn_pt2B/cn_step006000, 166 pairs)
this classifier puts 65.7% of pairs in (1) where Vicson's own labels put 50.0% -- so the
absolute level reads roughly 16 points optimistic.  That run is the BASELINE a post-train
has to beat; compare a new checkpoint's (1) share against 65.7%, not against 50%.

These are still AUTOMATIC numbers off a gaze estimator: good for watching a trend across
checkpoints, not for a claim about any one pair.  Recalibrate the absolute level against a
fresh batch of manual labels before quoting it (CLAUDE.md 3.5).
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, os
import numpy as np, pandas as pd

GT = GT_EVAL
HEADS = EVAL + '/headwidths.csv'

TGT_ABS = 50.0    # output within this many degrees of the request -> turned
TGT_REL = 0.8     # or the error fell to this fraction of the do-nothing error
OTH_DEG = 30.0    # bystander moved more than this -> it turned
MIN_W   = 0.5     # a head narrower than this fraction of the target's is not a bystander


def classify(d, tgt_abs=TGT_ABS, tgt_rel=TGT_REL, oth_deg=OTH_DEG, min_w=MIN_W):
    """d needs: gaze_err, base_err, oth (bystander deg vs its own source), w, tw.
    Returns d with `moved`, `kept`, `q` added."""
    d = d.copy()
    d['moved'] = (d.gaze_err < tgt_abs) | (d.gaze_err < tgt_rel * d.base_err)
    real = d.w >= min_w * d.tw          # is there a bystander worth scoring at all
    d['kept'] = (~real) | (d.oth < oth_deg) | d.oth.isna()
    d['q'] = np.where(d.moved & d.kept, 1, np.where(d.moved, 2, np.where(d.kept, 3, 4)))
    return d


def load(name, ogaze, heads=HEADS, gt=GT, require_heads=True):
    e = pd.read_csv(f'{gt}/{name}_shard0.csv')
    e['pid'] = e.scene + '__' + e.view_a + '__to__' + e.view_b
    o = pd.read_csv(ogaze)
    col = f'{name}__vs_src'
    if col not in o.columns:
        raise SystemExit(f'{ogaze} has no column {col}; it holds {list(o.columns)}')
    o = o[['pid', col]].rename(columns={col: 'oth'})
    h = pd.read_csv(heads)[['pid', 'w', 'tw']]
    d = e.merge(o, on='pid', how='left').merge(h, on='pid', how='left')
    miss = int(d.w.isna().sum())
    if miss and require_heads:
        print(f'  {miss} of {len(d)} pairs are not in the head-width cache and are DROPPED, '
              f'so this run is scored on the same pairs the criteria were calibrated on. '
              f'Run hsz to extend the cache, or pass --require_heads 0 to score them as '
              f'"no bystander".', flush=True)
        d = d[d.w.notna()].reset_index(drop=True)
    elif miss:
        print(f'  WARNING {miss} pairs have no cached head width; scored as "no bystander"',
              flush=True)
        d['w'] = d.w.fillna(0.0); d['tw'] = d.tw.fillna(1.0)
    return d


def report(d, label=''):
    d = classify(d)
    n = len(d)
    c = d.q.value_counts().reindex([1, 2, 3, 4]).fillna(0).astype(int)
    print(f'\n=== {label or "run"}  n={n} ===')
    for q, nm in [(1, 'target turned / bystander still  <- want'),
                  (2, 'target turned / bystander turned also'),
                  (3, 'neither moved'),
                  (4, 'target still / bystander turned')]:
        print(f'  ({q}) {nm:40s} {c[q]:4d}  {100*c[q]/n:5.1f}%')
    print(f'  gaze_err median {d.gaze_err.median():5.1f}deg   '
          f'base_err median {d.base_err.median():5.1f}deg   '
          f'bystander median {d.oth.median():5.1f}deg')
    return dict(n=n, q1=int(c[1]), q2=int(c[2]), q3=int(c[3]), q4=int(c[4]),
                q1_frac=round(c[1] / n, 4), gaze_err=round(float(d.gaze_err.median()), 2),
                oth=round(float(d.oth.median()), 2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='run name, i.e. <name>_shard0.csv in gt_eval')
    ap.add_argument('--ogaze', required=True, help='other_gaze csv holding <name>__vs_src')
    ap.add_argument('--heads', default=HEADS)
    ap.add_argument('--require_heads', type=int, default=1,
                    help='1 = drop pairs missing from the head-width cache (the calibrated '
                         'set); 0 = keep them and assume they have no bystander')
    ap.add_argument('--append', default='', help='append one row of the summary to this csv')
    ap.add_argument('--step', type=int, default=-1, help='training step, for --append')
    a = ap.parse_args()
    s = report(load(a.name, a.ogaze, a.heads, require_heads=bool(a.require_heads)), a.name)
    if a.append:
        s = dict(step=a.step, name=a.name, **s)
        pd.DataFrame([s]).to_csv(a.append, mode='a', header=not os.path.exists(a.append),
                                 index=False)
        print(f'\nappended to {a.append}')

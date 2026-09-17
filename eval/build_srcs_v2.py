"""Build srcs_v2.json for demo v2: per identity pick front/side/back SOURCE frames
(each with its own GT gaze), plus the full GT frame gallery of that identity.
Labels: train ids -> hitl.csv (has manual labels); held-out ids -> phase3/id_final/*.json."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, math, os
import numpy as np, pandas as pd
from pathlib import Path

P3   = Path(RAW + "/phase3")
CROP = Path(RAW + "/crops")
HITL = WORK + "/_cn_hitl/hitl.csv"
OUT  = WORK + "/_cn_hitl/fullrot_srcs_v2.json"

TRAIN_IDS = ['coco_000000001146__p0', 'wf_13_Interview_Interview_On_Location_13_289__p0']
HELD_IDS  = ['coco_000000186368__p0', 'coco_000000491223__p0', 'wf_36_Football_americanfootball_ball_36_149__p0']
CROP_ONLY = [('coco_000000386739__p1','coco','unseen-id'),
             ('wf_53_Raid_policeraid_53_552__p1','wf','unseen-id'),
             ('shhq_image_039695__p0','shhq','unseen-domain'),
             ('shhq_image_036822__p0','shhq','unseen-domain'),
             ('shhq_image_036687__p0','shhq','unseen-domain')]

def yaw_of(g):  return math.degrees(math.atan2(g[0], g[2]))
def pit_of(g):  return math.degrees(math.asin(max(-1,min(1,g[1]))))

def frames_from_hitl(sid, df):
    g = df[df['sample'] == sid]
    return [dict(img=r['image'], gz=[float(r.dx),float(r.dy),float(r.dz)],
                 lab=str(r['source']), fid=r['id']) for _, r in g.iterrows()]

def frames_from_idfinal(sid):
    j = json.load(open(P3/'id_final'/f'{sid}.json'))
    out = []
    for f in j['frames']:
        out.append(dict(img=str(P3/f['img']), gz=[float(v) for v in f['chosen']],
                        lab=f.get('used','auto'), fid=f"{f['tag']}_f{f['frame']:03d}"))
    return out

def pick(frames, lo, hi, prefer_manual=True):
    """pick the frame whose |yaw| lies in [lo,hi], closest to the band centre."""
    c = (lo+hi)/2; cands = []
    for f in frames:
        a = abs(yaw_of(f['gz']))
        if lo <= a <= hi and abs(pit_of(f['gz'])) < 48:
            man = 1 if (prefer_manual and f['lab']=='manual') else 0
            cands.append((man, -abs(a-c), f))
    if not cands: return None
    cands.sort(key=lambda t:(-t[0], -t[1]))
    return cands[0][2]

def main():
    df = pd.read_csv(HITL)
    srcs, galleries = [], {}
    for sid, tag in [(s,'train-id') for s in TRAIN_IDS] + [(s,'held-out-id') for s in HELD_IDS]:
        frames = frames_from_hitl(sid, df) if tag=='train-id' else frames_from_idfinal(sid)
        frames = [f for f in frames if os.path.exists(f['img'])]
        if not frames: print('NO FRAMES', sid); continue
        dom = sid.split('_')[0]
        galleries[sid] = [dict(img=f['img'], gz=f['gz'], lab=f['lab'], fid=f['fid']) for f in frames]
        for pose, (lo, hi) in [('front',(0,25)), ('side',(60,120)), ('back',(145,180))]:
            f = pick(frames, lo, hi)
            if f is None:
                print(f'  MISS {pose:5s} {sid}'); continue
            srcs.append(dict(id=f['fid'], ident=sid, dom=dom, tag=tag, pose=pose,
                             path=f['img'], gt=f['gz'], lab=f['lab']))
            print(f'  {pose:5s} {tag:12s} {sid[:40]:40s} yaw={yaw_of(f["gz"]):+7.1f} pit={pit_of(f["gz"]):+6.1f} lab={f["lab"]}')
    for cid, dom, tag in CROP_ONLY:
        p = CROP/f'{cid}.jpg'
        if not p.exists(): print('MISS crop', p); continue
        srcs.append(dict(id=cid, ident=cid, dom=dom, tag=tag, pose='front',
                         path=str(p), gt=None, lab='none'))
        print(f'  front {tag:12s} {cid[:40]:40s} (crop only, no GT gallery)')
    json.dump(dict(srcs=srcs, galleries=galleries), open(OUT,'w'))
    ng = sum(len(v) for v in galleries.values())
    print(f'\nSAVED {OUT}\n  {len(srcs)} sources | {len(galleries)} galleries | {ng} GT frames')

if __name__ == '__main__': main()

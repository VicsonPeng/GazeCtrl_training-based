"""Build heldout.csv: the 8 identities that have phase3/id_final labels but were
NEVER in the training csv (hitl.csv). Schema mirrors hitl.csv so eval_true.py reads it.
NOTE: labels here are AUTOMATIC (3DGazeNet / 6DRepNet), not human-verified."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, glob, os
import pandas as pd
from pathlib import Path
P3=Path(RAW + "/phase3")
HITL=WORK + "/_cn_hitl/hitl.csv"
OUT=WORK + "/_cn_hitl/heldout.csv"
train=set(pd.read_csv(HITL)['sample'])
idf=sorted({os.path.basename(p)[:-5] for p in glob.glob(str(P3/'id_final'/'*.json'))}-train)
rows=[]
for sid in idf:
    j=json.load(open(P3/'id_final'/f'{sid}.json'))
    for f in j['frames']:
        img=P3/f['img']
        if not img.exists(): continue
        dx,dy,dz=[float(v) for v in f['chosen']]
        rows.append(dict(id=f"{f['tag']}_f{f['frame']:03d}", sample=sid, mode=f['tag'].rsplit('__',1)[-1],
                         frame=f['frame'], dx=dx, dy=dy, dz=dz, source=f.get('used','auto'),
                         head_angle=0.0, image=str(img), viz=''))
d=pd.DataFrame(rows)
d.to_csv(OUT,index=False)
print(f"SAVED {OUT}\n  {len(d)} frames | {d['sample'].nunique()} held-out identities")
print("  label source:", d['source'].value_counts().to_dict())
print("\n  per-identity frame count:"); print(d.groupby('sample').size().to_string())

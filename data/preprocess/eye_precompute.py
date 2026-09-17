"""Precompute eye boxes (insightface buffalo_l 5-pt kps -> both eyes -> box in
512 training space). CPU. Writes eye_boxes.json + progress log with flush."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, math, sys, time
import numpy as np, pandas as pd, cv2
from insightface.app import FaceAnalysis
D=WORK + "/_cn_hitl"
app=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(384,384))
df=pd.read_csv(f"{D}/hitl.csv")
def to512(x,y):  # frame 512w x 768h -> pad width to 768 (x+128) -> scale 512/768
    s=512.0/768.0; return (x+128)*s, y*s
boxes={}; noface=0; t0=time.time()
for i,r in enumerate(df.itertuples()):
    img=cv2.imread(r.image)
    if img is None: noface+=1; continue
    faces=app.get(img)
    if not faces: noface+=1; continue
    f=max(faces, key=lambda x:float(x.det_score))
    e0,e1=f.kps[0],f.kps[1]
    d=float(np.hypot(e0[0]-e1[0],e0[1]-e1[1]))+1e-6
    minx,maxx=min(e0[0],e1[0]),max(e0[0],e1[0]); miny,maxy=min(e0[1],e1[1]),max(e0[1],e1[1])
    px,py=0.5*d,0.45*d
    x1,y1=to512(minx-px,miny-py); x2,y2=to512(maxx+px,maxy+py)
    boxes[r.id]=[float(x1),float(y1),float(x2),float(y2)]
    if (i+1)%200==0:
        print(f"{i+1}/{len(df)}  eye={len(boxes)} noface={noface}  {time.time()-t0:.0f}s",flush=True)
json.dump(boxes,open(f"{D}/eye_boxes.json","w"))
ws=[b[2]-b[0] for b in boxes.values()]; hs=[b[3]-b[1] for b in boxes.values()]
print(f"DONE frames={len(df)} eye_boxes={len(boxes)} noface={noface}",flush=True)
print(f"eye box 512px: w~{np.mean(ws):.0f} h~{np.mean(hs):.0f}  patches w~{np.mean(ws)/16:.1f} h~{np.mean(hs)/16:.1f}",flush=True)

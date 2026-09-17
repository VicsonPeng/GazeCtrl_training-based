"""Precompute head bbox (in the 512x512 pad-to-square training space) for every
labels_top20 frame -> head_boxes.json {id: [x1,y1,x2,y2]}.  Used for head-weighted
loss and the gaze-consistency head crop."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps
from ultralytics import YOLO
DATA=Path(DATASET); YOLO_P=YOLO_HEAD
def sq(p,s=512):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))
def main():
    det=YOLO(YOLO_P); df=pd.read_csv(str(DATA/'labels_top20.csv'))
    out={}; miss=0
    for i,r in df.iterrows():
        bgr=cv2.cvtColor(np.array(sq(str(DATA/r['image']))),cv2.COLOR_RGB2BGR)
        res=det(bgr,verbose=False)[0].boxes
        if res is None or len(res)==0: miss+=1; continue
        xy=res.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])))
        out[r['id']]=[float(v) for v in xy[k]]
        if (i+1)%500==0: print(i+1,'/',len(df),flush=True)
    json.dump(out,open(str(DATA/'head_boxes.json'),'w'))
    print(f'saved {len(out)} boxes, {miss} missed -> head_boxes.json')
if __name__=='__main__': main()

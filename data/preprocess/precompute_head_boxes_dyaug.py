"""Fill the head-box gap for 50id_final_clean.csv: the 507 dy-aug frames that
precompute_head_boxes.py never covered (it ran off labels_top20.csv, i.e. frames/ only).
Writes a SEPARATE json so the original head_boxes.json stays untouched; also records the
detection count and confidence so a wrong pick can be spotted during verification."""

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
CSV=WORK + "/_cn_50id/50id_final_clean.csv"
OUT=DATA/'head_boxes_dyaug.json'
def sq(p,s=512):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))
def main():
    have=json.load(open(DATA/'head_boxes.json'))
    df=pd.read_csv(CSV); todo=df[~df['id'].astype(str).isin(have)]
    print(f'{len(todo)} frames need a head box (of {len(df)})',flush=True)
    det=YOLO(YOLO_P); out={}; miss=[]
    for n,(_,r) in enumerate(todo.iterrows()):
        bgr=cv2.cvtColor(np.array(sq(str(DATA/r['image']))),cv2.COLOR_RGB2BGR)
        res=det(bgr,verbose=False)[0].boxes
        if res is None or len(res)==0: miss.append(str(r['id'])); continue
        xy=res.xyxy.cpu().numpy(); cf=res.conf.cpu().numpy()
        k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])))
        out[str(r['id'])]={"box":[float(v) for v in xy[k]],"n_det":int(len(xy)),"conf":float(cf[k])}
        if (n+1)%100==0: print(n+1,'/',len(todo),flush=True)
    json.dump(out,open(OUT,'w'))
    print(f'saved {len(out)} boxes, {len(miss)} with NO detection -> {OUT}',flush=True)
    if miss: print('  no-detection ids:',miss[:20],flush=True)
if __name__=='__main__': main()

"""Payload for the head-box verification artifact: every frame of 50id_final_clean.csv
with its head box drawn, plus the k=2.0 crop rectangle that box would produce, so a
wrong box is judged by its actual consequence and not just by the rectangle."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, base64, io, os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw

DATA=Path(DATASET)
CSV=WORK + "/_cn_50id/50id_final_clean.csv"
OUT=OUT + "/headbox_verify_data.json"
TH=190; Q=74; K=2.0; MIN_FRAC=0.45

HB1=json.load(open(DATA/'head_boxes.json'))
HB2=json.load(open(DATA/'head_boxes_dyaug.json'))

def padinfo(p):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return im,w,h,m,((m-w)//2,(m-h)//2)

def crop_rect_orig(hb_o,W,H):
    """representative (median-scale, centred) k=2.0 crop in ORIGINAL coords."""
    x1,y1,x2,y2=hb_o; hmax=max(x2-x1,y2-y1); M=min(W,H)
    lo=max(K*hmax, MIN_FRAC*M); hi=float(M)
    if lo>=hi: return None
    s=(lo+hi)/2
    def ax(a1,a2,L):
        l=max(0.0,a2-s); h=min(L-s,a1)
        return None if l>h else (l+h)/2
    x=ax(x1,x2,W); y=ax(y1,y2,H)
    return None if x is None or y is None else (x,y,s)

d=pd.read_csv(CSV); rows=[]
for i,r in d.iterrows():
    fid=str(r['id'])
    im,W,H,m,(ox,oy)=padinfo(DATA/r['image'])
    sqim=ImageOps.expand(im,(ox,oy,m-W-ox,m-H-oy)).resize((TH,TH))
    k=TH/512.0
    src_hb='orig' if fid in HB1 else ('dyaug' if fid in HB2 else None)
    conf=None; ndet=None; box=None
    if src_hb=='orig': box=HB1[fid]
    elif src_hb=='dyaug':
        e=HB2[fid]; box=e['box']; conf=round(e['conf'],3); ndet=e['n_det']
    dr=ImageDraw.Draw(sqim); crop_frac=None
    if box:
        dr.rectangle([box[0]*k,box[1]*k,box[2]*k,box[3]*k],outline=(80,220,140),width=3)
        # map padded-512 box -> original coords, compute crop, map back for display
        sc=m/512.0
        hb_o=[box[0]*sc-ox, box[1]*sc-oy, box[2]*sc-ox, box[3]*sc-oy]
        cr=crop_rect_orig(hb_o,W,H)
        if cr:
            cx,cy,cs=cr; crop_frac=round(cs/min(W,H),3)
            px=[(cx+ox)/sc*k,(cy+oy)/sc*k,(cx+cs+ox)/sc*k,(cy+cs+oy)/sc*k]
            dr.rectangle(px,outline=(246,183,74),width=3)
    b=io.BytesIO(); sqim.save(b,'JPEG',quality=Q)
    rows.append({"id":fid,"s":str(r['sample']),"m":str(r['mode']),"lab":str(r['source']),
                 "hb":src_hb,"conf":conf,"nd":ndet,"cf":crop_frac,
                 "im":base64.b64encode(b.getvalue()).decode()})
    if (i+1)%250==0: print(i+1,'/',len(d),flush=True)
json.dump({"rows":rows,"n":len(rows),"k":K,"csv":Path(CSV).name},open(OUT,'w'),separators=(',',':'))
nb=sum(1 for x in rows if x['hb'] is None)
print(f"SAVED {OUT}  {os.path.getsize(OUT)/1e6:.2f} MB | no-box {nb} | dyaug {sum(1 for x in rows if x['hb']=='dyaug')} | orig {sum(1 for x in rows if x['hb']=='orig')}")

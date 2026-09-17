"""Variant B: take the random square crop in ORIGINAL image coordinates instead of
in the pad-to-square 512 space, so a crop never contains the letterbox padding and
the result is a natural head+torso framing (closer to the coco/widerface data).
Head boxes are stored in 512-padded space, so they are mapped back to original coords first."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import json, random, argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw

DATA=Path(DATASET)
CSV=WORK + "/_cn_50id/50id_final_clean.csv"
HB=json.load(open(DATA/'head_boxes.json'))
P_FULL=0.35; MIN_HEAD_MULT=3.5; MIN_FRAC=0.45; MARGIN=0.15

def sq(im,s=512):
    w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))

def hb_to_orig(hb,W,H,S=512):
    """undo pad-to-square + resize: padded512 -> original pixel coords."""
    m=max(W,H); k=m/S; ox=(m-W)//2; oy=(m-H)//2
    return [hb[0]*k-ox, hb[1]*k-oy, hb[2]*k-ox, hb[3]*k-oy]

def crop_rect(hs,ht,W,H,rng=random):
    if rng.random()<P_FULL: return None
    x1=min(hs[0],ht[0]); y1=min(hs[1],ht[1]); x2=max(hs[2],ht[2]); y2=max(hs[3],ht[3])
    hmax=max(x2-x1,y2-y1); M=min(W,H)
    lo=max(MIN_HEAD_MULT*hmax, MIN_FRAC*M); hi=float(M)
    if lo>=hi: return None
    m=MARGIN*hmax; s=rng.uniform(lo,hi)
    def axis(a1,a2,L):
        lo_=max(0.0,a2-s); hi_=min(L-s,a1)
        if lo_>hi_: return None
        l2=max(lo_,a2+m-s); h2=min(hi_,a1-m)
        return (l2,h2) if l2<=h2 else (lo_,hi_)
    ax=axis(x1,x2,W); ay=axis(y1,y2,H)
    if ax is None or ay is None: return None
    return (rng.uniform(*ax), rng.uniform(*ay), s)

def render(im,hb,rect,label,S=512):
    if rect is None:
        out=sq(im,S); k=S/max(im.size); ox=(max(im.size)-im.size[0])//2; oy=(max(im.size)-im.size[1])//2
        box=[(hb[0]+ox)*k,(hb[1]+oy)*k,(hb[2]+ox)*k,(hb[3]+oy)*k]
    else:
        x,y,s=rect
        out=im.crop((int(x),int(y),int(x+s),int(y+s))).resize((S,S),Image.BICUBIC)
        k=S/s; box=[(hb[0]-x)*k,(hb[1]-y)*k,(hb[2]-x)*k,(hb[3]-y)*k]
    d=ImageDraw.Draw(out); d.rectangle(box,outline=(80,220,140),width=4)
    d.rectangle([0,0,S-1,40],fill=(0,0,0)); d.text((8,10),label,fill=(255,255,255))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=6)
    ap.add_argument('--draws',type=int,default=2); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',default=OUT + "/crop_preview_orig.png")
    a=ap.parse_args(); rng=random.Random(a.seed)
    d=pd.read_csv(CSV); d=d[d['id'].astype(str).isin(HB)]
    ok=[s for s,g in d.groupby('sample') if len(g)>=2]; rng.shuffle(ok); picks=ok[:a.n]
    C=200; cols=2+2*a.draws
    sheet=Image.new('RGB',(C*cols,C*len(picks)+28),(16,20,26)); dr=ImageDraw.Draw(sheet)
    hdr=["SRC full(pad)","TGT full(pad)"]+sum([[f"SRC crop#{i+1}",f"TGT crop#{i+1}"] for i in range(a.draws)],[])
    for j,t in enumerate(hdr): dr.text((j*C+8,8),t,fill=(220,230,240))
    fr=[]
    for r,sid in enumerate(picks):
        g=d[d['sample']==sid].sample(2,random_state=a.seed+r); rs,rt=g.iloc[0],g.iloc[1]
        ims=Image.open(DATA/rs['image']).convert('RGB'); imt=Image.open(DATA/rt['image']).convert('RGB')
        W,H=ims.size
        hs=hb_to_orig(HB[str(rs['id'])],W,H); ht=hb_to_orig(HB[str(rt['id'])],*imt.size)
        cells=[render(ims,hs,None,f"{sid[-12:]} {rs['mode']} {W}x{H}"), render(imt,ht,None,f"{rt['mode']}")]
        for k in range(a.draws):
            rect=crop_rect(hs,ht,W,H,rng=rng)
            f=1.0 if rect is None else rect[2]/min(W,H); fr.append(f)
            tag="FULL (no crop)" if rect is None else f"crop {rect[2]:.0f}px = {f*100:.0f}% of width"
            cells.append(render(ims,hs,rect,tag)); cells.append(render(imt,ht,rect,tag))
        for c,im in enumerate(cells): sheet.paste(im.resize((C,C)),(c*C,28+r*C))
    sheet.save(a.out); fr=np.array(fr)
    print(f"SAVED {a.out} ({sheet.size[0]}x{sheet.size[1]})")
    cr=fr[fr<1.0]
    print(f"draws={len(fr)} full-frame={int((fr==1.0).sum())} cropped: median={np.median(cr)*100:.0f}% min={cr.min()*100:.0f}% max={cr.max()*100:.0f}% (of image width)")
if __name__=='__main__': main()

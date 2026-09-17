"""Re-label dy of the 400 selected pitch frames using insightface pitch
(reliable at profile, unlike 6DRep). Drop frames insightface can't detect.
Calibrate insightface pitch -> dy using frontal frames (old dy reliable there).
Preserve yaw direction (dx,dz) and rescale by sqrt(1-dy^2). Also zero-out the
old yaw frames' noisy dy. Emits 50id_dyaug_relabel.csv + report."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, glob, math, csv
import numpy as np, cv2, pandas as pd
from insightface.app import FaceAnalysis
SEL=WAN_DIR + "/dy_aug/selected_frames"
BASE=WORK + "/_cn_50id"
app=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))
def if_pitch(img):
    faces=app.get(img)
    if not faces: return None
    fc=max(faces,key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    p=getattr(fc,'pose',None)
    return float(p[0]) if p is not None else None   # pitch deg (up>0)
# 1) gather per-frame: old dy/dx/dz + insightface pitch (cached)
CACHE=WORK + "/pitch_ifp_cache.csv"
if os.path.exists(CACHE):
    df=pd.read_csv(CACHE); print(f"loaded cache {len(df)}")
else:
    recs=[]
    for lc in sorted(glob.glob(f'{SEL}/*_labels.csv')):
        for r in csv.DictReader(open(lc)):
            img=cv2.imread(r['path'])
            ip=if_pitch(img) if img is not None else None
            recs.append(dict(path=r['path'],video=r['video'],dx=float(r['dx']),dy=float(r['dy']),
                             dz=float(r['dz']),ifp=ip))
    df=pd.DataFrame(recs); df.to_csv(CACHE,index=False)
n=len(df); det=df['ifp'].notna().sum()
print(f"pitch帧 {n}: insightface检测到 {det} ({100*det/n:.0f}%), 丢弃 {n-det}")
# 2) calibrate on FRONTAL detected frames: old dy (reliable) vs insightface pitch
front=df[(df['ifp'].notna())&(df['dx'].abs()<0.35)&(df['dz']>0.55)]
a,b=np.polyfit(front['ifp'].values, front['dy'].values, 1)
print(f"标定(正面 {len(front)}帧): dy = {a:.5f}*iF_pitch + {b:+.4f}")
# 3) recompute dy for detected frames, preserve yaw dir
def relab(row):
    if pd.isna(row['ifp']): return None
    dyn=float(np.clip(a*row['ifp']+b,-0.98,0.98))
    hx,hz=row['dx'],row['dz']; hn=math.hypot(hx,hz) or 1e-6
    r=math.sqrt(max(0,1-dyn*dyn))
    return (hx/hn*r, dyn, hz/hn*r)
keep=[]
for _,row in df.iterrows():
    v=relab(row)
    if v is None: continue
    keep.append((row['video'],row['path'],*v))
print(f"重标注保留 {len(keep)} 帧 (dy范围 {min(k[3] for k in keep):+.2f}~{max(k[3] for k in keep):+.2f})")
# 4) build new csv: base yaw frames (dy->0) + relabeled pitch frames
base=pd.read_csv(f'{BASE}/50id_dyaug.csv')
yaw=base[base['mode']!='pitch'].copy()
# clean yaw dy -> 0, renormalize dx,dz
hn=np.hypot(yaw['dx'],yaw['dz']).replace(0,1e-6)
yaw['dx']=yaw['dx']/hn; yaw['dz']=yaw['dz']/hn; yaw['dy']=0.0
cols=list(base.columns)
rows=[]
for vid,path,dx,dy,dz in keep:
    idnum=vid.split('_')[1]; sample=f'shhq_image_{idnum}__p0'
    nn=os.path.basename(path).replace('frame','').replace('.jpg','')  # e.g. '03'
    fdname=f'{vid}_f{nn}.jpg'                                         # matches frames_dy naming
    rows.append({'id':f'{vid}_f{nn}','sample':sample,'mode':'pitch',
                 'frame':int(nn),'dx':round(dx,4),'dy':round(dy,4),'dz':round(dz,4),'source':'insightface',
                 'head_angle':0,'image':'frames_dy/'+fdname,'viz':''})
add=pd.DataFrame(rows,columns=cols)
out=pd.concat([yaw,add],ignore_index=True)
out.to_csv(f'{BASE}/50id_dyaug_relabel.csv',index=False)
print(f"-> 50id_dyaug_relabel.csv : yaw {len(yaw)}(dy清零) + pitch {len(add)}(重标) = {len(out)}")
# save relabel map for the verification montage
pd.DataFrame([{'path':k[1],'video':k[0],'dx':k[2],'dy':k[3],'dz':k[4]} for k in keep]).to_csv(WORK + "/pitch_relabeled.csv",index=False)

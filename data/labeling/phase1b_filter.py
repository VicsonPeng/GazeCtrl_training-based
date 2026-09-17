"""Phase1 (refined): pick 200 src satisfying single-person + eyes-visible (no
sunglasses) + more coco. Face must be detectable & clear. Rank by sharpness."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, glob, random, shutil, cv2, numpy as np, csv
from insightface.app import FaceAnalysis
ROOT=RAW
CROPS=f'{ROOT}/crops'; OUT=f'{ROOT}/selected_wan_src'
FACE_MIN=45; CONF_MIN=0.6; WF_POOL=3500; TARGET_COCO=100; TARGET_WF=100
app=FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(640,640))
def eyes_visible(img,fc):
    kps=fc.kps.astype(int); le,re=kps[0],kps[1]  # left/right eye
    fh=int(fc.bbox[3]-fc.bbox[1]); pr=max(4,fh//12)
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    def patch(pt):
        y1,y2,x1,x2=max(0,pt[1]-pr),pt[1]+pr,max(0,pt[0]-pr),pt[0]+pr
        p=g[y1:y2,x1:x2]; return p.mean() if p.size else 0
    x1,y1,x2,y2=[int(v) for v in fc.bbox]; face=g[max(0,y1):y2,max(0,x1):x2]
    fm=face.mean() if face.size else 1
    le_m,re_m=patch(le),patch(re)
    # 墨镜: 双眼区显著暗于整脸
    return not (le_m<0.5*fm and re_m<0.5*fm)
def eval_img(fn):
    img=cv2.imread(f'{CROPS}/{fn}')
    if img is None: return None
    faces=app.get(img)
    if not faces: return None
    faces.sort(key=lambda x:-(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    fc=faces[0]; A=(fc.bbox[2]-fc.bbox[0])*(fc.bbox[3]-fc.bbox[1])
    # 单人: 无其他 det>0.5 且面积>0.3主脸 的脸
    for o in faces[1:]:
        if float(o.det_score)>0.5 and (o.bbox[2]-o.bbox[0])*(o.bbox[3]-o.bbox[1])>0.3*A: return None
    fw=int(fc.bbox[2]-fc.bbox[0])
    if fw<FACE_MIN or float(fc.det_score)<CONF_MIN: return None
    if not eyes_visible(img,fc): return None
    x1,y1,x2,y2=[int(v) for v in fc.bbox]; fcrop=img[max(0,y1):y2,max(0,x1):x2]
    if fcrop.size==0: return None
    sharp=float(cv2.Laplacian(cv2.cvtColor(cv2.resize(fcrop,(112,112)),cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    return dict(fn=fn,fw=fw,det=float(fc.det_score),sharp=sharp)
def pick(prefix,pool,target):
    cands=[os.path.basename(p) for p in glob.glob(f'{CROPS}/{prefix}*.jpg')]
    random.seed(42); random.shuffle(cands); cands=cands[:pool]
    kept=[]
    for i,fn in enumerate(cands):
        r=eval_img(fn)
        if r: kept.append(r)
        if (i+1)%500==0: print(f"  {prefix}: {i+1}/{len(cands)} kept {len(kept)}",flush=True)
    kept.sort(key=lambda r:-r['sharp'])
    return kept
print("=== coco ===")
coco=pick('coco',2000,TARGET_COCO)
print("=== wf ===")
wf=pick('wf',WF_POOL,TARGET_WF)
sel=coco[:TARGET_COCO]+wf[:TARGET_WF]
if len(sel)<200: sel+=wf[TARGET_WF:TARGET_WF+(200-len(sel))]  # coco不足用wf补
# 清旧 + 复制
shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT,exist_ok=True)
for r in sel: shutil.copy(f'{CROPS}/{r["fn"]}',f'{OUT}/{r["fn"]}')
with open(f'{OUT}/_manifest.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['fn','face_w','det','sharp']); [w.writerow([r['fn'],r['fw'],round(r['det'],3),round(r['sharp'],1)]) for r in sel]
print(f"\n选出 {len(sel)}: coco {sum(1 for r in sel if r['fn'].startswith('coco'))}, wf {sum(1 for r in sel if r['fn'].startswith('wf'))}")
from PIL import Image
n=len(sel); ncol=16; nrow=(n+ncol-1)//ncol; TS=100; cv=Image.new('RGB',(ncol*TS,nrow*TS),'black')
for i,r in enumerate(sel):
    im=cv2.cvtColor(cv2.imread(f'{CROPS}/{r["fn"]}'),cv2.COLOR_BGR2RGB); h,w=im.shape[:2]; sc=min(TS/w,TS/h)
    im=cv2.resize(im,(int(w*sc),int(h*sc))); c=Image.new('RGB',(TS,TS),'black'); c.paste(Image.fromarray(im),((TS-im.shape[1])//2,(TS-im.shape[0])//2)); cv.paste(c,((i%ncol)*TS,(i//ncol)*TS))
cv.save(OUT + "/phase1_selected200.jpg",quality=88); print("contact sheet saved")

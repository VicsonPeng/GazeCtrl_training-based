"""Final demo candidates: every source x the full gaze grid, at the chosen resolution.

Sources come from a json list where each entry may carry a crop rect in ORIGINAL image
coordinates (the head-preserving crop the model saw during training) or None for the plain
pad-to-square framing. Outputs are written as PNG so nothing is lost before the selection is
made; every generation is also measured (YOLO head -> 6DRepNet for direction, ArcFace for
identity) so the contact sheets can be annotated rather than merely admired.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
import os,sys,math,argparse,json,time
import numpy as np, torch, cv2, pandas as pd
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageOps
from safetensors.torch import load_file
sys.path.insert(0, _p)
sys.path.insert(0, SIXDREP_DIR)
from qwen_models import (QwenBackbone,QwenControlNet,load_qwen_vae,vae_encode,vae_decode,
                         patchify,unpatchify,compute_rope_freqs_3d,compute_text_rope_freqs)
from model import SixDRepNet2; from torchvision.models.resnet import Bottleneck; import utils as u6d
from ultralytics import YOLO
from insightface.app import FaceAnalysis
R=Path(ROOT); M=Path(MODELS)
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA=M/'vae/split_files/vae/qwen_image_vae.safetensors'
YAWS=[-180,-135,-90,-65,-40,-15,0,15,40,65,90,135]
PITCHES=[60,30,0,-30,-60]
STEPS=16
ap=argparse.ArgumentParser()
ap.add_argument('--ckpts',required=True,help='comma list of name=path; the backbone loads once and only the ControlNet is swapped')
ap.add_argument('--srcs',required=True); ap.add_argument('--outdir',required=True)
ap.add_argument('--size',type=int,default=768); ap.add_argument('--cn_scale',type=float,default=1.0)
ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshard',type=int,default=1)
a=ap.parse_args()
CKS=[c.split('=',1) for c in a.ckpts.split(',') if c]
os.makedirs(a.outdir,exist_ok=True)
dev=torch.device('cuda:0'); SZ=a.size; G=SZ//8//2
def load_src(e):
    im=Image.open(e['path']).convert('RGB')
    if e.get('rect'):
        x,y,s=e['rect']; im=im.crop((int(x),int(y),int(x+s),int(y+s)))
    else:
        w,h=im.size; m=max(w,h); im=ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2))
    return im.resize((SZ,SZ),Image.LANCZOS)
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
def w180(x): return (x+180.)%360.-180.
vae=load_qwen_vae(str(VA),device=dev); bb=QwenBackbone(str(BB),device=dev)
cn=QwenControlNet().to(dev)
m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
yd=YOLO(YOLO_HEAD)
fa=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider']); fa.prepare(ctx_id=0,det_size=(640,640))
IMN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
imf=compute_rope_freqs_3d(G,G,t_patches=2,device=dev); imfc=compute_rope_freqs_3d(G,G,t_patches=1,device=dev)
txf=compute_text_rope_freqs(7,device=dev)
@torch.no_grad()
def gen(ref,gz):
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
    col=(np.array([(dx/n+1)/2,(dy/n+1)/2,(dz/n+1)/2])*255).clip(0,255)
    c=(torch.from_numpy(np.full((SZ,SZ,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    cp=patchify(vae_encode(vae,c)).to(torch.bfloat16)
    torch.manual_seed(a.seed); x=torch.randn(1,16,1,SZ//8,SZ//8,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
    for i,sg in enumerate(S):
        si=sg.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
        res=[q*a.cn_scale for q in cn.forward(patchify(x),cp,te,si,img_freqs=imfc,txt_freqs=txf)]
        v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        x=x-(sg-sn).view(1,1,1,1,1)*unpatchify(v,G,G)
    return vae_decode(vae,x)[0]
@torch.no_grad()
def measure(arr):
    bgr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR); my=None
    r=yd(bgr,verbose=False)[0].boxes
    if r is not None and len(r)>0:
        xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); x1,y1,x2,y2=xy[k]
        pad=int(.35*max(x2-x1,y2-y1)); H,W=bgr.shape[:2]
        c=bgr[max(0,int(y1-pad)):min(H,int(y2+pad)),max(0,int(x1-pad)):min(W,int(x2+pad))]
        if c.size:
            t=torch.from_numpy(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)/255.
            t=F.interpolate(t,(224,224),mode='bilinear',align_corners=False); t=(t-IMN)/ISD
            e=u6d.compute_euler_angles_from_rotation_matrices(m6(t.float()))[0].cpu().numpy()
            pit,yaw=float(e[0]),float(e[1])
            dx=-math.sin(yaw)*math.cos(pit); dy=math.sin(pit); dz=math.cos(yaw)*math.cos(pit)
            if dz<0: dx=-dx
            my=(math.degrees(math.atan2(dx,dz)), math.degrees(math.asin(max(-1,min(1,dy)))))
    fs=fa.get(bgr)
    emb=None if not fs else max(fs,key=lambda q:(q.bbox[2]-q.bbox[0])*(q.bbox[3]-q.bbox[1])).normed_embedding
    return my,emb
srcs=[e for i,e in enumerate(json.load(open(a.srcs))) if i%a.nshard==a.shard]
rows=[]; nskip=[0]
# cache each source's encoded reference once; it is identical across checkpoints
CACHE={}
for e in srcs:
    sid=e['sample']; src=load_src(e); CACHE[sid]=src
for name,path in CKS:
    OUT=os.path.join(a.outdir,name); os.makedirs(os.path.join(OUT,'img'),exist_ok=True)
    cn.load_state_dict(load_file(path)); cn.to(torch.bfloat16).eval()
    print(f'===== checkpoint {name}',flush=True)
    for e in srcs:
        sid=e['sample']; src=CACHE[sid]
        src.save(os.path.join(OUT,'img',f'{sid}__src.png'))
        x0=(torch.from_numpy(np.array(src)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
        with torch.no_grad(): ref=patchify(vae_encode(vae,x0)).to(torch.bfloat16)
        _,se=measure(np.array(src))
        t0=time.time()
        for pi,pit in enumerate(PITCHES):
            for yi,yaw in enumerate(YAWS):
                ry,rp=math.radians(yaw),math.radians(pit)
                gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
                fp=os.path.join(OUT,'img',f'{sid}__y{yi}_p{pi}.png')
                if os.path.exists(fp) and os.path.getsize(fp)>1024:
                    # resume: a completed PNG is authoritative, re-read it instead of regenerating.
                    # The seed is fixed, so the image would be identical anyway, and an interrupted
                    # run of this size (3000 images, ~10h) is too expensive to restart from zero.
                    arr=np.array(Image.open(fp).convert('RGB')); nskip[0]+=1
                else:
                    im=gen(ref,gz)
                    arr=(((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())
                    Image.fromarray(arr).save(fp)
                mv,em=measure(arr)
                rows.append(dict(ckpt=name,size=SZ,tag=e['tag'],mode=e['mode'],sample=sid,yi=yi,pi=pi,
                    req_yaw=yaw,req_pitch=pit,
                    meas_yaw=(None if mv is None else round(mv[0],1)),
                    meas_pitch=(None if mv is None else round(mv[1],1)),
                    yaw_err=(None if mv is None else round(abs(w180(mv[0]-yaw)),1)),
                    id_sim=(None if (em is None or se is None) else round(float(np.dot(em,se)),3)),
                    det_head=int(mv is not None), det_face=int(em is not None)))
            pd.DataFrame(rows).to_csv(os.path.join(a.outdir,f'demo_shard{a.shard}.csv'),index=False)
        print(f'   [{name}] {sid[:40]:40s} {e["tag"]}/{e["mode"]}  60 cells {time.time()-t0:.0f}s (reused {nskip[0]})',flush=True)
        nskip[0]=0
print('DONE',len(rows))

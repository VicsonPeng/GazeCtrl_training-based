"""Checkpoint x cn_scale grid on a single user-supplied photo.

cn_scale multiplies the ControlNet residuals before they reach the frozen backbone.
Training used 1.0, so everything else is out of distribution: raising it usually buys
condition adherence at the cost of identity and image quality. The point of the grid is to
see where that trade sits for THIS model, not to assume a higher value is better.
The backbone is loaded once; only the ControlNet state dict is swapped per checkpoint.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
import os,sys,math,json,argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageOps
from safetensors.torch import load_file
sys.path.insert(0, SIXDREP_DIR)
from qwen_models import (QwenBackbone,QwenControlNet,load_qwen_vae,vae_encode,vae_decode,
                         patchify,unpatchify,compute_rope_freqs_3d,compute_text_rope_freqs)
from model import SixDRepNet2; from torchvision.models.resnet import Bottleneck; import utils as u6d
from ultralytics import YOLO
from insightface.app import FaceAnalysis
R=Path(ROOT); M=Path(MODELS)
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA=M/'vae/split_files/vae/qwen_image_vae.safetensors'
GRID=32; STEPS=16
CONDS=[(0,0),(-45,45),(0,60),(0,-60),(150,0)]     # (yaw, pitch) degrees
def sq(p,s):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
ap=argparse.ArgumentParser()
ap.add_argument('--src',required=True); ap.add_argument('--outdir',required=True)
ap.add_argument('--ckpts',required=True,help='comma-separated safetensors paths')
ap.add_argument('--scales',default='0.6,0.8,1.0,1.2,1.5'); ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args()
os.makedirs(os.path.join(a.outdir,'img'),exist_ok=True)
CKS=[c for c in a.ckpts.split(',') if c]; SCALES=[float(s) for s in a.scales.split(',')]
dev=torch.device('cuda:0')
vae=load_qwen_vae(str(VA),device=dev); bb=QwenBackbone(str(BB),device=dev)
cn=QwenControlNet().to(dev)
m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
yd=YOLO(YOLO_HEAD)
fa=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider']); fa.prepare(ctx_id=0,det_size=(640,640))
IMN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev); imfc=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev)
txf=compute_text_rope_freqs(7,device=dev)
img512=sq(a.src,512)
x0=(torch.from_numpy(np.array(img512)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
with torch.no_grad(): ref=patchify(vae_encode(vae,x0)).to(torch.bfloat16)
def femb(bgr):
    fs=fa.get(bgr)
    if not fs: return None
    return max(fs,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding
SRCE=femb(cv2.cvtColor(np.array(img512),cv2.COLOR_RGB2BGR))
@torch.no_grad()
def run(gz,scale):
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
    col=(np.array([(dx/n+1)/2,(dy/n+1)/2,(dz/n+1)/2])*255).clip(0,255)
    c=(torch.from_numpy(np.full((512,512,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    cp=patchify(vae_encode(vae,c)).to(torch.bfloat16)
    torch.manual_seed(a.seed); x=torch.randn(1,16,1,64,64,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
    for i,s in enumerate(S):
        si=s.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
        res=[r*scale for r in cn.forward(patchify(x),cp,te,si,img_freqs=imfc,txt_freqs=txf)]
        v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        x=x-(s-sn).view(1,1,1,1,1)*unpatchify(v,GRID,GRID)
    return vae_decode(vae,x)[0]
@torch.no_grad()
def meas(im):
    bgr=cv2.cvtColor((((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()),cv2.COLOR_RGB2BGR)
    e=femb(bgr); r=yd(bgr,verbose=False)[0].boxes
    my=mp=None
    if r is not None and len(r)>0:
        xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); x1,y1,x2,y2=xy[k]
        pad=int(.35*max(x2-x1,y2-y1)); H,W=bgr.shape[:2]
        cc=bgr[max(0,int(y1-pad)):min(H,int(y2+pad)),max(0,int(x1-pad)):min(W,int(x2+pad))]
        if cc.size:
            t=torch.from_numpy(cv2.cvtColor(cc,cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)/255.
            t=F.interpolate(t,(224,224),mode='bilinear',align_corners=False); t=(t-IMN)/ISD
            eu=u6d.compute_euler_angles_from_rotation_matrices(m6(t.float()))[0].cpu().numpy()
            pit,yaw=float(eu[0]),float(eu[1])
            dx=-math.sin(yaw)*math.cos(pit); dy=math.sin(pit); dz=math.cos(yaw)*math.cos(pit)
            if dz<0: dx=-dx
            my=math.degrees(math.atan2(dx,dz)); mp=math.degrees(math.asin(max(-1,min(1,dy))))
    sim=None if (e is None or SRCE is None) else float(np.dot(e,SRCE))
    return my,mp,sim
def w180(x): return (x+180.)%360.-180.
rows=[]
for ckp in CKS:
    step=int(''.join(ch for ch in os.path.basename(ckp) if ch.isdigit()))
    cn.load_state_dict(load_file(ckp)); cn.to(torch.bfloat16).eval()
    print(f'== ckpt {step}',flush=True)
    for sc in SCALES:
        for (yaw,pit) in CONDS:
            ry,rp=math.radians(yaw),math.radians(pit)
            gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
            im=run(gz,sc)
            fn=f'{a.outdir}/img/s{step}_c{sc}_y{yaw}_p{pit}.jpg'
            Image.fromarray((((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())).resize((320,320),Image.LANCZOS).save(fn,quality=88)
            my,mp,sim=meas(im)
            rows.append(dict(step=step,scale=sc,req_yaw=yaw,req_pitch=pit,
                             meas_yaw=my,meas_pitch=mp,id_sim=sim,
                             yaw_err=(None if my is None else abs(w180(my-yaw))),
                             pitch_err=(None if mp is None else abs(mp-pit)),file=os.path.basename(fn)))
        print(f'   scale {sc} done ({len(rows)})',flush=True)
    import pandas as pd; pd.DataFrame(rows).to_csv(f'{a.outdir}/grid_{os.getpid()}.csv',index=False)
print('DONE',len(rows))

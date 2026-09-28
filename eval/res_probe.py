"""How high can the resolution go before the ControlNet stops following the condition?

The adapter was trained only at a 32x32 patch grid (512px). It extrapolates -- plausibly
because the gaze condition is a spatially UNIFORM colour, so there is no spatial structure to
misalign -- but the residuals are added per token, and the token count grows quadratically,
so the effective control strength can drift without the image visibly breaking. Extreme yaw
is where that would show first, so the probe includes back and profile targets, not just the
easy frontal ones.
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
import numpy as np, torch, cv2
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
R=Path(ROOT); M=Path(MODELS)
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA=M/'vae/split_files/vae/qwen_image_vae.safetensors'
SIZES=[512,768,1024]
CONDS=[(0,0),(-90,0),(150,0),(0,45)]          # frontal, profile, back, look-up
STEPS=16
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt',default='/home/vicson/gaze_diag/_cn_stage2/cn_step040000.safetensors')
ap.add_argument('--srcs',nargs='+',default=[os.path.join(ROOT,'gaze_controlnet','before.png'),
                                            os.path.join(ROOT,'gaze_controlnet','zeroshots','zs03.jpg')])
ap.add_argument('--outdir',required=True); a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
dev=torch.device('cuda:0')
def padsq(p):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2))
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
vae=load_qwen_vae(str(VA),device=dev); bb=QwenBackbone(str(BB),device=dev)
cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
yd=YOLO(YOLO_HEAD)
IMN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
def w180(x): return (x+180.)%360.-180.
@torch.no_grad()
def meas(arr):
    bgr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR); r=yd(bgr,verbose=False)[0].boxes
    if r is None or len(r)==0: return None
    xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); x1,y1,x2,y2=xy[k]
    pad=int(.35*max(x2-x1,y2-y1)); H,W=bgr.shape[:2]
    c=bgr[max(0,int(y1-pad)):min(H,int(y2+pad)),max(0,int(x1-pad)):min(W,int(x2+pad))]
    if c.size==0: return None
    t=torch.from_numpy(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)/255.
    t=F.interpolate(t,(224,224),mode='bilinear',align_corners=False); t=(t-IMN)/ISD
    e=u6d.compute_euler_angles_from_rotation_matrices(m6(t.float()))[0].cpu().numpy()
    pit,yaw=float(e[0]),float(e[1])
    dx=-math.sin(yaw)*math.cos(pit); dy=math.sin(pit); dz=math.cos(yaw)*math.cos(pit)
    if dz<0: dx=-dx
    return math.degrees(math.atan2(dx,dz))
@torch.no_grad()
def gen(base,size,gz):
    G=size//8//2
    imf=compute_rope_freqs_3d(G,G,t_patches=2,device=dev); imfc=compute_rope_freqs_3d(G,G,t_patches=1,device=dev)
    txf=compute_text_rope_freqs(7,device=dev)
    src=base.resize((size,size),Image.LANCZOS)
    x0=(torch.from_numpy(np.array(src)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    ref=patchify(vae_encode(vae,x0)).to(torch.bfloat16)
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
    col=(np.array([(dx/n+1)/2,(dy/n+1)/2,(dz/n+1)/2])*255).clip(0,255)
    c=(torch.from_numpy(np.full((size,size,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    cp=patchify(vae_encode(vae,c)).to(torch.bfloat16)
    torch.manual_seed(0); x=torch.randn(1,16,1,size//8,size//8,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
    for i,sg in enumerate(S):
        si=sg.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
        res=[q*1.0 for q in cn.forward(patchify(x),cp,te,si,img_freqs=imfc,txt_freqs=txf)]
        v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        x=x-(sg-sn).view(1,1,1,1,1)*unpatchify(v,G,G)
    return vae_decode(vae,x)[0]
rows=[]
for sp in a.srcs:
    base=padsq(sp); tag=Path(sp).stem
    for size in SIZES:
        for (y,p) in CONDS:
            ry,rp=math.radians(y),math.radians(p)
            gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
            try:
                torch.cuda.reset_peak_memory_stats(); t0=time.time()
                im=gen(base,size,gz); dt=time.time()-t0
                arr=(((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())
                Image.fromarray(arr).save(f'{a.outdir}/{tag}__{size}__y{y}_p{p}.png')
                my=meas(arr); err=None if my is None else abs(w180(my-y))
                pk=torch.cuda.max_memory_allocated()/2**30
                rows.append(dict(src=tag,size=size,req_yaw=y,req_pitch=p,meas_yaw=my,yaw_err=err,peak=pk,sec=dt))
                print(f'{tag:10s} {size:5d} y{y:+4d} p{p:+3d}  meas={"n/a" if my is None else f"{my:+6.1f}"}  '
                      f'err={"n/a" if err is None else f"{err:5.1f}"}  peak={pk:5.2f}GiB  {dt:5.1f}s',flush=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(dict(src=tag,size=size,req_yaw=y,req_pitch=p,meas_yaw=None,yaw_err=None,peak=None,sec=None))
                print(f'{tag:10s} {size:5d} y{y:+4d} p{p:+3d}  OUT OF MEMORY',flush=True)
import pandas as pd; d=pd.DataFrame(rows); d.to_csv(f'{a.outdir}/probe.csv',index=False)
print('\n=== median |yaw err| by size (lower better) ===')
print(d.dropna(subset=['yaw_err']).groupby('size').yaw_err.agg(['median','mean','count']).round(1).to_string())
print('\n=== peak GiB / seconds by size ===')
print(d.dropna(subset=['peak']).groupby('size')[['peak','sec']].mean().round(2).to_string())
print('PROBE_DONE')

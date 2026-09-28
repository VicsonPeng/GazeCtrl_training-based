"""Is the eye artifact a resolution problem, and can the resolution be raised?

Two questions, one script, four modes on the same photo and the same three gaze targets:
  full512   the pipeline as it ships                      -- both eyes span ~66 px
  crop512   head-preserving crop (k=2.0), then 512         -- same token budget, more px on the face
  full768   native 768 generation, full frame              -- 48x48 patches instead of 32x32
  crop768   head crop AND 768                              -- best case
The ControlNet was only ever trained at 512/32x32, so the 768 modes are out of distribution:
RoPE extrapolates in principle, but whether the control survives is an empirical question and
this is the cheapest way to answer it.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
import os,sys,math,argparse,json
import numpy as np, torch
from pathlib import Path
from PIL import Image, ImageOps
from safetensors.torch import load_file
sys.path.insert(0, _p)
from qwen_models import (QwenBackbone,QwenControlNet,load_qwen_vae,vae_encode,vae_decode,
                         patchify,unpatchify,compute_rope_freqs_3d,compute_text_rope_freqs)
from ultralytics import YOLO
R=Path(ROOT); M=Path(MODELS)
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA=M/'vae/split_files/vae/qwen_image_vae.safetensors'
CONDS=[(0,0),(15,0),(0,-30)]
STEPS=16
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt',default='/home/vicson/gaze_diag/_cn_stage2/cn_step040000.safetensors')
ap.add_argument('--src',default=os.path.join(ROOT,'gaze_controlnet','before.png'))
ap.add_argument('--outdir',required=True); ap.add_argument('--scale',type=float,default=1.0)
a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
dev=torch.device('cuda:0')
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
def padsq(im):
    w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2))
yd=YOLO(YOLO_HEAD)
raw=Image.open(a.src).convert('RGB'); W,H=raw.size
r=yd(np.array(raw)[:,:,::-1],verbose=False)[0].boxes
assert r is not None and len(r)>0, 'no head detected in the source'
xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); bx=xy[k]
hmax=max(bx[2]-bx[0],bx[3]-bx[1]); s=min(max(2.0*hmax,0.45*min(W,H)),min(W,H))
cx,cy=(bx[0]+bx[2])/2,(bx[1]+bx[3])/2
cxr=min(max(cx-s/2,0),W-s); cyr=min(max(cy-s/2,0),H-s)
head=raw.crop((int(cxr),int(cyr),int(cxr+s),int(cyr+s)))
print(f'source {W}x{H}  head box {[round(v) for v in bx]}  head {hmax:.0f}px  crop side {s:.0f}px',flush=True)
vae=load_qwen_vae(str(VA),device=dev); bb=QwenBackbone(str(BB),device=dev)
cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
@torch.no_grad()
def gen(img_pil,size,gz):
    G=size//8//2                                   # patch grid; 512->32, 768->48
    imf=compute_rope_freqs_3d(G,G,t_patches=2,device=dev)
    imfc=compute_rope_freqs_3d(G,G,t_patches=1,device=dev)
    txf=compute_text_rope_freqs(7,device=dev)
    src=img_pil.resize((size,size),Image.LANCZOS)
    x0=(torch.from_numpy(np.array(src)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    ref=patchify(vae_encode(vae,x0)).to(torch.bfloat16)
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
    col=(np.array([(dx/n+1)/2,(dy/n+1)/2,(dz/n+1)/2])*255).clip(0,255)
    c=(torch.from_numpy(np.full((size,size,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    cp=patchify(vae_encode(vae,c)).to(torch.bfloat16)
    torch.manual_seed(0); x=torch.randn(1,16,1,size//8,size//8,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
    for i,sg in enumerate(S):
        si=sg.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
        res=[q*a.scale for q in cn.forward(patchify(x),cp,te,si,img_freqs=imfc,txt_freqs=txf)]
        v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        x=x-(sg-sn).view(1,1,1,1,1)*unpatchify(v,G,G)
    return vae_decode(vae,x)[0]
MODES=[('full512',padsq(raw),512),('crop512',head,512),('full768',padsq(raw),768),('crop768',head,768)]
for name,base,size in MODES:
    base.resize((size,size),Image.LANCZOS).save(f'{a.outdir}/{name}__source.png')
    for (y,p) in CONDS:
        ry,rp=math.radians(y),math.radians(p)
        gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
        try:
            torch.cuda.reset_peak_memory_stats()
            im=gen(base,size,gz)
            arr=(((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())
            Image.fromarray(arr).save(f'{a.outdir}/{name}__y{y}_p{p}.png')
            print(f'{name:8s} y{y:+d} p{p:+d}  peak {torch.cuda.max_memory_allocated()/2**30:5.2f} GiB',flush=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f'{name:8s} y{y:+d} p{p:+d}  OUT OF MEMORY',flush=True)
print('RES_TEST_DONE')

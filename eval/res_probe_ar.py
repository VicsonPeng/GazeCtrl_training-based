"""Does the ControlNet survive a NON-SQUARE patch grid, and is it worth it?

The pipeline supports it already -- compute_rope_freqs_3d and unpatchify both take separate
h_patches and w_patches; passing (G,G) everywhere was a choice, not a constraint. It matters
because these are portrait photographs: padding them to square spends a measured ~65% of the
tokens on black bars. Generating at the source's own aspect ratio puts every token on the
subject, for the same budget. The ControlNet has only ever seen a 32x32 square grid, so
whether control survives a rectangular one is an empirical question.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
import os,sys,math,argparse,time
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
R=Path(ROOT); M=Path(MODELS)
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA=M/'vae/split_files/vae/qwen_image_vae.safetensors'
CONDS=[(0,0),(-90,0),(150,0),(0,45)]
STEPS=16
def grid_for(W,H,budget):
    """patch grid closest to the source aspect ratio at ~budget patches; 16px per patch."""
    ar=H/W; best=None
    for wp in range(16,161,1):
        hp=int(round(wp*ar))
        if hp<16: continue
        n=wp*hp
        if abs(n-budget)/budget>0.12: continue
        sc=abs(math.log((hp/wp)/ar))+abs(n-budget)/budget*0.35
        if best is None or sc<best[0]: best=(sc,hp,wp)
    if best is None:
        g=int(round(math.sqrt(budget))); return g,g
    return best[1],best[2]
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt',default='/home/vicson/gaze_diag/_cn_stage2/cn_step040000.safetensors')
ap.add_argument('--srcs',nargs='+',default=[os.path.join(ROOT,'gaze_controlnet','before.png'),
                                            os.path.join(ROOT,'gaze_controlnet','zeroshots','zs03.jpg')])
ap.add_argument('--outdir',required=True); a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
dev=torch.device('cuda:0')
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
def w180(x): return (x+180.)%360.-180.
vae=load_qwen_vae(str(VA),device=dev); bb=QwenBackbone(str(BB),device=dev)
cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
yd=YOLO(YOLO_HEAD)
IMN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
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
def gen(img,hp,wp,gz):
    Hp,Wp=hp*16,wp*16
    imf=compute_rope_freqs_3d(hp,wp,t_patches=2,device=dev); imfc=compute_rope_freqs_3d(hp,wp,t_patches=1,device=dev)
    txf=compute_text_rope_freqs(7,device=dev)
    src=img.resize((Wp,Hp),Image.LANCZOS)
    x0=(torch.from_numpy(np.array(src)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    ref=patchify(vae_encode(vae,x0)).to(torch.bfloat16)
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
    col=(np.array([(dx/n+1)/2,(dy/n+1)/2,(dz/n+1)/2])*255).clip(0,255)
    c=(torch.from_numpy(np.full((Hp,Wp,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
    cp=patchify(vae_encode(vae,c)).to(torch.bfloat16)
    torch.manual_seed(0); x=torch.randn(1,16,1,Hp//8,Wp//8,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
    for i,sg in enumerate(S):
        si=sg.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
        res=[q*1.0 for q in cn.forward(patchify(x),cp,te,si,img_freqs=imfc,txt_freqs=txf)]
        v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        x=x-(sg-sn).view(1,1,1,1,1)*unpatchify(v,hp,wp)
    return vae_decode(vae,x)[0]
def padsq(im):
    w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2))
rows=[]
for sp in a.srcs:
    raw=Image.open(sp).convert('RGB'); W,H=raw.size; tag=Path(sp).stem
    for budget,blab in [(2304,'~2.3k'),(4096,'~4.1k')]:
        hp,wp=grid_for(W,H,budget)
        sq=int(round(math.sqrt(budget)))
        for mode,img,(HP,WP) in [('square',padsq(raw),(sq,sq)),('native',raw,(hp,wp))]:
            for (y,p) in CONDS:
                ry,rp=math.radians(y),math.radians(p)
                gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
                try:
                    torch.cuda.reset_peak_memory_stats(); t0=time.time()
                    im=gen(img,HP,WP,gz); dt=time.time()-t0
                    arr=(((im.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())
                    Image.fromarray(arr).save(f'{a.outdir}/{tag}__{blab}_{mode}_{HP*16}x{WP*16}__y{y}_p{p}.png')
                    my=meas(arr); err=None if my is None else abs(w180(my-y))
                    pk=torch.cuda.max_memory_allocated()/2**30
                    rows.append(dict(src=tag,budget=blab,mode=mode,H=HP*16,W=WP*16,tokens=HP*WP,
                                     req_yaw=y,meas_yaw=my,yaw_err=err,peak=pk,sec=dt))
                    print(f'{tag:8s} {blab:6s} {mode:6s} {HP*16:4d}x{WP*16:<4d} tok={HP*WP:5d} y{y:+4d}  '
                          f'err={"n/a" if err is None else f"{err:5.1f}"}  {pk:5.2f}GiB {dt:5.1f}s',flush=True)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); print(f'{tag} {blab} {mode} y{y} OOM',flush=True)
d=pd.DataFrame(rows); d.to_csv(f'{a.outdir}/probe_ar.csv',index=False)
print('\n=== square vs native, by token budget ===')
print(d.dropna(subset=['yaw_err']).groupby(['budget','mode']).agg(
    med_err=('yaw_err','median'), mean_err=('yaw_err','mean'),
    tokens=('tokens','mean'), peak=('peak','mean'), sec=('sec','mean'), n=('yaw_err','size')).round(2).to_string())
print('AR_PROBE_DONE')

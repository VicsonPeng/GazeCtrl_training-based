"""TRUE eval v2. Same generation / metric definitions as eval_true.py, with three changes:
  1) --n_ids 0 (default) = evaluate EVERY identity in the csv. The old script silently
     used list(groupby)[:n_ids], i.e. the FIRST n ids of the TRAINING csv.
  2) adds id_sim  = ArcFace(buffalo_l) cosine between the generated image and the SOURCE
     image -> the identity-preservation number the old metrics could not express.
     ArcFace fails on profile/back-facing faces, so id_sim is averaged ONLY over
     generations where a face was found, and face_det (detection rate) is reported too.
  3) cn_scale defaults to 1.0 to match training (the old default was 1.5).
Metric notes: gaze_err is a PROXY -- 6DRepNet measures HEADPOSE, which differs from gaze
by construction; that offset is expected. cross_gaze_lpips mixes wanted pose change with
unwanted appearance drift and must not be read as an identity metric on its own."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, sys, argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import torch, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0,SIXDREP_DIR)
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode, vae_decode,
                         patchify, unpatchify, compute_rope_freqs_3d, compute_text_rope_freqs)
import cv2
from model import SixDRepNet2; from torchvision.models.resnet import Bottleneck; import utils as u6d
from ultralytics import YOLO
import lpips
from insightface.app import FaceAnalysis
ROOT=Path(ROOT); M=ROOT/'gaze_controlnet/models'
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VAE=M/'vae/split_files/vae/qwen_image_vae.safetensors'; DATA=ROOT/'gaze_dataset'
GRID=32; STEPS=16
def sq(p,s):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))
def limg(p,dev):
    return (torch.from_numpy(np.array(sq(p,512))).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def rgb_cond(gz,dev):
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0; dx,dy,dz=dx/n,dy/n,dz/n
    col=(np.array([(dx+1)/2,(dy+1)/2,(dz+1)/2])*255).clip(0,255)
    rgb=np.full((512,512,3),col,dtype=np.uint8)
    return (torch.from_numpy(rgb).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
def to_bgr(img):
    return cv2.cvtColor((((img.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()),cv2.COLOR_RGB2BGR)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--csv',required=True)
    ap.add_argument('--cn_scale',type=float,default=1.0)
    ap.add_argument('--n_ids',type=int,default=0,help='0 = all identities in the csv')
    ap.add_argument('--tag',default=''); a=ap.parse_args()
    dev=torch.device('cuda:0')
    vae=load_qwen_vae(str(VAE),device=dev); bb=QwenBackbone(str(BB),device=dev)
    cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
    lpnet=lpips.LPIPS(net='alex').to(dev)
    m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
    m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
    ydet=YOLO(YOLO_HEAD)
    fa=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    fa.prepare(ctx_id=0,det_size=(640,640))
    IMEAN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISTD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
    imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev)
    imf_cn=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev); txf=compute_text_rope_freqs(7,device=dev)
    def gen(ref,gz):
        cp=patchify(vae_encode(vae,rgb_cond(gz,dev))).to(torch.bfloat16)
        torch.manual_seed(0); x=torch.randn(1,16,1,64,64,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
        for i,s in enumerate(S):
            si=s.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
            res=cn.forward(patchify(x),cp,te,si,img_freqs=imf_cn,txt_freqs=txf); res=[r*a.cn_scale for r in res]
            v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
            x=x-(s-sn).view(1,1,1,1,1)*unpatchify(v,GRID,GRID)
        return vae_decode(vae,x)[0]
    def head_dir(img):
        bgr=to_bgr(img); r=ydet(bgr,verbose=False)[0].boxes
        if r is None or len(r)==0: return None
        xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); x1,y1,x2,y2=xy[k]
        pad=int(0.35*max(x2-x1,y2-y1)); H,W=bgr.shape[:2]
        c=bgr[max(0,int(y1-pad)):min(H,int(y2+pad)),max(0,int(x1-pad)):min(W,int(x2+pad))]
        if c.size==0: return None
        t=torch.from_numpy(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)/255.0
        t=F.interpolate(t,size=(224,224),mode='bilinear',align_corners=False); t=(t-IMEAN)/ISTD
        R=m6(t.float()); eul=u6d.compute_euler_angles_from_rotation_matrices(R)[0].cpu().numpy()
        pit,yaw=float(eul[0]),float(eul[1])
        dx=-math.sin(yaw)*math.cos(pit); dy=math.sin(pit); dz=math.cos(yaw)*math.cos(pit)
        if dz<0: dx=-dx
        return np.array([dx,dy,dz])
    def face_emb(bgr):
        fs=fa.get(bgr)
        if not fs: return None
        f=max(fs,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return f.normed_embedding
    def ang(u,v):
        u=u/(np.linalg.norm(u)+1e-9); v=np.array(v,float)/(np.linalg.norm(v)+1e-9); return math.degrees(math.acos(np.clip(np.dot(u,v),-1,1)))
    df=pd.read_csv(a.csv); divs=[]; errs=[]; slopes=[]; sims=[]; ndet=0; ntot=0; nosrc=0
    groups=list(df.groupby('sample'))
    if a.n_ids>0: groups=groups[:a.n_ids]
    for sid,g in groups:
        g=g.sort_values('dz',ascending=False); idx=np.linspace(0,len(g)-1,5).astype(int); picks=g.iloc[idx]
        spath=str(DATA/g.iloc[0]['image'])
        ref=patchify(vae_encode(vae,limg(spath,dev))).to(torch.bfloat16)
        src_emb=face_emb(cv2.cvtColor(np.array(sq(spath,512)),cv2.COLOR_RGB2BGR))
        if src_emb is None: nosrc+=1
        imgs=[gen(ref,(float(r.dx),float(r.dy),float(r.dz))) for _,r in picks.iterrows()]
        dd=[lpnet(imgs[i].unsqueeze(0).float(),imgs[j].unsqueeze(0).float()).item() for i in range(5) for j in range(i+1,5)]
        divs.append(float(np.mean(dd)))
        req=[]; meas=[]; isim=[]
        for im,(_,r) in zip(imgs,picks.iterrows()):
            ntot+=1
            if src_emb is not None:
                e=face_emb(to_bgr(im))
                if e is not None:
                    ndet+=1; s=float(np.dot(e,src_emb)); isim.append(s); sims.append(s)
            hd=head_dir(im)
            if hd is None: continue
            gz=[float(r.dx),float(r.dy),float(r.dz)]; errs.append(ang(hd,gz))
            req.append(math.degrees(math.atan2(gz[0],gz[2]))); meas.append(math.degrees(math.atan2(float(hd[0]),float(hd[2]))))
        if len(req)>=3: slopes.append(float(np.polyfit(req,meas,1)[0]))
        sm=f"{np.mean(isim):+.3f}" if isim else " n/a "
        print(f"  {sid[:46]:46s} err={np.mean(errs[-5:]):5.1f} slope={slopes[-1] if slopes else float('nan'):+.2f} id_sim={sm} ({len(isim)}/5 faces)",flush=True)
    print(f"\n[TRUE EVAL v2 {Path(a.ckpt).stem}{(' '+a.tag) if a.tag else ''}]  csv={Path(a.csv).name}  cn_scale={a.cn_scale}  n_ids={len(groups)}")
    print(f"  gaze_err          = {np.mean(errs):.2f} deg      (headpose proxy, lower better)")
    print(f"  ctrl_slope        = {np.mean(slopes):+.4f}        (1.0 = ideal control response)")
    print(f"  cross_gaze_lpips  = {np.mean(divs):.4f}        (output spread across gazes; NOT an identity metric)")
    print(f"  id_sim (ArcFace)  = {np.mean(sims):+.4f}        (vs source, higher better; over {len(sims)} detected faces)")
    print(f"  face_det          = {ndet}/{ntot} = {ndet/max(1,ntot)*100:.1f}%   ({nosrc} identities had no face in the SOURCE image)")
if __name__=='__main__': main()

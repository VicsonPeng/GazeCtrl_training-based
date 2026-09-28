"""Dense gaze sweep for visual inspection: 12 yaw x 5 pitch per identity.

Unlike eval_true_v3.py this asks for a REGULAR GRID of directions rather than the 5 labelled
targets an identity happens to have, so the whole sphere of requests is covered uniformly --
including directions no frame of that identity was ever labelled with. Every output is also
measured (YOLO head -> 6DRepNet) and scored for identity (ArcFace) so the grid can be read
as evidence, not just looked at.
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
import os, sys, argparse, math, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import torch, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, SIXDREP_DIR)
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode, vae_decode,
                         patchify, unpatchify, compute_rope_freqs_3d, compute_text_rope_freqs)
import cv2
from model import SixDRepNet2; from torchvision.models.resnet import Bottleneck; import utils as u6d
from ultralytics import YOLO
from insightface.app import FaceAnalysis
ROOT=Path(ROOT); M=ROOT/'gaze_controlnet/models'
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VAE=M/'vae/split_files/vae/qwen_image_vae.safetensors'
GRID=32; STEPS=16
YAWS=[-180,-135,-90,-65,-40,-15,0,15,40,65,90,135]
PITCHES=[60,30,0,-30,-60]

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
def save_img(img,path,size=256):
    Image.fromarray((((img.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())).resize((size,size),Image.LANCZOS).save(path,quality=88)
def wrap180(a): return (a+180.0)%360.0-180.0

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True); ap.add_argument('--srcs',required=True)
    ap.add_argument('--outdir',required=True); ap.add_argument('--cn_scale',type=float,default=1.0)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshard',type=int,default=1)
    a=ap.parse_args()
    os.makedirs(os.path.join(a.outdir,'img'),exist_ok=True)
    dev=torch.device('cuda:0')
    vae=load_qwen_vae(str(VAE),device=dev); bb=QwenBackbone(str(BB),device=dev)
    cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
    m6=SixDRepNet2(Bottleneck,[3,4,6,3]); sd=torch.load(SIXDREP_CKPT,map_location='cpu',weights_only=False)
    m6.load_state_dict(sd.get('model_state_dict',sd),strict=False); m6.to(dev).eval()
    ydet=YOLO(YOLO_HEAD)
    fa=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider']); fa.prepare(ctx_id=0,det_size=(640,640))
    IMEAN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); ISTD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
    imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev)
    imf_cn=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev); txf=compute_text_rope_freqs(7,device=dev)

    def gen(ref,gz):
        cp=patchify(vae_encode(vae,rgb_cond(gz,dev))).to(torch.bfloat16)
        torch.manual_seed(a.seed); x=torch.randn(1,16,1,64,64,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
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
        eul=u6d.compute_euler_angles_from_rotation_matrices(m6(t.float()))[0].cpu().numpy()
        pit,yaw=float(eul[0]),float(eul[1])
        dx=-math.sin(yaw)*math.cos(pit); dy=math.sin(pit); dz=math.cos(yaw)*math.cos(pit)
        if dz<0: dx=-dx
        return np.array([dx,dy,dz])
    def emb(bgr):
        fs=fa.get(bgr)
        if not fs: return None
        return max(fs,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding

    srcs=json.load(open(a.srcs))
    srcs=[s for i,s in enumerate(srcs) if i%a.nshard==a.shard]
    rows=[]
    for s in srcs:
        sid=s['sample']; sq(s['path'],256).save(os.path.join(a.outdir,'img',f'{sid}__src.jpg'),quality=90)
        ref=patchify(vae_encode(vae,limg(s['path'],dev))).to(torch.bfloat16)
        se=emb(cv2.cvtColor(np.array(sq(s['path'],512)),cv2.COLOR_RGB2BGR))
        print(f'[{sid}] src_face={"yes" if se is not None else "NO"}',flush=True)
        for pi,pit in enumerate(PITCHES):
            for yi,yaw in enumerate(YAWS):
                ry,rp=math.radians(yaw),math.radians(pit)
                gz=(math.sin(ry)*math.cos(rp), math.sin(rp), math.cos(ry)*math.cos(rp))
                im=gen(ref,gz)
                save_img(im,os.path.join(a.outdir,'img',f'{sid}__y{yi}_p{pi}.jpg'))
                hd=head_dir(im); e=emb(to_bgr(im))
                my=mp=ge=float('nan')
                if hd is not None:
                    my=math.degrees(math.atan2(float(hd[0]),float(hd[2])))
                    mp=math.degrees(math.asin(float(np.clip(hd[1],-1,1))))
                    ge=math.degrees(math.acos(float(np.clip(np.dot(hd/np.linalg.norm(hd),np.array(gz)/np.linalg.norm(gz)),-1,1))))
                rows.append(dict(tag=s['tag'],sample=sid,yi=yi,pi=pi,req_yaw=yaw,req_pitch=pit,
                                 dx=round(gz[0],4),dy=round(gz[1],4),dz=round(gz[2],4),
                                 meas_yaw=my,meas_pitch=mp,gaze_err=ge,
                                 yaw_err=(float('nan') if hd is None else abs(wrap180(my-yaw))),
                                 id_sim=(float('nan') if (e is None or se is None) else float(np.dot(e,se))),
                                 det_head=int(hd is not None),det_face=int(e is not None)))
            print(f'   pitch {pit:+4d} done ({len(rows)} cells)',flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.outdir,f'sweep_shard{a.shard}.csv'),index=False)
    print('DONE',len(rows),'cells')
if __name__=='__main__': main()

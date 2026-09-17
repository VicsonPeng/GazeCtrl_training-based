"""Gaze control via HSV-map + InstantX ControlNet (Option A).
  Condition = whole-image HSV solid color encoding (dx,dy,dz) [gaze_hsv_renderer].
  ControlNet (QwenControlNet, 5 blocks) reads the HSV cond + noisy latent -> 5
  residuals injected into the frozen backbone.  Source image kept as in-context
  reference (identity).  Train ControlNet params; backbone + VAE frozen.
  Loss = head-weighted flow-matching denoising MSE.
  Rationale: a spatial dense ControlNet signal is much harder for the model to
  ignore than the global adaLN gaze vector (which got drowned by the source-ref
  when identities varied).
"""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import argparse, os, math, time, random, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import torch, torch.nn.functional as F
from torch.optim import AdamW
from safetensors.torch import save_file
sys.path.insert(0, SIXDREP_DIR)
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode, vae_decode,
                         patchify, unpatchify, compute_rope_freqs_3d, compute_text_rope_freqs, PATCH_SIZE)
from gaze_hsv_renderer import GazeHSVRenderer

ROOT=Path(ROOT); MODELS=ROOT/'gaze_controlnet/models'
BACKBONE=MODELS/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VAE_PATH=MODELS/'vae/split_files/vae/qwen_image_vae.safetensors'
CN_PATH=MODELS/'controlnet/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors'
DATA=ROOT/'gaze_dataset'
IMSZ=512; GRID=IMSZ//8//PATCH_SIZE  # 32
_REND=GazeHSVRenderer(height=IMSZ, width=IMSZ)
COND_MODE='rgb'   # 'rgb' (per-axis linear, no clamp) or 'hsv' (legacy)

def auraflow_sigma(t,shift=3.0): return shift/(shift+1.0/t-1.0)
def pad_sq(img):
    w,h=img.size; s=max(w,h); return ImageOps.expand(img,((s-w)//2,(s-h)//2,s-w-(s-w)//2,s-h-(s-h)//2),fill=0)
def load_image(path,dev):
    img=pad_sq(Image.open(path).convert('RGB')).resize((IMSZ,IMSZ),Image.BILINEAR)
    return (torch.from_numpy(np.array(img)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def gaze_cond(dx,dy,dz,dev):
    if COND_MODE=='rgb':
        n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0; dx,dy,dz=dx/n,dy/n,dz/n
        col=(np.array([(dx+1)/2,(dy+1)/2,(dz+1)/2])*255).clip(0,255)   # R=dx G=dy B=dz
        rgb=np.full((IMSZ,IMSZ,3),col,dtype=np.uint8)
    else:
        rgb=_REND.render_rgb(float(dx),float(dy),float(dz))
    return (torch.from_numpy(rgb).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def box_to_patchmask(box,dev):
    m=torch.zeros(GRID,GRID,device=dev)
    if box is None: return m.flatten()
    px=IMSZ/GRID
    x1,y1,x2,y2=[max(0,int(box[0]//px)),max(0,int(box[1]//px)),min(GRID,int(box[2]//px)+1),min(GRID,int(box[3]//px)+1)]
    if x2>x1 and y2>y1: m[y1:y2,x1:x2]=1.0
    return m.flatten()

class PairedGaze:
    def __init__(self,csv,dev,fixed_src=False,boxes_path=None,eye_boxes_path=None):
        df=pd.read_csv(csv); self.dev=dev; self.fixed_src=fixed_src
        self.by_id={s:g.to_dict('records') for s,g in df.groupby('sample') if len(g)>=2}
        self.ids=list(self.by_id); self.boxes=json.load(open(boxes_path or str(DATA/'head_boxes.json')))
        self.eye_boxes=json.load(open(eye_boxes_path)) if eye_boxes_path else {}
        self.src_of={s:max(recs,key=lambda r:r['dz']) for s,recs in self.by_id.items()}
        print(f'[PairedGaze] {len(self.ids)} identities | fixed_src={fixed_src}')
    def sample(self):
        sid=random.choice(self.ids); recs=self.by_id[sid]
        if self.fixed_src:
            src=self.src_of[sid]; tgt=random.choice(recs)
        else:
            src,tgt=random.sample(recs,2)
        si=load_image(str(DATA/src['image']),self.dev); ti=load_image(str(DATA/tgt['image']),self.dev)
        gz=(tgt['dx'],tgt['dy'],tgt['dz']); hm=box_to_patchmask(self.boxes.get(tgt['id']),self.dev)
        em=box_to_patchmask(self.eye_boxes.get(tgt['id']),self.dev)
        return si,ti,gz,hm,em

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=str(DATA/'labels_top20_hp.csv'))
    ap.add_argument('--out',default=str(ROOT/'gaze_controlnet/_cn_test'))
    ap.add_argument('--steps',type=int,default=8000); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--head_boost',type=float,default=8.0); ap.add_argument('--gaze_drop',type=float,default=0.0)
    ap.add_argument('--cn_scale',type=float,default=1.0)
    ap.add_argument('--save_every',type=int,default=2000); ap.add_argument('--eval_every',type=int,default=1000)
    ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--smoke',type=int,default=0)
    ap.add_argument('--cond',default='rgb',choices=['rgb','hsv'])
    ap.add_argument('--fixed_src',type=int,default=0)
    ap.add_argument('--no_ref',type=int,default=0)
    ap.add_argument('--resume',default=''); ap.add_argument('--start_step',type=int,default=0)
    ap.add_argument('--boxes',default=''); ap.add_argument('--eye_boxes',default='')
    ap.add_argument('--eye_boost',type=float,default=0.0); args=ap.parse_args()
    global COND_MODE; COND_MODE=args.cond
    os.makedirs(args.out,exist_ok=True); dev=torch.device(f'cuda:{args.gpu}')

    vae=load_qwen_vae(str(VAE_PATH),device=dev); bb=QwenBackbone(str(BACKBONE),device=dev)
    cn=QwenControlNet().to(dev)
    if args.resume:
        from safetensors.torch import load_file as _lf; cn.load_state_dict(_lf(args.resume)); print(f'[resume] loaded {args.resume}')
    else:
        cn.load_pretrained(str(CN_PATH))
    cn.to(torch.bfloat16); cn.train()
    txt_emb=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)  # neutral text (backbone/VAE frozen)
    ds=PairedGaze(args.csv,dev,fixed_src=bool(args.fixed_src),boxes_path=(args.boxes or None),eye_boxes_path=(args.eye_boxes or None))
    imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev)      # backbone: noisy+ref
    imf_cn=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev)   # controlnet: noisy only
    txf=compute_text_rope_freqs(txt_emb.shape[1],device=dev)
    params=[p for p in cn.parameters() if p.requires_grad]
    opt=AdamW(params,lr=args.lr,betas=(0.9,0.999),weight_decay=1e-2)
    print(f'trainable {sum(p.numel() for p in params)/1e6:.1f}M | ControlNet gaze inject | head_boost={args.head_boost}')

    import lpips as _lp; lpnet=_lp.LPIPS(net='alex').to(dev).eval()
    # editing-accuracy metric: 6DRepNet head-pose on GENERATED image vs requested gaze
    import cv2, math as _math
    from model import SixDRepNet2; from torchvision.models.resnet import Bottleneck; import utils as u6d
    from ultralytics import YOLO as _Y
    _SNAP=SIXDREP_CKPT
    m6=SixDRepNet2(Bottleneck,[3,4,6,3]); _sdd=torch.load(_SNAP,map_location='cpu',weights_only=False)
    m6.load_state_dict(_sdd.get('model_state_dict',_sdd),strict=False); m6.to(dev).eval()
    _ydet=_Y(YOLO_HEAD)
    _IMEAN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); _ISTD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    def _head_dir(bgr):
        r=_ydet(bgr,verbose=False)[0].boxes
        if r is None or len(r)==0: return None
        xy=r.xyxy.cpu().numpy(); k=int(np.argmax((xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]))); x1,y1,x2,y2=xy[k]
        pad=int(0.35*max(x2-x1,y2-y1)); H,W=bgr.shape[:2]
        c=bgr[max(0,int(y1-pad)):min(H,int(y2+pad)),max(0,int(x1-pad)):min(W,int(x2+pad))]
        if c.size==0: return None
        t=torch.from_numpy(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)/255.0
        t=F.interpolate(t,size=(224,224),mode='bilinear',align_corners=False); t=(t-_IMEAN)/_ISTD
        R=m6(t.float()); eul=u6d.compute_euler_angles_from_rotation_matrices(R)[0].cpu().numpy()
        pit,yaw=float(eul[0]),float(eul[1])
        # headpose.csv uses hp_dx=sin(yaw)cos(pit), hp_dy=-sin(pit), hp_dz=cos(yaw)cos(pit).
        # This rdx/rdy/rdz already equals (-hp_dx,-hp_dy,hp_dz) = the fixed-label convention
        # for 6DRep frames. Only apply the SAME dz<0 dx-flip the labels use; do NOT re-negate.
        rdx=-_math.sin(yaw)*_math.cos(pit); rdy=_math.sin(pit); rdz=_math.cos(yaw)*_math.cos(pit)
        dx,dy,dz=rdx,rdy,rdz
        if dz<0: dx=-dx
        return np.array([dx,dy,dz])
    def _to_bgr(img): return cv2.cvtColor((((img.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()),cv2.COLOR_RGB2BGR)
    def _ang(a,b):
        a=a/(np.linalg.norm(a)+1e-9); b=np.array(b,dtype=float)/(np.linalg.norm(b)+1e-9)
        return _math.degrees(_math.acos(np.clip(np.dot(a,b),-1,1)))

    @torch.no_grad()
    def gen(ref, gz, steps=16):
        def sched(nn,st=0.999,sh=3.0):
            ts=st/(sh*(1-st)+st+1e-9); tt=torch.linspace(ts,1e-5,nn+1)[:-1]; return sh/(sh+(1.0/tt.clamp(1e-6,1-1e-6)-1.0))
        torch.manual_seed(0); x=torch.randn(1,16,1,64,64,device=dev,dtype=torch.bfloat16); S=sched(steps)
        cl=vae_encode(vae, gaze_cond(*gz,dev)); cp=patchify(cl).to(torch.bfloat16)
        for i,s in enumerate(S):
            si=s.unsqueeze(0).to(dev,x.dtype); sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<steps else torch.zeros(1,device=dev,dtype=x.dtype))
            res=cn.forward(patchify(x),cp,txt_emb,si,img_freqs=imf_cn,txt_freqs=txf); res=[r*args.cn_scale for r in res]
            v=bb.forward(patchify(x),txt_emb,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
            x=x-(s-sn).view(1,1,1,1,1)*unpatchify(v,GRID,GRID)
        return vae_decode(vae,x)[0]

    @torch.no_grad()
    def eval_metric(n_ids=6):
        df=pd.read_csv(args.csv); divs=[]; errs=[]; slopes=[]
        _grp=list(df.groupby('sample'))                    # evenly span sorted ids so coco+widerface both covered
        _sel=[_grp[i] for i in np.linspace(0,len(_grp)-1,min(n_ids,len(_grp))).astype(int)]
        for sid,g in _sel:
            g=g.sort_values('dz',ascending=False); idx=np.linspace(0,len(g)-1,5).astype(int); picks=g.iloc[idx]
            src=str(DATA/g.iloc[0]['image']); ref=patchify(vae_encode(vae,load_image(src,dev))).to(torch.bfloat16)  # frontal src
            imgs=[gen(ref,(float(r.dx),float(r.dy),float(r.dz))) for _,r in picks.iterrows()]
            dd=[lpnet(imgs[i].unsqueeze(0).float(),imgs[j].unsqueeze(0).float()).item() for i in range(5) for j in range(i+1,5)]
            divs.append(float(np.mean(dd)))
            req=[]; meas=[]
            for im,(_,r) in zip(imgs,picks.iterrows()):
                hd=_head_dir(_to_bgr(im))
                if hd is None: continue
                gz=[float(r.dx),float(r.dy),float(r.dz)]; errs.append(_ang(hd,gz))
                req.append(_math.degrees(_math.atan2(gz[0],gz[2]))); meas.append(_math.degrees(_math.atan2(float(hd[0]),float(hd[2]))))
            if len(req)>=3: slopes.append(float(np.polyfit(req,meas,1)[0]))
        return (float(np.mean(divs)) if divs else float('nan'),
                float(np.mean(errs)) if errs else float('nan'),
                float(np.mean(slopes)) if slopes else float('nan'))

    _m='a' if args.start_step>0 else 'w'
    logf=open(os.path.join(args.out,'loss.csv'),_m); 
    mlog=open(os.path.join(args.out,'metrics_train.csv'),_m)
    if args.start_step==0:
        logf.write('step,mse,sigma\n'); mlog.write('step,cross_gaze_lpips,gaze_err,ctrl_slope\n')
    t0=time.time(); rm=0.0
    for step in range(args.start_step+1,args.steps+1):
        si,ti,gz,hm,em=ds.sample()
        if args.gaze_drop>0 and random.random()<args.gaze_drop: gz=(0.0,0.0,0.0)
        with torch.no_grad():
            clean=vae_encode(vae,ti); ref=patchify(vae_encode(vae,si)).to(torch.bfloat16)
            cond_lat=vae_encode(vae, gaze_cond(*gz,dev)); cond_p=patchify(cond_lat).to(torch.bfloat16)
        t=torch.rand(1,device=dev); sigma=auraflow_sigma(t); noise=torch.randn_like(clean)
        noisy=((1-sigma.view(1,1,1,1,1))*clean+sigma.view(1,1,1,1,1)*noise).to(torch.bfloat16)
        target=(noise-clean).to(torch.bfloat16)
        res=cn.forward(patchify(noisy),cond_p,txt_emb,sigma,img_freqs=imf_cn,txt_freqs=txf); res=[r*args.cn_scale for r in res]
        if args.no_ref:
            v=bb.forward(patchify(noisy),txt_emb,sigma,res,img_freqs=imf_cn,txt_freqs=txf,ref_packed=None)
        else:
            v=bb.forward(patchify(noisy),txt_emb,sigma,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
        err=(v.float()-patchify(target).float())**2
        w=(1.0+args.head_boost*hm.view(1,-1,1)+args.eye_boost*em.view(1,-1,1)).float(); mse=(err*w).mean()/w.mean()
        mse.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
        rm+=mse.item(); logf.write(f'{step},{mse.item():.5f},{sigma.item():.3f}\n'); logf.flush()
        if step%50==0: print(f'step {step:5d}/{args.steps} mse={rm/50:.4f} {time.time()-t0:.0f}s',flush=True); rm=0.0
        if step%args.eval_every==0:
            cn.eval(); dv,ge,sl=eval_metric(); cn.train()
            mlog.write(f'{step},{dv:.4f},{ge:.2f},{sl:.4f}\n'); mlog.flush()
            print(f'  [METRIC step {step}] cross_gaze_lpips={dv:.4f}  gaze_err={ge:.1f}deg  ctrl_slope={sl:.3f}',flush=True)
        if step%args.save_every==0 or step==args.steps:
            save_file({k:vv.contiguous().cpu() for k,vv in cn.state_dict().items()}, os.path.join(args.out,f'cn_step{step:06d}.safetensors'))
            print(f'  saved step {step}',flush=True)
    print('done')
if __name__=='__main__': main()

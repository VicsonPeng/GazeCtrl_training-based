"""Stage-1 pretraining: SHHQ full-body Wan-turnaround pairs, with a head-preserving
random crop, mixed with a minority of HITL pairs.

Differs from train_gaze_controlnet.py in exactly three places:
  1. HEAD-PRESERVING RANDOM CROP. A square crop is taken in ORIGINAL image coordinates
     (never in the pad-to-square space, which would drag the letterbox bars in). It must
     contain the head boxes of BOTH src and tgt, its side is never below
     max(crop_k * head size, crop_minfrac * short side) so it cannot collapse onto the
     face, and the SAME rect is applied to src and tgt -- a different framing per side
     would make the pair unlearnable. Side and position are re-drawn every sample;
     crop_p of samples keep the full frame.
  2. THE HEAD AND EYE BOXES ARE RE-MAPPED THROUGH THAT CROP. They are stored in the 512
     pad-to-square space, so head_boost/eye_boost would weight the wrong patches after a
     crop. Boxes are mapped 512-pad -> original -> cropped-512 for every sample.
  3. TWO DATASETS. --csv is sampled with probability 1-mix2 and --csv2 with mix2, each
     with its own head/eye boxes, so stage 1 sees a minority of the stage-2 distribution
     and cannot settle into an SHHQ-only or copy-the-source shortcut.
Evaluation never crops: inference feeds whole pad-to-square frames, so the metric must too.
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
def load_image(path,dev,rect=None):
    """rect=(x,y,s) square in ORIGINAL image coords; None keeps the pad-to-square framing."""
    im=Image.open(path).convert('RGB')
    if rect is None: img=pad_sq(im).resize((IMSZ,IMSZ),Image.BILINEAR)
    else:
        x,y,s=rect; img=im.crop((int(x),int(y),int(x+s),int(y+s))).resize((IMSZ,IMSZ),Image.BILINEAR)
    return (torch.from_numpy(np.array(img)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)

def box_pad512_to_orig(b,W,H):
    m=max(W,H); sc=m/IMSZ; ox=(m-W)//2; oy=(m-H)//2
    return [b[0]*sc-ox, b[1]*sc-oy, b[2]*sc-ox, b[3]*sc-oy]
def box_orig_to_out512(b,W,H,rect):
    if rect is None:
        m=max(W,H); sc=IMSZ/m; ox=(m-W)//2; oy=(m-H)//2
        return [(b[0]+ox)*sc,(b[1]+oy)*sc,(b[2]+ox)*sc,(b[3]+oy)*sc]
    x,y,s=rect; k=IMSZ/s
    return [(b[0]-x)*k,(b[1]-y)*k,(b[2]-x)*k,(b[3]-y)*k]
def remap_box(b,W,H,rect):
    """a box stored in the 512 pad-to-square space -> the 512 space AFTER this crop."""
    if b is None: return None
    return box_orig_to_out512(box_pad512_to_orig(b,W,H),W,H,rect)

def sample_crop_rect(hs,ht,W,H,k,minfrac,margin,rng):
    """square crop in ORIGINAL coords containing both head boxes; None when impossible."""
    if hs is None or ht is None: return None
    x1=min(hs[0],ht[0]); y1=min(hs[1],ht[1]); x2=max(hs[2],ht[2]); y2=max(hs[3],ht[3])
    hmax=max(x2-x1,y2-y1); M=min(W,H)
    lo=max(k*hmax, minfrac*M); hi=float(M)
    if lo>=hi: return None
    s=rng.uniform(lo,hi); m=margin*hmax
    def axis(a1,a2,L):
        # plain containment first: a head touching the frame edge has no room for a
        # hard margin, and demanding one there would silently disable cropping.
        l=max(0.0,a2-s); h=min(L-s,a1)
        if l>h: return None
        l2=max(l,a2+m-s); h2=min(h,a1-m)
        return (l2,h2) if l2<=h2 else (l,h)
    ax=axis(x1,x2,W); ay=axis(y1,y2,H)
    if ax is None or ay is None: return None
    return (rng.uniform(*ax), rng.uniform(*ay), s)
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

class MixedPairedGaze:
    """Two paired-gaze pools sampled by ratio, with the head-preserving crop applied
    identically to src and tgt and every box re-mapped through it."""
    def __init__(self,dev,csv,boxes,eye_boxes,csv2=None,boxes2=None,eye_boxes2=None,mix2=0.0,
                 fixed_src=False,crop_p=0.0,crop_k=2.0,crop_minfrac=0.45,crop_margin=0.15,seed=0):
        self.dev=dev; self.fixed_src=fixed_src; self.mix2=mix2
        self.crop_p=crop_p; self.k=crop_k; self.minfrac=crop_minfrac; self.margin=crop_margin
        self.rng=random.Random(seed); self._size={}
        def load(c,b,e):
            if not c: return None
            df=pd.read_csv(c)
            by={s:g.to_dict('records') for s,g in df.groupby('sample') if len(g)>=2}
            return {"by":by,"ids":list(by),
                    "boxes":(json.load(open(b)) if b else {}),
                    "eye":(json.load(open(e)) if e else {}),
                    "src_of":{s:max(r,key=lambda x:x['dz']) for s,r in by.items()}}
        self.A=load(csv,boxes,eye_boxes); self.B=load(csv2,boxes2,eye_boxes2)
        na=len(self.A['ids']); nb=len(self.B['ids']) if self.B else 0
        print(f'[MixedPairedGaze] A={na} ids | B={nb} ids @ mix2={mix2} | crop_p={crop_p} k={crop_k} '
              f'minfrac={crop_minfrac} | fixed_src={fixed_src}',flush=True)
    def _wh(self,p):
        p=str(p)
        if p not in self._size:
            with Image.open(p) as im: self._size[p]=im.size
        return self._size[p]
    def sample(self):
        P=self.B if (self.B and self.rng.random()<self.mix2) else self.A
        sid=self.rng.choice(P['ids']); recs=P['by'][sid]
        if self.fixed_src: src=P['src_of'][sid]; tgt=self.rng.choice(recs)
        else: src,tgt=self.rng.sample(recs,2)
        sp=DATA/src['image']; tp=DATA/tgt['image']
        Ws,Hs=self._wh(sp); Wt,Ht=self._wh(tp)
        rect=None
        if self.crop_p>0 and (Ws,Hs)==(Wt,Ht) and self.rng.random()<self.crop_p:
            # a rect is only valid on both frames when they share a resolution
            hs=P['boxes'].get(str(src['id'])); ht=P['boxes'].get(str(tgt['id']))
            rect=sample_crop_rect(
                box_pad512_to_orig(hs,Ws,Hs) if hs else None,
                box_pad512_to_orig(ht,Wt,Ht) if ht else None,
                Ws,Hs,self.k,self.minfrac,self.margin,self.rng)
        si=load_image(str(sp),self.dev,rect); ti=load_image(str(tp),self.dev,rect)
        gz=(tgt['dx'],tgt['dy'],tgt['dz'])
        hm=box_to_patchmask(remap_box(P['boxes'].get(str(tgt['id'])),Wt,Ht,rect),self.dev)
        em=box_to_patchmask(remap_box(P['eye'].get(str(tgt['id'])),Wt,Ht,rect),self.dev)
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
    ap.add_argument('--eye_boost',type=float,default=0.0)
    ap.add_argument('--csv2',default='',help='secondary pool (HITL) mixed in at --mix2')
    ap.add_argument('--boxes2',default=''); ap.add_argument('--eye_boxes2',default='')
    ap.add_argument('--mix2',type=float,default=0.2,help='probability a sample comes from --csv2')
    ap.add_argument('--crop_p',type=float,default=0.65,help='probability of cropping (1-crop_p keeps the full frame)')
    ap.add_argument('--crop_k',type=float,default=2.0,help='crop side >= crop_k x head size')
    ap.add_argument('--crop_minfrac',type=float,default=0.45,help='crop side >= this fraction of the short side')
    ap.add_argument('--crop_margin',type=float,default=0.15,help='slack around the head, in head sizes')
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
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
    ds=MixedPairedGaze(dev,args.csv,(args.boxes or str(DATA/'head_boxes.json')),(args.eye_boxes or None),
                       csv2=(args.csv2 or None),boxes2=(args.boxes2 or None),eye_boxes2=(args.eye_boxes2 or None),
                       mix2=args.mix2,fixed_src=bool(args.fixed_src),crop_p=args.crop_p,crop_k=args.crop_k,
                       crop_minfrac=args.crop_minfrac,crop_margin=args.crop_margin,seed=args.seed)
    imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev)      # backbone: noisy+ref
    imf_cn=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev)   # controlnet: noisy only
    txf=compute_text_rope_freqs(txt_emb.shape[1],device=dev)
    params=[p for p in cn.parameters() if p.requires_grad]
    opt=AdamW(params,lr=args.lr,betas=(0.9,0.999),weight_decay=1e-2)
    print(f'trainable {sum(p.numel() for p in params)/1e6:.1f}M | ControlNet gaze inject | head_boost={args.head_boost} | crop_p={args.crop_p} k={args.crop_k}')

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

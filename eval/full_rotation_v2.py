"""Demo v2: full-360 sweep with SIDE / BACK sources (not just frontal) and a GT
reference gallery per identity, so each output can be compared against the real
frame of that same person whose labelled gaze is closest to the condition.
CONVENTION: dx>0 right, dy>0 UP, dz>0 toward camera."""

import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD,
                        SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, WAN_DIR)
import os, sys, math, argparse, json, base64, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors.torch import load_file
sys.path.insert(0,SIXDREP_DIR)
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode, vae_decode,
                         patchify, unpatchify, compute_rope_freqs_3d, compute_text_rope_freqs)
ROOT=Path(ROOT); M=ROOT/'gaze_controlnet/models'
BB=M/'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VAE=M/'vae/split_files/vae/qwen_image_vae.safetensors'
GRID=32; STEPS=16
YAWS=[-180,-135,-90,-45,0,45,90,135]
PITCHES=[25,0,-25]
def sq(p,s):
    im=Image.open(p).convert('RGB'); w,h=im.size; m=max(w,h)
    return ImageOps.expand(im,((m-w)//2,(m-h)//2,m-w-(m-w)//2,m-h-(m-h)//2)).resize((s,s))
def limg(p,dev):
    return (torch.from_numpy(np.array(sq(p,512))).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def gz_of(yaw,pit):
    y=math.radians(yaw); p=math.radians(pit)
    return (math.sin(y)*math.cos(p), math.sin(p), math.cos(y)*math.cos(p))
def rgb_cond(gz,dev):
    dx,dy,dz=gz; n=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0; dx,dy,dz=dx/n,dy/n,dz/n
    col=(np.array([(dx+1)/2,(dy+1)/2,(dz+1)/2])*255).clip(0,255)
    return (torch.from_numpy(np.full((512,512,3),col,dtype=np.uint8)).float().permute(2,0,1)/127.5-1).unsqueeze(0).to(dev,torch.bfloat16)
def sched(n,st=0.999,sh=3.0):
    ts=st/(sh*(1-st)+st+1e-9); t=torch.linspace(ts,1e-5,n+1)[:-1]; return sh/(sh+(1.0/t.clamp(1e-6,1-1e-6)-1.0))
def b64(pil,q=85):
    b=io.BytesIO(); pil.save(b,format='JPEG',quality=q); return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--srcs',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--cn_scale',type=float,default=1.0)
    ap.add_argument('--cell',type=int,default=224); a=ap.parse_args()
    dev=torch.device('cuda:0')
    vae=load_qwen_vae(str(VAE),device=dev); bb=QwenBackbone(str(BB),device=dev)
    cn=QwenControlNet().to(dev); cn.load_state_dict(load_file(a.ckpt)); cn.to(torch.bfloat16).eval()
    te=torch.zeros((1,7,3584),device=dev,dtype=torch.bfloat16)
    imf=compute_rope_freqs_3d(GRID,GRID,t_patches=2,device=dev)
    imf_cn=compute_rope_freqs_3d(GRID,GRID,t_patches=1,device=dev); txf=compute_text_rope_freqs(7,device=dev)
    def gen(ref,gz):
        cp=patchify(vae_encode(vae,rgb_cond(gz,dev))).to(torch.bfloat16)
        torch.manual_seed(0); x=torch.randn(1,16,1,64,64,device=dev,dtype=torch.bfloat16); S=sched(STEPS)
        for i,s in enumerate(S):
            si=s.unsqueeze(0).to(dev,x.dtype)
            sn=(S[i+1].unsqueeze(0).to(dev,x.dtype) if i+1<STEPS else torch.zeros(1,device=dev,dtype=x.dtype))
            res=cn.forward(patchify(x),cp,te,si,img_freqs=imf_cn,txt_freqs=txf); res=[r*a.cn_scale for r in res]
            v=bb.forward(patchify(x),te,si,res,img_freqs=imf,txt_freqs=txf,ref_packed=ref)
            x=x-(s-sn).view(1,1,1,1,1)*unpatchify(v,GRID,GRID)
        img=vae_decode(vae,x)[0]
        return Image.fromarray(((img.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy())

    J=json.load(open(a.srcs)); srcs=J['srcs']; gal=J['galleries']
    data={"yaws":YAWS,"pitches":PITCHES,"cell":a.cell,"ckpt":Path(a.ckpt).stem,
          "cn_scale":a.cn_scale,"steps":STEPS,"seed":0,"sources":[],"galleries":{}}
    # GT galleries (encoded once per identity, shared by its front/side/back sources)
    for ident,frames in gal.items():
        out=[]
        for f in frames:
            if not os.path.exists(f['img']): continue
            out.append({"gz":f['gz'],"lab":f['lab'],"fid":f['fid'],"img":b64(sq(f['img'],a.cell))})
        data["galleries"][ident]=out
        print(f"gallery {ident}: {len(out)} GT frames",flush=True)
    for s in srcs:
        if not os.path.exists(s['path']): print("MISS",s['path'],flush=True); continue
        ref=patchify(vae_encode(vae,limg(s['path'],dev))).to(torch.bfloat16)
        outs={}
        for p in PITCHES:
            for y in YAWS:
                outs[f"{y}_{p}"]=b64(gen(ref,gz_of(y,p)).resize((a.cell,a.cell)))
        data["sources"].append({"id":s['id'],"ident":s['ident'],"dom":s['dom'],"tag":s['tag'],
                                "pose":s['pose'],"gt":s['gt'],"lab":s['lab'],
                                "src":b64(sq(s['path'],a.cell)),"outs":outs})
        print(f"done [{s['tag']}/{s['pose']}] {s['id']}",flush=True)
    json.dump(data,open(a.out,'w'),separators=(',',':'))
    print("SAVED",a.out,os.path.getsize(a.out),flush=True)
if __name__=='__main__': main()

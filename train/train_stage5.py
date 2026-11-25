"""Stage-5 post-training: edit the person the mask points at, and leave everyone else alone.

Everything before this stage assumed one person in the frame, so "who should move" was never
a question the model had to answer. The mask channel added to QwenControlNet asks it, and this
run is what teaches it to read the answer. Three things are deliberately NOT changed against
stage 2, so that whatever happens is attributable to the mask and the corpus:

  * the gaze condition is still the flat RGB map, on its own latent path. The mask is a
    SEPARATE input summed into the ControlNet's image embedding through a zero-initialised
    linear layer, never mixed into the gaze condition. Space says WHERE, colour says WHICH WAY.
  * head_boost, eye_boost and the perceptual term all default to OFF here. They are available
    but unused, because stage 3 established that stacking changes makes the result unreadable.
  * the ControlNet alone is trained; backbone and VAE stay frozen in fp8/bf16 as before.

Because mask_embedder is zero-initialised and load_state_dict zero-fills it for any older
checkpoint, step 0 of this run reproduces the stage-2 model exactly, mask or no mask. Any
change in behaviour is therefore bought by this training and nothing else.

The corpus is read from disk (make_multiperson.py), not assembled here, so that the whole of
it can be -- and has been -- checked before a single gradient step:
  multi    : two people, one masked. GT moves the masked one and leaves the other byte-identical.
  single   : the one-person pairs the model already handles, so that ability cannot decay.
  identity : someone is masked and asked for the gaze they already have. The correct answer is
             to change nothing; without these the model can learn that a mask always means motion.
No crop is taken. The composites are built in the 512 pad-to-square space and a crop would cut
B or the mask out of a frame whose whole point is where the two people are.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, json, os, time, random, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn.functional as F
from torch.optim import AdamW
from safetensors.torch import save_file, load_file
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode,
                         patchify, pack_mask, compute_rope_freqs_3d, compute_text_rope_freqs,
                         PATCH_SIZE)

ROOT = Path(ROOT); MODELS = ROOT / 'gaze_controlnet/models'
BACKBONE = MODELS / 'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VAE_PATH = MODELS / 'vae/split_files/vae/qwen_image_vae.safetensors'
CN_PATH = MODELS / 'controlnet/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors'
IMSZ = 512; LAT = IMSZ // 8; GRID = LAT // PATCH_SIZE          # 64 latent cells, 32 token grid


def auraflow_sigma(t, shift=3.0): return shift / (shift + 1.0 / t - 1.0)


def load_rgb(path, dev):
    a = np.array(Image.open(path).convert('RGB'))
    return (torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(dev, torch.bfloat16)


def load_gray(path, dev):
    a = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    return torch.from_numpy(a).view(1, 1, IMSZ, IMSZ).to(dev)


def gaze_colour(dx, dy, dz):
    n = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
    return (np.array([(dx / n + 1) / 2, (dy / n + 1) / 2, (dz / n + 1) / 2]) * 255).clip(0, 255)


def gaze_cond(dx, dy, dz, dev, box=None, pad=0.0):
    """The gaze map.

    Without a box: the flat fill stage 2 was trained on -- a direction about the PICTURE. In a
    multi-person frame that is exactly what the model does with it: it turns everyone. Measured
    on 166 eval pairs, the second person ends up 42.7 deg (newlabel 40k) to 57.6 deg (stage2
    40k) away from the gaze they started with, where the real target frame moves them 0.6 deg.

    With a box: the colour fills the target's bounding box and the rest of the frame holds the
    neutral value (128,128,128), the midpoint every axis maps zero to, so "no instruction here"
    is expressible. A box, not the mask itself: the mask's outline is the person's current pose,
    which the edit is supposed to change, so painting inside it would hand the model its own
    answer to copy. The box says where and how big, and nothing about which way they face.
    """
    col = gaze_colour(dx, dy, dz)
    if box is None:
        rgb = np.full((IMSZ, IMSZ, 3), col, dtype=np.uint8)
    else:
        rgb = np.full((IMSZ, IMSZ, 3), 128, dtype=np.uint8)
        x0, y0, x1, y1 = box
        if pad:
            w, h = x1 - x0, y1 - y0
            x0 -= pad * w; x1 += pad * w; y0 -= pad * h; y1 += pad * h
        x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
        x1 = min(IMSZ, int(round(x1))); y1 = min(IMSZ, int(round(y1)))
        if x1 > x0 and y1 > y0:
            rgb[y0:y1, x0:x1] = col
    return (torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(dev, torch.bfloat16)


def to_tokens(m):
    """A 512 pixel map -> one weight per token of the 32x32 grid, by area."""
    return F.adaptive_avg_pool2d(m, (GRID, GRID)).view(1, -1, 1)


class Corpus:
    """The on-disk corpus, split into train and a held-out tail.

    The split is by position in index.csv, which is the order the examples were generated, and
    the generator draws identities independently per example -- so the tail is held out by
    EXAMPLE, not by identity, and the validation numbers say "can it do this on pairs it has
    not seen", not "can it generalise to new people". The second question is the eval set's job.
    """
    def __init__(self, root, dev, val, seed=0, boxes='boxes.json'):
        self.root = Path(root); self.dev = dev
        df = pd.read_csv(self.root / 'index.csv')
        bp = Path(boxes) if os.path.isabs(boxes) else self.root / boxes
        self.boxes = json.load(open(bp)) if bp.exists() else None
        print(f'[corpus] boxes: {len(self.boxes) if self.boxes else 0} from {bp}'
              f'{"" if self.boxes else "  (MISSING -- the box falls back to the mask bbox)"}',
              flush=True)
        self.val = df.iloc[len(df) - val:].to_dict('records') if val else []
        self.train = df.iloc[:len(df) - val].to_dict('records')
        self.rng = random.Random(seed)
        k = pd.DataFrame(self.train)['kind'].value_counts().to_dict()
        print(f'[corpus] {len(self.train)} train + {len(self.val)} held out | {k}', flush=True)

    def get(self, r):
        n = r['name']
        si = load_rgb(self.root / 'input' / f'{n}.png', self.dev)
        ti = load_rgb(self.root / 'gt' / f'{n}.png', self.dev)
        mk = load_gray(self.root / 'mask' / f'{n}.png', self.dev)
        bm = load_gray(self.root / 'bmask' / f'{n}.png', self.dev)
        gz = (float(r['dx']), float(r['dy']), float(r['dz']))
        bx = self.boxes.get(n, {}).get('a') if self.boxes else None
        return si, ti, gz, mk, bm, r['kind'], bx

    def sample(self): return self.get(self.rng.choice(self.train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'datasets/mp_v5'))
    ap.add_argument('--out', default=str(ROOT / 'gaze_controlnet/_cn_stage5'))
    ap.add_argument('--init', default='', help='stage-2 ControlNet to start from; mask_embedder is zero-filled')
    ap.add_argument('--resume', default=''); ap.add_argument('--start_step', type=int, default=0)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--accum', type=int, default=8, help='gradient accumulation; the effective batch')
    ap.add_argument('--cn_scale', type=float, default=1.0)
    ap.add_argument('--cond_box', type=int, default=0,
                    help='1 = paint the gaze colour only inside the target box, neutral grey '
                         'elsewhere. 0 = the flat whole-frame fill stages 1-5 used.')
    ap.add_argument('--boxes', default='boxes.json',
                    help="which box file to read, relative to --data or absolute. Use "
                         "boxes_headrule.json for the head-box rule; those boxes already "
                         "have the 5%% pad baked in, so pair them with --box_pad 0")
    ap.add_argument('--box_pad', type=float, default=0.0,
                    help='grow the box by this fraction of its own width/height on each side')
    ap.add_argument('--b_boost', type=float, default=0.0,
                    help='extra loss weight on person B. 0 keeps plain MSE, which already '
                         'supervises B over ~8%% of the canvas; raise it only if B turns out '
                         'to move, and say so when reporting, because it is a second change')
    ap.add_argument('--head_boost', type=float, default=0.0)
    ap.add_argument('--mask_drop', type=float, default=0.1,
                    help='probability of passing no mask at all, so the one-person behaviour '
                         'stays reachable when nothing is marked')
    ap.add_argument('--spatial_cond', type=int, default=0,
                    help='ARM A: paint the gaze colour only inside the mask, neutral (zero '
                         'vector = mid grey) elsewhere, so WHERE and WHICH-WAY ride the already '
                         'trained controlnet_x_embedder path instead of mask_embedder. '
                         'Falls back to the flat map whenever no mask is passed.')
    ap.add_argument('--mask_lr', type=float, default=0.0,
                    help='ARM B: separate LR for mask_embedder.* (0 = same as --lr). The 12k '
                         'stage-5 run left it at absmean 0.0023, 1/60 of img_in, i.e. inert.')
    ap.add_argument('--val', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', type=int, default=0, help='run this many steps and stop')
    a = ap.parse_args()
    if a.smoke: a.steps = a.smoke; a.eval_every = max(1, a.smoke // 2); a.save_every = a.smoke
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f'cuda:{a.gpu}')

    vae = load_qwen_vae(str(VAE_PATH), device=dev)
    bb = QwenBackbone(str(BACKBONE), device=dev)
    cn = QwenControlNet().to(dev)
    if a.resume:
        cn.load_state_dict(load_file(a.resume)); print(f'[resume] {a.resume}', flush=True)
    elif a.init:
        # the override in QwenControlNet.load_state_dict zero-fills mask_embedder.*, so a
        # checkpoint from before the mask channel loads and behaves exactly as it did
        cn.load_state_dict(load_file(a.init)); print(f'[init] {a.init} (mask_embedder zero-filled)', flush=True)
    else:
        cn.load_pretrained(str(CN_PATH)); print('[init] stock InstantX ControlNet', flush=True)
    cn.to(torch.bfloat16); cn.train()

    txt_emb = torch.zeros((1, 7, 3584), device=dev, dtype=torch.bfloat16)   # neutral text, as in stage 2
    imf = compute_rope_freqs_3d(GRID, GRID, t_patches=2, device=dev)        # backbone: noisy + ref
    imf_cn = compute_rope_freqs_3d(GRID, GRID, t_patches=1, device=dev)     # controlnet: noisy only
    txf = compute_text_rope_freqs(txt_emb.shape[1], device=dev)

    ds = Corpus(a.data, dev, a.val, seed=a.seed, boxes=a.boxes)
    params = [p for p in cn.parameters() if p.requires_grad]
    if a.mask_lr > 0:
        me = [p for n, p in cn.named_parameters() if p.requires_grad and n.startswith('mask_embedder')]
        rest = [p for n, p in cn.named_parameters() if p.requires_grad and not n.startswith('mask_embedder')]
        # no weight decay on the mask embedder: it starts at zero, and decay only fights the growth
        opt = AdamW([{'params': rest, 'lr': a.lr, 'weight_decay': 1e-2},
                     {'params': me, 'lr': a.mask_lr, 'weight_decay': 0.0}],
                    lr=a.lr, betas=(0.9, 0.999))
        print(f'[arm B] mask_embedder on its own group, lr={a.mask_lr} wd=0', flush=True)
    else:
        opt = AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    if a.spatial_cond:
        print('[arm A] gaze condition painted inside the mask only', flush=True)

    def mask_norm():
        w = dict(cn.named_parameters())['mask_embedder.weight']
        return float(w.detach().float().abs().mean())
    nme = sum(p.numel() for n, p in cn.named_parameters() if n.startswith('mask_embedder'))
    print(f'trainable {sum(p.numel() for p in params) / 1e6:.1f}M, of which mask_embedder {nme / 1e3:.1f}K | '
          f'accum={a.accum} lr={a.lr} b_boost={a.b_boost} mask_drop={a.mask_drop}', flush=True)

    def one(si, ti, gz, mk, bm, use_mask, sigma=None, box=None):
        """Velocity loss on one example, plus the three region errors worth reading separately."""
        with torch.no_grad():
            clean = vae_encode(vae, ti)
            ref = patchify(vae_encode(vae, si)).to(torch.bfloat16)
            cond_img = gaze_cond(*gz, dev, box=(box if a.cond_box else None), pad=a.box_pad)
            if a.spatial_cond and use_mask:
                # zero is the neutral colour here: (dx,dy,dz)=(0,0,0) maps to mid grey, which
                # normalises to exactly 0, so multiplying by the mask leaves "no direction asked"
                # outside it rather than some arbitrary hue.
                cond_img = cond_img * mk.to(cond_img.dtype)
            cond_p = patchify(vae_encode(vae, cond_img)).to(torch.bfloat16)
        if sigma is None:
            sigma = auraflow_sigma(torch.rand(1, device=dev))
        noise = torch.randn_like(clean)
        s5 = sigma.view(1, 1, 1, 1, 1)
        noisy = ((1 - s5) * clean + s5 * noise).to(torch.bfloat16)
        target = patchify((noise - clean).to(torch.bfloat16)).float()
        mp = pack_mask(mk, LAT, LAT).to(torch.bfloat16) if use_mask else None
        res = cn.forward(patchify(noisy), cond_p, txt_emb, sigma,
                         img_freqs=imf_cn, txt_freqs=txf, mask_packed=mp)
        res = [r * a.cn_scale for r in res]
        v = bb.forward(patchify(noisy), txt_emb, sigma, res, img_freqs=imf, txt_freqs=txf, ref_packed=ref)
        err = (v.float() - target) ** 2
        tb, tm = to_tokens(bm), to_tokens(mk)
        w = 1.0 + a.b_boost * tb + a.head_boost * tm
        loss = (err * w).mean() / w.mean()
        with torch.no_grad():
            e = err.mean(-1, keepdim=True)
            rb = float((e * tb).sum() / tb.sum().clamp_min(1e-6))          # on person B: must stay put
            rm = float((e * tm).sum() / tm.sum().clamp_min(1e-6))          # on person A: must move
            out = 1.0 - torch.clamp(tb + tm, 0, 1)
            rk = float((e * out).sum() / out.sum().clamp_min(1e-6))        # background
        return loss, rb, rm, rk

    @torch.no_grad()
    def validate():
        """Held-out error per region, at FIXED sigmas so the number is comparable step to step.

        Noise is redrawn each call but the seed is pinned, so two evals differ because the
        model differs. Reported per kind as well: a single average would hide the one failure
        that matters, a model that does well on the composites by moving everybody.
        """
        cn.eval(); g = torch.Generator(device='cpu').manual_seed(1234)
        acc = {}
        for r in ds.val:
            si, ti, gz, mk, bm, kind, bx = ds.get(r)
            for sg in (0.25, 0.55, 0.85):
                sd = int(torch.randint(0, 2**31 - 1, (1,), generator=g))
                torch.manual_seed(sd)
                sigma = torch.tensor([sg], device=dev)
                l, rb, rm, rk = one(si, ti, gz, mk, bm, True, sigma=sigma, box=bx)
                for key, val in (('loss', float(l)), ('B', rb), ('A', rm), ('bg', rk)):
                    acc.setdefault((kind, key), []).append(val)
                if kind == 'multi':
                    # the decisive number: the SAME example, same noise, with the mask withheld.
                    # If the mask is doing any work, errB must rise when it is taken away.
                    torch.manual_seed(sd)
                    l2, rb2, rm2, rk2 = one(si, ti, gz, mk, bm, False, sigma=sigma, box=bx)
                    for key, val in (('loss', float(l2)), ('B', rb2), ('A', rm2), ('bg', rk2)):
                        acc.setdefault(('multi_nomask', key), []).append(val)
        cn.train()
        return {k: float(np.mean(v)) for k, v in acc.items()}

    m = 'a' if a.start_step > 0 else 'w'
    logf = open(os.path.join(a.out, 'loss.csv'), m)
    vlog = open(os.path.join(a.out, 'val.csv'), m)
    if a.start_step == 0:
        logf.write('step,loss,errB,errA,errbg\n')
        vlog.write('step,kind,loss,errB,errA,errbg\n')
    t0 = time.time(); run = np.zeros(4)
    for step in range(a.start_step + 1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = np.zeros(4)
        for _ in range(a.accum):
            si, ti, gz, mk, bm, kind, bx = ds.sample()
            use_mask = random.random() >= a.mask_drop
            loss, rb, rm, rk = one(si, ti, gz, mk, bm, use_mask, box=bx)
            (loss / a.accum).backward()
            tot += np.array([float(loss), rb, rm, rk])
        tot /= a.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run += tot
        logf.write('%d,%.5f,%.5f,%.5f,%.5f\n' % (step, *tot)); logf.flush()
        if step % 50 == 0:
            r = run / 50
            print(f'step {step:5d}/{a.steps} loss={r[0]:.4f} B={r[1]:.4f} A={r[2]:.4f} bg={r[3]:.4f} '
                  f'{time.time() - t0:.0f}s', flush=True)
            run = np.zeros(4)
        if step % a.eval_every == 0:
            v = validate()
            for kind in sorted({k for k, _ in v}):
                vlog.write('%d,%s,%.5f,%.5f,%.5f,%.5f\n' % (
                    step, kind, v[(kind, 'loss')], v[(kind, 'B')], v[(kind, 'A')], v[(kind, 'bg')]))
                print(f'  [val {step}] {kind:9s} loss={v[(kind,"loss")]:.4f} '
                      f'B={v[(kind,"B")]:.4f} A={v[(kind,"A")]:.4f} bg={v[(kind,"bg")]:.4f}', flush=True)
            if ('multi_nomask', 'B') in v:
                gap = v[('multi_nomask', 'B')] - v[('multi', 'B')]
                print(f'  [MASK GAP {step}] errB nomask-mask = {gap:+.5f}  '
                      f'({v[("multi_nomask","B")]:.5f} vs {v[("multi","B")]:.5f})  '
                      f'|W_mask| = {mask_norm():.6f}', flush=True)
            vlog.flush()
        if step % a.save_every == 0 or step == a.steps:
            save_file({k: t.contiguous().cpu() for k, t in cn.state_dict().items()},
                      os.path.join(a.out, f'cn_step{step:06d}.safetensors'))
            print(f'  saved step {step}', flush=True)
    print('done')


if __name__ == '__main__':
    main()

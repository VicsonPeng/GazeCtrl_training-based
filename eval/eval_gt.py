"""Evaluate gaze redirection against GazeCtrl-Eval's human labels, not against our teacher.

Every pair is two edits of the SAME photograph, so the two views share an image coordinate
system. That is what makes this set worth more than the teacher-based benchmark: crop both
views with ONE box (the union of their person masks) and the comparison is pixel-aligned,
so an appearance metric finally means something. Under the old LPIPS-vs-teacher the best
method scored worst, because that metric rewarded imitating our own 16-step sampler.

Per pair (A -> B): feed view A's crop, ask for view B's labelled gaze, and score
  gaze_err  angle between the generated person's measured gaze and B's label  (primary)
  id_sim    ArcFace(generated, A)            -- did the person survive the edit
  lpips_gt  LPIPS(generated, B's crop)       -- how close to how B actually looks
  base_err  angle between A's and B's labels -- what doing nothing would score
A method only earns its keep where gaze_err is well under base_err.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import os, sys, json, math, argparse, time, random
import numpy as np, pandas as pd, torch, cv2
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageOps
from safetensors.torch import load_file
sys.path.insert(0, _os.path.join(_p, "src"))
sys.path.insert(0, SIXDREP_DIR)
from qwen_models import (QwenBackbone, QwenControlNet, load_qwen_vae, vae_encode, vae_decode, pack_mask,
                         patchify, unpatchify, compute_rope_freqs_3d, compute_text_rope_freqs,
                         MODS_12)
from model import SixDRepNet2
from torchvision.models.resnet import Bottleneck
import utils as u6d
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import lpips as _lpips

R = Path(ROOT); M = R / 'gaze_controlnet/models'
BB = M / 'diffusion_models/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors'
VA = M / 'vae/split_files/vae/qwen_image_vae.safetensors'
EV = R / 'datasets/gazectrl_eval'

ap = argparse.ArgumentParser()
ap.add_argument('--cn', default=str(R / 'ckpt_archive/stage2_cn_step040000.safetensors'))
ap.add_argument('--backbone', default=str(BB))
ap.add_argument('--ext_lora', default='', help='diffusers-style LoRA via the 12-family hook')
ap.add_argument('--lora_scale', type=float, default=1.0)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--size', type=int, default=768)
ap.add_argument('--cn_scale', type=float, default=1.0)
ap.add_argument('--shift', type=float, default=3.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--manifest', default='manifest.json',
                help="the view labels to score against; manifest_g3d.json is the audited set")
ap.add_argument('--pairs', default='', help='json list of pairs to run; built by --sample if absent')
ap.add_argument('--sample', type=int, default=400, help='stratified pairs to draw when --pairs is absent')
ap.add_argument('--margin', type=float, default=0.35, help='box padding, in box sizes')
ap.add_argument('--name', required=True)
ap.add_argument('--use_mask', action='store_true',
                help="feed view_a's person mask to the ControlNet's mask channel, as stage-5 "
                     "was trained: SOURCE mask only, never the union with the target, because "
                     "at inference the target does not exist")
ap.add_argument('--outdir', default=str(R / 'gaze_controlnet/eval_samples/gt_eval'))
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--save_img', type=int, default=1)
ap.add_argument('--cond_box', type=int, default=0,
                help='1 = paint the gaze colour only inside the target person\'s box, neutral '
                     'grey elsewhere -- the conditioning arm B was trained with. Must match '
                     'the checkpoint: a box-trained model fed a flat map, or the reverse, is '
                     'being asked a question it was never trained on.')
ap.add_argument('--box_pad', type=float, default=0.05)
ap.add_argument('--box_json', default='', help='pre-built boxes keyed by pid '
                '(scene__view_a__to__view_b); each entry {"target":[x0,y0,x1,y1],"size":[w,h]}. '
                'Overrides the mask bounding box when --cond_box is on.')
ap.add_argument('--box_from', default='a', choices=['a', 'b'],
                help="whose mask the box comes from: 'a' the source view's person (what "
                     "training used), 'b' the target view's. Swapping them is how to test "
                     "whether the box actually points the edit at a person.")
ap.add_argument('--shuffle_gaze', type=int, default=0,
                help='CONTROL: feed another pair\'s target gaze instead of this one\'s. '
                     'The seed of a derangement over the pair list; 0 is off. Everything else '
                     '-- source image, crop, seed, checkpoint -- is unchanged, so the only '
                     'variable is which gaze the model is asked for. If the model is driven by '
                     'the gaze signal, the error against the FED gaze stays where it was and the '
                     'error against the pair\'s own target blows up; if it ignores the signal, '
                     'the opposite.')
a = ap.parse_args()

dev = torch.device('cuda:0'); SZ = a.size; G = SZ // 8 // 2
BOXJ = json.load(open(a.box_json)) if a.box_json else None
BOXMISS = []
if BOXJ is not None:
    print(f'box_json: {len(BOXJ)} boxes from {a.box_json}', flush=True)
_RES = float(os.environ.get('RESERVE_GIB', '26'))
if _RES > 0:
    _r = torch.empty(int(_RES * (1024 ** 3)) // 2, dtype=torch.float16, device=dev); del _r

OUT = os.path.join(a.outdir, a.name); os.makedirs(os.path.join(OUT, 'img'), exist_ok=True)
man = json.load(open(EV / a.manifest))
SC = {s['image']: s for s in man['scenes']}
VIEW = {(s['image'], v['view_id']): v for s in man['scenes'] for v in s['views']}
# A frozen pair list outlives the manifest it was drawn from: pairs_multi200.json names views
# that the cull later removed, and every run already measured on it stays comparable only if
# those views can still be read. The pre-cull backups are kept beside the manifest for that;
# they fill in views the live one no longer lists, and never override a view it does.
for _bak in sorted(EV.glob(a.manifest + '.bak_*')):
    for _sc in json.load(open(_bak))['scenes']:
        for _v in _sc['views']:
            VIEW.setdefault((_sc['image'], _v['view_id']), _v)

# ---- pair selection: stratified over category and over how far the gaze has to move ----
if a.pairs:
    PAIRS = json.load(open(a.pairs))
else:
    df = pd.read_csv(EV / 'pairs.csv')
    df = df[df.apply(lambda r: all(os.path.exists(EV / VIEW[(r['scene'], v)]['file'])
                                   and os.path.exists(EV / VIEW[(r['scene'], v)]['mask'])
                                   for v in (r['view_a'], r['view_b'])), axis=1)]
    df['bin'] = pd.cut(df.angle_deg, [0, 30, 60, 90, 120, 180], labels=False)
    rng = random.Random(0); PAIRS = []
    per = max(1, a.sample // (df.category.nunique() * 5))
    for (_c, _b), g in df.groupby(['category', 'bin']):
        PAIRS += g.iloc[rng.sample(range(len(g)), min(per, len(g)))].to_dict('records')
    rng.shuffle(PAIRS); PAIRS = PAIRS[:a.sample]
    json.dump(PAIRS, open(os.path.join(a.outdir, 'pairs_sample.json'), 'w'), indent=1, default=str)
PAIRS = [p for i, p in enumerate(PAIRS) if i % a.nshard == a.shard]
print(f'{len(PAIRS)} pairs on shard {a.shard}/{a.nshard}', flush=True)

FED = None
if a.shuffle_gaze:
    assert a.nshard == 1, 'shuffle_gaze needs the whole pair list in one shard'
    rng = random.Random(a.shuffle_gaze)
    idx = list(range(len(PAIRS)))
    for _ in range(10000):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
    else:
        raise SystemExit('could not build a derangement')
    FED = idx
    print(f'SHUFFLE CONTROL: each pair is fed the target gaze of another pair '
          f'(derangement, seed {a.shuffle_gaze}); no pair keeps its own', flush=True)

def box_of(mask_path):
    m = np.array(Image.open(mask_path).convert('L'))
    ys, xs = np.where(m > 127)
    if not len(xs): return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def union_square(b1, b2, W, H):
    """One box for both views, square and padded, clipped to the photo."""
    x1 = min(b1[0], b2[0]); y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2]); y2 = max(b1[3], b2[3])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + a.margin)
    s = min(s, min(W, H))
    cx = min(max(cx, s / 2), W - s / 2); cy = min(max(cy, s / 2), H - s / 2)
    return int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)

def crop(path, box):
    return Image.open(path).convert('RGB').crop(box).resize((SZ, SZ), Image.LANCZOS)

def crop_mask(path, box, dilate=9):
    m = Image.open(path).convert('L').crop(box).resize((SZ, SZ), Image.NEAREST)
    m = np.array(m) > 127
    if dilate:
        k = np.ones((dilate, dilate), np.uint8)
        m = cv2.dilate(m.astype(np.uint8), k, 1) > 0
    return m[..., None]

def apply_mask(arr, m):
    """Grey outside the person.

    A whole-crop appearance score is dominated by background here: the two views are the same
    photograph, so an unedited source already matches the target background perfectly while
    our pipeline regenerates it. That makes doing nothing beat every edit, which says nothing
    about whether the person was edited correctly. Scoring inside the person mask asks the
    question the task actually poses.
    """
    return np.where(m, arr, 128).astype(np.uint8)

# ---- models ----
vae = load_qwen_vae(str(VA), device=dev)
bb = QwenBackbone(str(a.backbone), device=dev)
LORA = None
if a.ext_lora:
    from safetensors import safe_open
    with safe_open(a.ext_lora, framework='pt', device='cpu') as f:
        ks = set(f.keys()); LORA = []
        for i in range(60):
            d = {}
            for loc, tgt in MODS_12.items():
                b = f'transformer_blocks.{i}.{tgt}'
                if b + '.lora_down.weight' not in ks: continue
                A = f.get_tensor(b + '.lora_down.weight'); B = f.get_tensor(b + '.lora_up.weight')
                al = b + '.alpha'
                s = float(f.get_tensor(al)) / A.shape[0] if al in ks else 1.0
                d[loc] = (A.to(dev, torch.bfloat16), (B.float() * s).to(dev, torch.bfloat16))
            LORA.append(d)
    print(f'[ext_lora] {a.ext_lora} mods {sorted(LORA[0])}', flush=True)
cn = QwenControlNet().to(dev); cn.load_state_dict(load_file(a.cn)); cn.to(torch.bfloat16).eval()
m6 = SixDRepNet2(Bottleneck, [3, 4, 6, 3])
_sd = torch.load(SIXDREP_CKPT,
                 map_location='cpu', weights_only=False)
m6.load_state_dict(_sd.get('model_state_dict', _sd), strict=False); m6.to(dev).eval()
yd = YOLO(YOLO_HEAD)
fa = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(640, 640))
percep = _lpips.LPIPS(net='alex').to(dev).eval()
for p in percep.parameters(): p.requires_grad_(False)
IMN = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1)
ISD = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
te = torch.zeros((1, 7, 3584), device=dev, dtype=torch.bfloat16)
imf = compute_rope_freqs_3d(G, G, t_patches=2, device=dev)
imfc = compute_rope_freqs_3d(G, G, t_patches=1, device=dev)
txf = compute_text_rope_freqs(7, device=dev)

def sched(n, st=0.999, sh=None):
    sh = a.shift if sh is None else sh
    ts = st / (sh * (1 - st) + st + 1e-9)
    t = torch.linspace(ts, 1e-5, n + 1)[:-1]
    return sh / (sh + (1.0 / t.clamp(1e-6, 1 - 1e-6) - 1.0))

@torch.no_grad()
def gen(ref, gz, mp=None, box=None):
    dx, dy, dz = gz; n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    col = (np.array([(dx / n + 1) / 2, (dy / n + 1) / 2, (dz / n + 1) / 2]) * 255).clip(0, 255)
    if box is None:
        arr = np.full((SZ, SZ, 3), col, dtype=np.uint8)
    else:
        # 128 is where every axis encodes 0, so the frame outside the box says "no instruction
        # here" rather than "look along this particular direction"
        arr = np.full((SZ, SZ, 3), 128, dtype=np.uint8)
        x0, y0, x1, y1 = box
        arr[max(0, y0):min(SZ, y1), max(0, x0):min(SZ, x1)] = col
    c = (torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1
         ).unsqueeze(0).to(dev, torch.bfloat16)
    cp = patchify(vae_encode(vae, c)).to(torch.bfloat16)
    torch.manual_seed(a.seed)
    x = torch.randn(1, 16, 1, SZ // 8, SZ // 8, device=dev, dtype=torch.bfloat16)
    S = sched(a.steps)
    for i, sg in enumerate(S):
        si = sg.unsqueeze(0).to(dev, x.dtype)
        sn = (S[i + 1].unsqueeze(0).to(dev, x.dtype) if i + 1 < a.steps
              else torch.zeros(1, device=dev, dtype=x.dtype))
        res = [q * a.cn_scale for q in cn.forward(patchify(x), cp, te, si, img_freqs=imfc, txt_freqs=txf,
                                                  mask_packed=mp)]
        v = bb.forward(patchify(x), te, si, res, img_freqs=imf, txt_freqs=txf,
                       ref_packed=ref, lora_blocks=LORA, lora_scale=a.lora_scale)
        x = x - (sg - sn).view(1, 1, 1, 1, 1) * unpatchify(v, G, G)
    return vae_decode(vae, x)[0]

@torch.no_grad()
def other_person_l1(out, src, mU, anchor, pad=0.25):
    """How much the OTHER people changed: heads the label does not point at.

    keep_l1 averages bystanders together with a large, easy background. This looks only where
    someone else is standing, which is the thing the mask is supposed to protect. Returns nan
    when the crop holds nobody else.
    """
    r = yd(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), verbose=False)[0].boxes
    if r is None or len(r) == 0: return float('nan')
    sel = np.zeros((SZ, SZ), bool)
    for b in r.xyxy.cpu().numpy():
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if anchor is not None and math.hypot(cx - anchor[0], cy - anchor[1]) < max(b[2] - b[0], b[3] - b[1]):
            continue                                  # this is the labelled person
        w, h = b[2] - b[0], b[3] - b[1]
        x1 = int(max(0, b[0] - pad * w)); y1 = int(max(0, b[1] - pad * h))
        x2 = int(min(SZ, b[2] + pad * w)); y2 = int(min(SZ, b[3] + pad * h))
        sel[y1:y2, x1:x2] = True
    sel &= ~mU[..., 0]
    if sel.sum() < 500: return float('nan')
    return float(np.abs(out.astype(int) - src.astype(int))[sel].mean())

@torch.no_grad()
def measure(arr, anchor=None):
    """anchor = the labelled face point in CROP pixels.

    Half of these scenes are couples, families or meetings, so the crop usually holds more
    than one person and 'the largest head' is regularly the wrong one -- a parent instead of
    the labelled partner. With the anchor the head nearest the label is taken instead, which
    is the only way the gaze number refers to the person the label describes. Without an
    anchor (e.g. scoring the source) the largest head is still the fallback.
    """
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR); gz = None
    r = yd(bgr, verbose=False)[0].boxes
    nhead = 0 if (r is None) else len(r)
    if r is not None and len(r) > 0:
        xy = r.xyxy.cpu().numpy()
        if anchor is not None:
            cx = (xy[:, 0] + xy[:, 2]) / 2; cy = (xy[:, 1] + xy[:, 3]) / 2
            k = int(np.argmin((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2))
        else:
            k = int(np.argmax((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])))
        x1, y1, x2, y2 = xy[k]; pad = int(.35 * max(x2 - x1, y2 - y1)); H, W = bgr.shape[:2]
        c = bgr[max(0, int(y1 - pad)):min(H, int(y2 + pad)), max(0, int(x1 - pad)):min(W, int(x2 + pad))]
        if c.size:
            t = torch.from_numpy(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(dev) / 255.
            t = F.interpolate(t, (224, 224), mode='bilinear', align_corners=False); t = (t - IMN) / ISD
            e = u6d.compute_euler_angles_from_rotation_matrices(m6(t.float()))[0].cpu().numpy()
            pit, yaw = float(e[0]), float(e[1])
            dx = -math.sin(yaw) * math.cos(pit); dy = math.sin(pit); dz = math.cos(yaw) * math.cos(pit)
            if dz < 0: dx = -dx
            gz = (dx, dy, dz)
    fs = fa.get(bgr)
    if not fs:
        emb = None
    elif anchor is not None:
        emb = min(fs, key=lambda q: ((q.bbox[0] + q.bbox[2]) / 2 - anchor[0]) ** 2
                                    + ((q.bbox[1] + q.bbox[3]) / 2 - anchor[1]) ** 2).normed_embedding
    else:
        emb = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])).normed_embedding
    return gz, emb, nhead, len(fs)

def ang(u, v):
    u = np.array(u, float); v = np.array(v, float)
    u /= (np.linalg.norm(u) or 1); v /= (np.linalg.norm(v) or 1)
    return float(math.degrees(math.acos(max(-1, min(1, float(u @ v))))))

def lp(p, q):
    def t(x): return (torch.from_numpy(np.array(x)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).to(dev)
    with torch.no_grad(): return float(percep(t(p), t(q)).item())

rows = []
t0 = time.time()
for n, pr in enumerate(PAIRS):
    sc, va_, vb = pr['scene'], pr['view_a'], pr['view_b']
    A, B = VIEW[(sc, va_)], VIEW[(sc, vb)]
    fa_, fb = EV / A['file'], EV / B['file']
    ba, bbx = box_of(EV / A['mask']), box_of(EV / B['mask'])
    if ba is None or bbx is None: continue
    W, H = Image.open(fa_).size
    box = union_square(ba, bbx, W, H)
    srcA, gtB = crop(fa_, box), crop(fb, box)
    mB = crop_mask(EV / B['mask'], box)
    mA = crop_mask(EV / A['mask'], box)
    mU = mA | mB
    gzOwn = (B['dx'], B['dy'], B['dz'])
    if FED is None:
        gzB = gzOwn
    else:
        q = PAIRS[FED[n]]
        Bf = VIEW[(q['scene'], q['view_b'])]
        gzB = (Bf['dx'], Bf['dy'], Bf['dz'])
    x0 = (torch.from_numpy(np.array(srcA)).float().permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(dev, torch.bfloat16)
    with torch.no_grad(): ref = patchify(vae_encode(vae, x0)).to(torch.bfloat16)
    mp = None
    if a.use_mask:
        mt = torch.from_numpy(mA[..., 0].astype(np.float32)).view(1, 1, SZ, SZ).to(dev)
        mp = pack_mask(mt, SZ // 8, SZ // 8).to(torch.bfloat16)
    cbox = None
    if a.cond_box and BOXJ is not None:
        e = BOXJ.get(f'{sc}__{va_}__to__{vb}')
        if e is None:
            BOXMISS.append(f'{sc}__{va_}__to__{vb}')
        else:
            k = SZ / float(e.get('size', [768, 768])[0])
            cbox = tuple(int(round(v * k)) for v in e['target'])
    if a.cond_box and cbox is None:
        msel = mA if a.box_from == 'a' else mB
        ys, xs = np.where(msel[..., 0])   # crop_mask returns bool; a bool > 127 is always False
        if len(xs):
            bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            pw, ph = a.box_pad * (bx1 - bx0), a.box_pad * (by1 - by0)
            cbox = (int(round(bx0 - pw)), int(round(by0 - ph)),
                    int(round(bx1 + pw)), int(round(by1 + ph)))
    if n < 3:
        print(f'[cbox] {sc}__{va_}__to__{vb}  cond_box={a.cond_box}  '
              f'from_json={BOXJ is not None}  box={cbox}', flush=True)
    im = gen(ref, gzB, mp, box=cbox)
    arr = (((im.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy())
    if a.save_img:
        Image.fromarray(arr).save(os.path.join(OUT, 'img', f'{sc}__{va_}__to__{vb}.jpg'), quality=92)
    # the labelled face point, carried from photo pixels into crop pixels
    sx = SZ / (box[2] - box[0])
    anc_b = ((B['face_px'][0] - box[0]) * sx, (B['face_px'][1] - box[1]) * sx)
    anc_a = ((A['face_px'][0] - box[0]) * sx, (A['face_px'][1] - box[1]) * sx)
    mg, em, nh, nf = measure(arr, anc_b)
    _, ea, _, nf_a = measure(np.array(srcA), anc_a)
    rows.append(dict(name=a.name, scene=sc, category=pr['category'], view_a=va_, view_b=vb,
                     kind_a=A['kind'], kind_b=B['kind'],
                     label_src_b=B['label_source'],
                     req_yaw=round(B['yaw_deg'], 1), req_pitch=round(B['pitch_deg'], 1),
                     base_err=round(ang((A['dx'], A['dy'], A['dz']), gzB), 1),
                     gaze_err=(None if mg is None else round(ang(mg, gzB), 1)),
                     fed_dx=round(gzB[0], 6), fed_dy=round(gzB[1], 6), fed_dz=round(gzB[2], 6),
                     fed_vs_own=round(ang(gzB, gzOwn), 1),
                     gaze_err_own=(None if mg is None else round(ang(mg, gzOwn), 1)),
                     id_sim=(None if (em is None or ea is None) else round(float(np.dot(em, ea)), 3)),
                     lpips_gt=round(lp(arr, gtB), 4),
                     lpips_src=round(lp(np.array(srcA), gtB), 4),
                     lpips_person=round(lp(apply_mask(arr, mU), apply_mask(np.array(gtB), mU)), 4),
                     lpips_src_person=round(lp(apply_mask(np.array(srcA), mU),
                                               apply_mask(np.array(gtB), mU)), 4),
                     person_frac=round(float(mU.mean()), 4),
                     keep_l1=round(float(np.abs(arr.astype(int) - np.array(srcA).astype(int))[
                         np.repeat(~mU, 3, 2)].mean()), 3),
                     keep_floor_l1=round(float(np.abs(np.array(gtB).astype(int) - np.array(srcA).astype(int))[
                         np.repeat(~mU, 3, 2)].mean()), 3),
                     keep_lpips=round(lp(apply_mask(arr, ~mU), apply_mask(np.array(srcA), ~mU)), 4),
                     keep_floor_lpips=round(lp(apply_mask(np.array(gtB), ~mU),
                                               apply_mask(np.array(srcA), ~mU)), 4),
                     other_l1=round(other_person_l1(arr, np.array(srcA), mU, anc_a), 3),
                     other_floor_l1=round(other_person_l1(np.array(gtB), np.array(srcA), mU, anc_a), 3),
                     det=int(mg is not None), n_head=nh, n_face=nf, n_face_src=nf_a,
                     multi=int(nf_a > 1), box=str(box)))
    if (n + 1) % 10 == 0:
        d = pd.DataFrame(rows); d.to_csv(os.path.join(a.outdir, f'{a.name}_shard{a.shard}.csv'), index=False)
        g = d.gaze_err.dropna()
        print(f'  {n+1}/{len(PAIRS)}  {(time.time()-t0)/(n+1):.1f}s/pair  '
              f'gaze_err med {g.median() if len(g) else float("nan"):.1f} (base {d.base_err.median():.1f})  '
              f'lpips_person {d.lpips_person.mean():.4f} (src {d.lpips_src_person.mean():.4f})  '
              f'det {100*d.det.mean():.0f}%', flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(a.outdir, f'{a.name}_shard{a.shard}.csv'), index=False)
g = d.gaze_err.dropna()
if BOXMISS:
    print(f'WARNING: {len(BOXMISS)} pairs had no box in box_json and fell back to the mask bbox: {BOXMISS[:5]}', flush=True)
print(f'== {a.name}  n={len(d)}  gaze_err median {g.median():.1f}  mean {g.mean():.1f}  '
      f'base median {d.base_err.median():.1f}  within30 {100*(g<30).mean():.1f}%  '
      f'id_sim {d.id_sim.dropna().mean():+.3f}  '
      f'lpips_person {d.lpips_person.mean():.4f} (src {d.lpips_src_person.mean():.4f})  '
      f'lpips_full {d.lpips_gt.mean():.4f} (src {d.lpips_src.mean():.4f})  '
      f'multi {100*d.multi.mean():.0f}%  det {100*d.det.mean():.0f}%', flush=True)
print(f'   keep_l1 {d.keep_l1.mean():.2f} (floor {d.keep_floor_l1.mean():.2f})  '
      f'keep_lpips {d.keep_lpips.mean():.4f} (floor {d.keep_floor_lpips.mean():.4f})  '
      f'other_l1 {d.other_l1.mean():.2f} (floor {d.other_floor_l1.mean():.2f}, '
      f'n={int(d.other_l1.notna().sum())})  mask_fed={a.use_mask}', flush=True)
print('DONE ->', os.path.join(a.outdir, f'{a.name}_shard{a.shard}.csv'))

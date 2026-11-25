"""Synthesise two-person training examples whose supervision is exact.

Take two single-person pairs that already exist. Person A is the one to edit: we have the
frame they start in, the frame they end in, and the gaze label of that end frame. Person B is
cut out of a frame of their own and pasted into A's scene at one position, into BOTH the input
and the target, from the SAME pixels. So "do not move the person who is not masked" is not an
approximation here -- B is bit-identical by construction, and the loss can say so exactly.

Why this is worth building rather than collecting: a real two-person photograph gives no
supervision for the person who must stay still, because no second photograph exists in which
only the other person moved. Compositing manufactures exactly that counterfactual.

The shortcut to design against: the model could learn "leave the pasted-looking one alone" or
"leave the right-hand side alone" instead of "leave whoever is not masked alone". Three
measures, all in --mix: the masked person is sometimes the pasted one rather than the scene
owner, some examples ask for the gaze the masked person already has so the correct answer is
to change nothing, and single-person examples stay in the pool so the original behaviour is
not forgotten.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import os, sys, json, math, argparse, random
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps

sys.path.insert(0, _os.path.join(_p, "src"))
R = ROOT
HITL = WORK + '/_cn_hitl/hitl.csv'
SHHQ = WORK + '/_cn_50id/50id_stage1.csv'
SHHQ_ROOT = f'{R}/gaze_dataset/'

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=f'{R}/datasets/multiperson')
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dev', default='cpu', help='cpu keeps this off the contended cards')
ap.add_argument('--min_free', type=float, default=0.22,
                help='fraction of the canvas width that must be clear of A before B is placed')
ap.add_argument('--btries', type=int, default=8,
                help='how many B candidates to try against one A before giving up on it')
ap.add_argument('--sheet', type=int, default=12, help='examples to draw into the check sheet')
ap.add_argument('--mix', default='multi=0.4,single=0.5,identity=0.1',
                help='proportions of the three example kinds. single keeps the behaviour the '
                     'ControlNet already has, so adding a mask channel does not cost it; '
                     'identity masks someone and asks for the gaze they already have, so the '
                     'correct answer is to change nothing -- without it the model can learn '
                     'that a mask always means motion.')
a = ap.parse_args()
SZ = a.size
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)
for d in ('input', 'gt', 'mask', 'bmask'):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)

def load_pool():
    rows = []
    for f, root in ((HITL, ''), (SHHQ, SHHQ_ROOT)):
        d = pd.read_csv(f)
        d['path'] = d['image'].map(lambda p: p if str(p).startswith('/') else root + str(p))
        d = d[d['path'].map(os.path.exists)]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    # a sample is usable only if it has at least two frames, so a source and a target exist
    keep = d.groupby('sample').size()
    d = d[d['sample'].isin(keep[keep >= 2].index)]
    return d

POOL = load_pool()
SAMPLES = sorted(POOL['sample'].unique())
print(f'pool: {len(POOL)} frames over {len(SAMPLES)} identities', flush=True)

from ultralytics import YOLO
seg = YOLO(f'{R}/yolo_seg/yolo26n-seg.pt')
head = YOLO(f'{R}/yolo_seg/yolov8_head.pt')

def content_box(path, size):
    """Where the real photograph sits on the padded square canvas.

    Computed from the original dimensions, not by looking for black pixels: a night shot or a
    dark studio backdrop would be mistaken for padding and the placement would get worse on
    exactly the images that are hardest already.
    """
    w, h = Image.open(path).size
    m = max(w, h); k = size / m
    x0 = ((m - w) // 2) * k; y0 = ((m - h) // 2) * k
    return x0, y0, x0 + w * k, y0 + h * k


_hcache = {}
def head_box(path, size):
    """Largest head on the padded square canvas, or None.

    Head size is the scale reference rather than person height: A is often a close-up and B a
    full-length shot, so matching their heights would put a doll beside a face. Two heads of
    the same size read as two people standing the same distance away, which is the only way
    the composite can pass for a photograph.
    """
    k = (path, size)
    if k in _hcache: return _hcache[k]
    im = np.array(sq(Image.open(path).convert('RGB'), size))
    r = head.predict(im[:, :, ::-1], verbose=False, device=a.dev)[0].boxes
    out = None
    if r is not None and len(r):
        xy = r.xyxy.cpu().numpy()
        j = int(np.argmax((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])))
        out = tuple(float(v) for v in xy[j])
    _hcache[k] = out
    return out


def match_colour(cut, cutm, scene, scenem):
    """Pull B's cut-out towards the lighting of the scene it is being dropped into.

    Per channel mean and standard deviation only. Anything stronger would start repainting the
    person, and the point is to stop an obvious lighting seam from becoming the cue the model
    keys on instead of the mask.
    """
    out = cut.astype(np.float32).copy()
    ref = scene[scenem] if scenem.sum() > 64 else scene.reshape(-1, 3)
    src = cut[cutm]
    if len(src) < 64: return cut
    for c in range(3):
        ms, ss = src[:, c].mean(), src[:, c].std() + 1e-5
        mr, sr = ref[:, c].mean(), ref[:, c].std() + 1e-5
        g = float(np.clip(sr / ss, 0.7, 1.4))
        out[..., c] = (out[..., c] - ms) * g + (ms + 0.6 * (mr - ms))
    return np.clip(out, 0, 255).astype(np.uint8)

def sq(img, size):
    w, h = img.size; m = max(w, h)
    return ImageOps.expand(img, ((m - w) // 2, (m - h) // 2,
                                 m - w - (m - w) // 2, m - h - (m - h) // 2)).resize((size, size),
                                                                                      Image.LANCZOS)

_mcache = {}
def person_mask(path, size, want_count=False):
    """Largest person in the frame, as a {0,1} map on the padded square canvas.

    Also returns how many people are in the frame. A's source has to hold exactly one: the
    frame pairs come from a generator that redrew the whole picture, so anyone else standing
    in A's own photograph is unmasked and still changes between source and target -- an
    example that teaches the precise opposite of the rule the mask is meant to establish.
    Small detections are ignored; a bystander a few pixels across is not what the loss sees.
    """
    k = (path, size)
    if k not in _mcache:
        im = Image.open(path).convert('RGB')
        r = seg.predict(path, verbose=False, classes=[0], device=a.dev)[0]
        if r.masks is None or len(r.masks) == 0:
            _mcache[k] = (None, 0)
        else:
            mk = r.masks.data.cpu().numpy()
            big = [x for x in mk if x.mean() > 0.01]
            j = int(np.argmax([x.sum() for x in mk]))
            m = Image.fromarray((mk[j] * 255).astype(np.uint8)).resize(im.size, Image.NEAREST)
            m = (np.array(sq(m.convert('L'), size)) > 127).astype(np.uint8)
            _mcache[k] = (m, len(big))
    m, c = _mcache[k]
    return (m, c) if want_count else m

def place(free_cols, bw):
    """Left-most run of clear columns wide enough for B; None if the scene is too full."""
    runs, s = [], None
    for i, v in enumerate(free_cols):
        if v and s is None: s = i
        elif not v and s is not None: runs.append((s, i)); s = None
    if s is not None: runs.append((s, len(free_cols)))
    runs = [r for r in runs if r[1] - r[0] >= bw]
    return rng.choice(runs) if runs else None

MIX = {}
for part in a.mix.split(','):
    k, v = part.split('='); MIX[k.strip()] = float(v)
tot = sum(MIX.values()); MIX = {k: v / tot for k, v in MIX.items()}
print('mix:', {k: round(v, 3) for k, v in MIX.items()}, flush=True)

rows, made, tried, skipped = [], 0, 0, {}
def skip(why): skipped[why] = skipped.get(why, 0) + 1

QUOTA = {k: int(round(a.n * v)) for k, v in MIX.items()}
_short = a.n - sum(QUOTA.values())
if _short: QUOTA[max(QUOTA, key=lambda k: QUOTA[k])] += _short
DONE = {k: 0 for k in MIX}

def pick_kind():
    """Fill each kind up to its quota rather than sampling the mix per attempt.

    Sampling skews the corpus, because the kinds do not fail equally often: a composite is
    rejected far more than a single-person example, so the realised mix drifts towards
    whatever is easiest to make. An earlier run asked for 50/40/10 and wrote 50/8/2.
    """
    short = [k for k in MIX if MIX[k] > 0 and DONE[k] < QUOTA[k]]
    if not short: short = [k for k in MIX if MIX[k] > 0]
    return max(short, key=lambda k: (QUOTA[k] - DONE[k]) / MIX[k])

def emit_simple(kind):
    """A one-person example in the same on-disk form as the composites.

    Written out rather than assembled during training so the whole corpus -- every kind --
    can be checked once, before a single gradient step, instead of trusting a dataloader.
    """
    sa = rng.choice(SAMPLES)
    ga = POOL[POOL['sample'] == sa]
    if len(ga) < 2: return None
    ra_src, ra_tgt = ga.iloc[rng.sample(range(len(ga)), 2)].to_dict('records')
    if kind == 'identity':
        ra_tgt = ra_src                      # asked for the gaze it already has: do nothing
    m, nm = person_mask(ra_src['path'], SZ, want_count=True)
    if m is None or m.sum() < 0.01 * SZ * SZ: return None
    if nm > 1: return None            # same rule for the single-person examples
    inp = np.array(sq(Image.open(ra_src['path']).convert('RGB'), SZ))
    gt = inp if kind == 'identity' else np.array(sq(Image.open(ra_tgt['path']).convert('RGB'), SZ))
    return ra_src, ra_tgt, m, inp, gt, sa

while made < a.n and tried < a.n * 40:
    tried += 1
    kind = pick_kind()
    if kind in ('single', 'identity'):
        got = emit_simple(kind)
        if got is None: skip(f'{kind}: no usable pair'); continue
        ra_src, ra_tgt, mA_src, inp, gt, sa = got
        name = f'{made:05d}_{kind}_{sa}__{ra_src["frame"]}to{ra_tgt["frame"]}'
        Image.fromarray(inp).save(os.path.join(a.out, 'input', name + '.png'))
        Image.fromarray(gt).save(os.path.join(a.out, 'gt', name + '.png'))
        Image.fromarray(mA_src * 255).save(os.path.join(a.out, 'mask', name + '.png'))
        Image.fromarray(np.zeros((SZ, SZ), np.uint8)).save(os.path.join(a.out, 'bmask', name + '.png'))
        rows.append(dict(name=name, kind=kind, sample_a=sa, sample_b='',
                         frame_src=ra_src['frame'], frame_tgt=ra_tgt['frame'], frame_b=-1,
                         dx=ra_tgt['dx'], dy=ra_tgt['dy'], dz=ra_tgt['dz'],
                         src_dx=ra_src['dx'], src_dy=ra_src['dy'], src_dz=ra_src['dz'],
                         b_x0=0, b_y0=0, b_w=0, b_h=0, b_px=0, a_px=int(mA_src.sum()),
                         a_union_px=int(mA_src.sum()), head_a=-1, head_b=-1, scale=-1, in_photo=-1,
                         path_src=ra_src['path'], path_tgt=ra_tgt['path'], path_b=''))
        made += 1; DONE[kind] += 1
        if made % 50 == 0: print(f'  {made}/{a.n}  (tried {tried})  {DONE}', flush=True)
        continue
    # A is drawn first and B is chosen to fit it, rather than both at once. The canvas is a
    # padded square: these frames are all 512x768, so the real photograph is only the middle
    # 341 of 512 columns and a third of the canvas is black bar. Dropping B on a bar keeps the
    # supervision exact (B is byte-identical in input and target either way) but teaches
    # nothing: the model learned in stage 2 that the bar region is copied verbatim, there is no
    # shared background, lighting or occlusion to reason about, and a real two-person
    # photograph never has anybody standing on a letterbox strip. So containment is now a hard
    # requirement, and when A is a close-up that leaves no room, a narrower B is drawn instead
    # of smearing this one onto the bar.
    sa = rng.choice(SAMPLES)
    ga = POOL[POOL['sample'] == sa]
    ra_src, ra_tgt = ga.iloc[rng.sample(range(len(ga)), 2)].to_dict('records')

    mA_src, nA_src = person_mask(ra_src['path'], SZ, want_count=True)
    mA_tgt, nA_tgt = person_mask(ra_tgt['path'], SZ, want_count=True)
    if mA_src is None or mA_tgt is None: skip('no person detected'); continue
    if nA_src > 1 or nA_tgt > 1: skip('A frame already has a second person'); continue

    hA = head_box(ra_src['path'], SZ)
    if hA is None: skip('no head for scale'); continue
    hA_h = hA[3] - hA[1]

    srcA = np.array(sq(Image.open(ra_src['path']).convert('RGB'), SZ))
    tgtA = np.array(sq(Image.open(ra_tgt['path']).convert('RGB'), SZ))
    occupied = (mA_src | mA_tgt)

    cx0, cy0, cx1, cy1 = content_box(ra_src['path'], SZ)
    icx0, icy0 = int(np.ceil(max(0, cx0))), int(np.ceil(max(0, cy0)))
    icx1, icy1 = int(np.floor(min(SZ, cx1))), int(np.floor(min(SZ, cy1)))
    content = np.zeros((SZ, SZ), bool)
    content[icy0:icy1, icx0:icx1] = True
    # the feather grows a few pixels outward from the hard mask, so the hard mask has to stay
    # that far clear of the edge of the photograph or the rim lands on the bar
    FEATHER = 3

    placed = None
    for _bt in range(a.btries):
        sb = rng.choice(SAMPLES)
        if sb == sa: continue
        gb = POOL[POOL['sample'] == sb]
        rb = gb.iloc[rng.randrange(len(gb))].to_dict()
        mB = person_mask(rb['path'], SZ)
        hB = head_box(rb['path'], SZ)
        if mB is None or hB is None: continue
        hB_h = hB[3] - hB[1]
        if hB_h < 4: continue
        ys, xs = np.where(mB > 0)
        if not len(xs): continue
        bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

        # Does B's silhouette end in feet, or was the body cut off by the edge of its own
        # frame? Almost every frame here runs the figure down to within a few percent of the
        # bottom, so the cut-out nearly always ends in a straight line -- and dropped into the
        # middle of the canvas that line is a torso stopping in mid-air, which is what makes
        # the composite read as a rectangle. Feet are narrow and a cut through the body is as
        # wide as the body, so the width of the lowest rows against the widest tells them
        # apart: measured over the pool it is strongly bimodal, 0.25 for the studio frames
        # that show feet and 0.80 for the cropped ones. A body-cut B is allowed only where the
        # cut falls on the bottom edge of the canvas, where a frame edge explains it.
        band = mB[max(0, by1 - 7):by1] > 0
        foot_end = band.any(0).sum() / max(1, int((mB > 0).any(0).sum())) < 0.45
        if not foot_end: continue
        # How much of the person do we actually have? Measured in head-heights, which is
        # free of scale: a whole figure is six to eight, head-and-shoulders is about two.
        # On the first sheet where every B ended in feet, the four that still read as
        # stickers measured 1.6 to 2.4 head-heights and the four that read as people
        # measured 5.0 to 7.8, so the two groups do not overlap anywhere near the cut.
        if (by1 - by0) / hB_h < 4.0: continue

        # Matching the two heads exactly puts B at the same apparent distance as A, which is
        # the most natural composite -- but it often will not fit. A is usually a whole figure
        # whose head is about 80px on this canvas, so a second person at the same distance
        # needs roughly four head-widths, some 320px, while the photograph is 341px wide in
        # total and keeps only ~190px of that once A is in it. Left as a hard rule it drags the
        # draw towards head-and-shoulders close-ups, the only crops narrow enough per head, and
        # those end in mid-air. So B may instead stand FURTHER BACK: smaller overall, head
        # still near A's head height, feet stopping higher up the frame -- which for a camera
        # at eye level is precisely what somebody standing behind looks like, and keeps B a
        # whole figure. Near-matched is tried first and only shrunk when it will not fit.
        for mult in (rng.uniform(0.9, 1.1), 0.8, 0.7, 0.6):
            scale = (hA_h / hB_h) * mult
            bw = max(1, int((bx1 - bx0) * scale)); bh = max(1, int((by1 - by0) * scale))
            if bw > 4 * SZ or bh > 4 * SZ: continue
            if hB_h * scale < 22: continue          # B's head must survive the downscale

            cutm = np.array(Image.fromarray((mB[by0:by1, bx0:bx1] * 255).astype(np.uint8))
                            .resize((bw, bh), Image.NEAREST)) > 127
            if not cutm.any(): continue
            # the support of the feathered alpha, so the soft rim is held to the same rule as the
            # body; dilating once is simpler than carrying the margin through the arithmetic
            cutd = cv2.dilate(cutm.astype(np.uint8),
                              np.ones((2 * FEATHER + 1,) * 2, np.uint8)) > 0

            # B's head goes at roughly A's head height
            hBy = (hB[1] - by0) * scale
            cy = (hA[1] + hA[3]) / 2 + rng.uniform(-0.05, 0.05) * SZ
            y0 = int(cy - hBy - (hA_h / 2))

            # A pixel of B is acceptable inside the photograph, or off the canvas altogether, and
            # nowhere else. Running off the bottom edge is exactly what a second person standing at
            # the edge of the frame looks like, so that stays allowed -- forbidding it forces B to
            # end in mid-air and biases the draw towards head-and-shoulders close-ups shrunk to a
            # quarter size. The bars lie between the photograph and the left and right edges, so
            # sideways there is no way out, and containment there is a hard requirement.
            rr = np.where(cutd.any(1))[0]; cc = np.where(cutd.any(0))[0]
            rd0, rd1 = int(rr[0]), int(rr[-1]) + 1
            cd0, cd1 = int(cc[0]), int(cc[-1]) + 1
            vr0, vr1 = max(0, y0 + rd0), min(SZ, y0 + rd1)      # the rows that land on the canvas
            if vr1 - vr0 < 16: continue
            if vr0 < icy0 or vr1 > icy1: continue               # on a bar, and no x can fix it
            lo = -SZ if icx0 <= 0 else icx0 - cd0
            hi = SZ if icx1 >= SZ else icx1 - cd1
            if hi < lo: continue                    # B is wider than the room in the photograph
            sub = cutm[vr0 - y0:vr1 - y0]                       # B's visible band, in B's rows
            # every x is checked rather than 48 random draws: whenever the photograph has room at
            # all a fully contained placement exists, and the old sampler was missing it. Shuffled
            # so the choice is still varied rather than always hugging one side.
            cand_x = list(range(lo, hi + 1))
            rng.shuffle(cand_x)
            x0 = None
            for xc in cand_x:
                vc0, vc1 = max(0, xc), min(SZ, xc + bw)
                vis = sub[:, vc0 - xc:vc1 - xc]
                if vis.sum() < 0.015 * SZ * SZ: continue        # too little of B is in frame
                if (occupied[vr0:vr1, vc0:vc1] & vis).any(): continue  # never covers A or its path
                x0 = xc; break
            if x0 is None: continue                 # the room there is taken by A

            srcB = np.array(sq(Image.open(rb['path']).convert('RGB'), SZ))
            cut = np.array(Image.fromarray(srcB[by0:by1, bx0:bx1]).resize((bw, bh), Image.LANCZOS))
            cut = match_colour(cut, cutm, srcA, mA_src.astype(bool))
            placed = (sb, rb, scale, hB_h, cut, cutm, bw, bh, x0, y0)
            break
        if placed is not None: break

    if placed is None: skip('no clear space for B'); continue
    sb, rb, scale, hB_h, cut, cutm, bw, bh, x0, y0 = placed

    # A soft edge, so a hard cut line does not become the giveaway the model keys on.
    # The blur is taken as a MAXIMUM against the hard mask: inside the person alpha has to be
    # exactly 1.0, not the 0.99999994 a Gaussian returns, because the leftover 6e-8 still
    # mixes in a trace of whatever lies underneath -- and what lies underneath is the one
    # thing that differs between the source and target frames. The feather then only grows
    # outward, which is what was wanted anyway.
    alpha = np.maximum(cv2.GaussianBlur(cutm.astype(np.float32), (0, 0), 1.2),
                       cutm.astype(np.float32))[..., None]

    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(SZ, x0 + bw), min(SZ, y0 + bh)

    def paste(canvas):
        out = canvas.astype(np.float32).copy()
        al = alpha[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        cu = cut[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0].astype(np.float32)
        out[ay0:ay1, ax0:ax1] = out[ay0:ay1, ax0:ax1] * (1 - al) + cu * al
        # round, never truncate: truncation turns a sub-ulp difference in the blend into a
        # whole level whenever the sum sits just under an integer
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    inp, gt = paste(srcA), paste(tgtA)
    amap = np.zeros((SZ, SZ), np.float32)
    amap[ay0:ay1, ax0:ax1] = alpha[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, 0]
    bmap = (amap > 0.02).astype(np.uint8)

    name = f'{made:05d}_{sa}__{ra_src["frame"]}to{ra_tgt["frame"]}__with__{sb}'
    Image.fromarray(inp).save(os.path.join(a.out, 'input', name + '.png'))
    Image.fromarray(gt).save(os.path.join(a.out, 'gt', name + '.png'))
    # The mask the model is given is A's mask in the SOURCE frame, never the union with the
    # target. At inference only the source exists, so a union mask would be an input that
    # cannot be produced when it matters, and the model would lean on knowing in advance where
    # the person ends up. Learning to move the person OUT of the marked region is the task.
    Image.fromarray(mA_src * 255).save(os.path.join(a.out, 'mask', name + '.png'))
    # B's footprint is written down rather than inferred later: recovering it by differencing
    # misses any pasted pixel that happens to match what it covered.
    # the alpha, not a binary map: the verifier needs to tell B's solid interior (which must be
    # bit-identical between input and target) from the soft rim (which blends with a background
    # that differs slightly between the two frames, and so cannot be)
    Image.fromarray((amap * 255).astype(np.uint8)).save(os.path.join(a.out, 'bmask', name + '.png'))
    in_photo = float((bmap.astype(bool) & content).sum()) / max(1, int(bmap.sum()))
    rows.append(dict(name=name, kind='multi', sample_a=sa, sample_b=sb,
                     frame_src=ra_src['frame'], frame_tgt=ra_tgt['frame'], frame_b=int(rb['frame']),
                     dx=ra_tgt['dx'], dy=ra_tgt['dy'], dz=ra_tgt['dz'],
                     src_dx=ra_src['dx'], src_dy=ra_src['dy'], src_dz=ra_src['dz'],
                     b_x0=int(x0), b_y0=int(y0), b_w=int(bw), b_h=int(bh),
                     b_px=int(bmap.sum()), a_px=int(mA_src.sum()),
                     head_a=round(float(hA_h), 1), head_b=round(float(hB_h), 1),
                     in_photo=round(in_photo, 3),
                     scale=round(float(scale), 3),
                     hratio=round(float(scale * hB_h / hA_h), 3),
                     heads_tall=round(float((by1 - by0) / hB_h), 2),
                     a_union_px=int(occupied.sum()),
                     path_src=ra_src['path'], path_tgt=ra_tgt['path'], path_b=rb['path']))
    made += 1; DONE['multi'] += 1
    if made % 50 == 0: print(f'  {made}/{a.n}  (tried {tried})  {DONE}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, 'index.csv'), index=False)
print(f'\nmade {made} examples in {tried} attempts -> {a.out}')
print('skips:', skipped)

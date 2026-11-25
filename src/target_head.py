"""Which detected head belongs to the person being edited, and the gaze box around them.

Why this exists: the target was picked as the head nearest the labelled face point, while
the gaze box was built from the target's segmentation mask. On 11 of 166 eval pairs those
two disagreed -- two people cheek to cheek, where the neighbour's head centre lands closer
to the target's labelled face point than the target's own centre does. A mask-built box
still framed the right person, but a head-led box followed the wrong one, and other_gaze.py
measured the target while calling it the bystander.

So: the mask decides. The face point only breaks ties and rescues the case where no head
sits on the mask at all (occluded or cropped-out head).
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import numpy as np

MIN_OV = 0.30   # a head this much covered by the mask counts as sitting on the target
TIE    = 0.10   # heads within this much overlap of the best are decided by the face point


def head_mask_overlap(h, mask):
    """fraction of head box h covered by the boolean mask"""
    H, W = mask.shape[:2]
    x0, y0, x1, y1 = (int(max(0, min(W, h[0]))), int(max(0, min(H, h[1]))),
                      int(max(0, min(W, h[2]))), int(max(0, min(H, h[3]))))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def pick_target(heads, mask, anchor=None):
    """Return (index of the target's head, overlaps, how it was decided).

    heads   list of (x0,y0,x1,y1) in the same pixel frame as mask
    mask    boolean array, True on the person to be edited
    anchor  (x,y) labelled face point of the target, or None
    """
    if not heads:
        return None, [], 'none'
    ov = [head_mask_overlap(h, mask) for h in heads]
    best = max(ov)
    if best < MIN_OV:
        # no head sits on the mask: the head is occluded, cropped, or the mask is wrong.
        # fall back to the face point, and say so, so the caller can refuse a head-led box.
        if anchor is None:
            return int(np.argmax(ov)), ov, 'weak-mask'
        c = np.array([[(h[0] + h[2]) / 2, (h[1] + h[3]) / 2] for h in heads])
        return int(np.argmin(((c - np.array(anchor)) ** 2).sum(1))), ov, 'weak-mask'
    near = [i for i, v in enumerate(ov) if v >= best - TIE]
    if len(near) == 1 or anchor is None:
        return int(near[0] if len(near) == 1 else np.argmax(ov)), ov, 'mask'
    c = np.array([[(heads[i][0] + heads[i][2]) / 2, (heads[i][1] + heads[i][3]) / 2] for i in near])
    return int(near[int(np.argmin(((c - np.array(anchor)) ** 2).sum(1)))]), ov, 'mask+face'


def gaze_box(mask, head, others=(), size=None, band=0.60, hw_mult=1.6, pad=0.05,
             mode='head_led'):
    """The rectangle the gaze colour is painted into.

    mode 'full'      the mask's bounding box (what shipped first)
    mode 'head_led'  head width x hw_mult centred on the head, from the mask's top down
                     through `band` of its height, then pulled back horizontally until each
                     bystander head is outside -- never cutting into the target's own head.
                     Falls back to 'band' when there is no trustworthy head.
    Returns (x0, y0, x1, y1) as floats, clamped to the frame.
    """
    H, W = mask.shape[:2]
    if size: W, H = size
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    def _pad(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return [b[0] - pad * w, b[1] - pad * h, b[2] + pad * w, b[3] + pad * h]

    def _clamp(b):
        return (max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3]))

    if mode == 'full' or head is None:
        if mode == 'full':
            return _clamp(_pad((mx0, my0, mx1, my1)))
        # head unusable: keep the band but take the width from the mask inside it
        yb = my0 + band * (my1 - my0)
        bx = np.where(mask[int(my0):int(yb)].any(0))[0]
        if not len(bx):
            return _clamp(_pad((mx0, my0, mx1, my1)))
        return _clamp(_pad((bx.min(), my0, bx.max() + 1, yb)))

    yb = my0 + band * (my1 - my0)
    hw, hc = head[2] - head[0], (head[0] + head[2]) / 2
    b = _pad((hc - hw * hw_mult / 2, my0, hc + hw * hw_mult / 2, yb))
    for o in others:
        ix = max(0.0, min(o[2], b[2]) - max(o[0], b[0]))
        iy = max(0.0, min(o[3], b[3]) - max(o[1], b[1]))
        if ix * iy <= 0.05 * max(1e-6, (o[2] - o[0]) * (o[3] - o[1])):
            continue
        if (o[0] + o[2]) / 2 > hc:            # bystander on the right: pull the right edge in
            b[2] = max(head[2] + 2, min(b[2], o[0] - 2))
        else:                                  # on the left: pull the left edge in
            b[0] = min(head[0] - 2, max(b[0], o[2] + 2))
    return _clamp(b)


def _cuts(om, tol):
    """Where each edge would have to go to leave at most `tol` of this person's mask inside.

    Taken from the person's own pixels, not their bounding box: a bystander with an arm out
    has a box that reaches across the target's head, and cutting at the box edge would throw
    the target's head away to escape a few pixels of sleeve.
    """
    ys, xs = np.where(om)
    if not len(xs):
        return None
    n = len(xs)
    k = max(0, int(round(tol * n)))
    sx = np.sort(xs); sy = np.sort(ys)
    return dict(right=float(sx[k]),               # box right edge here -> tol of them left inside
                left=float(sx[n - 1 - k] + 1),    # box left edge here
                bottom=float(sy[k]),
                top=float(sy[n - 1 - k] + 1))


def repel_box(mask, others, pad=0.05, size=None, keep=None, tol=0.02, max_others=4,
              head_keep=0.9):
    """The target mask's bounding box, pulled back until no other person is inside it.

    This is the rule Vicson asked for and it uses no heuristic about heads or body
    proportions: start from the box around the person the target mask covers, which can
    never be the wrong person, then take the boundary in until the others are outside.

    Each other person can only be escaped by moving one of the four edges past them, so the
    candidates are the 4**k ways of choosing an edge per intersecting person. Among the
    candidates that hold `keep` (the target's head, if given) the one covering the most of
    the target mask wins; area breaks ties.

    mask     boolean, True on the target
    others   list of boolean masks, one per other person, in the same frame
    tol      an other person may keep this fraction of their pixels inside before counting
    keep     a box that must stay `head_keep` inside every candidate (the target's head)
    Returns (box, coverage_of_target_mask, n_others_still_inside).
    """
    from itertools import product
    H, W = mask.shape[:2]
    if size: W, H = size
    ys, xs = np.where(mask)
    if not len(ys):
        return None, 0.0, 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    base = (max(0.0, x0 - pw), max(0.0, y0 - ph), min(float(W), x1 + pw), min(float(H), y1 + ph))
    tot = float(mask.sum())

    def frac_in(om, b):
        q = [int(round(v)) for v in b]
        if q[2] <= q[0] or q[3] <= q[1]:
            return 0.0
        return float(om[max(0, q[1]):q[3], max(0, q[0]):q[2]].sum()) / max(1.0, float(om.sum()))

    def cover(b):
        q = [int(round(v)) for v in b]
        if q[2] <= q[0] or q[3] <= q[1]:
            return -1.0
        return float(mask[max(0, q[1]):q[3], max(0, q[0]):q[2]].sum()) / max(1.0, tot)

    hit = [om for om in others if om.any() and frac_in(om, base) > tol]
    hit.sort(key=lambda om: -frac_in(om, base))
    hit = hit[:max_others]
    if not hit:
        return base, cover(base), 0
    cuts = [_cuts(om, tol) for om in hit]

    def head_ok(b):
        if keep is None:
            return True
        kx = max(0.0, min(keep[2], b[2]) - max(keep[0], b[0]))
        ky = max(0.0, min(keep[3], b[3]) - max(keep[1], b[1]))
        return kx * ky >= head_keep * max(1e-6, (keep[2] - keep[0]) * (keep[3] - keep[1]))

    best, best_key = None, (-1.0, -1.0)
    for choice in product(range(4), repeat=len(hit)):
        b = list(base)
        for cu, c in zip(cuts, choice):
            if c == 0:   b[2] = min(b[2], cu['right'])
            elif c == 1: b[0] = max(b[0], cu['left'])
            elif c == 2: b[3] = min(b[3], cu['bottom'])
            else:        b[1] = max(b[1], cu['top'])
        b = tuple(b)
        if b[2] - b[0] < 8 or b[3] - b[1] < 8:
            continue
        cv = cover(b)
        if cv < 0 or not head_ok(b):
            continue
        key = (round(cv, 4), (b[2] - b[0]) * (b[3] - b[1]))
        if key > best_key:
            best, best_key = b, key
    if best is None:
        # no rectangle both escapes them and holds the head: keep the head, take the one
        # that leaves the least of the others inside.
        for choice in product(range(4), repeat=len(hit)):
            b = list(base)
            for cu, c in zip(cuts, choice):
                if c == 0:   b[2] = min(b[2], cu['right'])
                elif c == 1: b[0] = max(b[0], cu['left'])
                elif c == 2: b[3] = min(b[3], cu['bottom'])
                else:        b[1] = max(b[1], cu['top'])
            b = tuple(b)
            if b[2] - b[0] < 8 or b[3] - b[1] < 8 or cover(b) < 0:
                continue
            kx = max(0.0, min(keep[2], b[2]) - max(keep[0], b[0])) if keep else 1.0
            ky = max(0.0, min(keep[3], b[3]) - max(keep[1], b[1])) if keep else 1.0
            hk = (kx * ky / max(1e-6, (keep[2]-keep[0])*(keep[3]-keep[1]))) if keep else 1.0
            key = (round(hk, 3), round(cover(b), 4))
            if key > best_key:
                best, best_key = b, key
        if best is None:
            return base, cover(base), len(hit)
        return best, cover(best), sum(1 for om in others if om.any() and frac_in(om, best) > tol)
    return best, best_key[0], sum(1 for om in others if om.any() and frac_in(om, best) > tol)


def headbox_box(mask, head, other_heads, pad=0.05, size=None, margin=2.0,
                max_heads=6, slack=0.0):
    """Vicson's rule (2026-11-22). Three boxes:

      box1(target)  the target's head box      -- the final box MUST contain it
      box1(other)   every other person's head  -- the final box must contain NONE of them
      box2          the target mask's bbox     -- the final box stays as close to it as it can

    Only the other people's HEADS constrain the box, never their bodies: an arm across the
    target's chest is allowed to stay inside, so a hug no longer collapses the box. Starting
    from box2 every blocking head gives four ways to take one edge past it; among the
    combinations that still hold the target's head whole, the one covering most of the target
    mask wins, which is what "stay close to box2" means in pixels.

    slack: fraction of an other-head's area that may remain inside (0 = none).
    Returns (box, coverage_of_target_mask, n_other_heads_still_inside, ok)
    where ok is False when the target's head could not be kept whole.
    """
    H, W = mask.shape[:2]
    if size:
        W, H = size
    ys, xs = np.where(mask)
    if not len(ys):
        return None, 0.0, 0, False
    my0, my1, mx0, mx1 = float(ys.min()), float(ys.max() + 1), float(xs.min()), float(xs.max() + 1)
    w, h = mx1 - mx0, my1 - my0
    base = (max(0.0, mx0 - pad * w), max(0.0, my0 - pad * h),
            min(float(W), mx1 + pad * w), min(float(H), my1 + pad * h))
    npx = float(mask.sum())

    def cover(b):
        q = [int(round(v)) for v in b]
        if q[2] <= q[0] or q[3] <= q[1]:
            return 0.0
        return float(mask[max(0, q[1]):q[3], max(0, q[0]):q[2]].sum()) / max(1.0, npx)

    def frac_in(i, o):                      # how much of box i sits inside box o
        ix = max(0.0, min(i[2], o[2]) - max(i[0], o[0]))
        iy = max(0.0, min(i[3], o[3]) - max(i[1], o[1]))
        return ix * iy / max(1e-6, (i[2] - i[0]) * (i[3] - i[1]))

    def holds_head(b):
        return head is None or frac_in(head, b) >= 0.999

    # only the heads that actually block box2 matter; nearest first if there are many
    blockers = [o for o in other_heads if frac_in(o, base) > slack]
    blockers.sort(key=lambda o: -frac_in(o, base))
    blockers = blockers[:max_heads]
    if not blockers:
        return base, cover(base), 0, holds_head(base)

    # Four ways to put one head outside: the box edge goes to the far side of that head --
    # but never past the target's own head. Other people's heads can overlap the target's
    # (a face just behind their shoulder), and there the two demands cannot both be met;
    # the target's head wins, because a box without it cannot say "turn this person".
    hk = head if head is not None else (base[0], base[1], base[0], base[1])
    opts = []
    for o in blockers:
        opts.append([(2, max(hk[2] + margin, o[0] - margin)),
                     (0, min(hk[0] - margin, o[2] + margin)),
                     (3, max(hk[3] + margin, o[1] - margin)),
                     (1, min(hk[1] - margin, o[3] + margin))])

    best = None
    import itertools
    for combo in itertools.product(*opts):
        b = list(base)
        for k, v in combo:
            b[k] = min(b[k], v) if k in (2, 3) else max(b[k], v)
        if b[2] - b[0] < 8 or b[3] - b[1] < 8:
            continue
        b = tuple(b)
        # rank: first how much of other people's heads is left inside, then how close the
        # box stays to box2 (= how much of the target mask it still covers).
        resid = sum(max(0.0, frac_in(o, b) - slack) for o in other_heads)
        key = (-round(resid, 3), cover(b), (b[2] - b[0]) * (b[3] - b[1]))
        if best is None or key > best[0]:
            best = (key, b)

    if best is None:
        return base, cover(base), len(blockers), holds_head(base)
    b = best[1]
    left = sum(1 for o in other_heads if frac_in(o, b) > max(slack, 0.05))
    return b, cover(b), left, holds_head(b)


def other_heads(heads, ti, omasks, thead, min_on_other=0.30, max_iou_target=0.40):
    """Which detected heads really belong to other people.

    Two kinds of noise make a head look like a bystander's when it is not, and both put a
    head inside the box that no cut can ever remove:
      * the detector fires twice on the target's own head -> high IoU with the target head;
      * a false positive lands on the target's shoulder   -> it sits on no other person.
    A real bystander head sits on a person instance that is not the target.
    """
    out = []
    for k, h in enumerate(heads):
        if k == ti:
            continue
        if thead is not None:
            ix = max(0.0, min(h[2], thead[2]) - max(h[0], thead[0]))
            iy = max(0.0, min(h[3], thead[3]) - max(h[1], thead[1]))
            inter = ix * iy
            uni = ((h[2]-h[0])*(h[3]-h[1]) + (thead[2]-thead[0])*(thead[3]-thead[1]) - inter)
            if inter / max(1e-6, uni) > max_iou_target:
                continue
        if omasks is not None and len(omasks):
            if max(head_mask_overlap(h, om) for om in omasks) < min_on_other:
                continue
        elif omasks is not None:
            continue
        out.append(h)
    return out

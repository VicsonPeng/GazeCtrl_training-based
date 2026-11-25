"""Run the idiap/gaze3d Gaze Transformer over a list of images, one row per detected head.

The repository ships a demo that detects heads with a yolov5-crowdhuman model, tracks them
with boxmot and renders a video. None of that is wanted here: this project already has a head
detector, and what is needed is the raw prediction for a head box we choose. So the model and
its preprocessing are imported from the repository unchanged -- same BboxReshape(square,
ratio=-0.1), same 224 crop, same ImageNet normalisation, same single-frame path through the
temporal model -- and only the surrounding plumbing is replaced. Anything that changed the
preprocessing would invalidate the comparison, so nothing here touches it.

Output convention, read off the repository's own renderer (demo.py:draw_arrow2D and the
colour map in draw_gaze):
    screen dx = -gaze[0],  screen dy = -gaze[1],  and gaze (0,0,-1) is "looking at the camera"
so +x is LEFT in the image, +y is UP, and -z points towards the camera. The conversion into
this project's own convention is NOT applied here; it is measured, in gaze3d_validate.py.
"""
import sys as _sys, os as _os
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != '/' and not _os.path.exists(_os.path.join(_p, 'gaze_paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p); _sys.path.insert(0, _os.path.join(_p, 'src'))
from gaze_paths import (ROOT, DATASET, RAW, MODELS, WORK, OUT, THIRD, EVAL, CORPORA,
                        GT_EVAL, SIXDREP_DIR, SIXDREP_CKPT, YOLO_HEAD, YOLO_SEG,
                        GAZE3D_DIR, WAN_DIR)

import argparse, os, sys, json
import numpy as np, pandas as pd
import torch
from PIL import Image

G3D = GAZE3D_DIR
sys.path.insert(0, G3D)
from functools import partial
from src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D       # noqa: E402
from src.data.components.transforms import (                          # noqa: E402
    BboxReshape, Concatenate, Crop, Normalize, ToImage, ToTensor)
from torchvision.transforms import Compose                            # noqa: E402

IMG_MEAN = [0.485, 0.456, 0.406]; IMG_STD = [0.229, 0.224, 0.225]
TRANSFORM = Compose([BboxReshape(square=True, ratio=-0.1), ToImage(), Crop(output_size=224),
                     Concatenate(), ToTensor(), Normalize(mean=IMG_MEAN, std=IMG_STD)])
CKPT = {'gf': f'{G3D}/checkpoints/gat_stwsge_gaze360_gf.ckpt',
        'g360': f'{G3D}/checkpoints/gat_gaze360.ckpt'}


def load_model(which='gf', device='cuda'):
    model = GaT(encoder=Swin3D(pretrained=False),
                head_dict=HeadDict(names=['gaze'],
                                   modules=[partial(MLPHead, hidden_dim=256, num_layers=1,
                                                    out_features=3)]))
    sd = torch.load(CKPT[which], map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, items, device='cuda', batch=16):
    """items: list of (image_path_or_array, head_box). Returns an (N,3) unit-vector array."""
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        xs = []
        for img, box in chunk:
            if isinstance(img, str):
                img = np.array(Image.open(img).convert('RGB'), copy=True)
            s = TRANSFORM({'images': [img], 'head_bbox': [list(box)]})
            xs.append(s['images'])
        x = torch.stack(xs).to(device)
        p = model(x)['gaze']
        p = torch.nn.functional.normalize(p, p=2, dim=2, eps=1e-8)
        t = p.size(1); t = (t // 2) if t == 1 or t % 2 != 0 else (t // 2) - 1
        out.append(p[:, t, :].float().cpu().numpy())
    return np.vstack(out)


_YOLO = {}


def head_boxes(path, device='cuda', conf=0.25):
    """Head boxes from this project's own detector, largest first."""
    from ultralytics import YOLO
    if 'm' not in _YOLO:
        _YOLO['m'] = YOLO(YOLO_HEAD)
    r = _YOLO['m'].predict(path, verbose=False, device=device, conf=conf)[0].boxes
    if r is None or len(r) == 0: return []
    xy = r.xyxy.cpu().numpy()
    order = np.argsort(-(xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1]))
    return [tuple(float(v) for v in xy[j]) for j in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True, help='text file of image paths, or a .json list of {path, box}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='gf', choices=['gf', 'g360'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--all_heads', type=int, default=0, help='1 keeps every head, 0 the largest only')
    a = ap.parse_args()

    if a.list.endswith('.json'):
        spec = json.load(open(a.list))
        items = [(s['path'], s['box']) for s in spec]
        keys = [s.get('key', s['path']) for s in spec]
    else:
        paths = [l.strip() for l in open(a.list) if l.strip()]
        items, keys = [], []
        for p in paths:
            bs = head_boxes(p, device=a.device)
            if not bs: continue
            for j, b in enumerate(bs if a.all_heads else bs[:1]):
                items.append((p, b)); keys.append(f'{p}#{j}')
        print(f'detected heads on {len(set(k.split("#")[0] for k in keys))}/{len(paths)} images', flush=True)

    model = load_model(a.ckpt, a.device)
    g = predict(model, items, a.device, a.batch)
    pd.DataFrame(dict(key=keys, path=[i[0] for i in items],
                      x0=[i[1][0] for i in items], y0=[i[1][1] for i in items],
                      x1=[i[1][2] for i in items], y1=[i[1][3] for i in items],
                      gx=g[:, 0], gy=g[:, 1], gz=g[:, 2])).to_csv(a.out, index=False)
    print(f'wrote {len(g)} predictions -> {a.out}')


if __name__ == '__main__':
    main()

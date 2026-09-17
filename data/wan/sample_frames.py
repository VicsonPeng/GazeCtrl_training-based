#!/usr/bin/env python
"""
Propose angularly-diverse candidate frames from a combined-gaze video.

Reads the per-frame gaze vectors (combined _vectors.txt), then runs
farthest-point sampling in gaze-vector space so the chosen frames span as many
distinct directions as possible (not clustered at whatever angle the video
lingered on). Outputs a montage of the candidates (with the drawn arrow) for a
human/Claude to confirm accuracy, plus the candidate frame indices.

Usage: sample_frames.py --viz viz_hc_<tag>.mp4 --vectors viz_hc_<tag>_vectors.txt
                        --out_prefix _sample/<tag> --n_candidates 16
"""
import argparse, os
import numpy as np
import cv2


def load_vectors(path):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            out.append([float(v) for v in ln.replace(",", " ").split()])
    return np.array(out)


def farthest_point_sample(vecs, idxs, k):
    # greedy: start from the most "extreme" vector, repeatedly add the frame
    # whose gaze is farthest (min-distance) from the already-selected set.
    if len(idxs) <= k:
        return list(idxs)
    pts = vecs[idxs]
    start = int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))
    chosen = [start]
    d = np.linalg.norm(pts - pts[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return [int(idxs[c]) for c in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--n_candidates", type=int, default=16)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    vecs = load_vectors(args.vectors)
    valid = [i for i in range(len(vecs)) if not (vecs[i] == 0).all()]
    cand = farthest_point_sample(vecs, valid, args.n_candidates)

    cap = cv2.VideoCapture(args.viz)
    frames = {}
    fi = 0
    want = set(cand)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fi in want:
            # crop top 38% (head region), label frame idx + gaze
            h, w = f.shape[:2]
            crop = f[: int(h * 0.24)]
            crop = cv2.resize(crop, (300, int(crop.shape[0] * 300 / w)))
            gx, gy, gz = vecs[fi]
            cv2.putText(crop, f"f{fi} ({gx:+.2f},{gy:+.2f},{gz:+.2f})", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, f"f{fi} ({gx:+.2f},{gy:+.2f},{gz:+.2f})", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            frames[fi] = crop
        fi += 1
    cap.release()

    # build montage grid
    cols = args.cols
    cand = [c for c in cand if c in frames]
    rows = (len(cand) + cols - 1) // cols
    hh = max(im.shape[0] for im in frames.values())
    ww = 300
    grid = np.full((rows * hh, cols * ww, 3), 255, np.uint8)
    for n, c in enumerate(cand):
        r, cc = divmod(n, cols)
        im = frames[c]
        grid[r * hh:r * hh + im.shape[0], cc * ww:cc * ww + im.shape[1]] = im
    cv2.imwrite(f"{args.out_prefix}_cands.png", grid)
    open(f"{args.out_prefix}_cands.txt", "w").write(
        "\n".join(f"{c} {vecs[c][0]:+.3f} {vecs[c][1]:+.3f} {vecs[c][2]:+.3f}" for c in cand))
    print(f"{len(cand)} candidates -> {args.out_prefix}_cands.png")


if __name__ == "__main__":
    main()

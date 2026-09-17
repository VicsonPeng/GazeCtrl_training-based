"""
HSV Color Wheel Gaze Condition Renderer

Encodes a unit gaze vector (dx, dy, dz) as a solid-color H×W×3 BGR image.

Encoding (optical-flow HSV convention):
  H = atan2(dy, dx)         → azimuth in image space  [0, 179] OpenCV
  S = sqrt(dx² + dy²)       → 2-D magnitude           [0, 255]
  V = max(0.3, dz) × 255    → depth component         [77, 255]  (avoids pure black)

Coordinate system:
  dx > 0  →  look right
  dy > 0  →  look down (image space)
  dz > 0  →  look toward camera

The entire condition image is one flat color — no spatial blobs.
ControlNet already sees the original image via TextEncodeQwenImageEdit; the
condition only needs to encode WHAT direction the gaze points.

Usage
-----
  from gaze_hsv_renderer import GazeHSVRenderer
  renderer = GazeHSVRenderer(height=1024, width=1024)
  bgr = renderer.render(dx=0.0, dy=0.0, dz=1.0)   # pure white (frontal)

  # CLI self-test
  python gaze_hsv_renderer.py --test
"""

import argparse
import math
import os

import cv2
import numpy as np


class GazeHSVRenderer:
    def __init__(self, height: int = 1024, width: int = 1024):
        self.height = height
        self.width = width

    def render(self, dx: float, dy: float, dz: float) -> np.ndarray:
        """Return (H, W, 3) uint8 BGR image filled with the gaze HSV color."""
        dx, dy, dz = self._normalize(dx, dy, dz)

        hue_cv = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)   # [0, 179]
        sat    = int(math.sqrt(dx ** 2 + dy ** 2) * 255)              # [0, 255]
        val    = int(max(0.3, dz) * 255)                               # [77, 255]

        hsv = np.full((self.height, self.width, 3),
                      [hue_cv, sat, val], dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def render_rgb(self, dx: float, dy: float, dz: float) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB image."""
        return cv2.cvtColor(self.render(dx, dy, dz), cv2.COLOR_BGR2RGB)

    def render_match_image(
        self,
        reference_image_bgr: np.ndarray,
        dx: float, dy: float, dz: float,
    ) -> np.ndarray:
        """Render at the same resolution as reference_image_bgr."""
        h, w = reference_image_bgr.shape[:2]
        self.height, self.width = h, w
        return self.render(dx, dy, dz)

    @staticmethod
    def _normalize(dx, dy, dz):
        norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if norm < 1e-8:
            return 0.0, 0.0, 1.0
        return dx / norm, dy / norm, dz / norm

    @staticmethod
    def hsv_pixel(dx: float, dy: float, dz: float) -> tuple[int, int, int]:
        """Return (H_cv, S, V) uint8 for a gaze vector (no image allocation)."""
        dx, dy, dz = GazeHSVRenderer._normalize(dx, dy, dz)
        h = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)
        s = int(math.sqrt(dx ** 2 + dy ** 2) * 255)
        v = int(max(0.3, dz) * 255)
        return h, s, v


# ---------------------------------------------------------------------------
# CLI self-test: renders 8 canonical directions and saves a color-wheel strip
# ---------------------------------------------------------------------------
def _self_test(out_dir: str = "."):
    cases = [
        ("frontal",    0.000,  0.000,  1.000),
        ("right_90",   1.000,  0.000,  0.000),
        ("left_90",   -1.000,  0.000,  0.000),
        ("up_90",      0.000, -1.000,  0.000),
        ("down_90",    0.000,  1.000,  0.000),
        ("back",       0.000,  0.000, -1.000),
        ("right_45",   0.707,  0.000,  0.707),
        ("left_front",-0.500,  0.000,  0.866),
    ]

    tile_size = 128
    renderer = GazeHSVRenderer(tile_size, tile_size)
    strip = []
    for name, dx, dy, dz in cases:
        bgr = renderer.render(dx, dy, dz)
        h, s, v = GazeHSVRenderer.hsv_pixel(dx, dy, dz)
        print(f"  {name:12s}  dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}"
              f"  →  HSV({h:3d}, {s:3d}, {v:3d})  BGR={tuple(bgr[0,0])}")
        strip.append(bgr)

    row = np.concatenate(strip, axis=1)
    out = os.path.join(out_dir, "gaze_hsv_test_strip.png")
    cv2.imwrite(out, row)
    print(f"\nSaved color strip → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width",  type=int, default=256)
    ap.add_argument("--output", default="gaze_condition.png")
    args = ap.parse_args()

    if args.test:
        _self_test(args.out_dir)
    else:
        r = GazeHSVRenderer(args.height, args.width)
        img = r.render(args.dx, args.dy, args.dz)
        cv2.imwrite(args.output, img)
        h, s, v = GazeHSVRenderer.hsv_pixel(args.dx, args.dy, args.dz)
        print(f"Saved {args.output}  HSV=({h},{s},{v})")

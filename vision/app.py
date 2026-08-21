#!/usr/bin/env python3
"""Overhead tracker.

    python -m vision.app --source synthetic          # no camera, no robots
    python -m vision.app --source 0                  # your webcam
    python -m vision.app --source 0 --calibrate      # click 4 corners first
    python -m vision.app --source 0 --tune           # tune colour thresholds
    python -m vision.app --source synthetic --selftest

Keys while running: c recalibrate, t tune, m toggle mask view, q quit.
"""

import argparse
import time

import cv2
import numpy as np

from . import config
from .detect import Detector, tune
from .homography import Homography, calibrate
from .synthetic import SyntheticSource, open_source
from .track import Tracker

CHALK = (228, 240, 248)
DIM = (204, 179, 143)
CYAN = (232, 210, 99)
CORAL = (107, 107, 255)


def draw_overlay(frame, tracker, raw, show_grid=True):
    vis = frame.copy()
    H = tracker.H

    if H.ready and show_grid:
        a = tracker.H.arena
        for i in range(0, 5):
            f = i * a / 4
            p = H.to_px([[0, f], [a, f]]).astype(int)
            q = H.to_px([[f, 0], [f, a]]).astype(int)
            cv2.line(vis, tuple(p[0]), tuple(p[1]), (70, 66, 60), 1)
            cv2.line(vis, tuple(q[0]), tuple(q[1]), (70, 66, 60), 1)
        box = H.to_px([[0, 0], [a, 0], [a, a], [0, a]]).astype(int)
        cv2.polylines(vis, [box], True, (110, 104, 96), 2)

    for name, (x, y, area) in raw.items():
        col = config.COLORS[tracker.tracks[name].color]["draw"] \
            if name in tracker.tracks else CHALK
        cv2.circle(vis, (int(x), int(y)), int(np.sqrt(area / np.pi)) + 4, col, 1)

    for name, t in tracker.tracks.items():
        col = config.COLORS[t.color]["draw"]
        p = t.kf.pos
        px = H.to_px([p])[0] if H.ready else p
        px = tuple(int(v) for v in px)
        cv2.circle(vis, px, 5, col, -1)
        if t.missing > 0:
            cv2.circle(vis, px, 13, CORAL, 1)
        label = f"{name} {p[0]:.0f},{p[1]:.0f}" if H.ready else name
        cv2.putText(vis, label, (px[0] + 12, px[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        v = t.kf.vel
        if H.ready and np.linalg.norm(v) > 3:
            tip = H.to_px([p + v * 0.4])[0]
            cv2.arrowedLine(vis, px, tuple(int(z) for z in tip), col, 1, tipLength=0.3)

    bar = [f"{tracker.fps:4.1f} fps",
           f"tracked {len(tracker.read())}/{len(tracker.assignment)}",
           "cm" if H.ready else "PIXELS - press c to calibrate"]
    cv2.putText(vis, "   ".join(bar), (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, CHALK if H.ready else CORAL, 1)
    cv2.putText(vis, "c calibrate   t tune   m mask   q quit", (14, vis.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1)
    return vis


def selftest(n=5, frames=300):
    """Run against synthetic ground truth and report tracking error in cm."""
    src = SyntheticSource(n=n)
    h = Homography()
    h.set_corners(src.true_corners)
    tr = Tracker(homography=h, detector=Detector())
    tr.assignment = {c: c for c in list(config.COLORS)[:n]}

    errs, misses = [], 0
    for i in range(frames):
        ok, frame = src.read()
        pos, _ = tr.step(frame)
        truth = src.truth()
        if i < 20:
            continue
        for name, p in truth.items():
            if name in pos:
                errs.append(float(np.linalg.norm(pos[name] - p)))
            else:
                misses += 1

    errs = np.array(errs) if errs else np.array([np.nan])
    total = (frames - 20) * n
    print(f"frames {frames}  robots {n}")
    print(f"  tracked      {100*(1-misses/total):5.1f}% of robot-frames")
    print(f"  mean error   {errs.mean():5.2f} cm")
    print(f"  median       {np.median(errs):5.2f} cm")
    print(f"  95th pct     {np.percentile(errs,95):5.2f} cm")
    print(f"  worst        {errs.max():5.2f} cm")
    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--robots", type=int, default=5)
    a = p.parse_args()

    if a.selftest:
        selftest(n=a.robots)
        return

    src = open_source(a.source)
    det = Detector()
    H = Homography.load()

    if isinstance(src, SyntheticSource) and not H.ready:
        H = Homography()
        H.set_corners(src.true_corners)     # synthetic knows its own geometry

    if a.calibrate or not H.ready:
        got = calibrate(src)
        if got:
            H = got
    if a.tune:
        det = tune(src, det)

    tracker = Tracker(homography=H, detector=det)
    tracker.assignment = {c: c for c in list(config.COLORS)[:a.robots]}

    win = "tracker"
    cv2.namedWindow(win)
    show_mask = False
    t0, n = time.time(), 0

    while True:
        ok, frame = src.read()
        if not ok:
            break
        pos, raw = tracker.step(frame)
        n += 1

        if show_mask:
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            m = np.zeros(frame.shape[:2], np.uint8)
            for c in tracker.colors:
                m = cv2.bitwise_or(m, det.mask_for(hsv, c))
            vis = cv2.bitwise_and(frame, frame, mask=m)
        else:
            vis = draw_overlay(frame, tracker, raw)
        cv2.imshow(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            got = calibrate(src)
            if got:
                tracker.H = got
        elif k == ord("t"):
            det = tune(src, det)
            tracker.det = det
        elif k == ord("m"):
            show_mask = not show_mask

    src.release()
    cv2.destroyAllWindows()
    print(f"{n} frames, {n/(time.time()-t0):.1f} fps average")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Why is the tracker not seeing my robot?

    python -m vision.diagnose --source 0
    python -m vision.diagnose --source 0 --colors cyan,red,yellow

Grabs one frame and reports, per colour, exactly which gate the robot failed:
hue, saturation, value or blob area. "Not detected" on its own sends you
adjusting things at random — a lit ball can fail for four different reasons and
three of them look identical in the preview.

The usual answer on a bright floor is saturation: auto-exposure meters for the
white surface, the LED core blows out to near-white, and white has no hue at
all. That is what `s_min` rejects.
"""

import argparse

import cv2
import numpy as np

from . import config
from .detect import Detector
from .synthetic import open_source


def brightest_blob(hsv, frame, min_v=200):
    """The most likely LED: the brightest compact bright region."""
    v = hsv[:, :, 2]
    mask = (v >= min_v).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    return cx, cy, float(cv2.contourArea(c))


def ring_sample(hsv, cx, cy, r_in=6, r_out=18):
    """HSV around the core, where the usable colour actually is."""
    h, w = hsv.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    if not ring.any():
        return None
    px = hsv[ring]
    return (float(np.median(px[:, 0])), float(np.median(px[:, 1])),
            float(np.median(px[:, 2])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0")
    p.add_argument("--colors", default=None, help="comma separated; default all")
    p.add_argument("--warmup", type=int, default=15,
                   help="frames to discard while auto-exposure settles")
    a = p.parse_args(argv)

    src = open_source(a.source)
    frame = None
    for _ in range(max(1, a.warmup)):
        ok, f = src.read()
        if ok:
            frame = f
    src.release()
    if frame is None:
        print("no frame from the camera")
        return 1

    det = Detector()
    t = det.thresh
    blur = cv2.GaussianBlur(frame, (t["blur"] | 1, t["blur"] | 1), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    print(f"frame {frame.shape[1]}x{frame.shape[0]}")
    print(f"thresholds: s_min={t['s_min']} v_min={t['v_min']} "
          f"min_area={t['min_area']} max_area={t['max_area']}")

    spot = brightest_blob(hsv, frame)
    if spot:
        cx, cy, area = spot
        core = hsv[cy, cx]
        print(f"\nbrightest spot at ({cx},{cy}), area {area:.0f}px")
        print(f"  core  H={core[0]:3d} S={core[1]:3d} V={core[2]:3d}"
              + ("   <-- blown out: S below "
                 f"{t['s_min']} is colourless" if core[1] < t["s_min"] else ""))
        ring = ring_sample(hsv, cx, cy)
        if ring:
            print(f"  ring  H={ring[0]:3.0f} S={ring[1]:3.0f} V={ring[2]:3.0f}"
                  "   <-- this is the signal the detector wants")
            for name, c in config.COLORS.items():
                lo, hi = c["hue"] - c["tol"], c["hue"] + c["tol"]
                near = lo <= ring[0] <= hi or (c["hue"] == 0 and ring[0] >= 172)
                if near:
                    print(f"        hue {ring[0]:.0f} matches '{name}' "
                          f"({c['hue']}±{c['tol']})")
    else:
        print("\nno bright spot at all — is the LED on? try: set_led cyan SSMK")

    wanted = ([c.strip() for c in a.colors.split(",")] if a.colors
              else list(config.COLORS))
    print("\nper colour:")
    for name in wanted:
        if name not in det.colors:
            print(f"  {name:<8} unknown colour")
            continue
        mask = det.mask_for(hsv, name)
        hits = int((mask > 0).sum())
        found = det.detect(frame, only=[name])
        if name in found:
            x, y, area = found[name]
            print(f"  {name:<8} DETECTED at ({x:.0f},{y:.0f}) area {area:.0f}px")
        elif hits == 0:
            print(f"  {name:<8} no pixels pass hue+saturation")
        else:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            biggest = max((cv2.contourArea(c) for c in cnts), default=0.0)
            why = ("blob too small" if biggest < t["min_area"]
                   else "blob too large" if biggest > t["max_area"] else "?")
            print(f"  {name:<8} {hits} px pass, biggest blob {biggest:.0f}px "
                  f"— rejected: {why} (min_area={t['min_area']})")

    print("\nif saturation is the problem, in order of what usually works:")
    print("  1. lower the room lights, or point the camera away from the window")
    print("  2. python -m vision.app --source 0 --tune   (drag s_min down, 's' saves)")
    print("  3. reduce camera exposure if your driver exposes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does this camera let you set the exposure at all, and how?

    python -m vision.expose --camera 0

An exposure slider that does nothing is the most common wall in this project's
bring-up, and it has three quite different causes that look identical from
behind the UI:

    the property is ignored      the backend accepts the write and drops it
    auto-exposure is still on    it accepts the write and immediately overrides
    the value is out of range    it accepts a number the camera cannot use

Reading the property back cannot tell these apart, because a camera that
ignored a write will happily report the value you asked for. The only thing
that settles it is whether the PICTURE changed -- so this sets each
combination, grabs frames, and measures the mean brightness.

`CAP_PROP_AUTO_EXPOSURE` deserves its own warning. The 0.25-means-manual
convention is V4L2's. macOS AVFoundation, DirectShow and several UVC drivers
each use something different, and 0.25 on the wrong one is silently a no-op --
which is a slider that moves and a picture that does not.
"""

import argparse
import time

import cv2
import numpy as np

# The manual-mode values worth trying, and where each convention comes from.
AUTO_VALUES = [(0.25, "V4L2 manual"), (0.0, "off"),
               (1.0, "manual on some UVC"), (3.0, "auto, for comparison")]
# Two ranges, because cameras disagree. Negative values are log2 seconds;
# 0-255 is the other common scheme, and a camera using one ignores the other.
LOG_RANGE = [-1, -4, -7, -10]
ABS_RANGE = [5, 40, 120, 250]
SETTLE_S = 0.45
GRABS = 4
# A lit floor sits near 150 and a usable one near 30, so a control worth having
# moves the mean by most of that. Anything less cannot do the job however
# real the effect is.
USABLE = 60.0
FLICKER = 8.0


def mean_brightness(cap):
    """Mean V after letting the camera settle. Frames are dropped first
    because a change lands several frames later on a buffered stream."""
    time.sleep(SETTLE_S)
    last = None
    for _ in range(GRABS):
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
    if last is None:
        return None
    return float(cv2.cvtColor(last, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def probe(index=0, width=None, height=None):
    cap = cv2.VideoCapture(index)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"camera {index} would not open. On macOS, grant camera access "
              "to your terminal in System Settings > Privacy & Security.")
        return 1
    print(f"camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
          f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
          f"backend {cap.getBackendName()}")
    base = mean_brightness(cap)
    print(f"baseline mean brightness {base:.1f}\n")

    worked = []
    for auto, label in AUTO_VALUES:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
        got = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"auto_exposure = {auto}  ({label}) — reads back {got}")
        for name, values in (("exposure(log2)", LOG_RANGE),
                             ("exposure(abs)", ABS_RANGE)):
            means = []
            for v in values:
                cap.set(cv2.CAP_PROP_EXPOSURE, v)
                m = mean_brightness(cap)
                means.append(m)
            if any(m is None for m in means):
                print(f"    {name:>15}: no frames")
                continue
            spread = max(means) - min(means)
            row = "  ".join(f"{v}->{m:5.1f}" for v, m in zip(values, means))
            # Graded against what the JOB needs, not against zero. Getting a
            # lit floor down to a black one is a swing of 100+ counts; a
            # control that moves the mean by nine is technically responding
            # and useless, and calling that a success sends somebody off to
            # build on a lever that cannot carry them.
            if spread >= USABLE:
                verdict = "USABLE"
            elif spread >= FLICKER:
                verdict = f"responds, but {spread:.0f} counts is far too little"
            else:
                verdict = "no effect"
            print(f"    {name:>15}: {row}   spread {spread:5.1f}  {verdict}")
            if spread >= USABLE:
                worked.append((auto, label, name, spread))
        print()

    cap.release()
    if worked:
        best = max(worked, key=lambda r: r[3])
        print(f"USE THIS: auto_exposure={best[0]} ({best[1]}) with {best[2]}, "
              f"which moved the mean by {best[3]:.0f}")
    else:
        print("Nothing moved the picture. This camera does not expose its "
              "exposure through this backend.\n"
              "Darken the room instead: the LEDs emit and the floor reflects, "
              "so ambient light is the only thing you lose.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--size", default=None, help="e.g. 1920x1080")
    a = p.parse_args(argv)
    w = h = None
    if a.size:
        w, h = (int(v) for v in a.size.lower().split("x"))
    return probe(a.camera, w, h)


if __name__ == "__main__":
    raise SystemExit(main())

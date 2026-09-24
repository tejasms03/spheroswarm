#!/usr/bin/env python3.13
"""Check the arena's cm scale against a tape measure.

Every margin the bench works in is a centimetre: how close two balls may
pass, how far an orbit sits from its centre, how near "arrived" is. All of
them come from four corner clicks and two typed numbers, and nothing has
ever checked those two numbers against the floor.

The measurement across 197 recorded runs says they are wrong: a ball reads
about 1.28x faster along y than along x, the same on every ball and in every
part of the arena. That is the signature of a declared arena whose shape
does not match the taped one. It cannot say WHICH number is off, because a
speed only ever gives a ratio. One tape measure settles it.

How to use it
-------------
1. Tape two marks exactly 100 cm apart along the arena's x axis (left to
   right, as the camera sees it).
2. Press `r` in the bench to start recording tracks.
3. Sit a ball on the first mark. Leave it still for 5 seconds.
4. Move it to the second mark. Leave it still for 5 seconds.
5. Repeat along the y axis (top to bottom).
6. Press `r` again, then run:

       python3.13 check_scale.py

It finds the still spots in the newest recording and prints what the bench
thought the distance was. Pass a file to read a different one, and --true to
use a spacing other than 100 cm.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

STILL_S = 1.5      # a spot has to be held at least this long
STILL_CM = 2.0     # and move less than this while held
GAP_CM = 20.0      # two spots must be at least this far apart to count


def load(path):
    """Every seen frame of every ball: (t, name, x_cm, y_cm)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            balls = row.get("balls")
            if not isinstance(balls, dict):
                continue
            for name, ball in balls.items():
                xy = ball.get("cm")
                if xy and not ball.get("lost"):
                    out.append((row.get("t", 0.0), name,
                                float(xy[0]), float(xy[1])))
    return out


def still_spots(rows):
    """Places the ball was parked, in order: (x, y, seconds held)."""
    spots, run = [], []

    def close_run():
        if len(run) < 3:
            return
        held = run[-1][0] - run[0][0]
        if held < STILL_S:
            return
        xs = [r[2] for r in run]
        ys = [r[3] for r in run]
        if max(xs) - min(xs) > STILL_CM or max(ys) - min(ys) > STILL_CM:
            return
        spots.append((st.median(xs), st.median(ys), held))

    for r in rows:
        if run:
            xs = [q[2] for q in run] + [r[2]]
            ys = [q[3] for q in run] + [r[3]]
            if max(xs) - min(xs) > STILL_CM or max(ys) - min(ys) > STILL_CM:
                close_run()
                run = []
        run.append(r)
    close_run()
    return spots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="a runs/tracks/*.jsonl file")
    ap.add_argument("--true", type=float, default=100.0, dest="truth",
                    help="the taped distance in cm (default 100)")
    args = ap.parse_args()

    path = args.path
    if path is None:
        got = sorted(glob.glob("runs/tracks/*.jsonl"), key=os.path.getmtime)
        if not got:
            raise SystemExit(
                "no recordings in runs/tracks/ — press `r` in the bench, "
                "park the ball on each tape mark, then press `r` again")
        path = got[-1]

    rows = load(path)
    if not rows:
        raise SystemExit(f"{path} has no tracked frames in it")
    print(f"{path}: {len(rows)} tracked frames")

    # One ball carries the tape measure. Mixing two balls' frames would read
    # the distance between the balls as a move.
    names = {}
    for r in rows:
        names[r[1]] = names.get(r[1], 0) + 1
    name = max(names, key=names.get)
    if len(names) > 1:
        others = ", ".join(f"{k} {v}" for k, v in names.items() if k != name)
        print(f"using {name} ({names[name]} frames); ignoring {others}")
    rows = [r for r in rows if r[1] == name]

    spots = still_spots(rows)
    if len(spots) < 2:
        raise SystemExit(
            f"found {len(spots)} still spot(s) — park the ball on each mark "
            f"and leave it alone for a good 5 seconds")

    print(f"\n{len(spots)} still spots:")
    for i, (x, y, held) in enumerate(spots):
        print(f"  {i + 1}. ({x:6.1f}, {y:6.1f}) cm   held {held:.1f}s")

    print(f"\nlegs between them, against a taped {args.truth:.0f} cm:")
    seen = False
    for i in range(len(spots) - 1):
        ax, ay, _ = spots[i]
        bx, by, _ = spots[i + 1]
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        if d < GAP_CM:
            continue
        seen = True
        axis = "x" if abs(dx) > abs(dy) * 2 else (
            "y" if abs(dy) > abs(dx) * 2 else "diagonal")
        off = math.degrees(math.atan2(dx, dy)) % 180
        print(f"  {i + 1} -> {i + 2}: bench reads {d:6.1f} cm along {axis} "
              f"({off:.0f} deg)   error {100 * (d / args.truth - 1):+5.1f}%")
        if axis in ("x", "y"):
            print(f"      {axis} scale is off by x{args.truth / d:.3f} — "
                  f"multiply the arena's {axis} size by that and re-pick "
                  f"the corners")

    if not seen:
        raise SystemExit(
            f"no two spots more than {GAP_CM:.0f} cm apart — the ball has to "
            f"actually move between the marks")

    cal = "calib/homography.json"
    if os.path.exists(cal):
        with open(cal) as fh:
            h = json.load(fh)
        print(f"\ndeclared arena: {h.get('width')} x {h.get('height')} cm")
        print("the bench takes these from what was typed when the corners "
              "were picked — correct them there and pick the corners again")


if __name__ == "__main__":
    main()

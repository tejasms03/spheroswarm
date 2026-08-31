#!/usr/bin/env python3
"""Generate matched pairs of test frames: the exposure, and the gate.

    python -m vision.shots --pairs 10 --out runs/shots

Every pair is ONE scene rendered once in HDR and read twice, so a ball sits at
the same pixel in both images. That is what makes them a test rather than two
pictures: any method that reads a heading from one can be scored against the
same truth in the other, and the truth is written alongside as JSON.

The two readings model what a camera really does.

    EXPOSURE  clips. The LED cores are orders of magnitude brighter than the
              shell, so they saturate into the same flat colour as the lobes
              around them: two overlapping discs and a mixed lens, with no
              trace of where the lights are. Colour survives; peaks do not.
    GATE      reads the intensity underneath. The lobes fall below the gate,
              only the cores clear it, and the three collinear dots come back.
              Peaks survive; colour does not, because the false-colour map
              throws the hue away.
    SHORT     the one you would actually shoot. Cores sharp AND still coloured,
              lobes and floor gone dark -- so the dots give a position and an
              axis, and the colour at the ends says which way along it the
              robot points. Both halves in one frame, and it freezes motion.

Scale comes from the workspace rather than from taste: the arena is rendered at
`PX_CM` pixels per centimetre, so a 7.4cm shell is ~68px and the two LEDs sit
~20px apart -- comfortably above the 3px floor `vision/facing.py` needs, and
close to what the real camera delivers over this arena.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import config

PX_CM = 9.2
BALL_CM = 7.4
# The real arrangement, and it is not one LED plus a reflection. TWO tag-colour
# LEDs sit equidistant either side of the centre, and the blue aiming light
# sits behind them -- so the three dots along the axis are
#
#     front tag LED  ...  CENTRE  ...  rear tag LED  ...  taillight
#
# which makes the centre the MIDPOINT OF THE TWO SAME-COLOURED DOTS rather than
# anything that has to be estimated. Both distances want measuring on a real
# frame; these are placeholders and every geometric result scales off them.
TAG_R_CM = 1.1                  # each tag LED, from the centre along the axis
TAIL_R_CM = 3.3                 # the taillight, behind the centre
# Why 3.3 and not 2.2. The three dots photograph EVENLY SPACED, and that fixes
# the taillight's distance once the tag pair is chosen: tags at +a and -a means
# the gap between them is 2a, so an equal gap behind puts the tail at -3a. With
# 2.2 the rear gap was half the front one -- 4px at the real camera scale, under
# what peak-finding can split -- and the three-dot method appeared to die at
# range. It was this number dying, not the method.
SEP_CM = TAG_R_CM               # kept for callers that still ask
TAIL_BGR = (255, 120, 40)       # blue, and the same on every ball
CEILING = 255.0                 # the sensor's ceiling; everything above clips
GATE_HI = 6000.0
GATE_GAMMA = 0.7
# Blue is the taillight and may not also be a tag, or a ball's nose and tail
# are the same colour and there is no heading to read.
TAGS = [n for n in config.COLORS if n != "blue"]


def tag_bgr(name):
    hsv = np.uint8([[[config.COLORS[name]["hue"], 235, 250]]])
    return tuple(float(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


REACH = 3.2             # radii out to which a glow is still worth adding


def _glow(canvas, centre, radius, bgr, amp, reach=REACH):
    """One light, added only where it can actually be seen.

    A gaussian never reaches zero, so the honest version of this evaluates it
    over the whole frame -- and at thirty lights a frame that is thirty full
    -- canvas exponentials, which is fine for writing files and far too slow to
    watch. Past three radii the term is under a thousandth of the peak and
    disappears into the noise floor anyway, so the tail is dropped and the
    render runs in a box around each light instead.
    """
    h, w = canvas.shape[:2]
    r = int(math.ceil(radius * reach))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    y = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(centre[1])
    x = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(centre[0])
    fall = np.exp((x * x + y * y) * np.float32(-1.0 / (2.0 * radius ** 2)),
                  dtype=np.float32) * np.float32(amp)
    tint = np.array([bgr[0], bgr[1], bgr[2]], dtype=np.float32) / np.float32(255.0)
    canvas[y0:y1, x0:x1] += fall[:, :, None] * tint


def place(rng, n, arena_w, arena_h, margin=16.0, apart=26.0, tries=400):
    """`n` positions in centimetres, none of them touching.

    Overlapping balls are a different problem -- one blob with two robots in it
    -- and mixing that into a heading test would score the heading code for a
    failure that belongs to the detector.
    """
    out = []
    for _ in range(n):
        for _ in range(tries):
            p = np.array([rng.uniform(margin, arena_w - margin),
                          rng.uniform(margin, arena_h - margin)])
            if all(np.linalg.norm(p - q) >= apart for q in out):
                out.append(p)
                break
        else:
            raise RuntimeError("no room left for another ball — fewer bots, "
                               "or a bigger arena")
    return out


def scene(rng, n_bots, arena_w, arena_h):
    """One randomised layout: where, which way, and which robot."""
    tags = list(TAGS)
    rng.shuffle(tags)
    return [{"x_cm": float(p[0]), "y_cm": float(p[1]),
             "heading_deg": float(rng.uniform(0.0, 360.0)),
             "tag": tags[i % len(tags)]}
            for i, p in enumerate(place(rng, n_bots, arena_w, arena_h))]


def draw_ball(canvas, x_cm, y_cm, heading_deg, tag, px_cm, scale=1.0):
    """One ball into an HDR canvas: two tag LEDs about the centre, blue behind.

    `scale` shrinks the ball without moving it, which is what a camera further
    back does -- the arena keeps its size and the shell loses pixels.
    """
    cx, cy = x_cm * px_cm, y_cm * px_cm
    r_px = BALL_CM * px_cm * scale / 2.0
    tag_px, tail_px = TAG_R_CM * px_cm * scale, TAIL_R_CM * px_cm * scale
    rad = math.radians(heading_deg)
    axis = np.array([math.cos(rad), math.sin(rad)])
    tag_c = tag_bgr(tag)
    mix = tuple((a + c) / 2.0 for a, c in zip(tag_c, TAIL_BGR))
    front = (cx + axis[0] * tag_px, cy + axis[1] * tag_px)
    rear = (cx - axis[0] * tag_px, cy - axis[1] * tag_px)
    tail = (cx - axis[0] * tail_px, cy - axis[1] * tail_px)

    # All three cores the same size and the same brightness, which is what a
    # real ball looks like under a brightness gate -- three alike dots, no
    # bright one and no dim one. Modelling the tail as dimmer was an invention,
    # and an expensive one: it made blue the first colour to lose its pixels as
    # the ball shrank, which is what the minimum-pixel threshold then tripped on.
    #
    # It also settles the brightness question completely. Two equal tag LEDs
    # made "the brighter core is the nose" a coin toss; three equal cores make
    # it meaningless outright. Nothing but colour can say which end is which.
    _glow(canvas, (cx, cy), r_px * 0.95, mix, 90.0)
    _glow(canvas, front, r_px * 0.68, tag_c, 600.0)
    _glow(canvas, rear, r_px * 0.68, tag_c, 600.0)
    _glow(canvas, tail, r_px * 0.68, TAIL_BGR, 600.0)
    _glow(canvas, front, max(0.7, 3.0 * scale), tag_c, 10000.0)
    _glow(canvas, rear, max(0.7, 3.0 * scale), tag_c, 10000.0)
    _glow(canvas, tail, max(0.7, 3.0 * scale), TAIL_BGR, 10000.0)
    return canvas


def render_hdr(bots, arena_w, arena_h, rng, px_cm=PX_CM):
    """Linear light, unbounded. Neither image is the truth; this is."""
    w, h = int(arena_w * px_cm), int(arena_h * px_cm)
    canvas = np.zeros((h, w, 3), np.float32)
    canvas[:] = (6.0, 5.0, 4.0)
    for cm in np.arange(20.0, arena_w, 20.0):
        cv2.line(canvas, (int(cm * px_cm), 0), (int(cm * px_cm), h), (14, 13, 12), 2)
    for cm in np.arange(20.0, arena_h, 20.0):
        cv2.line(canvas, (0, int(cm * px_cm)), (w, int(cm * px_cm)), (14, 13, 12), 2)

    for b in bots:
        draw_ball(canvas, b["x_cm"], b["y_cm"], b["heading_deg"], b["tag"], px_cm)

    return canvas + rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)


def as_exposed(hdr, ceiling=CEILING):
    """A normal frame. Clipped per PIXEL, so a core keeps its lobe's hue."""
    top = hdr.max(axis=2, keepdims=True)
    scaled = np.where(top > ceiling, hdr * (ceiling / np.maximum(top, 1e-6)), hdr)
    return np.clip(scaled, 0, 255).astype(np.uint8)


SHORT_GAIN = 1.0 / 50.0


def as_short(hdr, gain=SHORT_GAIN):
    """A SHORT exposure: the frame the dots-plus-colour method needs.

    Long enough that the cores are well clear of the noise, short enough that
    they have not clipped -- so each dot keeps the colour of the LED that made
    it, and the tag lobe and the blue tail are still distinguishable AT the
    cores rather than only in the glow.

    This is the middle setting between the other two, and on real hardware it
    is the useful one: the diffuse lobes fall to almost nothing, the floor goes
    black, and what is left is three coloured points on a dark field. It is
    also what freezes a moving ball, which is the other reason to want it.
    """
    return np.clip(hdr * gain, 0, 255).astype(np.uint8)


def as_gated(hdr, gate_hi=GATE_HI, gamma=GATE_GAMMA):
    """The brightness gate: intensity, floored hard, then false-coloured."""
    grey = 0.114 * hdr[:, :, 0] + 0.587 * hdr[:, :, 1] + 0.299 * hdr[:, :, 2]
    g = np.clip(grey / gate_hi, 0.0, 1.0) ** gamma
    return cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def make(out_dir, pairs=10, bots=5, seed=0, arena=None, px_cm=PX_CM):
    """Write `pairs` matched image pairs plus the truth they were drawn from."""
    from workspace.space import Workspace
    if arena is None:
        ws = Workspace.load()
        arena = (ws.width, ws.height)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    index = []
    for i in range(pairs):
        bots_here = scene(rng, bots, arena[0], arena[1])
        hdr = render_hdr(bots_here, arena[0], arena[1], rng, px_cm=px_cm)
        stem = f"scene{i + 1:02d}"
        cv2.imwrite(str(out / f"{stem}_exposure.png"), as_exposed(hdr))
        cv2.imwrite(str(out / f"{stem}_gate.png"), as_gated(hdr))
        cv2.imwrite(str(out / f"{stem}_short.png"), as_short(hdr))
        truth = {"scene": stem, "arena_cm": list(arena), "px_cm": px_cm,
                 "ball_cm": BALL_CM, "sep_cm": SEP_CM, "bots": bots_here}
        (out / f"{stem}.json").write_text(json.dumps(truth, indent=2) + "\n")
        index.append(truth)
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--bots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/shots")
    a = p.parse_args(argv)
    index = make(a.out, pairs=a.pairs, bots=a.bots, seed=a.seed)
    print(f"{len(index)} pairs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

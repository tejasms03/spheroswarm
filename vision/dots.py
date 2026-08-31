"""Finding lit balls, and reading a heading from their dots and their colour.

Named `dots` rather than `lighting`: `vision/lighting.py` is a different
module about whether the EXPOSURE is usable, and the two are easy to
confuse by name alone.

`vision/facing.py` reads a heading from the two brightest peaks in the patch.
That is the right instrument for a short exposure, where the LED cores are
sharp points and everything else is dark. It has nothing to work with on an
ordinary frame, because an ordinary frame CLIPS: the cores are orders of
magnitude brighter than the shell, so they saturate to the same flat value as
the lobes around them and the peak-finder sees one plateau.

The colour survives that. A ball photographed at normal exposure is two
overlapping discs -- the robot's tag colour and the blue aiming light -- and the
line between their centroids is the heading, whether or not the cores are
distinguishable. Centroids do not need a gap between the regions, which is why
the overlap that defeats the peak method does not matter here.

Reading the two methods against each other is the point of having both. They
fail in opposite conditions, so where they agree the answer is solid, and where
they disagree the disagreement itself says which one to distrust.

One rule makes this work, and it is a rule about the ROSTER rather than about
the optics: blue belongs to the taillight and to nothing else. Every ball's
tail is blue, so blue is the back, and no robot may be tagged blue -- otherwise
its nose and tail are the same colour and there is no heading to read.
"""

import math

import cv2
import numpy as np

from . import config

TAIL_COLOR = "blue"
S_MIN = 90              # a lit shell is saturated; the floor is not
V_MIN = 60
MIN_AREA_PX = 120
MIN_SEP_PX = 2.0        # centroids closer than this are one light, not two
MIN_SIDE_PX = 8         # pixels a side needs before its centroid means anything
"""Why 8 and not 12, and not 3.

The guard exists so a sliver of blue -- a reflection off the floor, a
neighbour's taillight clipped into this blob -- cannot produce a confident
centroid a long way from the truth. So it wants to be as high as it can be.

But it is a count of PIXELS, and the taillight is one LED against the tag's
two, so it is the first thing to fall below any fixed count as the ball shrinks.
At 12 it was refusing four frames in five on a 29px ball -- all of them "no blue
in this blob" -- which reads as the method failing at range when it is only
this number failing. At 8 the same ball answers 99.3% of frames.

Below 8 buys nothing measurable (99.5% at 5 and at 3) while steadily weakening
the guard, so the knee is where this sits."""


def tail_range():
    """(lo, hi) hue for the taillight, from the palette the tracker hunts."""
    c = config.COLORS[TAIL_COLOR]
    return c["hue"] - c["tol"], c["hue"] + c["tol"]


SPARSE_FRACTION = 0.004     # of the frame lit; below this it is a short exposure
# Must exceed the OUTER dot separation, which is the ball's diameter in
# pixels, not the gap between neighbouring dots -- otherwise a ball splits into
# a nose group and a tail group and each half reports "one dot". At 9.2px/cm a
# 7.4cm shell is 68px across and its end LEDs sit ~40px apart.
MERGE_PX = 56               # how far apart dots may be and still be one ball
SPARSE_MIN_AREA = 24


def find_balls(frame, min_area=MIN_AREA_PX, v_min=V_MIN, max_balls=32,
               merge_px=MERGE_PX, sparse_min_area=SPARSE_MIN_AREA):
    """Every lit ball in the frame, as (centre_px, radius_px, area).

    By BRIGHTNESS, not by colour, so the same call works on a normal exposure
    and on a false-coloured brightness gate -- and on a ball whose tag colour
    nobody has calibrated yet. Identifying which robot it is comes later and
    from somewhere else; this only answers "there is a lit thing here".

    Bright regions are found and then CLUSTERED, which covers both shapes a
    ball can take: one disc on a normal exposure, or three specks a couple of
    centimetres apart on a short one. `merge_px` has to span the end dots and
    stay under the distance between two balls, so it follows the ball's size in
    pixels rather than being a constant.

    The radius ENCLOSES the lit region rather than being its area-equivalent.
    That distinction matters for a cluster: three dots in a line have a small
    area and a large extent, and a patch cut to the area-equivalent radius
    contains one dot and reports that the lights have merged.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] >= v_min).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    spots = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < sparse_min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        (_, _), enclosing = cv2.minEnclosingCircle(c)
        spots.append((np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]),
                      float(enclosing), area))

    # Cluster, always. There used to be two code paths here -- one for discs on
    # a normal exposure, one for dots on a short one -- chosen by a threshold,
    # and the threshold was wrong at both ends. Judged by how much of the frame
    # was lit, a big ball on a short exposure tipped over to "disc" and was
    # then measured against a disc-sized area floor its dots could not clear.
    # Judged by contour size instead, a big DOT clears the disc floor and each
    # dot becomes its own ball.
    #
    # Clustering needs neither judgement. One disc is a cluster of one; three
    # dots are a cluster of three; and `merge_px` is smaller than a ball, so
    # two separate balls only merge when they are close enough to be one blob
    # anyway -- which is a case no method here survives regardless.
    parent = list(range(len(spots)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            if float(np.linalg.norm(spots[i][0] - spots[j][0])) <= merge_px:
                parent[root(i)] = root(j)

    groups = {}
    for i, spot in enumerate(spots):
        groups.setdefault(root(i), []).append(spot)

    out = []
    for members in groups.values():
        pts = np.array([m[0] for m in members])
        area = sum(m[2] for m in members)
        centre = pts.mean(axis=0)
        reach = max((float(np.linalg.norm(p - centre)) for p in pts), default=0.0)
        reach += max(m[1] for m in members)
        out.append(((float(centre[0]), float(centre[1])), float(reach), area))
    out.sort(key=lambda b: -b[2])
    return out[:max_balls]


def explain_hue_px(frame, centre_px, radius_px, s_min=S_MIN, v_min=V_MIN):
    """Heading from the two coloured lobes, plus the reason when it declines.

    Returns `(reading, why)` -- exactly one is None. Degrees run the way image
    coordinates do, x right and y DOWN, the same convention `facing_px` answers
    in, so the two are directly comparable and one homography converts either.

    `radius_px` is forgiving here, and it did not used to be. With one tag LED
    opposite the tail, a patch cut tight to the shell clipped the two lobes
    unevenly and cost 4.5deg of bias -- so the rule was to pass the LIT radius
    that `find_balls` returns. Two tag LEDs equidistant from the centre clip
    symmetrically, so that bias is gone: measured at 0.36deg worst on the shell
    radius against 1.74deg on the lit one, the tighter patch is now marginally
    the better of the two.
    """
    r = int(max(5, round(radius_px * 1.35)))
    x, y = int(round(centre_px[0])), int(round(centre_px[1]))
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, "blob is at the frame edge — no room to read it"

    patch = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    lit = (val >= v_min) & (sat >= s_min)
    if lit.sum() < 2 * MIN_SIDE_PX:
        return None, "nothing saturated enough to be a lit shell"

    lo, hi = tail_range()
    is_tail = lit & (hue >= lo) & (hue <= hi)
    is_tag = lit & ~is_tail
    if is_tail.sum() < MIN_SIDE_PX:
        return None, ("no blue in this blob — the taillight is off, or this "
                      "robot is tagged blue and its nose and tail match")
    if is_tag.sum() < MIN_SIDE_PX:
        return None, "all blue — that is a taillight with no tag colour beside it"

    def centroid(mask):
        # Weighted by brightness, so the lobe's lit core pulls the centroid
        # rather than the ragged edge where the shell fades into the floor.
        wgt = (val * mask).astype(np.float64)
        total = wgt.sum()
        ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        return np.array([(xs * wgt).sum() / total, (ys * wgt).sum() / total])

    tail, nose = centroid(is_tail), centroid(is_tag)
    step = nose - tail
    sep = float(np.linalg.norm(step))
    if sep < MIN_SEP_PX:
        return None, (f"the two colours share a centre ({sep:.1f}px apart) — "
                      "that is one light, or a blob with two robots in it")

    deg = math.degrees(math.atan2(step[1], step[0])) % 360.0
    # Separation against the ball's own size, and how evenly the two lobes
    # split it. A sliver of blue on one edge is a reflection, not a taillight,
    # and it produces a confident-looking centroid a long way from the truth.
    room = min(1.0, sep / max(radius_px * 0.5, 1e-6))
    share = min(is_tail.sum(), is_tag.sum()) / float(max(lit.sum(), 1))
    return (deg, round(float(room * min(share * 3.0, 1.0)), 3)), None


def facing_hue_px(frame, centre_px, radius_px, **kw):
    """The heading alone, or None. See `explain_hue_px`."""
    return explain_hue_px(frame, centre_px, radius_px, **kw)[0]


DOTS_FLOOR = 0.28           # of the brightest core; see `explain_dots_px`


def explain_dots_px(frame, centre_px, radius_px, floor=DOTS_FLOOR,
                    s_min=S_MIN):
    """Position, heading and identity from the three dots and their colours.

    The arrangement on the ball is what makes this work, and it is worth
    stating because the obvious guess is wrong. There are not one LED and a
    reflection. There are TWO tag-colour LEDs, equidistant either side of the
    centre, with the blue aiming light behind them:

        front tag LED  ...  CENTRE  ...  rear tag LED  ...  taillight

    Three consequences, and each one removes a problem this module used to have.

    POSITION IS EXACT, not estimated. The two tag LEDs are symmetric about the
    centre, so their midpoint IS the centre -- no offset to calibrate, and
    nothing that leans with the heading. Every other candidate was biased: the
    blob centroid slides as the lobes change shape, and the middle dot sits off
    centre by a centimetre in a direction that rotates with the robot.

    DIRECTION CANNOT FLIP. The blue dot is the odd one out and it is at the
    back, so the heading is simply "away from blue". No brightness ordering, so
    no reversal when a reflection outshines something.

    IDENTITY GETS TWO SAMPLES. Both tag dots are the same colour, so the tag
    can be read from their combined pixels rather than from one dot's worth.

    It also stops needing the dots to be collinear, which a real ball under a
    tilted camera will not quite be.
    """
    from .facing import _local_maxima

    r = int(max(5, round(radius_px * 1.05)))
    x, y = int(round(centre_px[0])), int(round(centre_px[1]))
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, "blob is at the frame edge — no room to read it"

    patch = frame[y0:y1, x0:x1]
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    top = float(grey.max())
    if top < 40:
        return None, f"too dark to read (peak {top:.0f}, needs 40)"

    peaks = _local_maxima(grey, top * floor)
    if len(peaks) < 3:
        return None, (f"{len(peaks)} dot(s), need 3 — the cores are merging. "
                      "Shorten the exposure, or the ball is too small to read")
    peaks.sort(key=lambda p: -p[3])
    pts = [np.array([p[0], p[1]], dtype=float) for p in peaks[:3]]

    # Which dot is the blue one. Sampled in a small disc rather than at the
    # single peak pixel, which on a bright core carries little colour.
    lo, hi = tail_range()
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    blueness = []
    for pt in pts:
        near = ((xx - pt[0]) ** 2 + (yy - pt[1]) ** 2) <= (4.0 ** 2)
        sel = near & (hsv[:, :, 1] >= s_min)
        if sel.sum() < 3:
            blueness.append(0.0)
            continue
        hue = hsv[:, :, 0][sel].astype(int)
        blueness.append(float(((hue >= lo) & (hue <= hi)).mean()))

    order = sorted(range(3), key=lambda i: -blueness[i])
    tail_i = order[0]
    if blueness[tail_i] < 0.25:
        return None, ("no blue at any dot — this frame has no colour in its "
                      "cores, so the axis is there but its direction is not")
    if blueness[order[1]] > 0.6 * blueness[tail_i]:
        return None, "two dots look blue — cannot tell the taillight from a tag"

    tags = [pts[i] for i in order[1:]]
    centre = (tags[0] + tags[1]) / 2.0
    step = centre - pts[tail_i]
    span = float(np.linalg.norm(step))
    if span < MIN_SEP_PX:
        return None, "the taillight sits on the centre — no direction in that"

    deg = math.degrees(math.atan2(step[1], step[0])) % 360.0
    conf = min(1.0, blueness[tail_i]) * min(
        1.0, span / max(radius_px * 0.35, 1e-6))
    return {"deg": deg,
            "centre": (float(centre[0] + x0), float(centre[1] + y0)),
            "tags": [(float(t[0] + x0), float(t[1] + y0)) for t in tags],
            "tail": (float(pts[tail_i][0] + x0), float(pts[tail_i][1] + y0)),
            "dots": 3, "conf": round(float(conf), 3)}, None


def identify(frame, reading, colors=None, s_min=40, v_min=40, radius=None):
    """Which robot this is, from the colour at its two tag dots.

    Sampled AT the LEDs rather than over the shell, which is what makes this
    work on an underexposed frame. The tag colour there is a couple of hundred
    pixels rather than a couple of thousand, so a detector tuned to find a big
    coloured disc sees nothing -- but the pixels that remain are the purest
    ones in the frame, because they came straight from the emitter rather than
    through a diffusing shell that has already mixed them with the taillight.

    Chromaticity rather than hue: each pixel is divided by its own total, so
    brightness drops out and what is left is the ratio between the LED's
    channels. Near-neutral pixels are dropped first -- a blown core is white,
    and white is equally close to every colour in the palette.

    Returns (name, distance) or None. The distance is what it is: small is a
    confident match, and two palette colours that sit close together will both
    be small, so a caller comparing robots should look at the gap between the
    best two rather than at the best alone.
    """
    from . import config
    colors = colors or [n for n in config.COLORS if n != TAIL_COLOR]
    ref = {}
    for n in colors:
        bgr = cv2.cvtColor(np.uint8([[[config.COLORS[n]["hue"], 235, 250]]]),
                           cv2.COLOR_HSV2BGR)[0, 0].astype(float)
        ref[n] = bgr / max(bgr.sum(), 1e-6)

    spots = reading.get("tags") if isinstance(reading, dict) else None
    if not spots:
        return None
    r = int(max(1, round(radius if radius is not None else 2.0)))
    keep = []
    h, w = frame.shape[:2]
    for (x, y) in spots:
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = frame[y0:y1, x0:x1].reshape(-1, 3)
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        keep.append(patch[(hsv[:, 1] >= s_min) & (hsv[:, 2] >= v_min)])
    keep = [k for k in keep if len(k)]
    if not keep:
        return None
    px = np.vstack(keep).astype(float)
    chroma = (px / px.sum(axis=1, keepdims=True)).mean(axis=0)
    best = sorted(((float(np.linalg.norm(chroma - c)), n) for n, c in ref.items()))
    return best[0][1], round(best[0][0], 4)


def gate_view(frame, lo=40, hi=200, gamma=0.7):
    """The brightness gate, for looking at rather than for measuring.

    Deliberately separate from anything that reads a heading. False colour has
    no hue in it, so a frame rendered this way is for eyes only -- every method
    here is given the real capture and this is given to the screen.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip((grey - float(lo)) / max(float(hi) - float(lo), 1.0), 0.0, 1.0)
    return cv2.applyColorMap((g ** gamma * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def read_all(frame, use_peaks=True, use_hue=True, use_dots=True,
             stop_at_first=False, **kw):
    """Every ball in the frame with whatever headings could be read from it.

    The disagreement between the two is carried rather than resolved. Which one
    to believe depends on how the frame was taken, and the caller knows that
    and this does not.
    """
    from .facing import explain_px
    out = []
    for centre, radius, area in find_balls(frame, **kw):
        row = {"centre": centre, "radius": radius, "area": area,
               "peaks": None, "peaks_why": None, "hue": None, "hue_why": None,
               "dots": None, "dots_why": None}
        if use_dots:
            row["dots"], row["dots_why"] = explain_dots_px(frame, centre, radius)
            if row["dots"]:
                # The dots know better than the blob does. A centroid moves
                # with the shape of the glow; the middle core does not.
                row["centre"] = row["dots"]["centre"]
        # Order matters when only the first answer is kept, and it is not the
        # order you would guess. The peak method ANSWERS on a normal exposure
        # -- the clipped lobes still have local maxima -- it just answers
        # wrongly, by about ninety degrees. Running it before the hue method
        # therefore throws away a correct reading in favour of a confident
        # wrong one. Hue first: it refuses cleanly on a false-coloured frame,
        # so it never pre-empts the peaks where the peaks are the right tool.
        if use_hue and not (stop_at_first and row["dots"]):
            row["hue"], row["hue_why"] = explain_hue_px(frame, centre, radius)
        if use_peaks and not (stop_at_first and (row["dots"] or row["hue"])):
            row["peaks"], row["peaks_why"] = explain_px(frame, centre, radius)
        if row["peaks"] and row["hue"]:
            row["disagree_deg"] = round(
                abs((row["peaks"][0] - row["hue"][0] + 180) % 360 - 180), 1)
        else:
            row["disagree_deg"] = None
        out.append(row)
    return out

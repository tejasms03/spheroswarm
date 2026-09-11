"""Heading from the camera, because a Sphero has two lights and not one.

The premise everything else in this project rests on is that the camera cannot
see which way a robot is pointing — "a glowing sphere has no facing" — and so
the aim frame has to be inferred by driving known legs and watching where the
ball went. That inference is where the pain lives: it needs floor, it needs a
tracker that is following the right ball, it goes stale every time the robot
reconnects, and it cannot distinguish a rotated frame from a mirrored one
without legs pointing several different ways.

But the premise is only true if you ignore the tail light. A Sphero has the
main RGB LED and a separate blue aim LED at the back. Two lights on one shell
is an orientation, directly: the vector from the tail to the main LED is where
the robot is pointing, measured rather than deduced, every frame, with no legs
driven and nothing to go stale.

**What this needs from the camera, and it is not optional.** The two lights are
a few centimetres apart on a 7.4cm ball, so at a typical mounting height they
are single-figure pixels apart. Autofocus hunting on a dark floor smears them
into one blob, and auto exposure meters a mostly-black frame and opens up until
the shell blows out to a single white disc with no separable peaks at all. See
`CameraSource.manual`. If the peaks cannot be separated this returns None, which
is the honest answer and leaves the driven-leg method to do its job.
"""

import math

import numpy as np

# A SPRK+ is 7.4cm across; the two lights sit well inside that. Anything
# claiming a wider separation is two robots, or a reflection.
MIN_SEP_PX = 3.0
MAX_SEP_FRAC = 0.75         # of the blob's own diameter
PEAK_FLOOR = 0.55           # of the brightest pixel, before a peak counts


def _local_maxima(patch, floor):
    """Peaks that are the brightest thing in their own neighbourhood.

    Plain thresholding gives one fat region per light and no centres. This is
    the non-maximum suppression: dilate, keep pixels equal to their local max,
    and every surviving cluster is one light.
    """
    import cv2
    k = np.ones((5, 5), np.uint8)
    peak = (patch >= cv2.dilate(patch, k)) & (patch >= floor)
    n, _, stats, cents = cv2.connectedComponentsWithStats(
        peak.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        cx, cy = cents[i]
        out.append((float(cx), float(cy), float(stats[i, 4]),
                    float(patch[int(round(cy)), int(round(cx))])))
    return out


def facing_px(frame, centre_px, radius_px, floor=PEAK_FLOOR):
    """Which way the robot at `centre_px` is pointing, in image degrees.

    Returns (degrees, confidence) or None. Degrees are measured the way image
    coordinates run — x right, y DOWN — so the caller converts to arena
    coordinates through the same homography that moves everything else, and a
    mirrored calibration mirrors this too. That is deliberate: one frame, one
    convention, no second place for a sign to be wrong.
    """
    return explain_px(frame, centre_px, radius_px, floor=floor)[0]


def explain_px(frame, centre_px, radius_px, floor=PEAK_FLOOR):
    """`facing_px`, plus the reason when it declines to answer.

    Returns `(reading, why)` — exactly one of the two is None. The refusals
    here are the whole value of this module on a bench: "no reading" is a
    useless thing to put in front of somebody trying to fix their lighting,
    while "one peak — the two lights are merging, defocus less or drop the
    peak floor" names the knob to turn. `facing_px` stays the thin answer for
    callers that only want the number.
    """
    import cv2

    r = int(max(4, round(radius_px * 1.4)))
    x, y = int(round(centre_px[0])), int(round(centre_px[1]))
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None, "blob is at the frame edge — no room to read it"

    patch = frame[y0:y1, x0:x1]
    grey = (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if patch.ndim == 3 else patch).astype(np.uint8)
    top = float(grey.max())
    if top < 40:
        return None, f"too dark to read (peak {top:.0f}, needs 40)"

    peaks = _local_maxima(grey, top * floor)
    if len(peaks) < 2:
        return None, ("one peak — the lights are merging. Defocus less, "
                      "shorten the exposure, or lower the peak floor")

    # The two brightest. A third peak is a reflection or a neighbour, and
    # picking the two brightest is what keeps it out.
    peaks.sort(key=lambda p: -p[3])
    (ax, ay, _, av), (bx, by, _, bv) = peaks[0], peaks[1]
    sep = math.hypot(ax - bx, ay - by)
    if sep < MIN_SEP_PX:
        return None, (f"peaks {sep:.1f}px apart, under the {MIN_SEP_PX:.0f}px "
                      "floor — that is one light seen twice")
    if sep > MAX_SEP_FRAC * (2 * radius_px):
        return None, (f"peaks {sep:.1f}px apart on a {2 * radius_px:.0f}px "
                      "ball — that is two robots, or a reflection")

    # Front is the brighter light: the main RGB LED is driven hard and the aim
    # light is a dim blue marker. Using brightness rather than colour keeps
    # this working when the main LED happens to be set to blue.
    fx, fy, bx2, by2 = (ax, ay, bx, by) if av >= bv else (bx, by, ax, ay)
    deg = math.degrees(math.atan2(fy - by2, fx - bx2)) % 360.0

    # How much brighter the front is than the back, and how cleanly the two
    # separated. Both matter: two equally bright peaks could be either way
    # round, and a separation near the floor is one light seen twice.
    contrast = abs(av - bv) / max(av, 1.0)
    room = min(1.0, (sep - MIN_SEP_PX) / max(radius_px, 1.0))
    return (deg, round(float(min(contrast, 1.0) * room), 3)), None


def facing_cm(frame, centre_px, radius_px, homography, floor=PEAK_FLOOR):
    """The same thing in arena coordinates, as a compass heading.

    Taken through the homography as a pair of points rather than as an angle,
    because a perspective map does not preserve angles — rotating a bearing by
    whatever the matrix does at the image centre is wrong everywhere else in a
    tilted view, and a tilted view is what an overhead camera on a tripod is.
    """
    return explain_cm(frame, centre_px, radius_px, homography, floor=floor)[0]


def explain_cm(frame, centre_px, radius_px, homography, floor=PEAK_FLOOR):
    """`facing_cm`, plus the reason. See `explain_px`."""
    got, why = explain_px(frame, centre_px, radius_px, floor=floor)
    if got is None:
        return None, why
    if homography is None or not homography.ready:
        return None, "no arena calibration — set the arena on the COLOUR tab"
    deg, conf = got
    r = math.radians(deg)
    a = np.array(centre_px, dtype=float)
    b = a + np.array([math.cos(r), math.sin(r)]) * max(radius_px, 2.0)
    pa, pb = homography.to_cm([a.tolist(), b.tolist()])
    d = np.asarray(pb, dtype=float) - np.asarray(pa, dtype=float)
    if float(np.linalg.norm(d)) < 1e-9:
        return None, "the homography collapsed the heading to a point"
    # Compass: zero is +y, increasing clockwise — the frame `velocity_to_command`
    # speaks, so the answer can be used without another conversion.
    return (float(math.degrees(math.atan2(d[0], d[1])) % 360.0), conf), None


# -- reading the lights directly, without finding a blob first ------------
#
# The route above needs a hue blob first and then reads a heading inside it,
# which makes the heading only as good as the colour tracking — and colour is
# the weak link on this bench. On a brightness map a lit Sphero is not a subtle
# thing: it is the brightest object in a dark room by a wide margin, and its
# lights are already separate peaks before anybody has thresholded a hue.
#
# So this route inverts the order. Find the bright spots, group the ones close
# enough to be on one shell, and read the heading off their arrangement. The
# colour is then used for the one thing it is genuinely good at — saying which
# end is the front — rather than for finding the robot at all.

LIGHT_MIN_V = 200           # a lit LED against a dark floor, not a bright floor
ANNULUS = (1.1, 2.4)        # radii, in units of the core's own radius
# The outer radius above is a multiple of the core, and a multiple of the core
# is not where the light ends. Measured on a bloomed LED, the coloured halo
# runs out at about 1.3x the outer radius that rule picks, so the far half of
# the ring lands on floor — dark pixels whose hue is noise. Keep only ring
# pixels still carrying a real fraction of this light's own peak.
RING_V_FRAC = 0.15
RING_MIN_PX = 8             # below this, the clip has taken too much
BALL_SPAN_FRAC = 1.0        # lights further apart than a ball are not one ball
# How collinear three lights must be to be three lights on one shell. Measured
# rather than guessed: a real arrangement with a couple of pixels of noise
# across a 50px span reads 0.94-1.0, and a scattered triangle reads 0.37. The
# first value tried here was 0.35, which accepted the triangle — it sat below
# the whole range instead of inside the gap.
STRAIGHT_MIN = 0.75
# How much the two ends must differ before "which is the front" is an answer
# rather than a coin toss. Both LEDs clip to 255 on a bright frame, so the
# brightness fallback can find NO difference at all — and it was then picking
# whichever end came first along the axis and reporting it with confidence
# zero. A heading 180 degrees out is worse than no heading, so this refuses.
MIN_END_CERTAINTY = 0.04
# The light separation, in pixels, at which the geometry stops limiting
# confidence. Below it the heading is measured on too short a baseline: the
# angular error of a two-point bearing goes as 1/span, so a 20px span carries
# twice the noise of a 40px one and is scored accordingly.
SPAN_FOR_FULL_CONF = 40.0
# How much bigger than the taillight a single blob must be before it is taken
# to be the two tag LEDs fused rather than one of them. They carry about twice
# the lit pixels; the floor sits below that so a dim frame still qualifies.
MERGED_PAIR_RATIO = 1.5


def _flux(light):
    """How much light a blob carries: its size times its height.

    Neither factor is enough alone. A clipped core stops growing in height and
    grows in WIDTH instead, so peak goes flat exactly when the exposure is
    short enough to be useful; and two lights of equal size are told apart
    only by height. The product degrades to whichever one still has signal.
    """
    return float(light.get("area", 0.0)) * float(light.get("peak", 0.0))


def light_mask(frame, min_v=LIGHT_MIN_V):
    """The V channel, and the pixels `lights_px` will consider lights.

    Split out so a bench can SHOW what the detector is working from without
    re-deriving the threshold beside it. Two derivations of "what counts as a
    light" is how a mask view ends up reassuring you about a picture the
    detector never saw — and this bench exists to be believed.

    Returns `(grey, mask)`. `grey` is V, not a BGR-to-grey luma: V is the
    channel that saturates, so it is what a blown LED core maxes out in and
    what the threshold below has always meant.
    """
    import cv2
    grey = (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
            if frame.ndim == 3 else frame.astype(np.uint8))
    return grey, grey >= min_v


def lights_px(frame, min_v=LIGHT_MIN_V, min_area=3, max_area=6000,
              colour_frame=None):
    """Every bright spot in the frame, with the colour in the ring around it.

    `colour_frame` samples the COLOUR from a different image than the one the
    positions came from — pass a defocused copy and the halo spreads, which
    averages down sensor noise and gives a steadier hue, while the positions
    stay as sharp as the sensor delivered them. Blur helps a colour and hurts a
    geometry, so the two are read from different images rather than one
    compromise between them. It must be the same size as `frame`.

    The core of a lit LED is blown out and has no hue left — that is what
    `vision/detect.py` already knows about lit shells, and it is more true of a
    bare LED than of a shell. So position comes from the blown core, which is
    crisp and stable, and colour comes from an ANNULUS outside it, where the
    light is still coloured and not yet floor.

    Returns dicts, brightest first: x, y, area, peak, bgr, hue, sat.
    """
    import cv2

    # One conversion, used twice. V is the channel that saturates, so it is
    # both the right thing to threshold on and the right thing to call
    # brightness — and converting the frame a second time to get it back is a
    # full-resolution pass for nothing, which at 1080p is most of the budget.
    if colour_frame is None:
        colour_frame = frame
    elif colour_frame.shape[:2] != frame.shape[:2]:
        colour_frame = frame
    hsv = (cv2.cvtColor(colour_frame, cv2.COLOR_BGR2HSV)
           if frame.ndim == 3 else None)
    # Positions and the threshold come from the frame as delivered; only the
    # ring sample below reads `hsv`, which may be the defocused copy.
    grey, mask = light_mask(frame, min_v)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)

    # How far each light is from its nearest neighbour, before any sampling.
    # The lights on one shell are CLOSE — about 13px apart on this arena — and
    # an annulus of 1.1 to 2.4 core radii around a 7px core reaches 17px, which
    # lands squarely on the light next door. Sampling a neighbour's colour and
    # calling it your own reverses front and back whenever the tail is the one
    # that got read, which is a heading exactly 180 degrees wrong.
    keep = [i for i in range(1, n)
            if min_area <= float(stats[i, 4]) <= max_area]
    spots = np.array([[float(cents[i][0]), float(cents[i][1])] for i in keep]) \
        if keep else np.zeros((0, 2))
    gap = {}
    for k, i in enumerate(keep):
        if len(spots) < 2:
            gap[i] = float("inf")
            continue
        d = np.linalg.norm(spots - spots[k], axis=1)
        d[k] = np.inf
        gap[i] = float(d.min())
    fh, fw = grey.shape[:2]
    out = []
    for i in range(1, n):
        area = float(stats[i, 4])
        if area < min_area or area > max_area:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        core = math.sqrt(area / math.pi)

        # Everything below works in a window around the light rather than over
        # the frame. The obvious version — a full-frame coordinate grid and a
        # full-frame `labels == i` — is correct and unusable: at 1080p that is
        # four 2-megapixel arrays PER LIGHT, so a five-light frame does tens of
        # millions of element-ops before anything is drawn. It is invisible in
        # a 640x480 test and it will not hold 30fps in the bench.
        bx, by = int(stats[i, 0]), int(stats[i, 1])
        bw, bh = int(stats[i, 2]), int(stats[i, 3])
        box = labels[by:by + bh, bx:bx + bw] == i
        light = {
            "x": cx, "y": cy, "area": area, "core_px": core,
            "peak": float(grey[by:by + bh, bx:bx + bw][box].max()),
            "bgr": None, "hue": None, "sat": None,
        }

        # Never sample past halfway to the next light. Half the gap is the
        # furthest a ring can reach and still be unambiguously this light's
        # own, and it is a bound rather than a tuning knob for that reason.
        r_out = min(ANNULUS[1] * core, 0.5 * gap.get(i, float("inf")))
        r_in = min(ANNULUS[0] * core, r_out * 0.7)
        x0, x1 = max(0, int(cx - r_out) - 1), min(fw, int(cx + r_out) + 2)
        y0, y1 = max(0, int(cy - r_out) - 1), min(fh, int(cy + r_out) + 2)
        if frame.ndim != 3 or x1 <= x0 or y1 <= y0:
            out.append(light)
            continue
        # `ogrid` rather than `mgrid`: two 1-D arrays that broadcast, instead of
        # two 2-D ones materialised in full.
        wy, wx = np.ogrid[y0:y1, x0:x1]
        d2 = (wx - cx) ** 2 + (wy - cy) ** 2
        ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        if ring.any():
            win_hsv = hsv[y0:y1, x0:x1]
            lit_enough = win_hsv[:, :, 2] >= light["peak"] * RING_V_FRAC
            clipped = ring & lit_enough
            # Falling back rather than refusing: on a small or dim light the
            # clip can take nearly everything, and a rough colour from the
            # whole ring beats no colour at all — `sat` still reports how much
            # there was to read, and `identify_cluster` refuses on that.
            if int(clipped.sum()) >= RING_MIN_PX:
                ring = clipped
            light["ring_px"] = int(ring.sum())
            px = colour_frame[y0:y1, x0:x1][ring]
            light["bgr"] = tuple(float(v) for v in px.mean(axis=0))
            hp = win_hsv[ring]
            # Circular mean of hue, weighted by SATURATION TIMES VALUE. A flat
            # average is wrong across the 179/0 seam, which is exactly where red
            # sits on this bench. Weighting by saturation alone is wrong too:
            # a dark floor pixel can be fully saturated and its hue is noise,
            # so brightness has to count as well as colourfulness.
            ang = hp[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
            wgt = (hp[:, 1].astype(np.float64)
                   * hp[:, 2].astype(np.float64) / 255.0)
            if wgt.sum() > 0:
                sin_, cos_ = float((np.sin(ang) * wgt).sum()), float((np.cos(ang) * wgt).sum())
                light["hue"] = float(
                    (math.degrees(math.atan2(sin_, cos_)) % 360.0) / 2.0)
                light["sat"] = float(hp[:, 1].mean())
        out.append(light)
    out.sort(key=lambda l: -l["peak"])
    return out


# How much longer than it is wide a single blob must be before its long axis
# is taken to mean anything. A round blob has no axis, and second moments will
# happily hand you one made of noise.
MERGED_ELONGATION = 1.25


def blob_axis_px(frame, light, reach=2.6):
    """The long axis of ONE blob, from its second moments.

    For when the lights have bloomed into each other. Peak-finding needs the
    lights to RESOLVE — two maxima have to survive non-maximum suppression —
    and on a bright white ball at this scale they routinely do not: all three
    fuse into a single region and the reader refuses, which freezes the
    heading until a frame happens to separate. That reads, from outside, as a
    tracker that only notices large rotations.

    A streak still has an axis whether or not its peaks resolve, and second
    moments recover it from every pixel at once rather than needing two
    maxima. Measured against the peak reader on rendered balls: 100% of frames
    read at every geometry tried, including ones where peak-finding returned
    nothing at all, and the axis itself came out four to seven times more
    accurate.

    Returns `(axis, elongation)` or None. The axis is a LINE — a unit vector
    with an arbitrary sign — because a fused blob cannot say which end is the
    front. That is the caller's problem, and continuity solves it: a ball
    turns about 12 degrees between frames while a flip is 180.
    """
    import cv2

    cx, cy = float(light["x"]), float(light["y"])
    core = max(2.0, float(light.get("core_px") or 2.0))
    r = int(max(6, round(core * reach)))
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    win = frame[y0:y1, x0:x1]
    v = (cv2.cvtColor(win, cv2.COLOR_BGR2HSV)[:, :, 2]
         if win.ndim == 3 else win).astype(np.float64)
    # Weighted by how far each pixel is ABOVE the floor, not by its value: the
    # floor is not part of the shape, and letting it vote drags the axis
    # toward whichever way the window happens to be cropped.
    floor = float(np.median(v))
    wgt = np.clip(v - floor, 0.0, None)
    total = float(wgt.sum())
    if total <= 1e-9:
        return None
    yy, xx = np.mgrid[0:wgt.shape[0], 0:wgt.shape[1]]
    mx = float((xx * wgt).sum() / total)
    my = float((yy * wgt).sum() / total)
    dx, dy = xx - mx, yy - my
    cov = np.array([[float((dx * dx * wgt).sum() / total),
                     float((dx * dy * wgt).sum() / total)],
                    [float((dx * dy * wgt).sum() / total),
                     float((dy * dy * wgt).sum() / total)]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 1e-9:
        return None
    elong = math.sqrt(max(vals[0], 1e-12) / max(vals[1], 1e-12))
    if elong < MERGED_ELONGATION:
        return None                 # round: it has no axis worth reporting
    axis = vecs[:, 0]
    n = float(np.linalg.norm(axis))
    return (axis / n, float(elong)) if n > 1e-9 else None


def cluster_px(lights, span_px):
    """Group lights that are close enough to be on one shell.

    Single-link on distance: a Sphero's lights are strung along one axis, so
    two of them may be further apart than each is from the middle one, and a
    centroid-radius test would split exactly the three-light case this exists
    to handle.
    """
    groups = []
    for light in lights:
        joined = None
        for g in groups:
            if any(math.hypot(light["x"] - o["x"], light["y"] - o["y"]) <= span_px
                   for o in g):
                if joined is None:
                    g.append(light)
                    joined = g
                else:                       # bridges two groups; merge them
                    joined.extend(g)
                    g.clear()
        if joined is None:
            groups.append([light])
    return [g for g in groups if g]


def _axis(points):
    """The long axis of a set of points, and where each sits along it."""
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    u, s, vt = np.linalg.svd(p - c, full_matrices=False)
    direction = vt[0]
    t = (p - c) @ direction
    straightness = 1.0
    if len(s) > 1 and s[0] > 1e-9:
        # How much of the spread is off the axis. Three lights on a shell are
        # collinear; three things that are not collinear are not one robot.
        straightness = float(1.0 - min(1.0, s[1] / s[0]))
    return c, direction, t, straightness


def _body_centre(group, back):
    """Where the BALL is, given which of its lights is the tail.

    The two tag LEDs sit symmetrically about the centre of the shell, so the
    midpoint BETWEEN THEM is the centre. The tail sits behind both, and
    averaging it in with them drags the answer backwards along the heading —
    about 7mm on a SPRK+, measured in `blob_test.py`. That offset ROTATES with
    the robot, so no constant can cancel it: a controller sees a position that
    leans whichever way the ball happens to be pointing, which is the error
    least likely to be spotted and most likely to be blamed on the drive.

    Returns `(centre, basis)`. The basis is reported rather than assumed,
    because the two-light case cannot give a centre at all: with one tag LED
    visible there is nothing to take a midpoint with, and the honest answer is
    that light's own position — half the LED spacing FORWARD of the truth. A
    caller that cares can refuse on it, and one that does not at least stops
    presenting a biased number as if it were the centre.
    """
    body = [l for l in group if l is not back]
    if not body:                      # every light is the tail: nothing better
        body = list(group)
    x = sum(l["x"] for l in body) / len(body)
    y = sum(l["y"] for l in body) / len(body)
    if len(body) >= 2:
        basis = "tag pair"
    elif (back is not None and body[0] is not back
          and float(body[0]["area"]) >= MERGED_PAIR_RATIO * float(back["area"])):
        # One blob, but far too big to be one LED. The tag pair has fused,
        # and because the two LEDs straddle the shell symmetrically their
        # merged centroid IS the centre — this is the good case, not a
        # degraded one, and calling it "one LED" would send somebody looking
        # for an offset that is not there.
        basis = "tag pair, merged"
    else:
        basis = "one tag LED — half the LED spacing ahead of the true centre"
    return (float(x), float(y)), basis


def heading_from_lights(group, blue_is_back=True, signatures=None):
    """Which way a cluster of lights points, and which of them is the front.

    Returns `(reading, why)`. The reading carries the heading in IMAGE degrees
    — x right, y down, the same convention as `facing_px` — along with the
    front and back lights themselves so a caller can show its working.

    Front is decided by COLOUR here, not by brightness. `set_back_led` is
    `Color(0, 0, n)` on every toy that has one, so the tail is the bluest thing
    on the shell, and blue is a much sharper discriminator than "brighter" once
    there are three lights and one of them is a reflection. When nothing is
    convincingly blue it falls back to brightness, which is what the two-light
    reader has always done.
    """
    if len(group) < 2:
        return None, "one light — that is a position, not an orientation"

    pts = [(l["x"], l["y"]) for l in group]
    centre, direction, t, straightness = _axis(pts)
    span = float(t.max() - t.min())
    if span < MIN_SEP_PX:
        return None, (f"lights span {span:.1f}px, under the {MIN_SEP_PX:.0f}px "
                      "floor — that is one light seen twice")
    if straightness < STRAIGHT_MIN and len(group) > 2:
        return None, (f"{len(group)} lights but not in a line "
                      f"({straightness:.2f}) — a reflection, or two robots")

    ends = (group[int(np.argmin(t))], group[int(np.argmax(t))])
    blueness = [_blueness(l) for l in ends]

    tag, tag_why = (identify_cluster(group, signatures) if signatures
                    else (None, None))
    # The tag decides the front only when the light it identified is actually
    # an END of the axis. A main LED read in the MIDDLE of the cluster names
    # the robot perfectly well and says nothing about which way it points, and
    # treating it as the front would put the arrow across the ball instead of
    # along it.
    by_tag = tag is not None and any(tag["front"] is e for e in ends)
    by_colour = (not by_tag) and blue_is_back and max(blueness) > 0.35 and \
        abs(blueness[0] - blueness[1]) > 0.15
    if by_tag:
        front = tag["front"]
        back = ends[1] if front is ends[0] else ends[0]
    elif by_colour:
        back = ends[0] if blueness[0] > blueness[1] else ends[1]
        front = ends[1] if back is ends[0] else ends[0]
    else:
        # TOTAL LIGHT, not peak height. Both cores clip on a short shutter
        # — the tag pair and the tail each read 255 — so comparing peaks is a
        # coin toss: measured on rendered balls with clipped cores it names
        # the tail correctly 46-61% of the time. That is the "both are equally
        # bright" refusal, and the 180-degree flips that get past it.
        #
        # Area survives the clip, because clipping is what makes a bright blob
        # BIGGER: light that can no longer raise a pixel's value spreads into
        # its neighbours instead. On this rig the two tag LEDs sit close
        # enough to fuse into one blob carrying about twice the lit pixels of
        # the lone taillight.
        #
        # The product of the two is what is actually compared, so neither has
        # to be the discriminator on its own: it falls back to area when the
        # peaks have clipped level, and to peak when the areas match. Measured
        # 100% correct in both regimes.
        back = ends[0] if _flux(ends[0]) <= _flux(ends[1]) else ends[1]
        front = ends[1] if back is ends[0] else ends[0]

    deg = math.degrees(math.atan2(front["y"] - back["y"],
                                  front["x"] - back["x"])) % 360.0
    # Confidence: how cleanly the two ends separated, and how sure we are which
    # way round they go. A pair that is only just distinguishable front-to-back
    # is a heading that may be 180 degrees out, which is worse than no heading.
    if by_tag:
        certainty = tag["conf"]
    elif by_colour:
        certainty = abs(blueness[0] - blueness[1])
    else:
        # As a FRACTION of the total, so it does not depend on the exposure.
        # A merged tag pair against a single tail is about 2:1, which lands
        # near 0.33 — comfortably clear of MIN_END_CERTAINTY.
        f0, f1 = _flux(ends[0]), _flux(ends[1])
        certainty = abs(f0 - f1) / max(f0 + f1, 1e-9)
    if certainty < MIN_END_CERTAINTY:
        return None, ("cannot tell the front from the back: no colour tag, "
                      "neither end is blue, and both are equally bright"
                      + (f" ({tag_why})" if tag_why else "")
                      + ". Turn the taillight on, or fix the tag")
    # Confidence is a PRODUCT of two independent things, and reporting only
    # the product tells nobody which one to fix. A ball 20px across and
    # perfectly tagged scores the same 0.5 as one 60px across whose front and
    # back are nearly indistinguishable — and those want opposite actions:
    # move the camera, or fix the colour. So both terms are carried out.
    reach = min(1.0, span / SPAN_FOR_FULL_CONF)
    sure = min(1.0, certainty * 3.0)
    conf = round(float(reach * sure), 3)
    # The centre of the BALL, not the centroid of the lights — see
    # `_body_centre`. `lights_centre` keeps the raw axis centroid available,
    # because it is what the span and straightness above were measured about.
    body, basis = _body_centre(group, back)
    return {"deg": deg, "conf": conf,
            "conf_span": round(float(reach), 3),
            "conf_ends": round(float(sure), 3),
            "certainty": round(float(certainty), 3),
            "front": front, "back": back,
            "centre": body, "centre_from": basis,
            "lights_centre": (float(centre[0]), float(centre[1])),
            "span_px": span, "straightness": round(straightness, 3),
            "by_colour": bool(by_colour), "by_tag": bool(by_tag),
            "color": tag["color"] if tag else None,
            "tag_conf": tag["conf"] if tag else None,
            "tag_why": tag_why, "lights": list(group)}, None


def _blueness(light):
    """How much this light looks like the blue aim LED. 0 to 1.

    From the BGR of the ring rather than the hue, because a nearly-white ring
    has a meaningless hue with a large error bar, and blue-minus-the-others is
    stable whether or not there is much saturation left.
    """
    bgr = light.get("bgr")
    if not bgr:
        return 0.0
    b, g, r = bgr
    total = max(b + g + r, 1.0)
    return float(max(0.0, (b - max(g, r)) / total * 3.0))


def bearing_cm(a_px, b_px, homography):
    """Compass bearing from one pixel point to another, through the arena map.

    As two POINTS, never as an angle. A perspective map does not preserve
    angles, so rotating a bearing by whatever the matrix does at the image
    centre is wrong everywhere else in a tilted view — and a tilted view is
    what an overhead camera on a tripod is.
    """
    if homography is None or not homography.ready:
        return None
    pa, pb = homography.to_cm([list(a_px), list(b_px)])
    d = np.asarray(pb, dtype=float) - np.asarray(pa, dtype=float)
    if float(np.linalg.norm(d)) < 1e-9:
        return None
    return float(math.degrees(math.atan2(d[0], d[1])) % 360.0)


# -- naming a cluster from the colour of its own lights -------------------

MIN_TAG_SAT = 40.0          # below this the annulus hue is noise, not a colour
# The runner-up slot must be this many hue units further away than the winner.
# ABSOLUTE rather than proportional, and the difference matters: a ratio rule
# accepts 2-away over 4-away, because 4 is twice 2 — but the gap is two units,
# and hue noise is a few units whatever the distances are. Measured on the
# yellow-60/cyan-62 pair this bench actually shipped, a proportional rule named
# hue 58 "yellow" and hue 64 "cyan", so a ball sitting still and wobbling by
# four would change its own name.
TAG_GAP = 8.0


def hue_err(a, b):
    """Distance between two OpenCV hues (0-179), the short way round.

    A subtraction is wrong here and wrong in a way that hides: red currently
    sits at hue 172 on this bench, five units from the seam, so `abs(172 - 2)`
    says 170 when the true distance is 10. That is a colour the roster is
    actually using, so the naive version fails on the live calibration rather
    than on some hypothetical one.
    """
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def identify_cluster(group, signatures, min_sat=MIN_TAG_SAT):
    """Which colour slot this cluster is wearing, and which light is the front.

    Colour is used for identity here, not for detection — the lights were
    already found without it. That is the whole point of the arrangement: hue
    has to survive a blown-out core to find a robot, but only has to be roughly
    right to name one you have already located.

    Returns `(reading, why)`. The reading names a colour SLOT, not a robot —
    the roster owns the slot-to-code mapping and this layer has no business
    knowing about it.
    """
    if not signatures:
        return None, "no colour signatures to match against"

    # The tail first, so it cannot be mistaken for an identity. It is blue on
    # every toy that has one, and blue is also a slot a robot can be wearing —
    # which is the one genuine ambiguity here and is reported rather than
    # guessed at.
    blues = sorted(group, key=lambda l: -_blueness(l))
    tail = blues[0] if _blueness(blues[0]) > 0.35 else None

    # Readability is judged among the lights that could BE the identity, not
    # across the whole cluster. Checking the cluster lets a perfectly readable
    # taillight vouch for a main LED that has blown out to white, and the
    # refusal then blames hue matching for a exposure problem — which is the
    # wrong knob, and this bench exists to name the right one.
    candidates = [l for l in group if l is not tail]
    lit = [l for l in candidates if l.get("hue") is not None
           and (l.get("sat") or 0.0) >= min_sat]
    if not lit:
        best_sat = max((l.get("sat") or 0.0) for l in candidates) if candidates else 0.0
        return None, (f"no light has a readable colour — best saturation "
                      f"{best_sat:.0f}, need {min_sat:.0f}. The LEDs are "
                      "blowing out: dim them, or shorten the exposure")

    best, contested, nearest = None, None, None
    for light in lit:
        ranked = sorted(((hue_err(light["hue"], spec.get("hue", 0)), name)
                         for name, spec in signatures.items()),
                        key=lambda t: t[0])
        if not ranked:
            continue
        err, name = ranked[0]
        if nearest is None or err < nearest[0]:
            nearest = (err, name, light["hue"])
        tol = float(signatures[name].get("tol", 10))
        if err > tol:
            continue
        runner_err, runner_name = (ranked[1] if len(ranked) > 1
                                   else (180.0, None))
        # A match that is barely closer than the next slot is not a match.
        # Picking whichever won by a hair is how a robot gets called by its
        # neighbour's name every few frames.
        if runner_err - err < TAG_GAP:
            if contested is None or err < contested[1]:
                contested = (name, err, runner_name, runner_err, light["hue"])
            continue
        score = (1.0 - err / max(tol, 1e-6)) * min(1.0, light["peak"] / 255.0)
        if best is None or score > best[0]:
            best = (score, name, light, err)

    if best is None and contested is not None:
        # Name the two slots and the gap, rather than saying "no match". The
        # fix for a contested hue is a better-separated palette, and that is
        # only obvious if the message says which pair is fighting.
        name, err, runner_name, runner_err, hue = contested
        apart = hue_err(signatures[name].get("hue", 0),
                        signatures[runner_name].get("hue", 0))
        return None, (f"reads hue {hue:.0f}: {name} is {err:.0f} away and "
                      f"{runner_name} is {runner_err:.0f} — too close to call. "
                      f"Those two slots are {apart:.0f} apart and want "
                      f"{TAG_GAP:.0f}+. Run `optimise hues`")
    if best is None:
        near = (f" (nearest is {nearest[1]}, {nearest[0]:.0f} away, "
                f"outside its tolerance)" if nearest else "")
        return None, ("no light matches a known colour within its tolerance"
                      + near + " — tune the signature, or this is not an "
                      "enrolled robot")

    score, name, front, err = best
    if tail is not None and name == "blue":
        return None, ("this cluster reads BLUE and so does its taillight — "
                      "which end is the front cannot be told apart. Give this "
                      "robot another colour, or turn the taillight off")
    return {"color": name, "front": front, "back": tail,
            "hue_err": round(err, 1), "conf": round(float(score), 3)}, None


def radial_profile(frame, centre_px, max_r, step=1.0):
    """Value and saturation against radius, for one light.

    The instrument for choosing `ANNULUS`. Those radii are currently a guess —
    1.1 to 2.4 times the core — and the right values depend on the lens, the
    exposure and how hard the LED is driven, none of which this module can
    know. Rather than defend the guess, this lets a person look at where the
    colour actually lives on their own camera: the white core is where `sat`
    collapses, the halo is where `sat` is high and `val` is still up, and the
    floor is where `val` falls away.

    Returns a list of dicts: r, val, sat, hue, n.
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 else None
    if hsv is None:
        return []
    fh, fw = hsv.shape[:2]
    cx, cy = float(centre_px[0]), float(centre_px[1])
    x0, x1 = max(0, int(cx - max_r) - 1), min(fw, int(cx + max_r) + 2)
    y0, y1 = max(0, int(cy - max_r) - 1), min(fh, int(cy + max_r) + 2)
    if x1 <= x0 or y1 <= y0:
        return []
    win = hsv[y0:y1, x0:x1]
    wy, wx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)

    out = []
    r = 0.0
    while r <= max_r:
        band = (d >= r) & (d < r + step)
        n = int(band.sum())
        if n:
            hp = win[band]
            ang = hp[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
            wgt = (hp[:, 1].astype(np.float64)
                   * hp[:, 2].astype(np.float64) / 255.0)
            hue = None
            if wgt.sum() > 0:
                sin_ = float((np.sin(ang) * wgt).sum())
                cos_ = float((np.cos(ang) * wgt).sum())
                hue = float((math.degrees(math.atan2(sin_, cos_)) % 360.0) / 2.0)
            out.append({"r": round(r, 2), "n": n,
                        "val": float(hp[:, 2].mean()),
                        "sat": float(hp[:, 1].mean()), "hue": hue})
        r += step
    return out

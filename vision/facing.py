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
BALL_SPAN_FRAC = 1.0        # lights further apart than a ball are not one ball
# How collinear three lights must be to be three lights on one shell. Measured
# rather than guessed: a real arrangement with a couple of pixels of noise
# across a 50px span reads 0.94-1.0, and a scattered triangle reads 0.37. The
# first value tried here was 0.35, which accepted the triangle — it sat below
# the whole range instead of inside the gap.
STRAIGHT_MIN = 0.75


def lights_px(frame, min_v=LIGHT_MIN_V, min_area=3, max_area=6000):
    """Every bright spot in the frame, with the colour in the ring around it.

    The core of a lit LED is blown out and has no hue left — that is what
    `vision/detect.py` already knows about lit shells, and it is more true of a
    bare LED than of a shell. So position comes from the blown core, which is
    crisp and stable, and colour comes from an ANNULUS outside it, where the
    light is still coloured and not yet floor.

    Returns dicts, brightest first: x, y, area, peak, bgr, hue, sat.
    """
    import cv2

    grey = (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
            if frame.ndim == 3 else frame.astype(np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (grey >= min_v).astype(np.uint8), connectivity=8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 else None
    h, w = grey.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for i in range(1, n):
        area = float(stats[i, 4])
        if area < min_area or area > max_area:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        core = math.sqrt(area / math.pi)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = ((d2 >= (ANNULUS[0] * core) ** 2)
                & (d2 <= (ANNULUS[1] * core) ** 2))
        light = {
            "x": cx, "y": cy, "area": area, "core_px": core,
            "peak": float(grey[labels == i].max()),
            "bgr": None, "hue": None, "sat": None,
        }
        if frame.ndim == 3 and ring.any():
            px = frame[ring]
            light["bgr"] = tuple(float(v) for v in px.mean(axis=0))
            hp = hsv[ring]
            # Circular mean of hue, weighted by saturation. A flat average of
            # OpenCV hue is wrong across the 179/0 seam, which is exactly where
            # red sits — and red is a colour these balls are often set to.
            ang = hp[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
            wgt = hp[:, 1].astype(np.float64)
            if wgt.sum() > 0:
                s = float((np.sin(ang) * wgt).sum())
                c = float((np.cos(ang) * wgt).sum())
                light["hue"] = float((math.degrees(math.atan2(s, c)) % 360.0) / 2.0)
                light["sat"] = float(hp[:, 1].mean())
        out.append(light)
    out.sort(key=lambda l: -l["peak"])
    return out


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


def heading_from_lights(group, blue_is_back=True):
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
    by_colour = blue_is_back and max(blueness) > 0.35 and \
        abs(blueness[0] - blueness[1]) > 0.15
    if by_colour:
        back = ends[0] if blueness[0] > blueness[1] else ends[1]
    else:
        back = ends[0] if ends[0]["peak"] <= ends[1]["peak"] else ends[1]
    front = ends[1] if back is ends[0] else ends[0]

    deg = math.degrees(math.atan2(front["y"] - back["y"],
                                  front["x"] - back["x"])) % 360.0
    # Confidence: how cleanly the two ends separated, and how sure we are which
    # way round they go. A pair that is only just distinguishable front-to-back
    # is a heading that may be 180 degrees out, which is worse than no heading.
    certainty = (abs(blueness[0] - blueness[1]) if by_colour
                 else abs(ends[0]["peak"] - ends[1]["peak"]) / 255.0)
    conf = round(float(min(1.0, span / 40.0) * min(1.0, certainty * 3.0)), 3)
    return {"deg": deg, "conf": conf, "front": front, "back": back,
            "centre": (float(centre[0]), float(centre[1])),
            "span_px": span, "straightness": round(straightness, 3),
            "by_colour": bool(by_colour), "lights": list(group)}, None


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

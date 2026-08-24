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
    import cv2

    r = int(max(4, round(radius_px * 1.4)))
    x, y = int(round(centre_px[0])), int(round(centre_px[1]))
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    patch = frame[y0:y1, x0:x1]
    grey = (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if patch.ndim == 3 else patch).astype(np.uint8)
    top = float(grey.max())
    if top < 40:
        return None

    peaks = _local_maxima(grey, top * floor)
    if len(peaks) < 2:
        return None

    # The two brightest. A third peak is a reflection or a neighbour, and
    # picking the two brightest is what keeps it out.
    peaks.sort(key=lambda p: -p[3])
    (ax, ay, _, av), (bx, by, _, bv) = peaks[0], peaks[1]
    sep = math.hypot(ax - bx, ay - by)
    if sep < MIN_SEP_PX or sep > MAX_SEP_FRAC * (2 * radius_px):
        return None

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
    return deg, round(float(min(contrast, 1.0) * room), 3)


def facing_cm(frame, centre_px, radius_px, homography, floor=PEAK_FLOOR):
    """The same thing in arena coordinates, as a compass heading.

    Taken through the homography as a pair of points rather than as an angle,
    because a perspective map does not preserve angles — rotating a bearing by
    whatever the matrix does at the image centre is wrong everywhere else in a
    tilted view, and a tilted view is what an overhead camera on a tripod is.
    """
    got = facing_px(frame, centre_px, radius_px, floor=floor)
    if got is None or homography is None or not homography.ready:
        return None
    deg, conf = got
    r = math.radians(deg)
    a = np.array(centre_px, dtype=float)
    b = a + np.array([math.cos(r), math.sin(r)]) * max(radius_px, 2.0)
    pa, pb = homography.to_cm([a.tolist(), b.tolist()])
    d = np.asarray(pb, dtype=float) - np.asarray(pa, dtype=float)
    if float(np.linalg.norm(d)) < 1e-9:
        return None
    # Compass: zero is +y, increasing clockwise — the frame `velocity_to_command`
    # speaks, so the answer can be used without another conversion.
    return float(math.degrees(math.atan2(d[0], d[1])) % 360.0), conf

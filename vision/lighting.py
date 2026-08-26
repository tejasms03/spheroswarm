"""Where the light falls, over the floor the robots actually use.

Split out of `taillight.py` so the bench and the lab share one implementation
rather than growing two. That matters more here than it usually would: the
whole point of a lighting readout is to tell you whether the exposure is in
the window where a hue survives, and two readouts that disagree about what
"blown out" means would be worse than having none.

The arena restriction is the substance of it. A bright window behind the floor
is not a lighting problem, and averaged into a whole-frame number it hides one
that is — so the frame is warped flat through the homography first, and only
the arena rectangle is measured.
"""

import numpy as np

BLOWN = 250         # V at or above this has no colour left in it
DARK = 30           # ...and below this there is nothing to key on


def arena_view(frame, hom, out_w, out_h):
    """The frame warped flat to the arena rectangle, top-down.

    Same composition the main app's floor view uses: the camera's pixel->cm
    matrix, then cm->local pixels. Returns `(image, why)` — exactly one of the
    two is None.
    """
    import cv2

    if frame is None:
        return None, "no frame yet"
    if hom is None or not hom.ready:
        return None, "no arena calibration — this is the whole frame"
    scale = min(out_w / hom.width, out_h / hom.height)
    w, h = int(hom.width * scale), int(hom.height * scale)
    if w < 2 or h < 2:
        return None, "no room to draw the arena"
    to_local = np.array([[scale, 0.0, 0.0],
                         [0.0, scale, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        return cv2.warpPerspective(frame, to_local @ hom.M, (w, h)), None
    except Exception as e:
        return None, f"warp failed: {e}"


def arena_scale(hom, out_w, out_h):
    """Pixels per centimetre in an `arena_view` of this size, or None.

    Exposed because a caller drawing on top of the map needs the same number,
    and recomputing it at the call site is how an overlay ends up half a ball
    out from the thing it is labelling.
    """
    if hom is None or not hom.ready:
        return None
    return min(out_w / hom.width, out_h / hom.height)


def brightness_map(frame, hom, out_w, out_h):
    """Where the light falls, as a heat map, with the numbers that matter.

    Value from HSV rather than a grey mix, because V is max(r,g,b) — which is
    precisely the channel that saturates. A shell reading 255 has no hue left
    for the detector and no separable peaks for a heading, so the number worth
    watching is not the average but the blown fraction.

    Returns `(image, stats, why)`. `why` is set when the map had to fall back
    to the whole frame, and the image and stats are still valid in that case.
    """
    import cv2

    warped, why = arena_view(frame, hom, out_w, out_h)
    src = warped if warped is not None else frame
    if src is None:
        return None, None, why
    v = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[:, :, 2]
    flat = v.reshape(-1)
    stats = {
        "mean": float(flat.mean()),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
        "max": int(flat.max()),
        "blown": float((flat >= BLOWN).mean() * 100.0),
        "dark": float((flat <= DARK).mean() * 100.0),
    }
    img = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
    # Blown pixels in flat red, which is nowhere on the inferno ramp. The
    # ramp's own top end is a pale yellow that reads as "bright" — the one
    # thing that must not be mistaken for "bright" is "clipped".
    img[v >= BLOWN] = (0, 0, 255)
    return img, stats, why

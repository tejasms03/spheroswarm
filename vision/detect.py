"""Colour blob detection.

Tuned for glowing translucent shells: a lit Sphero blows out to near-white in
the centre, so the usable colour signal is the saturated ring around the core.
Detection therefore keys on hue with a saturation floor, and uses the blob's
outer contour centroid rather than the brightest pixel.
"""

import cv2
import numpy as np

from . import config


class Detector:
    def __init__(self, colors=None, thresh=None):
        # Per-colour signatures come from `calib/colors.json` when one exists,
        # so a tuning session in calib.py is picked up by every app that builds
        # a Detector — which is the whole point of tuning it.
        self.colors = colors or config.load_signatures()
        self.thresh = dict(config.DEFAULT_THRESH)
        self.thresh.update(config.load("thresholds", {}) or {})
        if thresh:
            self.thresh.update(thresh)

    def save_thresholds(self):
        config.save("thresholds", self.thresh)

    def save_signatures(self):
        config.save_signatures(self.colors)

    def limits(self, name):
        """Effective thresholds for one colour: per-colour override, else global."""
        c, t = self.colors[name], self.thresh
        return {k: int(c.get(k, t[k])) for k in
                ("s_min", "v_min", "min_area", "max_area")}

    def mask_for(self, hsv, name):
        c = self.colors[name]
        t = self.limits(name)
        lo_s, lo_v = t["s_min"], t["v_min"]
        hue, tol = c["hue"], c["tol"]

        if hue - tol < 0 or hue + tol > 179:          # red wraps around 0
            a = cv2.inRange(hsv, (0, lo_s, lo_v), ((hue + tol) % 180, 255, 255))
            b = cv2.inRange(hsv, ((hue - tol) % 180, lo_s, lo_v), (179, 255, 255))
            m = cv2.bitwise_or(a, b)
        else:
            m = cv2.inRange(hsv, (hue - tol, lo_s, lo_v), (hue + tol, 255, 255))

        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m

    def detect(self, frame, only=None):
        """Return {color_name: (x_px, y_px, area)} for the best blob per colour."""
        b = self.thresh["blur"]
        blur = cv2.GaussianBlur(frame, (b | 1, b | 1), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        out = {}
        for name in (only or self.colors):
            m = self.mask_for(hsv, name)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best, best_area = None, 0
            lim = self.limits(name)
            for c in cnts:
                a = cv2.contourArea(c)
                if a < lim["min_area"] or a > lim["max_area"]:
                    continue
                if a > best_area:
                    mm = cv2.moments(c)
                    if mm["m00"] > 0:
                        best = (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"], a)
                        best_area = a
            if best:
                out[name] = best
        return out


def tune(source, detector=None):
    """Live threshold tuner. Trackbars for saturation/value floors and area."""
    d = detector or Detector()
    win = "tune — s/v floors and area. s = save, q = quit"
    cv2.namedWindow(win)
    names = list(d.colors)
    cv2.createTrackbar("s_min", win, d.thresh["s_min"], 255, lambda v: None)
    cv2.createTrackbar("v_min", win, d.thresh["v_min"], 255, lambda v: None)
    cv2.createTrackbar("min_area", win, d.thresh["min_area"], 2000, lambda v: None)
    cv2.createTrackbar("colour", win, 0, len(names) - 1, lambda v: None)

    while True:
        ok, frame = source.read()
        if not ok:
            break
        d.thresh["s_min"] = cv2.getTrackbarPos("s_min", win)
        d.thresh["v_min"] = cv2.getTrackbarPos("v_min", win)
        d.thresh["min_area"] = max(10, cv2.getTrackbarPos("min_area", win))
        name = names[cv2.getTrackbarPos("colour", win)]

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = d.mask_for(hsv, name)
        found = d.detect(frame, only=[name])
        vis = cv2.bitwise_and(frame, frame, mask=m)
        if name in found:
            x, y, a = found[name]
            cv2.circle(vis, (int(x), int(y)), 14, d.colors[name]["draw"], 2)
            cv2.putText(vis, f"{name} area={int(a)}", (14, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (228, 240, 248), 1)
        else:
            cv2.putText(vis, f"{name}: not found", (14, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 235), 1)
        cv2.putText(vis, "s = save thresholds", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (228, 240, 248), 1)
        cv2.imshow(win, np.hstack([frame, vis]) if frame.shape[1] < 900 else vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            d.save_thresholds()
        elif k in (ord("q"), 27):
            break
    cv2.destroyWindow(win)
    return d


# -- automatic signature learning -------------------------------------------
#
# Sliders are the wrong instrument for this. The trainer cannot see a hue
# histogram, so tuning by eye means nudging a number until the mask stops
# flickering — which optimises for the one frame on screen. What the trainer
# *can* do reliably is light one ball and leave it still, so that is all this
# asks of them: two short bursts of frames, LED off then LED on, and the ball
# is whatever changed between them. Background, lighting and exposure all
# cancel in the difference, so no assumption about the floor is needed.

HUE_TOL_MIN, HUE_TOL_MAX = 5, 22
CORE_PERCENTILE = 88        # drop the blown-out white core above this V


def _circular_hue_stats(hues):
    """(centre, spread) in OpenCV hue units, correct across the red wrap."""
    ang = np.radians(np.asarray(hues, dtype=float) * 2.0)     # 0-179 -> 0-358
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    centre = (np.degrees(np.arctan2(s, c)) % 360.0) / 2.0
    dev = np.abs((np.degrees(ang) - centre * 2.0 + 180.0) % 360.0 - 180.0) / 2.0
    return float(centre), float(np.percentile(dev, 90))


def locate_change(frames_off, frames_on, min_area=40):
    """Where did the picture change? Returns (mask, area, centroid) or None.

    Medians rather than means: one frame in which the shutter caught the LED
    mid-transition would drag a mean badly, and a median simply ignores it.
    """
    off = np.median(np.stack(frames_off).astype(np.float32), axis=0)
    on = np.median(np.stack(frames_on).astype(np.float32), axis=0)
    diff = np.abs(on - off).max(axis=2).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    peak = float(diff.max())
    if peak < 18:                       # nothing lit up; probably no ball
        return None
    _, m = cv2.threshold(diff, max(12.0, peak * 0.45), 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < min_area:
        return None
    # A ball is round. Anything that changed but is not round — a shifting
    # shadow, a person walking past, a whole frame that brightened — is not the
    # thing we are trying to learn the colour of.
    perim = float(cv2.arcLength(best, True))
    if perim <= 0 or (4.0 * np.pi * area / (perim * perim)) < 0.45:
        return None
    mm = cv2.moments(best)
    if mm["m00"] <= 0:
        return None
    solid = np.zeros(diff.shape, np.uint8)
    cv2.drawContours(solid, [best], -1, 255, -1)
    return solid, area, (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"])


def autotune_signature(frames_off, frames_on, blur=5):
    """Learn one colour signature from an LED-off / LED-on pair of bursts.

    Returns a dict with the signature and enough diagnostics to say whether to
    trust it — `pixels` and `area` are how a caller tells "learned a ball" from
    "learned a reflection on the floor".
    """
    found = locate_change(frames_off, frames_on)
    if found is None:
        return {"ok": False,
                "error": "nothing changed between LED off and LED on — is the "
                         "robot connected, lit, and inside the camera's view?"}
    solid, area, centroid = found

    on = np.median(np.stack(frames_on).astype(np.float32), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.GaussianBlur(on, (blur | 1, blur | 1), 0), cv2.COLOR_BGR2HSV)
    px = hsv[solid > 0]
    if len(px) < 30:
        return {"ok": False, "error": f"only {len(px)} pixels changed — too small to trust"}

    # A lit translucent shell blows out to near-white in the middle, where hue
    # is meaningless noise. The colour lives in the saturated ring, so the core
    # is dropped before any statistics are taken.
    v_cut = np.percentile(px[:, 2], CORE_PERCENTILE)
    ring = px[(px[:, 2] <= v_cut) & (px[:, 1] > 40)]
    if len(ring) < 20:
        ring = px

    hue, spread = _circular_hue_stats(ring[:, 0])
    tol = int(np.clip(round(spread * 1.6), HUE_TOL_MIN, HUE_TOL_MAX))
    s_min = int(np.clip(np.percentile(ring[:, 1], 8) * 0.85, 30, 200))
    v_min = int(np.clip(np.percentile(ring[:, 2], 8) * 0.85, 30, 200))

    sig = {
        "hue": int(round(hue)) % 180,
        "tol": tol,
        "s_min": s_min,
        "v_min": v_min,
        # Half the observed blob, so a partly-occluded ball still passes, and a
        # speck of reflected light still does not.
        "min_area": int(max(25, area * 0.4)),
        "max_area": int(max(2000, area * 6)),
    }

    # Verify before believing. A signature is only useful if it then finds the
    # ball and nothing else, and that is cheap to check against the very frames
    # it was learned from. Skipping this check is how a run against a robot the
    # camera cannot see — a simulated one, or a real one outside the frame —
    # reports a confident hue learned from whatever else moved.
    probe = Detector(colors={"x": {**sig, "draw": (255, 255, 255)}})
    mask = probe.mask_for(cv2.cvtColor(cv2.GaussianBlur(on, (blur | 1, blur | 1), 0),
                                       cv2.COLOR_BGR2HSV), "x")
    covered = float((mask > 0).mean())
    if covered > 0.06:
        return {"ok": False, "covered": round(covered, 4),
                "error": f"that signature selects {covered * 100:.0f}% of the "
                         "frame — it matched the background, not a ball"}
    found = probe.detect(on, only=["x"])
    if "x" not in found:
        return {"ok": False, "error": "the learned signature does not find the "
                                      "blob it was learned from"}
    fx, fy, _ = found["x"]
    radius = max(6.0, np.sqrt(area / np.pi))
    if np.hypot(fx - centroid[0], fy - centroid[1]) > 2.5 * radius:
        return {"ok": False, "error": "the learned signature locks onto "
                                      "something other than the ball"}

    return {
        "ok": True,
        **sig,
        "area": int(area),
        "covered": round(covered, 4),
        "pixels": int(len(ring)),
        "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
    }

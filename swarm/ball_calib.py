"""What a calibration run around the arena says about one ball.

The run is point-to-point legs corner to corner and across a diagonal. Each
leg is driven in three thirds: straight at heading 0, then held at a small
steer, then finished by ordinary point-to-point. The camera positions from the
first two thirds are what is measured here:

- **sign**   which way round a heading turns the ball, compared with the arena
             compass. Fixed by how the camera and arena were set up.
- **speed**  cm/s at the speed byte used, from the straight third.
- **delay**  seconds from a steer command until the path actually bends,
             camera latency included — which is the delay a follower lives with.
- **gain**   how much of the commanded steer the path really turned.
- **coast**  cm rolled after the stop at each corner.

The OFFSET — where the ball's heading 0 points — is deliberately NOT saved: it
changes on every reconnect and every re-aim, so a follower measures it fresh.

Everything here is a pure function of recorded samples, so a real run's log
can be re-analysed after the fact.
"""

import hashlib
import math
import time

import numpy as np

from vision import config

SCHEMA = 1
ACCEL_S = 0.8          # ignore the speed-up at the start of the straight third
BEND_SETTLE_S = 0.6    # ignore the turn itself when fitting the steered third.
                       # Measured on the bench: the path bends ~0.25s after the
                       # steer, and at 12-17cm/s the steered stretch is only
                       # ~2s long — 1.5s here left nothing to fit.
MIN_FIT_CM = 6.0       # a stretch shorter than this is noise, not a direction
MIN_LEGS = 4           # legs that must yield a sign before anything is saved
MAX_SPEED_SPREAD = 0.15
MIN_GAIN, MAX_GAIN = 0.4, 1.8


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def fingerprint(matrix):
    """A short id of the camera mapping the numbers were measured through."""
    m = np.round(np.asarray(matrix, float), 6)
    return hashlib.sha1(m.tobytes()).hexdigest()[:12]


def _fit(t, xy):
    """Direction (compass bearing), speed along it, and the fitted line.

    Direction from the principal axis of the points, oriented by time; speed
    from a regression of distance-along-that-axis on time. Returns None if the
    stretch is too short to mean anything.
    """
    t = np.asarray(t, float)
    xy = np.asarray(xy, float)
    if len(t) < 4:
        return None
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    u = vt[0]
    s = (xy - c) @ u
    if np.corrcoef(t, s)[0, 1] < 0:
        u, s = -u, -s
    if s.max() - s.min() < MIN_FIT_CM:
        return None
    slope = float(np.polyfit(t, s, 1)[0])
    bearing = math.degrees(math.atan2(u[0], u[1])) % 360.0
    return {"bearing": bearing, "speed": slope, "c": c, "u": u}


def analyse_leg(leg):
    """One leg's measurements, or a dict with `why` saying what was missing.

    `leg` is {"samples": [[t, x_cm, y_cm, seg], ...], "steer": deg,
    "steer_at": t or None, "go_at": t, "stop_at": t or None,
    "stop_xy": [x, y] or None, "end_xy": [x, y] or None, "mid": [x, y]}.
    """
    rows = leg.get("samples") or []
    out = {"mid": leg.get("mid"), "steer": leg.get("steer")}
    if leg.get("stop_xy") is not None and leg.get("end_xy") is not None:
        out["coast"] = float(np.linalg.norm(np.subtract(leg["end_xy"],
                                                        leg["stop_xy"])))
    a = [r for r in rows if r[3] == "A" and r[0] >= leg["go_at"] + ACCEL_S]
    fa = _fit([r[0] for r in a], [r[1:3] for r in a]) if a else None
    if fa is None:
        out["why"] = "straight third too short to measure"
        return out
    out["speed"] = fa["speed"]
    out["bearing_a"] = fa["bearing"]

    steer_at = leg.get("steer_at")
    if steer_at is None:
        out["why"] = "never reached the steered third"
        return out
    b = [r for r in rows if r[3] == "B" and r[0] >= steer_at + BEND_SETTLE_S]
    fb = _fit([r[0] for r in b], [r[1:3] for r in b]) if b else None
    if fb is None:
        out["why"] = "steered third too short to measure"
        return out
    turned = wrap180(fb["bearing"] - fa["bearing"])
    steer = float(leg["steer"])
    out["turned"] = turned
    out["sign"] = 1 if turned * steer > 0 else -1
    out["gain"] = abs(turned) / abs(steer)

    # Delay: where the two fitted lines cross is where an instant turn would
    # have happened. The time the ball was nearest that point, minus when the
    # steer was sent, is the delay a follower actually sees.
    d = np.array([[fa["u"][0], -fb["u"][0]], [fa["u"][1], -fb["u"][1]]])
    if abs(np.linalg.det(d)) > 1e-3:
        k = np.linalg.solve(d, fb["c"] - fa["c"])
        corner = fa["c"] + fa["u"] * k[0]
        near = [r for r in rows if r[3] in ("A", "B")]
        if near:
            gaps = [np.hypot(r[1] - corner[0], r[2] - corner[1]) for r in near]
            out["delay"] = max(0.0, near[int(np.argmin(gaps))][0] - steer_at)
    return out


def summarise(legs, byte, matrix, name, turn=None):
    """(record to save, None) or (None, why it must not be saved)."""
    got = [analyse_leg(g) for g in legs]
    signed = [g for g in got if "sign" in g]
    if len(signed) < MIN_LEGS:
        whys = sorted({g["why"] for g in got if "why" in g})
        return None, (f"only {len(signed)} of {len(got)} legs measured the "
                      f"steering (need {MIN_LEGS})"
                      + (f": {'; '.join(whys)}" if whys else ""))
    signs = {g["sign"] for g in signed}
    if len(signs) != 1:
        return None, ("legs disagree on which way it steers "
                      f"({[g['sign'] for g in signed]}) — not saving")
    gains = [g["gain"] for g in signed]
    gain = float(np.median(gains))
    if not MIN_GAIN <= gain <= MAX_GAIN:
        return None, (f"it turned {gain:.2f}x the steer asked — that is not "
                      "a heading being followed. Not saving.")
    speeds = [g["speed"] for g in got if "speed" in g]
    speed = float(np.median(speeds))
    spread = (max(speeds) - min(speeds)) / max(speed, 1e-9)
    warnings = []
    if spread > MAX_SPEED_SPREAD:
        # Reported, not refused: the bench's balls genuinely run 12-17cm/s
        # leg to leg, and the steering numbers do not depend on it.
        warnings.append(f"speed varies {spread * 100:.0f}% leg to leg "
                        f"({min(speeds):.1f}-{max(speeds):.1f}cm/s)")
    delays = [g["delay"] for g in signed if "delay" in g]
    coasts = [g["coast"] for g in got if "coast" in g]
    record = {
        "schema": SCHEMA, "name": name, "when": time.strftime("%Y-%m-%d %H:%M"),
        "homography": fingerprint(matrix),
        "sign": signs.pop(), "gain": gain, "byte": int(byte),
        "speed_cm_s": speed,
        "delay_s": float(np.median(delays)) if delays else None,
        "coast_cm": float(np.median(coasts)) if coasts else None,
        "turn": dict(turn or {}),
        "warnings": warnings,
        "regions": [{"mid": g["mid"], "speed": g.get("speed")} for g in got],
        "legs": got,
    }
    return record, None


def _key(name):
    return "ball_" + "".join(ch if ch.isalnum() or ch in "-_" else "_"
                             for ch in name)


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(record):
    config.save(_key(record["name"]), _plain(record))


def load(name, matrix):
    """(record, None) if usable with THIS camera mapping, else (None, why)."""
    rec = config.load(_key(name))
    if not rec:
        return None, f"{name} has no calibration — run calib first"
    if rec.get("schema") != SCHEMA:
        return None, f"{name}'s calibration is an old format — run calib again"
    if rec.get("homography") != fingerprint(matrix):
        return None, (f"{name} was calibrated with different corners — run "
                      "calib again")
    return rec, None

"""The ball is not on the floor, and the camera is not directly above it.

A homography maps one plane. Calibrating it on the floor — clicking tape marks
or tile corners — maps the floor, and then every ball is reported in the wrong
place, because a ball's centre sits about 3.7cm above the floor and the camera
sees it along a slanted ray. The reported position is where that ray meets the
floor, which is further from the camera than the ball really is.

The geometry is simple enough to write down. With the camera at height H over
a point N on the floor (its nadir), a ball whose centre is at height h and
whose true floor position is P is seen at

    P_obs = N + (P - N) * H / (H - h)

so everything is pushed radially away from N by a constant factor. That is
exactly the fingerprint `check pos` already reports: every error pointing away
from one place, growing with distance from it.

**Fitted, not specified.** The factor and the nadir come from measurements the
bench already takes — click a robot, say where it is, repeat. Asking for the
camera's height and position instead would mean a tape measure, a note of where
the tripod was, and a number that silently goes stale the first time somebody
nudges it. Three points of `check pos` give both, and they go stale together
with the calibration they were measured against.

The correction is the inverse and is exact for a ball at the height that was
fitted:

    P = N + (P_obs - N) / s

There is one honest alternative and it needs no arithmetic at all: calibrate
the arena by clicking a ROBOT on each corner. The homography then maps the
plane the balls travel in and the parallax is gone rather than corrected. This
module is for when that is inconvenient, or when the corners were done on the
floor and nobody wants to redo them.
"""

import numpy as np

# Below this the fit is indistinguishable from "the camera is far away or
# directly overhead", and applying it is fitting noise.
MIN_SCALE = 1.004
MIN_SAMPLES = 3


def fit(samples):
    """Least squares for (nadir, scale) from (true_cm, observed_cm) pairs.

    The model `obs = N + s(true - N)` rearranges to `obs = s*true + b` with
    `b = (1 - s)N`, which is linear in three unknowns and has a closed form —
    a uniform scaling about an unknown centre, with no rotation, because
    perspective from a point does not rotate anything.

    Returns None when there is nothing worth applying, which is the common and
    correct answer for a camera mounted high or nearly overhead.
    """
    pairs = [(np.asarray(t, dtype=float), np.asarray(o, dtype=float))
             for t, o in samples or []]
    pairs = [(t, o) for t, o in pairs
             if t.shape == (2,) and o.shape == (2,)
             and np.isfinite(t).all() and np.isfinite(o).all()]
    if len(pairs) < MIN_SAMPLES:
        return None

    u = np.array([t for t, _ in pairs])
    v = np.array([o for _, o in pairs])
    um, vm = u.mean(axis=0), v.mean(axis=0)
    spread = float(((u - um) ** 2).sum())
    if spread < 1e-6:
        return None                     # every sample in one place says nothing

    s = float(((u - um) * (v - vm)).sum() / spread)
    if not np.isfinite(s) or s <= MIN_SCALE:
        return None

    nadir = (vm - s * um) / (1.0 - s)
    if not np.isfinite(nadir).all():
        return None

    resid = v - (nadir + s * (u - nadir))
    before = float(np.linalg.norm(v - u, axis=1).mean())
    after = float(np.linalg.norm(resid, axis=1).mean())
    if after >= before:
        return None                     # the model does not explain the error

    return {"nadir_cm": [round(float(nadir[0]), 2), round(float(nadir[1]), 2)],
            "scale": round(s, 5),
            "samples": len(pairs),
            "residual_cm": round(after, 2),
            "was_cm": round(before, 2)}


def correct(p, fitted):
    """Pull one observed position back onto the plane the robot drives on."""
    if not fitted:
        return p
    try:
        n = np.asarray(fitted["nadir_cm"], dtype=float)
        s = float(fitted["scale"])
    except (KeyError, TypeError, ValueError):
        return p
    if not np.isfinite(n).all() or s <= MIN_SCALE:
        return p
    return n + (np.asarray(p, dtype=float) - n) / s


def ball_height_cm(fitted, camera_height_cm):
    """What the fitted scale implies the ball's centre height is.

    Not needed to apply the correction — it is a sanity check for a person. A
    Sphero is 7.4cm across, so a fit that implies 3-4cm is measuring the thing
    it claims to measure, and one implying 30cm is measuring something else and
    should not be trusted however well it fits.
    """
    if not fitted or not camera_height_cm:
        return None
    s = float(fitted["scale"])
    if s <= MIN_SCALE:
        return None
    return round(float(camera_height_cm) * (1.0 - 1.0 / s), 2)

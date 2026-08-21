"""Choosing which colours to use, for the room you are actually standing in.

The six hues in `config.COLORS` are a reasonable guess at a spread across the
circle. They are not a good answer for a specific room: a lab with orange
sodium lighting, a red floor mat, or a blue equipment case has background mass
sitting on some of those hues, and a robot lit to match its own background is
invisible however well its thresholds are tuned.

So rather than tuning six fixed colours until they cope, this picks the six the
room leaves free. Two things decide a hue's worth:

  SEPARATION  from the other five, because the tracker tells robots apart by
              hue and nothing else. This is the hard constraint — two robots
              closer than their tolerance windows will swap identities the
              first time they pass.

  CLARITY     from the background, weighted by how colourful that background
              is. A grey wall sits on no hue at all and costs nothing; a red
              mat occupies a stretch of the circle and anything placed there
              starts at a disadvantage no threshold can recover.

The output is a set of hues, each with the two scores that produced it, so a
person can see WHY a colour was chosen rather than being handed six numbers.
"""

import numpy as np

HUES = 180                  # OpenCV's hue range
MIN_SEPARATION = 22         # hue units; two default tolerance windows, plus room
BACKGROUND_WINDOW = 9       # hue units either side that a colour has to share


def _circ_dist(a, b, span=HUES):
    d = abs(float(a) - float(b)) % span
    return min(d, span - d)


def background_profile(frame, s_min=60, v_min=50):
    """How much colour the room already has, hue by hue.

    Weighted by saturation and brightness, because a washed-out grey pixel has
    a hue in the arithmetic sense and no hue in any sense that matters. Without
    the weighting a beige wall reads as a strong orange presence and pushes
    every robot away from a third of the circle for no reason.
    """
    import cv2
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    keep = (s >= s_min) & (v >= v_min)
    weight = (s[keep].astype(np.float64) / 255.0) * (v[keep].astype(np.float64) / 255.0)
    hist = np.bincount(h[keep], weights=weight, minlength=HUES).astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    # Smeared a little: a hue two units away is not a different colour, and a
    # spiky histogram would let a chosen hue sit in a one-unit gap that no real
    # camera could hold it in.
    k = np.ones(7) / 7.0
    return np.convolve(np.r_[hist[-3:], hist, hist[:3]], k, mode="same")[3:-3]


def room_light(frame):
    """Is the room bright enough, and is anything blown out?"""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float64)
    s = hsv[..., 1].astype(np.float64)
    blown = float((v > 250).mean())
    return {
        "brightness": round(float(v.mean()), 1),
        "saturation": round(float(s.mean()), 1),
        "blown_fraction": round(blown, 4),
        # A ball is a light source in a dark room and a coloured object in a
        # bright one, and those want opposite thresholds.
        "verdict": ("dim — the LED dominates, keep v_min low" if v.mean() < 60 else
                    "blown out — lower the exposure or the LEDs will read white"
                    if blown > 0.02 else "workable"),
    }


def collision(hist, hue, window=BACKGROUND_WINDOW):
    """How much of the room's colour sits where this hue wants to be."""
    idx = [(int(round(hue)) + d) % HUES for d in range(-window, window + 1)]
    return float(hist[idx].sum())


def clarity(hist, hue, window=BACKGROUND_WINDOW):
    """0 to 1. One means the room has nothing at this hue at all."""
    worst = max(float(hist.max()) * (2 * window + 1), 1e-9)
    return round(float(max(0.0, 1.0 - collision(hist, hue, window) / worst)), 3)


def optimise(n, hist=None, min_separation=MIN_SEPARATION, refine=True):
    """Choose `n` hues: spread apart first, then out of the room's way.

    Evenly spaced, rotated to whichever offset the background likes best, then
    nudged individually. Rotation before refinement matters: starting from an
    even spread guarantees the separation constraint is satisfiable at all, and
    every later step only has to avoid breaking it. Searching freely from
    nothing finds clusters that dodge the background beautifully and cannot be
    told apart.
    """
    n = int(n)
    if n <= 0:
        return []
    step = HUES / float(n)
    if hist is None:
        hist = np.zeros(HUES)

    best, best_cost = None, float("inf")
    for offset in range(HUES):
        hues = [(offset + i * step) % HUES for i in range(n)]
        cost = sum(collision(hist, h) for h in hues)
        if cost < best_cost:
            best_cost, best = cost, hues
    hues = list(best)

    if refine:
        # Never give up much of the even spread to dodge the background.
        # Separation is the hard constraint — two robots inside each other's
        # tolerance windows swap identities on contact, and no amount of
        # clarity compensates for the tracker calling one robot the other.
        # Clarity is soft: a colour sharing a hue with the floor is harder to
        # detect, and better thresholds still recover it.
        floor = max(min_separation, 0.75 * step)
        for _ in range(3):
            for i in range(n):
                current = hues[i]
                others = [h for j, h in enumerate(hues) if j != i]
                # Ties are broken by NOT moving. Ranking on cost alone breaks
                # them by whichever hue sorts lowest, so on a plain grey floor —
                # where every hue collides with nothing and every option ties —
                # the whole palette creeps downward and ends up crammed into
                # the first third of the circle for no reason at all.
                options = [(collision(hist, current), 0, current)]
                for delta in range(-12, 13):
                    if delta == 0:
                        continue
                    cand = (current + delta) % HUES
                    if others and min(_circ_dist(cand, o) for o in others) < floor:
                        continue
                    options.append((collision(hist, cand), abs(delta), cand))
                hues[i] = min(options)[2]

    hues = sorted(round(h) % HUES for h in hues)
    seps = [min(_circ_dist(h, o) for o in hues if o is not h) if n > 1 else HUES / 2
            for h in hues]
    return [{"hue": int(h),
             "clarity": clarity(hist, h),
             "separation": int(round(min(_circ_dist(h, o)
                                         for j, o in enumerate(hues) if j != i)))
                           if n > 1 else HUES // 2}
            for i, h in enumerate(hues)]


def score_existing(hues, hist=None):
    """Rate the palette already in use, so 'optimise' has something to beat."""
    hues = list(hues)
    if hist is None:
        hist = np.zeros(HUES)
    out = []
    for i, h in enumerate(hues):
        others = [o for j, o in enumerate(hues) if j != i]
        out.append({
            "hue": int(h),
            "clarity": clarity(hist, h),
            "separation": int(round(min((_circ_dist(h, o) for o in others),
                                        default=HUES // 2))),
        })
    return out


def summarise(entries, min_separation=MIN_SEPARATION):
    if not entries:
        return {"worst_separation": None, "worst_clarity": None, "safe": False}
    sep = min(e["separation"] for e in entries)
    cla = min(e["clarity"] for e in entries)
    return {"worst_separation": sep, "worst_clarity": round(cla, 3),
            "safe": sep >= min_separation and cla > 0.5}

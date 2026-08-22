"""Does the tracker's idea of who is who survive contact with the robots?

Colour tracking infers identity. It says "this blob is cyan, and Sunfyre wears
cyan, therefore this is Sunfyre" — and every step of that is a guess about the
room rather than a fact about the robot. When two hue windows overlap, one ball
answers to two names and the inference is wrong in a way nothing downstream can
detect: both detections are large, bright, well-formed blobs, and every static
measure agrees with both.

There is exactly one property of a Sphero that nothing else in the room shares:
**its brightness is ours to command**. That turns identity from an inference
into an experiment. Drive one robot's LED off and then on, and whichever blobs
follow belong to that robot. Blobs that do not follow are somebody else's, or
nobody's.

Run one robot at a time, and the verdict distinguishes cases that look
identical from the outside:

  CONFIRMED   its own colour responded, and only its own colour
  PHANTOM     its colour responded AND others did too — one ball, several
              names, which is the overlapping-hue failure caught red-handed
  MISASSIGNED a colour responded, but not the one this robot is wearing
  UNSEEN      nothing responded — the tracker is not watching this robot at
              all, whatever the blob count says

Cost is Bluetooth airtime, which is the scarce resource here, so this is not a
continuous check. It is worth running at the start of a session, and worth
running automatically whenever the blob count stops matching the robot count —
which is precisely the condition that has quietly spoiled previous sessions.
"""

import numpy as np

SETTLE = 0.7            # s for an LED change to reach the camera
BURST = 6               # frames averaged per phase
RESPONSE_FRACTION = 0.5  # of the lit area must appear when the LED comes on
RESPONSE_MIN_PX = 60     # ...and it must be a real blob, not sensor noise
SAMPLE_GAP = 1.0 / 45    # s between samples when the caller offers no frame id

CONFIRMED = "confirmed"
PHANTOM = "phantom"
MISASSIGNED = "misassigned"
UNSEEN = "unseen"


def mask_areas(detector, frame, colors=None):
    """Pixels passing each colour's full threshold. Not `detect()`.

    Deliberately the raw mask rather than the best contour: `detect()` filters
    on `min_area`/`max_area`, and half the palette has never been auto-tuned so
    those limits do not exist for it. A check that silently skipped the
    untuned colours would pass most confidently on exactly the colours least
    likely to be right.
    """
    import cv2
    b = detector.thresh["blur"]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (b | 1, b | 1), 0), cv2.COLOR_BGR2HSV)
    out = {}
    for name in (colors or detector.colors):
        out[name] = int(np.count_nonzero(detector.mask_for(hsv, name)))
    return out


def responded(dark_px, lit_px):
    """Did this colour follow the LED?

    Both conditions matter. The fraction rejects a blob that was already there
    and brightened a little — a reflection off a glossy floor does that. The
    floor rejects a handful of pixels flickering at the threshold, which any
    mask does frame to frame regardless of what the robot is doing.
    """
    gain = lit_px - dark_px
    return gain >= RESPONSE_MIN_PX and gain >= RESPONSE_FRACTION * lit_px


class IdentityCheck:
    """Blink each robot in turn; record which colours follow.

    A state machine stepped once per frame, like the characterisation stages,
    so the window keeps drawing and the check can be cancelled. It drives
    `set_led` and nothing else — it never moves a robot, so it is safe to run
    with balls anywhere in the arena, including in someone's hand.
    """

    def __init__(self, handles, detector, colors=None,
                 settle=SETTLE, burst=BURST):
        self.handles = list(handles)
        self.detector = detector
        self.colors = list(colors or detector.colors)
        self.settle, self.burst = settle, burst

        self.i = 0
        self.phase = "dark"
        self.t = 0.0
        self._dark, self._lit = [], []
        self._last_frame = None
        self._last_sample = -1e9
        self._restore = None
        self.results = {}
        self.done = not self.handles
        self.cancelled = False

    # -- progress --------------------------------------------------------

    @property
    def handle(self):
        return self.handles[self.i] if self.i < len(self.handles) else None

    @property
    def progress(self):
        h = self.handle
        if self.done:
            return "done"
        return f"{h.code} ({self.i + 1}/{len(self.handles)})"

    def cancel(self):
        """Stop, and give back the LED of whichever robot was mid-test."""
        self._restore_led()
        self.cancelled = True
        self.done = True

    # -- the loop --------------------------------------------------------

    def _grab(self, frame, bucket, stamp=None):
        """One sample per distinct camera frame, never the same one twice.

        A control loop runs far faster than a camera. Counting reads instead of
        frames fills both bursts from a single instant, and two averages of one
        instant differ by nothing — so every robot would read as UNSEEN.

        The caller says which frame this is. Three cheaper-looking answers were
        tried and are wrong:

        `id(frame)` identifies the object, not the picture. A frame that has
        been released is freed, and the next allocation of the same size lands
        on the same address — five separately created frames here share one id.
        Distinct pictures then read as duplicates and no burst ever fills.

        Hashing the pixels fails in the other direction: during the dark burst
        the scene is deliberately static, so consecutive genuine frames are
        near-identical and would be rejected as repeats.

        Elapsed `dt` alone cannot tell a fast loop from a fast camera.

        With no stamp the fallback is the time gate, which is right often
        enough to be useful and never silently wrong for long.
        """
        if stamp is not None:
            if stamp == self._last_frame:
                return False
            self._last_frame = stamp
        else:
            if self.t - self._last_sample < SAMPLE_GAP:
                return False
            self._last_sample = self.t
        bucket.append(mask_areas(self.detector, frame, self.colors))
        return True

    def _restore_led(self):
        h = self.handle
        if h is not None and self._restore is not None:
            try:
                h.set_led(self._restore)
            except Exception:
                pass
        self._restore = None

    def step(self, frame, dt, stamp=None):
        """One camera frame. `stamp` is anything that changes per frame — the
        tracker's `fixes_at` is exactly this and is what the bench passes."""
        if self.done or frame is None:
            return
        h = self.handle
        if h is None:
            self.done = True
            return

        self.t += dt

        if self.phase == "dark":
            if self._restore is None:
                self._restore = tuple(getattr(h, "rgb", (255, 255, 255)) or (255, 255, 255))
                h.set_led((0, 0, 0))
                self.t = 0.0
                return
            if self.t < self.settle:
                return
            if self._grab(frame, self._dark, stamp) and len(self._dark) >= self.burst:
                h.set_led(self._restore)
                self.phase, self.t = "lit", 0.0
                self._last_sample = -1e9
            return

        if self.phase == "lit":
            if self.t < self.settle:
                return
            if self._grab(frame, self._lit, stamp) and len(self._lit) >= self.burst:
                self.results[h.code] = self._verdict(h)
                self._restore = None
                self._dark, self._lit = [], []
                self.i += 1
                self.phase, self.t = "dark", 0.0
                if self.i >= len(self.handles):
                    self.done = True
            return

    # -- the verdict -----------------------------------------------------

    def _mean(self, bucket, color):
        vals = [b.get(color, 0) for b in bucket]
        return float(np.mean(vals)) if vals else 0.0

    def _verdict(self, h):
        followed = []
        detail = {}
        for c in self.colors:
            dark, lit = self._mean(self._dark, c), self._mean(self._lit, c)
            detail[c] = {"dark_px": round(dark, 1), "lit_px": round(lit, 1),
                         "gain_px": round(lit - dark, 1)}
            if responded(dark, lit):
                followed.append(c)

        own = h.color
        out = {"robot": h.code, "color": own, "followed": followed,
               "channels": detail}

        if not followed:
            out["verdict"] = UNSEEN
            out["why"] = (f"{h.code}'s LED went off and on and no colour "
                          "changed — the tracker is not watching this robot, "
                          "whatever the blob count says")
        elif own in followed and len(followed) == 1:
            out["verdict"] = CONFIRMED
            out["why"] = f"{own} followed {h.code}'s LED, and nothing else did"
        elif own in followed:
            others = [c for c in followed if c != own]
            out["verdict"] = PHANTOM
            out["why"] = (f"{h.code} is one ball answering to "
                          f"{len(followed)} names — {', '.join(others)} "
                          f"followed its LED as well as {own}. Any robot "
                          f"wearing {' or '.join(others)} is being tracked to "
                          f"this ball's position")
        else:
            out["verdict"] = MISASSIGNED
            out["why"] = (f"{h.code} wears {own}, but {', '.join(followed)} "
                          f"followed its LED — the roster and the floor "
                          "disagree about which ball this is")
        return out

    # -- the report ------------------------------------------------------

    def report(self):
        """What the whole pass found, and whether anything can be trusted."""
        rows = list(self.results.values())
        bad = [r for r in rows if r["verdict"] != CONFIRMED]

        # A colour claimed by more than one robot is the overlap seen from the
        # other side, and it is worth naming explicitly: it is the shape of
        # fault that makes two robots report one position.
        claimed = {}
        for r in rows:
            for c in r["followed"]:
                claimed.setdefault(c, []).append(r["robot"])
        contested = {c: who for c, who in claimed.items() if len(who) > 1}

        return {
            "checked": len(rows),
            "confirmed": len(rows) - len(bad),
            "results": self.results,
            "contested_colours": contested,
            "ok": not bad and not contested,
            "summary": _summarise(rows, bad, contested),
        }


def _summarise(rows, bad, contested):
    if not rows:
        return "nothing checked"
    if not bad and not contested:
        return f"all {len(rows)} robots are who the tracker thinks they are"
    parts = []
    for r in bad:
        parts.append(f"{r['robot']}: {r['verdict']}")
    for c, who in contested.items():
        parts.append(f"{c} is claimed by {' and '.join(who)}")
    return "; ".join(parts)

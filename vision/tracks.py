"""Identity a person hands over once, carried frame to frame.

Colour is the other way to know which ball is which, and it is the one the
rest of this project uses: a hue is re-derived from scratch every frame, so it
survives an occlusion, a restart, or somebody picking a robot up and putting it
down somewhere else. Nothing has to be remembered, so nothing can be forgotten.

That robustness is bought with light. A colour has to SURVIVE to the camera —
it needs saturation in the halo, which needs exposure, which needs the cores
not to blow out — and a primary spends two thirds of the LED to get there: pure
red drives one die of three. On a rig where the lights are already only a
handful of pixels across, that is the difference between a blob the reader can
work with and one it cannot.

So this is the other trade. A person says "that one is SK-4C2C, and THAT end is
its front" once, and from then on identity is maintained by association rather
than re-read. The robots can then all glow white at full power, because their
colour no longer has a job.

WHAT THIS BUYS, AND WHAT IT COSTS. Maintained identity cannot recover by
itself. Colour comes back after an occlusion; this does not — it has to be
told again. That is an acceptable trade only if the moment it breaks is
OBVIOUS, which is why most of the code below is about noticing rather than
about tracking:

    a track whose cluster left the gate is LOST, and says so. It does not
    keep its last position on screen looking live, because a stale dot a
    controller cannot distinguish from a fresh one is the one failure that
    controller cannot defend against.

    two tracks reaching for the same cluster is CONTENTION, and neither is
    updated. Picking the closer one is how a swap becomes permanent, and a
    swap that is merely unlikely is a swap that happens on the day it matters.

WHY THE HEADING CANNOT SIMPLY BE REMEMBERED. "Which blob is the front" is not
a property of a blob — the blobs move as the ball turns, and they are detected
afresh each frame with no identity of their own. What is remembered is a
DIRECTION. Each frame the cluster gives an axis, which is a line and therefore
two candidate headings; the one nearer last frame's heading wins. At 360 deg/s
and 30fps a ball turns about 12 degrees between frames while a flip is 180, so
that choice has a wide margin and cannot quietly go the wrong way.
"""

import math
from collections import deque

import numpy as np

# How far a track may reach for a cluster, in pixels, per second of elapsed
# time. A Sphero tops out near 60cm/s; at a typical 13px/cm that is 780px/s.
# The gate is deliberately generous — a missed association costs a LOST that a
# person must clear, while an over-tight gate does that on every fast pass.
GATE_PX_PER_S = 900.0
GATE_FLOOR_PX = 45.0
"""Below this the gate is noise rather than motion: a parked ball's centroid
still moves by a pixel or two, and a gate under that loses tracks standing
still."""

# Two tracks are contending when the same cluster is the best answer for both
# and neither is clearly closer. Measured as a ratio of distances rather than
# an absolute, because the whole point is that they are near each other.
CONTENTION_RATIO = 1.6

# How far the heading may swing in one second before the flip is not believed.
# A Sphero's drive assembly manages about 360 deg/s; this sits above it so a
# genuine hard turn is not rejected, and far under the 180 degrees that a
# front/back reversal would need.
MAX_TURN_DEG_PER_S = 540.0
MIN_TURN_GATE_DEG = 60.0

# Motion confirms heading: a Sphero rolls in the direction it faces, so a
# ball that has travelled has voted on which way it points. But only TRAVEL
# votes, and most movement is not travel.
#
# A ball coming to rest overshoots, rolls back, overshoots less — and the LEDs
# ride the internal chassis, which stays put, so the FACING never changes while
# the centre swings back and forth across it. Taking each swing as a vote reads
# every rebound as "you are pointing the wrong way": measured on a settling
# ball that never turned, the naive version flipped the heading four times and
# left it 180 degrees out for most of the settle.
#
# So a vote needs the motion to be going SOMEWHERE. Net displacement across a
# window, against the path length walked to get there: real driving scores near
# one, an oscillation scores near zero because it keeps coming back.
CONFIRM_MIN_PX = 25.0
CONFIRM_STRAIGHT = 0.7
CONFIRM_WINDOW_S = 0.6
# ...and even then, one disagreement is not enough to act on. A hand pushing
# the ball, a bump off a wall, or the tail of a turn all move a ball in a
# direction it is not facing, briefly. A wrong HEADING does not go away, so
# insisting the vote repeats costs a tenth of a second and rejects everything
# that was only passing.
#
# The votes have to be INDEPENDENT to mean anything. The history window slides,
# so a single shove sits inside it for as long as the window is deep and gets
# counted once per frame — three votes from one event, which is the repeat rule
# fooling itself. The trail is cleared after every vote, so each one is paid
# for with fresh travel.
CONFIRM_VOTES = 3

# A FLIP THAT DID NOT HELP IS NOT A FLIP TO REPEAT. Once turned round, the ball
# needs time to prove it; and if turning it round twice has not fixed the
# closing distance then the fault is not a reversal — a wrong scale, a wrong
# homography, or a ball against a wall all look like "it will not come to me",
# and none of them is cured by aiming the other way.
FLIP_COOLDOWN_S = 1.0
MAX_FLIPS = 2

# How much further from its setpoint a ball must get before that counts as
# going the wrong way rather than as tracker noise.
AWAY_PX = 18.0


def wrap180(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


class Track:
    """One robot, as this app currently believes it to be."""

    def __init__(self, name, centre, heading, at=0.0):
        self.name = name
        self.centre = tuple(float(v) for v in centre)
        self.heading = float(heading) % 360.0
        self.at = float(at)                 # when it was last actually seen
        self.lost = False
        self.contended = False
        self.why = None
        self.seen = 1                       # frames associated, ever
        self.confirmed_at = None            # last time motion agreed
        self.flips_caught = 0
        self.travelling = False             # is the motion going somewhere?
        self._last_flip_at = None
        self._trail = deque()               # (t, centre), recent history
        self._against = 0                   # consecutive disagreeing votes

    @property
    def live(self):
        return not self.lost and not self.contended

    @property
    def course(self):
        """Which way it has actually been going, in image degrees, or None.

        Net displacement over the history window, not the last step — one
        frame of travel at this scale is a couple of pixels and mostly noise.
        Returns None until it has gone somewhere worth measuring.
        """
        if len(self._trail) < 3:
            return None
        a = np.asarray(self._trail[0][1], float)
        b = np.asarray(self._trail[-1][1], float)
        d = b - a
        if float(np.linalg.norm(d)) < CONFIRM_MIN_PX:
            return None
        return math.degrees(math.atan2(d[1], d[0])) % 360.0

    def __repr__(self):                     # pragma: no cover - debugging aid
        return (f"<Track {self.name} {'LOST' if self.lost else 'live'} "
                f"at {self.centre[0]:.0f},{self.centre[1]:.0f} "
                f"{self.heading:.0f}deg>")


class Tracks:
    """Every assigned robot, and the association that keeps them assigned."""

    def __init__(self, gate_px_per_s=GATE_PX_PER_S):
        self.gate_rate = float(gate_px_per_s)
        self.by_name = {}
        self.t = 0.0

    # -- handing identity over -------------------------------------------

    def assign(self, name, cluster, front_px=None):
        """Say that this cluster is this robot, and which end is its front.

        `front_px` is a point the person clicked — the light they say is at the
        FRONT. The heading is taken from the cluster's own axis and pointed at
        whichever end that click is nearer, so a rough click on the right light
        is enough; the precision comes from the axis, not from the finger.
        """
        centre = _centre_of(cluster)
        axis = _axis_of(cluster)
        if centre is None or axis is None:
            return None, "that cluster has no axis to take a heading from"
        deg = math.degrees(math.atan2(axis[1], axis[0])) % 360.0
        if front_px is not None:
            to_click = np.asarray(front_px, dtype=float) - np.asarray(centre)
            if float(np.dot(to_click, axis)) < 0:
                deg = (deg + 180.0) % 360.0
        t = Track(name, centre, deg, at=self.t)
        self.by_name[name] = t
        return t, None

    def drop(self, name=None):
        if name is None:
            self.by_name.clear()
        else:
            self.by_name.pop(name, None)

    @property
    def names(self):
        return list(self.by_name)

    @property
    def live(self):
        return [t for t in self.by_name.values() if t.live]

    # -- carrying it ------------------------------------------------------

    def update(self, clusters, dt, targets=None):
        """Associate every track with a cluster, or lose it out loud.

        `clusters` are this frame's readings — anything with a `centre` and a
        `group`. `targets` optionally maps a name to where that robot is
        currently being DRIVEN, in the same pixel frame; see `_vote`, which
        much prefers that evidence to guessing from travel alone.
        """
        dt = max(float(dt), 1e-3)
        self.t += dt
        gate = max(self.gate_rate * dt, GATE_FLOOR_PX)

        usable = [c for c in (clusters or []) if _centre_of(c) is not None]
        # Every track-to-cluster distance inside the gate, nearest first. A
        # global sort rather than a per-track pick: taking each track's own
        # nearest in turn lets whichever track is considered first steal a
        # cluster that belonged to another, and the theft is order-dependent
        # and therefore invisible.
        pairs = []
        for t in self.by_name.values():
            for i, c in enumerate(usable):
                d = math.dist(t.centre, _centre_of(c))
                if d <= gate:
                    pairs.append((d, t.name, i))
        pairs.sort()

        taken_cluster, taken_track, chosen = set(), set(), {}
        for d, name, i in pairs:
            if name in taken_track or i in taken_cluster:
                continue
            # Is another track nearly as close to this cluster? If so neither
            # of them may have it: a swap that is merely unlikely is a swap.
            rival = next((dd for dd, nn, ii in pairs
                          if ii == i and nn != name and nn not in taken_track),
                         None)
            if rival is not None and rival <= d * CONTENTION_RATIO:
                continue
            taken_track.add(name)
            taken_cluster.add(i)
            chosen[name] = (usable[i], d)

        for name, t in self.by_name.items():
            got = chosen.get(name)
            if got is None:
                self._miss(t, usable, gate)
                continue
            cluster, d = got
            self._hit(t, cluster, d, dt, (targets or {}).get(name))
        return list(self.by_name.values())

    def _hit(self, t, cluster, d, dt, target=None):
        centre = _centre_of(cluster)
        axis = _axis_of(cluster)
        t.contended = False
        t.lost = False
        t.why = None
        t.seen += 1
        t.at = self.t

        if axis is not None:
            deg = math.degrees(math.atan2(axis[1], axis[0])) % 360.0
            # The axis is a line: take whichever end is nearer the heading we
            # already believe. This is the whole of "it remembers the front".
            if abs(wrap180(deg - t.heading)) > 90.0:
                deg = (deg + 180.0) % 360.0
            swing = abs(wrap180(deg - t.heading))
            limit = max(MAX_TURN_DEG_PER_S * dt, MIN_TURN_GATE_DEG)
            if swing <= limit:
                t.heading = deg
            else:
                # Too fast to be a turn. Keep the old heading and say so
                # rather than following the cluster into a reading that the
                # plant cannot have produced.
                t.why = (f"axis jumped {swing:.0f}deg in {dt*1000:.0f}ms — "
                         "holding the previous heading")

        t.centre = tuple(float(v) for v in centre)
        self._vote(t, centre, target)

    def _vote(self, t, centre, target=None):
        """Let TRAVEL confirm the facing — and nothing else.

        See CONFIRM_STRAIGHT: the question asked of the recent history is not
        "has it moved" but "has it gone anywhere", because a ball rocking
        itself to a standstill has done the first and not the second.
        """
        t._trail.append((self.t, tuple(float(v) for v in centre)))
        while t._trail and self.t - t._trail[0][0] > CONFIRM_WINDOW_S:
            t._trail.popleft()
        if len(t._trail) < 3:
            t.travelling = False
            return

        pts = [p for _, p in t._trail]
        net_v = np.asarray(pts[-1], float) - np.asarray(pts[0], float)
        net = float(np.linalg.norm(net_v))
        path = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        straight = net / path if path > 1e-9 else 0.0

        # WHERE IT WAS TOLD TO GO beats guessing from how it looks. Travel
        # against facing is vision checking vision, and it needs a straightness
        # rule to keep a settling ball from voting. Travel against the SETPOINT
        # needs none of that: a ball that is being driven somewhere and is
        # getting further from it is pointing the wrong way, and the fact that
        # it wobbled on the way is beside the point.
        #
        # `calib.py` already recognises this state and refuses on it — "drove
        # 12cm AWAY from the middle, that is the aim frame being wrong, not the
        # drive". The only new idea here is repairing it instead.
        if target is not None:
            was = math.dist(pts[0], target)
            now = math.dist(pts[-1], target)
            if net < CONFIRM_MIN_PX:
                return              # it has not gone far enough to have tried
            t._trail.clear()
            if now <= was - AWAY_PX:
                t._against = 0      # closing: whatever it is doing, it works
                t.confirmed_at = self.t
                t.travelling = True
                return
            if now < was + AWAY_PX:
                return              # sideways, or stalled on a wall
            t.travelling = True
            self._against_vote(t, "it is driving AWAY from its setpoint")
            return
        t.travelling = net >= CONFIRM_MIN_PX and straight >= CONFIRM_STRAIGHT
        if not t.travelling:
            # Two very different silences, and only one of them is evidence.
            #
            # A window that has simply not accumulated enough travel yet says
            # nothing — clearing the count there means a slow ball can never
            # reach the repeat threshold at all, and the flip guard becomes a
            # flip ban.
            #
            # A window with plenty of PATH and no net displacement is the
            # settle: the ball has been busy going nowhere, which actively
            # contradicts any disagreement already banked.
            if path >= CONFIRM_MIN_PX and straight < CONFIRM_STRAIGHT:
                t._against = 0
            return

        course = math.degrees(math.atan2(net_v[1], net_v[0])) % 360.0
        # Spent: whatever happens next, this stretch of travel has been counted
        # and the next vote must be earned with new movement.
        t._trail.clear()
        if abs(wrap180(course - t.heading)) <= 90.0:
            t._against = 0
            t.confirmed_at = self.t
            return

        self._against_vote(t, "travel disagrees with facing")

    def _against_vote(self, t, because):
        """Bank one vote that the heading is backwards, and act on enough."""
        t._against += 1
        if t._against < CONFIRM_VOTES:
            t.why = (f"{because} ({t._against} of {CONFIRM_VOTES}) — "
                     "watching before turning it round")
            return
        t._against = 0
        if t.flips_caught >= MAX_FLIPS:
            t.why = ("turning it round has not helped twice — this is not a "
                     "reversed heading. Check the arena calibration, or "
                     "whether the ball is against something")
            return
        if (t._last_flip_at is not None
                and self.t - t._last_flip_at < FLIP_COOLDOWN_S):
            t.why = "just turned round — giving it a moment to show"
            return
        t.heading = (t.heading + 180.0) % 360.0
        t.flips_caught += 1
        t._last_flip_at = self.t
        t.confirmed_at = self.t
        t._trail.clear()
        t.why = f"{because} — turned the heading round and carrying on"

    def _miss(self, t, usable, gate):
        """No cluster inside the gate, or the claim was contested."""
        near = min((math.dist(t.centre, _centre_of(c)) for c in usable),
                   default=None)
        t.lost = True
        t.contended = near is not None and near <= gate
        if not usable:
            t.why = "nothing lit in frame"
        elif near is None:
            t.why = "no cluster to reach for"
        elif t.contended:
            t.why = ("two robots are reaching for the same cluster — "
                     "neither has been moved, click to reassign")
        else:
            t.why = (f"nearest cluster is {near:.0f}px away, past the "
                     f"{gate:.0f}px gate — occluded, or it left the frame")


# -- reading a cluster, whatever shape the caller's dict is ----------------

def _centre_of(cluster):
    got = cluster.get("centre") if isinstance(cluster, dict) else None
    if got is None and isinstance(cluster, dict):
        got = cluster.get("centre_px")
    return tuple(float(v) for v in got) if got is not None else None


def _axis_of(cluster):
    """The cluster's long axis as a unit vector, front-first if it knows.

    Prefers the reader's own front/back call when there is one, because that
    is measured from the lights themselves. Falls back to the raw geometry,
    which is a line with no direction — and a line is all this needs, since
    the direction comes from continuity.
    """
    if not isinstance(cluster, dict):
        return None
    # An axis handed over directly wins: it is what a MERGED cluster has
    # instead of two blob centres, and the case the peak reader refuses on.
    given = cluster.get("axis")
    if given is not None:
        v = np.asarray(given, dtype=float)
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            return v / n
    front, back = cluster.get("front"), cluster.get("back")
    if front and back:
        v = np.array([front["x"] - back["x"], front["y"] - back["y"]], float)
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            return v / n
    pts = [(l["x"], l["y"]) for l in (cluster.get("group") or [])]
    if len(pts) < 2:
        return None
    p = np.asarray(pts, dtype=float)
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    v = vt[0]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None

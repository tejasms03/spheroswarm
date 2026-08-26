"""Turn a drawn path into the list of commands you would hand a Sphero.

A Sphero's entire motion API is `roll(heading, speed, duration)`: a compass
course, a speed byte, and how long to hold it. Two things follow from that, and
both are the reason this module exists.

The first is that heading and speed arrive TOGETHER. There is no separate turn
command — a roll at speed 0 turns the ball on the spot and goes nowhere, and a
roll at speed with a new heading turns it while it drives. So "yaw" and "roll"
here are the same command with the speed byte set differently, and the compiler
below chooses between them: a gentle bend is taken while moving, a sharp corner
is a stop, a turn, and a fresh leg.

The second is that a duration has to be chosen in ADVANCE, from a model of the
ball. It does not reach its commanded speed when the command lands — it reaches
it about a second later, having spent a fifth of a second not moving at all —
and it does not stop when the command ends either. `compile_path` sizes every
leg against those measured constants, which is why the arithmetic on each line
is not simply distance over speed.

Everything is measured rather than invented (see `calib/motion.json`):

    the link        commands land a few times a second, a fifth of a second late
    the deadband    a speed byte under ~18 turns the motors not at all
    the yaw         a new course is a physical turn, not a teleport
    the camera      a closed-loop pilot sees a noisy fix, never the truth

Two pilots run the same plant. `Program` executes a compiled list open-loop,
which is what you would actually send to a ball over BLE and the honest test of
whether the plan was any good. `RollFollower` closes the loop on the camera
instead. The gap between their trails is the value of the camera, in
centimetres.
"""

import math

import numpy as np

from fleet.handle import MAX_SPEED
from swarm.pd import Polyline, dedupe

DEFAULT_YAW_RATE = 180.0        # deg/s the drive assembly can swing through
DEFAULT_CMD_HZ = 6.0            # roll commands per second the link will carry
DEFAULT_FIX_SIGMA = 0.35        # cm, tracker noise; measured 0.31-0.34
DEFAULT_FIX_HZ = 30.0


def wrap180(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def bearing(v):
    """A vector in the arena frame as a Sphero heading: 0 is +y, clockwise."""
    v = np.asarray(v, dtype=float)
    if float(np.linalg.norm(v)) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360.0)


def heading_vector(deg):
    """The inverse of `bearing`: a unit vector for a Sphero heading."""
    rad = math.radians(float(deg))
    return np.array([math.sin(rad), math.cos(rad)])


# -- what the drawing hands over ---------------------------------------------

def simplify(points, tol_cm=1.0):
    """Douglas-Peucker. A scribble becomes the polyline somebody meant to draw.

    Freehand arrives at the frame rate of the mouse, so a slow hand produces
    hundreds of points a millimetre apart and every one of them is a corner the
    follower would try to take. Dropping the ones that lie on a line through
    their neighbours changes the shape by no more than `tol_cm` and leaves a
    path whose vertices mean something.
    """
    pts = dedupe(points)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        denom = float(ab @ ab)
        seg = pts[i + 1:j]
        if denom < 1e-12:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            u = np.clip(((seg - a) @ ab) / denom, 0.0, 1.0)
            d = np.linalg.norm(seg - (a + np.outer(u, ab)), axis=1)
        k = int(np.argmax(d))
        if float(d[k]) > float(tol_cm):
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return pts[keep]


# -- the units the radio speaks ----------------------------------------------

class SpeedMap:
    """Speed byte to cm/s: what a command MEANS, and the byte below which it
    means nothing at all.

    Two of these exist in any honest run and confusing them is the classic
    sim-to-real bug. One is the truth — what the ball does with a byte, which
    only the ball knows. The other is the BELIEF, which is whatever the last
    calibration wrote down, and it is the belief a plan is compiled against.
    On a ball with a 0.69 speed gain and nobody having measured it, the two
    differ by a third, and a plan is a third short before it is even sent.

    The deadband is the other half. Ask a Sphero for 2cm/s and it does not
    creep — it sits there, and a controller with no model of that sits there
    with it, politely asking. So a command either clears the deadband or is a
    stop, and that decision is made here rather than inside a control loop.
    """

    def __init__(self, max_speed=MAX_SPEED, min_moving_byte=0, cm_s_per_byte=None):
        self.min_moving_byte = int(min_moving_byte)
        self.slope = float(cm_s_per_byte or float(max_speed) / 255.0)

    @property
    def max_speed(self):
        return self.slope * 255.0

    @classmethod
    def from_motion(cls, fit):
        """Built from a `calib/motion.json` entry — the deadband, mostly.

        The measured speed CURVE is deliberately not used unless the fit stood
        behind it. On the ball this was written against, ten samples spanning
        18-76 bytes fitted at r2=0.23 and extrapolated to a 41cm/s top speed the
        run itself flagged as untrustworthy; taking that literally would put the
        deadband above the speed cap and the robot could not legally be asked to
        move. The deadband BYTE is a direct observation — the lowest byte the
        ball was seen to move at — and it survives its own fit being poor.
        """
        fit = fit or {}
        rec = fit.get("recommend") or {}
        sm = fit.get("speed_map") or {}
        slope = None
        if sm.get("max_speed_trusted") and sm.get("cm_s_per_byte"):
            slope = float(sm["cm_s_per_byte"])
        dead = rec.get("min_moving_byte") or sm.get("min_moving_byte") or 0
        return cls(min_moving_byte=int(dead), cm_s_per_byte=slope)

    @classmethod
    def truth_for(cls, robot, min_moving_byte=0):
        """What the ball ACTUALLY does with a byte — sim only, obviously.

        The simulator holds its speed gain openly, so this is the speed map a
        perfect calibration run would come back with. Handing it to a planner
        is the "you measured this ball" condition; handing the nominal one
        instead is the "you did not" condition, and the two are worth being
        able to run back to back.
        """
        gain = float(getattr(robot, "gain", 1.0) or 1.0)
        return cls(min_moving_byte=min_moving_byte,
                   cm_s_per_byte=MAX_SPEED / 255.0 * gain)

    @property
    def min_moving_cm_s(self):
        return self.min_moving_byte * self.slope

    def byte_for(self, cm_s):
        """Round a speed up to the deadband, or ask for nothing."""
        cm_s = float(cm_s)
        if cm_s <= 0.0:
            return 0
        byte = int(round(cm_s / max(self.slope, 1e-9)))
        if byte <= 0:
            return 0
        return int(min(max(byte, self.min_moving_byte), 255))

    def cm_s_for(self, byte):
        byte = int(min(max(int(byte), 0), 255))
        if byte < self.min_moving_byte:
            return 0.0
        return byte * self.slope


# -- the ball, spoken to only in roll commands -------------------------------

class RollPlant:
    """A robot handle that can only be addressed as `roll(heading, byte)`.

    It wraps a handle rather than replacing one, so the dynamics below are the
    measured ones — command latency, motor lag, per-ball gain, slip, heading
    drift — and a controller written here is not being flattered by a livelier
    machine than the hardware.

    Two things live at this level because they belong to the ball and not to
    the controller. The deadband, because the motors either turn or they do
    not; and the yaw rate, because a heading change is a drive assembly
    physically swinging round. A commanded course the ball has not reached yet
    is a course it is not driving, and on a tight corner that gap is most of
    what the tracking error is made of.
    """

    def __init__(self, robot, speeds=None, yaw_rate=DEFAULT_YAW_RATE):
        self.robot = robot
        # The NOMINAL map, always: this is the ball's own reading of a byte,
        # and whatever else it gets wrong it does not get its own units wrong.
        # The speed gain that makes a byte mean less than it says lives in the
        # handle underneath, which is where the plant's errors belong.
        self.speeds = speeds or SpeedMap()
        self.yaw_rate = float(yaw_rate)
        self.heading = 0.0              # where the assembly points NOW
        self.commanded = 0.0            # where it was last told to point
        self.byte = 0
        self.commands = 0               # roll commands accepted
        self.log = []                   # (t, heading, byte), newest last
        self.t = 0.0

    @property
    def pos(self):
        return self.robot.pos

    @property
    def vel(self):
        return self.robot.vel

    @property
    def speed(self):
        return float(np.linalg.norm(self.robot.vel))

    @property
    def turning(self):
        """Degrees still to swing before it is driving the course it was given."""
        return abs(wrap180(self.commanded - self.heading))

    def roll(self, heading_deg, speed_byte):
        """The whole API. Held until the next one, exactly like the hardware."""
        self.commanded = float(heading_deg) % 360.0
        self.byte = int(min(max(int(speed_byte), 0), 255))
        self.commands += 1
        self.log.append((self.t, self.commanded, self.byte))
        del self.log[:-200]

    def stop(self):
        self.roll(self.commanded, 0)

    def step(self, dt):
        # Swing toward the commanded course at a finite rate. A stopped ball
        # still turns — that is what a Sphero does when told to roll at zero,
        # and it is why aiming before moving is a thing you can do at all.
        err = wrap180(self.commanded - self.heading)
        if self.yaw_rate > 0:
            limit = self.yaw_rate * dt
            err = max(-limit, min(limit, err))
        self.heading = (self.heading + err) % 360.0

        cm_s = self.speeds.cm_s_for(self.byte)
        # Through the handle's velocity API because that is where the arena
        # frame, the heading offset and the plant live. The BYTE was already
        # turned into cm/s here, by the map that knows this ball's deadband and
        # ceiling, so nothing downstream converts it a second time.
        self.robot.set_velocity(heading_vector(self.heading) * cm_s)
        self.robot.step(dt)
        self.t += dt


class Camera:
    """What the controller is allowed to know: a noisy fix at a finite rate.

    Handing a controller the true position is the single easiest way to build a
    path follower that cannot be transferred. 30fps and a third of a centimetre
    are what the tracker on this arena actually delivers.
    """

    def __init__(self, sigma_cm=DEFAULT_FIX_SIGMA, fps=DEFAULT_FIX_HZ, seed=None):
        self.sigma = float(sigma_cm)
        self.fps = float(fps)
        self.rng = np.random.default_rng(seed)
        self.last = None
        self._since = 1e9

    def fix(self, true_pos, dt):
        self._since += dt
        if self.last is None or self._since >= 1.0 / max(self.fps, 1e-6):
            self._since = 0.0
            noise = self.rng.normal(0.0, self.sigma, 2) if self.sigma > 0 else 0.0
            self.last = np.asarray(true_pos, dtype=float) + noise
        return self.last


# -- the follower ------------------------------------------------------------

class RollFollower:
    """Pure pursuit over a `Polyline`, speaking only `roll(heading, byte)`.

    Pure pursuit rather than a PD loop, for the reason `TurnAndGo` exists: this
    link carries a handful of commands a second, so a controller that forms a
    fresh opinion every frame spends its whole budget arguing with itself. Aim
    at a point a fixed distance ahead ON THE PATH and the command stays correct
    for as long as it takes the next one to arrive — the lookahead is doing the
    job the missing bandwidth would otherwise have to do.

    The lookahead is the one knob that matters and it trades two failures
    against each other: short cuts no corners and wobbles, long is smooth and
    cuts every corner it meets. There is no setting that does neither, which is
    why it is on a slider rather than in a constant.
    """

    SEARCH_AHEAD = 4.0          # x lookahead; how far one projection may skip
    MIN_SPEED_FRAC = 0.35       # never slow below this fraction on a corner
    MAX_CUT_CM = 3.0            # of shape a lookahead may skip over
    MIN_LOOKAHEAD_CM = 4.0
    CUT_SAMPLES = 7

    def __init__(self, path, speed=20.0, lookahead=18.0, arrive_cm=6.0,
                 cmd_hz=DEFAULT_CMD_HZ, heading_deadband=4.0, byte_deadband=6,
                 speeds=None, corner_gain=1.0, stop_s=1.0, laps=None):
        self.path = path
        self.speed = float(speed)
        self.lookahead = float(lookahead)
        self.arrive = float(arrive_cm)
        self.cmd_hz = float(cmd_hz)
        self.heading_deadband = float(heading_deadband)
        self.byte_deadband = int(byte_deadband)
        self.speeds = speeds or SpeedMap()
        # How hard to slow for a corner, and how long the ball takes to stop.
        # The second is not a control gain but a measurement: dead time plus
        # motor lag, the distance it runs on after the last command, and the
        # only reason a path can be finished ON the end point rather than a
        # stopping distance past it.
        self.corner_gain = float(corner_gain)
        self.stop_s = float(stop_s)

        self.progress = 0.0             # arclength reached, cm
        self.target = None              # the lookahead point, for drawing
        self.cross_track = 0.0          # cm off the path, right now
        self.command = None             # (heading, byte) last SENT
        self.commands = 0
        self.since = 1e9                # s since the last command was sent
        self.done = False
        self.laps = 0
        self.laps_max = laps
        self.lap_s = 0.0                # position WITHIN the lap, cm

    @property
    def remaining(self):
        if self.path.loop:
            return math.inf
        return max(self.path.length - self.progress, 0.0)

    def step(self, pos, dt):
        """A fix in, a roll command out — or None, meaning 'nothing new to say'.

        None is not idleness. A Sphero holds its last command, so re-sending one
        that has not meaningfully changed buys nothing and costs airtime the
        next real correction needs.
        """
        if self.done:
            # The stop went out once, when it arrived. Repeating it every frame
            # while the ball coasts is not a controller doing anything — it is
            # a command count that flatters or damns nothing, on airtime a real
            # link would rather have back.
            return None
        self.since += dt
        pos = np.asarray(pos, dtype=float)

        length = max(self.path.length, 1e-9)
        window = self.SEARCH_AHEAD * self.lookahead
        if self.path.loop:
            # Never let one projection see more than half the loop, or a small
            # shape lets the search jump straight to the other side of itself.
            window = min(window, length / 2.0)
        # A projection is always a position WITHIN the shape, so that is what
        # the search is anchored to. `progress` on a loop is total distance
        # driven and can be several laps' worth, which no projection can match.
        s = self.path.project(pos, from_s=self.lap_s if self.path.loop
                              else self.progress, window=window)
        if self.path.loop:
            # Laps are counted from the JUMP, not from the arithmetic: crossing
            # the seam forwards reads as the position falling most of a lap,
            # and crossing it backwards as the position rising by the same.
            #
            # Deriving the lap from `progress % length` instead is subtly
            # broken and was: a projection that lands exactly ON the seam gives
            # a remainder of zero, which is indistinguishable from having just
            # started, and the base silently advances a full lap without any
            # crossing being seen. The ball then drove a second lap looking for
            # a finish line it had already gone past.
            delta = s - self.lap_s
            if delta < -length / 2.0:
                self.laps += 1
                if self.laps_max and self.laps >= self.laps_max:
                    self.done = True
                    self.target = None
                    self.lap_s = s
                    return self._send(self.command[0] if self.command else 0.0,
                                      0, force=True)
            elif delta > length / 2.0:
                self.laps = max(self.laps - 1, 0)
            self.lap_s = s
            self.progress = self.laps * length + s
        else:
            self.progress = max(self.progress, s)
        here, _ = self.path.point_at(self.progress)
        self.cross_track = float(np.linalg.norm(pos - here))

        end = self.path.points[-1]
        if not self.path.loop and self.remaining <= self.arrive \
                and float(np.linalg.norm(pos - end)) <= self.arrive:
            self.done = True
            self.target = None
            return self._send(self.command[0] if self.command else 0.0, 0, force=True)

        reach = self._reach_from(pos)
        target, _ = self.path.point_at(self.progress + reach)
        if not self.path.loop and self.progress + reach >= self.path.length:
            target = end
        self.target = target

        to = target - pos
        heading = bearing(to)
        if heading is None:
            return None

        speed = self._speed_for(heading, float(np.linalg.norm(to)))
        return self._send(heading, self.speeds.byte_for(speed))

    def _reach_from(self, pos):
        """How far ahead to aim, shortened until the shortcut stops being one.

        A fixed lookahead is two settings, both wrong. Long is smooth and cheap
        on commands and cuts every corner it meets — on a curve tighter than
        the lookahead the ball simply drives the chord, which is what a 12cm
        tracking error looks like on a hand-drawn S. Short follows anything and
        wobbles down a straight, spending the whole command budget correcting
        noise.

        So it is not fixed. The slider sets the most it may reach, and the
        reach is pulled in until the path between here and the aim point stays
        within `MAX_CUT_CM` of the straight line the ball would actually
        travel. Straights keep the full lookahead; a tight bend gets whatever
        it can have. The failure this removes is silent — a beautifully smooth
        trail through entirely the wrong place.
        """
        reach = self.lookahead
        for _ in range(4):
            if reach <= self.MIN_LOOKAHEAD_CM:
                break
            target, _ = self.path.point_at(self.progress + reach)
            if self._cut(pos, target, reach) <= self.MAX_CUT_CM:
                break
            reach *= 0.6
        return max(reach, self.MIN_LOOKAHEAD_CM)

    def _cut(self, pos, target, reach):
        """The most the shape strays from the line the ball would drive."""
        chord = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
        denom = float(chord @ chord)
        if denom < 1e-9:
            return 0.0
        worst = 0.0
        for f in np.linspace(0.0, 1.0, self.CUT_SAMPLES)[1:-1]:
            on_path, _ = self.path.point_at(self.progress + reach * f)
            u = float(np.clip((on_path - pos) @ chord / denom, 0.0, 1.0))
            worst = max(worst, float(np.linalg.norm(on_path - (pos + chord * u))))
        return worst

    def _speed_for(self, heading, gap):
        speed = self.speed

        # Ease off for the corner that is coming, not the one being taken. The
        # command is already a lookahead ahead of the ball, so the turn worth
        # slowing for is the one between here and there.
        if self.corner_gain > 0 and self.command is not None:
            turn = abs(wrap180(heading - self.command[0]))
            frac = 1.0 - self.corner_gain * min(turn, 90.0) / 90.0
            speed *= max(frac, self.MIN_SPEED_FRAC)

        # And ease into the end. A ball travels its stopping distance after the
        # last command whatever that command was, so the only way to finish on
        # the end point rather than beyond it is to be slow when arriving.
        if not self.path.loop:
            brake = max(self.speed * self.stop_s, self.arrive)
            if self.remaining < brake:
                speed = min(speed, max(self.speed * self.remaining / brake,
                                       self.speeds.min_moving_cm_s))
        return max(speed, 0.0)

    def _send(self, heading, byte, force=False):
        """Rate-limit and deadband, the way the radio and the robot both do."""
        if not force:
            if self.since < 1.0 / max(self.cmd_hz, 1e-6):
                return None
            if self.command is not None:
                turned = abs(wrap180(heading - self.command[0]))
                changed = abs(int(byte) - int(self.command[1]))
                stopping = (byte == 0) != (self.command[1] == 0)
                if not stopping and turned < self.heading_deadband \
                        and changed < self.byte_deadband:
                    return None
        self.command = (float(heading) % 360.0, int(byte))
        self.commands += 1
        self.since = 0.0
        return self.command


# -- the command list --------------------------------------------------------

YAW_SETTLE_S = 0.12         # after a turn in place, before the leg is trusted
TURN_THRESHOLD = 25.0       # deg; above this, stop and turn instead of bending
MIN_LEG_CM = 2.0


class Step:
    """One numbered line of the plan: one thing you would send the ball.

    `kind` is presentational only. A yaw IS a roll — a roll at speed zero,
    which turns the drive assembly and moves nothing — and naming the two
    separately is for whoever reads the list, not for the radio.
    """

    def __init__(self, kind, heading, byte, seconds, cm_s=0.0, turn=0.0,
                 distance=0.0, note=""):
        self.kind = kind
        self.heading = float(heading) % 360.0
        self.byte = int(byte)
        self.seconds = round(float(seconds), 2)
        self.cm_s = float(cm_s)
        self.turn = float(turn)
        self.distance = float(distance)
        self.note = note

    @property
    def command(self):
        """What actually goes out: heading and speed byte, held for `seconds`."""
        return (self.heading, self.byte)

    def line(self, n=None):
        head = f"{n:2d}) " if n is not None else ""
        if self.kind == "yaw":
            return (f"{head}yaw  {self.turn:+6.1f}deg  to {self.heading:5.1f}deg"
                    f"  for {self.seconds:5.2f}s   speed 0")
        return (f"{head}roll {self.heading:5.1f}deg  at speed {self.byte:3d}"
                f"  for {self.seconds:5.2f}s   ({self.cm_s:.1f}cm/s,"
                f" {self.distance:.0f}cm)")

    def __repr__(self):
        return f"<{self.line()}>"


def compile_path(path, speed=20.0, speeds=None, yaw_rate=DEFAULT_YAW_RATE,
                 dead_s=0.43, turn_threshold=TURN_THRESHOLD, start_pos=None,
                 start_heading=0.0, min_leg_cm=MIN_LEG_CM):
    """A drawn path in, a numbered list of roll commands out.

    Three decisions are made here, and they are the whole of the planner.

    WHERE TO TURN ON THE SPOT. A heading arrives attached to a speed, so a
    corner can be taken while moving — the ball simply curves through it, which
    is smooth and cuts the corner by roughly the distance covered while the
    assembly swings. Under `turn_threshold` that is what you want. Over it, the
    cut is bigger than the shape, so the plan stops, turns, and starts again.

    HOW LONG TO ROLL. Not distance over speed. The ball spends `dead_s` doing
    nothing at all after the command lands, then about `tau` seconds reaching
    speed, and then it runs on for about `tau` again after the command stops.
    Over a leg that starts from rest and ends at rest those last two cancel —
    what is lost accelerating is regained coasting — and what is left over is
    the dead time. Hence `length / speed + dead_s`, once per chain of legs
    rather than once per leg, because a chain with no yaw in it never stops.

    WHAT SPEED IS EVEN LEGAL. Below the deadband the motors do not turn, so a
    plan that asks for 3cm/s is a plan the ball ignores. `SpeedMap` rounds up to
    the lowest byte that moves, and the returned list says what it will actually
    travel at rather than what was asked for.
    """
    speeds = speeds or SpeedMap()
    pts = np.asarray(path.points, dtype=float)
    if start_pos is not None:
        start = np.asarray(start_pos, dtype=float)
        if float(np.linalg.norm(start - pts[0])) > min_leg_cm:
            # The lead-in is part of the plan, not a thing that happens before
            # it. A ball is wherever the last run left it, and a list that
            # assumes otherwise is a list that has to be driven to by hand.
            pts = np.vstack([start, pts])

    byte = speeds.byte_for(speed)
    cm_s = speeds.cm_s_for(byte)
    steps = []
    if byte == 0 or cm_s <= 0.0:
        return steps, {"error": f"{speed:.0f}cm/s is under the deadband — "
                                f"byte {speeds.min_moving_byte} is the slowest "
                                f"this ball moves at"}

    heading = float(start_heading) % 360.0
    from_rest = True
    for a, b in zip(pts[:-1], pts[1:]):
        leg = b - a
        length = float(np.linalg.norm(leg))
        if length < min_leg_cm:
            continue
        want = bearing(leg)
        if want is None:
            continue
        turn = wrap180(want - heading)
        if abs(turn) > turn_threshold:
            steps.append(Step("yaw", want, 0,
                              abs(turn) / max(yaw_rate, 1e-6) + YAW_SETTLE_S,
                              turn=turn))
            from_rest = True
        seconds = length / cm_s + (dead_s if from_rest else 0.0)
        steps.append(Step("roll", want, byte, seconds, cm_s=cm_s,
                          turn=turn if abs(turn) <= turn_threshold else 0.0,
                          distance=length))
        heading = want
        from_rest = False

    plan = {
        "steps": len(steps),
        "seconds": round(sum(s.seconds for s in steps), 2),
        "distance_cm": round(sum(s.distance for s in steps), 1),
        "speed_cm_s": round(cm_s, 1),
        "byte": byte,
        "turns": sum(1 for s in steps if s.kind == "yaw"),
    }
    if byte == speeds.min_moving_byte and cm_s > speed + 0.6:
        plan["note"] = (f"asked for {speed:.0f}cm/s; byte {byte} is the "
                        f"deadband, so it drives at {cm_s:.1f}cm/s or not at all")
    elif abs(cm_s - speed) > 0.6:
        plan["note"] = (f"asked for {speed:.0f}cm/s, byte {byte} is "
                        f"{cm_s:.1f}cm/s — the ball only has 255 speeds")
    return steps, plan


def format_plan(steps, plan=None):
    """The list as you would read it out, one numbered line each."""
    out = [s.line(i + 1) for i, s in enumerate(steps)]
    if plan:
        if plan.get("error"):
            return plan["error"]
        out.append(f"    {plan['steps']} commands, {plan['seconds']:.1f}s, "
                   f"{plan['distance_cm']:.0f}cm at {plan['speed_cm_s']:.1f}cm/s")
        if plan.get("note"):
            out.append(f"    {plan['note']}")
    return "\n".join(out)


def steps_from_log(log, speeds=None, until=None):
    """The commands that were actually SENT, as a list you could send again.

    A closed-loop run is not usually thought of as producing a program, but it
    does: every roll command it issued, in order, each held until the next one.
    Recovering that is what lets a drive be exported, read, checked by hand, or
    replayed open-loop on a ball with no camera watching it — which is the only
    honest way to ask how much of the tracking was the plan and how much was
    the feedback.
    """
    speeds = speeds or SpeedMap()
    out = []
    for i, (t, heading, byte) in enumerate(log):
        end = log[i + 1][0] if i + 1 < len(log) else until
        if end is None:
            end = t
        seconds = max(float(end) - float(t), 0.0)
        cm_s = speeds.cm_s_for(byte)
        turn = 0.0 if not out else wrap180(heading - out[-1].heading)
        out.append(Step("roll" if byte else "yaw", heading, byte, seconds,
                        cm_s=cm_s, turn=turn, distance=cm_s * seconds))
    return out


class Program:
    """Runs a compiled list open-loop. No camera, no corrections, no arguing.

    This is what a Sphero actually gets: a queue of headings, speeds and
    durations, executed on trust. Nothing here looks at where the ball ended
    up, which is exactly why running it is worth watching — every error the
    plant has is still on the floor at the end, uncorrected and cumulative.
    """

    def __init__(self, steps):
        self.steps = list(steps)
        self.i = -1
        self.left = 0.0
        self.commands = 0
        self.elapsed = 0.0
        self.done = not self.steps
        self.command = None

    @property
    def current(self):
        return self.steps[self.i] if 0 <= self.i < len(self.steps) else None

    @property
    def travelled(self):
        """Centimetres the PLAN believes it has covered — not a measurement."""
        return sum(s.distance for s in self.steps[:max(self.i, 0)])

    def step(self, pos, dt):
        """`pos` is ignored, and that is the entire point of this pilot."""
        if self.done:
            return None
        self.elapsed += dt
        if self.left > dt:
            self.left -= dt
            return None
        self.i += 1
        if self.i >= len(self.steps):
            self.done = True
            self.commands += 1
            self.command = ((self.command[0] if self.command else 0.0), 0)
            return self.command
        self.left = self.steps[self.i].seconds
        self.commands += 1
        self.command = self.steps[self.i].command
        return self.command


# -- one drive ---------------------------------------------------------------

class Drive:
    """A path, a ball, a camera and a pilot, stepped together.

    The pilot is either a `Program` — the numbered list, run on trust — or a
    `RollFollower`, which throws the list away and steers off the camera. They
    present the same two-line interface on purpose: the plant, the arena, the
    scoring and the window cannot tell which one is driving, so the difference
    between their trails is a difference between the pilots and nothing else.
    """

    def __init__(self, path, robot, pilot, speeds=None,
                 yaw_rate=DEFAULT_YAW_RATE, camera=None):
        self.path = path
        self.pilot = pilot
        self.speeds = speeds or SpeedMap()
        self.plant = RollPlant(robot, speeds=self.speeds, yaw_rate=yaw_rate)
        self.camera = camera if camera is not None else Camera()
        self.trail = []                 # where the ball really went
        self.fixes = []                 # what the pilot was shown
        self.cross_track = 0.0
        self.progress = 0.0             # cm along the shape, for reporting only
        self.joined_at = 0              # trail index where it reached the path
        self._joined = False
        self.t = 0.0

    @property
    def done(self):
        return bool(getattr(self.pilot, "done", False))

    def step(self, dt):
        pos = self.camera.fix(self.plant.pos, dt)
        cmd = self.pilot.step(pos, dt)
        if cmd is not None:
            self.plant.roll(*cmd)
        self.plant.step(dt)

        true = np.array(self.plant.pos, dtype=float)
        self.trail.append(true)
        self.fixes.append(np.asarray(pos, dtype=float))
        # Reported from the TRUE position, not the fix. This number is the
        # scoreboard, and a scoreboard fed by the same noisy measurement the
        # controller uses would flatter a controller that is simply wrong in
        # the direction the camera happens to be wrong in.
        self.progress = self.path.project(true, from_s=self.progress,
                                          window=4.0 * max(self.path.length, 1.0))
        here, _ = self.path.point_at(self.progress)
        self.cross_track = float(np.linalg.norm(true - here))
        if not self._joined and self.cross_track <= JOIN_CM:
            self._joined = True
            self.joined_at = len(self.trail) - 1
        self.t += dt
        return cmd

    def score(self):
        """Distance from the trail to the SHAPE, in cm, once it got there.

        Scored from where the ball joined the path rather than from where it
        happened to be sitting. Driving TO a shape is not driving it, and
        counting the lead-in makes a run look worse the further away the ball
        started — which measures the previous run, not this one.
        """
        from swarm.pd import tracking_error
        trail = self.trail[self.joined_at:] if self._joined else self.trail
        out = dict(tracking_error(self.path, trail) or {})
        end = self.path.points[0] if self.path.loop else self.path.points[-1]
        out.update({
            "commands": int(getattr(self.pilot, "commands", 0)),
            "seconds": round(self.t, 2),
            "cmd_hz": round(getattr(self.pilot, "commands", 0) / max(self.t, 1e-6), 2),
            "finished_cm": round(float(np.linalg.norm(self.plant.pos - end)), 2),
            "joined": bool(self._joined),
        })
        return out


JOIN_CM = 8.0           # inside this, the ball counts as being ON the path


def plan(points, speed=20.0, speeds=None, loop=False, simplify_cm=1.0,
         start_pos=None, start_heading=0.0, **kw):
    """Points in, (path, steps, summary) out. The whole planner, no simulation."""
    path = Polyline(simplify(points, simplify_cm), speed=speed, loop=loop)
    steps, summary = compile_path(path, speed=speed, speeds=speeds,
                                  start_pos=start_pos,
                                  start_heading=start_heading, **kw)
    return path, steps, summary


SETTLE_S = 3.0          # the longest a stopped ball is given to actually stop


def run(path, robot, pilot, dt=1.0 / 30.0, timeout_s=240.0, **kw):
    """Step a drive to completion, THEN let the ball coast to a stop.

    The coast is not an afterthought. A ball carrying 20cm/s into a stop
    command runs on for most of its motor lag — some 15cm on this plant — and a
    harness that stops stepping when the last command is sent scores the ball
    where it was told to be. That reads as a tenth of a metre of accuracy
    nobody has.
    """
    drive = Drive(path, robot, pilot, **kw)
    while drive.t < timeout_s and not drive.done:
        drive.step(dt)
    settle = 0.0
    while settle < SETTLE_S and drive.plant.speed > 1.0:
        drive.step(dt)
        settle += dt
    return drive


def follow(points, robot, speed=20.0, lookahead=18.0, loop=False,
           simplify_cm=1.0, speeds=None, dt=1.0 / 30.0, timeout_s=240.0,
           yaw_rate=DEFAULT_YAW_RATE, camera=None, **kw):
    """Closed loop: draw a shape, steer it off the camera until it is done."""
    path = Polyline(simplify(points, simplify_cm), speed=speed, loop=loop)
    speeds = speeds or SpeedMap()
    pilot = RollFollower(path, speed=speed, lookahead=lookahead, speeds=speeds, **kw)
    return run(path, robot, pilot, dt=dt, timeout_s=timeout_s, speeds=speeds,
               yaw_rate=yaw_rate, camera=camera)

"""Proportional-derivative position control, and the paths it follows.

Separate from `swarm/navigate.py` on purpose. `Navigate` is the swarm
controller: it assigns a set of robots to a set of targets, blends layers, and
bends everything around neighbours and obstacles. This is one robot, one
setpoint, no assignment and no avoidance — the thing you want when the question
is "does this ball go where I point it, and how well", which is a question about
the plant rather than about the swarm.

Gains come from measurement rather than from taste. `calib/motion.json` holds
the motor lag, the loop delay, the coast and the speed deadband for a specific
ball, and each of those maps onto a term here:

    tau + delay   sets how much gain the loop can carry before it rings
    deadband      sets the floor below which a command does nothing at all
    max_speed     sets the ceiling

A robot that has never been characterised still drives — the defaults are the
conservative end of the measured range — but it drives worse, and the point of
`calib.py` is that you can see the difference.
"""

import math

import numpy as np

# Gains are swept against the plant rather than chosen. What a position loop
# can carry is set by the phase the plant eats, and a first-order lag `tau`
# behind a dead time `L` eats roughly what a dead time of `L + tau/2` would —
# so that, and not `L` alone, is what the proportional gain scales against.
#
# Dividing by `L` alone was tried and is worse in a specific and instructive
# way: it is fine on the plant it was tuned on and diverges as the camera gets
# faster, because kp goes to infinity as L goes to zero while the motor lag
# that actually limits the loop stays put. Measured across five plants — from a
# 50ms camera to a 300ms link, and motors from 0.15s to 0.6s — this form is the
# only one that stays both quick and damped at every corner.
DAMPING_TARGET = 0.6
KD_LAG_MULTIPLE = 2.0
# Deliberately pessimistic, not typical. These are what an UNMEASURED robot
# drives on, and it drives with no delay compensation because there is no
# measured delay to compensate for — so the gain has to be low enough to be
# safe on a plant nobody has looked at. A characterised robot gets its own
# numbers and is visibly quicker, which is the argument for characterising it.
DEFAULT_TAU = 0.45          # s
DEFAULT_DELAY = 0.30        # s


def gains_from_motion(fit, damping=DAMPING_TARGET):
    """Turn a `calib/motion.json` entry into (kp, kd, deadband, max_speed).

    kp is in 1/s — centimetres of error become centimetres per second. kd is in
    seconds: it multiplies a velocity to produce a velocity, and setting it near
    the plant's own lag is what lets the loop anticipate instead of chase.
    """
    fit = fit or {}
    rec = fit.get("recommend", {}) or {}
    step = fit.get("step_response", {}) or {}
    lat = fit.get("latency", {}) or {}

    tau = step.get("tau_s") or DEFAULT_TAU
    delay = lat.get("loop_delay_s")
    if delay is None:
        delay = step.get("dead_s")
    if delay is None:
        delay = DEFAULT_DELAY
    dead = max(float(delay), 1e-3)
    # A loop delay this large is not a loop delay. It is a measurement taken
    # against a tracker that was not following the robot, and using it makes
    # the controller extrapolate most of a second ahead — which reads as the
    # ball orbiting its target at the prediction radius.
    implausible = dead > 0.45
    if implausible:
        dead = 0.45
    # The phase-equivalent lag: dead time in full, motor lag at about half.
    lag = max(dead + float(tau) / 2.0, 1e-3)

    measured = bool(step.get("tau_s") or lat.get("loop_delay_s"))
    # An arrival radius smaller than the robot can achieve is a radius it never
    # reports reaching. Two things set the floor: the position noise, since a
    # parked ball appears to move by that much anyway, and the distance covered
    # in one loop delay at the slowest speed the motors will accept — below the
    # deadband a Sphero does not creep in, it either lurches or sits.
    sigma = ((fit.get("position_noise") or {}).get("p95_cm") or 1.5)
    # The deadband in cm/s, from the BYTE that was measured rather than from a
    # top speed that may not have been. Converting a measured byte through the
    # nominal ceiling is a slightly rough number; converting it through an
    # extrapolated one manufactured a 20cm/s deadband out of a 288cm/s ceiling
    # that did not exist, and that sat above the speed cap so the ball could not
    # legally be commanded to move at all.
    dead_byte = rec.get("min_moving_byte")
    dead_cm_s = rec.get("min_moving_cm_s")
    if not dead_cm_s and dead_byte:
        dead_cm_s = round(float(dead_byte) / 255.0 * 60.0, 1)
    floor_speed = max(dead_cm_s or 6.0, 4.0)
    # The full stopping constant, not just the dead time: what limits how
    # precisely a ball can park is how far it runs on after the command to stop,
    # and that is the delay AND the roll-out. At the deadband speed — the
    # slowest it can be asked to move — that distance is the tightest circle it
    # can hold, and asking for tighter produces a robot that reaches the target
    # and then hunts around it indefinitely rather than one that parks.
    stop_total = rec.get("stopping_distance_s_per_cm_s") or (dead + float(tau))
    arrive = max(4.0, 2.0 * float(sigma), floor_speed * float(stop_total))
    return {
        "kp": round(damping / lag, 3),
        # The derivative term opposes the plant's dead time specifically, so it
        # is sized against that and not against the blended figure.
        "kd": round(KD_LAG_MULTIPLE * dead, 3),
        # Only predict with a lag somebody measured. Extrapolating along a
        # guessed delay is stable — mismatches from half to double stay stable
        # — but it is not honest, and an unmeasured robot driving conservatively
        # and visibly worse is the point of having a calibration bench.
        "predict_s": round(dead, 3) if measured else 0.0,
        "arrive_cm": round(float(arrive), 1),
        "deadband_cm_s": float(dead_cm_s or 0.0),
        "max_speed": rec.get("handle_max_speed_cm_s") or 60.0,
        "lag_s": round(lag, 3),
        "dead_s": round(dead, 3),
        "delay_clamped": bool(implausible),
        "tau_s": round(float(tau), 3),
        "measured": measured,
    }


class PDController:
    """One robot to one setpoint. Returns a velocity in cm/s."""

    # How much of the remaining error the prediction may consume. Extrapolating
    # a robot's position forward is what removes overshoot on a long move, and
    # close to the target it is what causes circling: a 0.6s prediction at
    # 10cm/s puts the estimate 6cm beyond a target 4cm away, the controller
    # commands back toward it, and the ball orbits at the prediction radius
    # forever. Never predicting past halfway to the target keeps the long-move
    # benefit and removes the orbit.
    PREDICT_FRACTION = 0.5

    def __init__(self, kp=1.0, kd=0.35, max_speed=60.0, deadband_cm_s=0.0,
                 tol=3.0, predict_s=0.0, release=1.7):
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_speed = float(max_speed)
        self.deadband = float(deadband_cm_s)
        self.tol = float(tol)
        self.predict = float(predict_s)
        # Once inside `tol` the robot stays stopped until the error grows past
        # `tol * release`. Without that hysteresis it creeps out of the circle,
        # gets a command, overshoots back in, stops, creeps out again — which
        # looks exactly like hunting and wears the motors doing nothing.
        self.release = float(release)
        self.error = None               # cm, last distance to setpoint
        self.arrived = False
        self.holding = False            # latched inside the arrival circle
        self.estimate = None            # where we believe the ball is NOW
        # Circling is not hunting and the difference matters: hunting is a
        # tolerance set tighter than the robot can hold, and widening it fixes
        # that. Circling is the commanded direction being wrong by enough that
        # the ball travels AROUND the target instead of toward it, and no
        # tolerance fixes that — the aim frame has to be calibrated. They look
        # identical from a distance, so the loop measures which it is.
        self._orbit_t = 0.0
        self._orbit_dist = []
        self.orbiting = False

    def step(self, pos, vel, setpoint, feedforward=None):
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        target = np.asarray(setpoint, dtype=float)

        # Control on where the ball will be when the command lands, not on
        # where the camera last saw it. Every position this loop receives is
        # already one loop delay old, and correcting an old position is what
        # produces the overshoot — measured at 8cm on a 150cm move, against
        # 0.3cm with this line in.
        if self.predict:
            step = vel * self.predict
            gap = float(np.linalg.norm(target - pos))
            reach = float(np.linalg.norm(step))
            if reach > self.PREDICT_FRACTION * gap:
                step = step / max(reach, 1e-9) * self.PREDICT_FRACTION * gap
            pos = pos + step
        self.estimate = pos

        err = target - pos
        d = float(np.linalg.norm(err))
        self.error = d

        # The derivative term is the robot's own velocity, negated: it opposes
        # whatever the ball is already doing. Using the error's derivative
        # instead would be identical for a still target and violently noisy for
        # a moving one, because the setpoint's own motion would differentiate
        # into the output.
        v = self.kp * err - self.kd * vel
        if feedforward is not None:
            # Following a path, the setpoint is moving. Without this the
            # controller is permanently behind by however far the target
            # travels in one lag, which on a circle reads as a smaller circle.
            v = v + np.asarray(feedforward, dtype=float)

        # Measured from the REAL position, not the predicted one. Arriving is a
        # fact about where the ball is; predicting is a guess about where it is
        # going, and a robot that has stopped inside the circle should not be
        # argued out of it by an extrapolation of its own remaining drift.
        if feedforward is None:
            inside = self.tol if not self.holding else self.tol * self.release
            if d <= inside:
                self.holding = True
                self.arrived = float(np.linalg.norm(vel)) < 4.0
                return np.zeros(2)
            self.holding = False
        self.arrived = d <= self.tol and float(np.linalg.norm(vel)) < 4.0

        self._watch_orbit(d, float(np.linalg.norm(vel)))
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            return np.zeros(2)
        if speed > self.max_speed:
            v = v / speed * self.max_speed
        elif self.deadband and speed < self.deadband:
            # Below the deadband the motors do not turn, so a command here is
            # indistinguishable from a stop — the ball simply parks short of
            # its target and the controller sits there asking politely. Either
            # ask for enough to move, or ask for nothing.
            if d > self.tol:
                v = v / speed * self.deadband
            else:
                return np.zeros(2)
        return v


    ORBIT_SECONDS = 6.0
    ORBIT_SPREAD = 0.30         # of the mean radius

    def _watch_orbit(self, d, speed, dt=1.0 / 30.0):
        """Moving steadily, at a distance that is not shrinking. That is a circle."""
        if speed < 3.0 or d < self.tol:
            self._orbit_t = 0.0
            self._orbit_dist = []
            self.orbiting = False
            return
        self._orbit_t += dt
        self._orbit_dist.append(d)
        del self._orbit_dist[:-int(self.ORBIT_SECONDS / dt)]
        if self._orbit_t < self.ORBIT_SECONDS or len(self._orbit_dist) < 60:
            return
        r = np.array(self._orbit_dist, dtype=float)
        # Distance neither shrinking nor varying much, while still moving.
        self.orbiting = bool(r.std() < self.ORBIT_SPREAD * max(r.mean(), 1e-9)
                             and r[-1] > 0.7 * r[0])


# -- paths -------------------------------------------------------------------
#
# A path is a setpoint that moves. Making it a moving setpoint rather than a
# sequence of waypoints is what keeps one controller for all three shapes: a
# point is a path that does not move, and the PD loop cannot tell the
# difference. It also gives a feedforward term for free — the setpoint's own
# velocity is known exactly, so the controller does not have to discover it by
# lagging behind.


class Point:
    """Go there and stop."""

    kind = "point"

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)
        self.t = 0.0

    def step(self, dt, pos=None):
        self.t += dt
        return self.target, None

    def describe(self):
        return f"point ({self.target[0]:.0f}, {self.target[1]:.0f})"

    def preview(self, n=2):
        return [self.target] * max(2, 2)


class Line:
    """Shuttle between two points at a fixed speed.

    The setpoint turns round at each end rather than easing through it, which
    is deliberate: a reversal is the hardest thing to ask of a ball with lag,
    so a line is the shape that shows up a badly tuned loop soonest.
    """

    kind = "line"

    def __init__(self, a, b, speed=25.0, bounce=True):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.speed = float(speed)
        self.bounce = bounce
        self.t = 0.0

    @property
    def length(self):
        return max(float(np.linalg.norm(self.b - self.a)), 1e-6)

    def step(self, dt, pos=None):
        self.t += dt
        period = self.length / max(self.speed, 1e-6)
        if self.bounce:
            phase = (self.t / period) % 2.0
            u = phase if phase <= 1.0 else 2.0 - phase
            direction = 1.0 if phase <= 1.0 else -1.0
        else:
            u = (self.t / period) % 1.0
            direction = 1.0
        unit = (self.b - self.a) / self.length
        return self.a + (self.b - self.a) * u, unit * self.speed * direction

    def describe(self):
        return (f"line ({self.a[0]:.0f},{self.a[1]:.0f})-"
                f"({self.b[0]:.0f},{self.b[1]:.0f}) at {self.speed:.0f}cm/s")

    def preview(self, n=48):
        # Sampled along the segment, not just its ends. A shape given as two
        # points makes "distance to the path" mean "distance to the nearest
        # endpoint", which is largest exactly in the middle where the tracking
        # is best.
        return [self.a + (self.b - self.a) * u for u in np.linspace(0.0, 1.0, n)]


class Circle:
    """Orbit a centre at a fixed radius and speed."""

    kind = "circle"

    def __init__(self, centre, radius=50.0, speed=25.0, clockwise=False,
                 phase=0.0):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)
        self.speed = float(speed)
        self.sign = -1.0 if clockwise else 1.0
        self.t = 0.0
        self.phase0 = float(phase)

    @property
    def omega(self):
        return self.sign * self.speed / max(self.radius, 1e-6)

    def step(self, dt, pos=None):
        self.t += dt
        a = self.phase0 + self.omega * self.t
        point = self.centre + self.radius * np.array([math.cos(a), math.sin(a)])
        # d/dt of the above. Handing this to the controller as feedforward is
        # the difference between tracking the circle and tracking a smaller
        # circle lagging behind it.
        tangent = self.radius * self.omega * np.array([-math.sin(a), math.cos(a)])
        return point, tangent

    def start_phase(self, pos):
        """Enter the circle at the nearest point, not at angle zero."""
        d = np.asarray(pos, dtype=float) - self.centre
        if float(np.linalg.norm(d)) > 1e-6:
            self.phase0 = math.atan2(d[1], d[0])
            self.t = 0.0
        return self

    def describe(self):
        return (f"circle r{self.radius:.0f} about "
                f"({self.centre[0]:.0f},{self.centre[1]:.0f}) at {self.speed:.0f}cm/s")

    def preview(self, n=48):
        return [self.centre + self.radius * np.array([math.cos(a), math.sin(a)])
                for a in np.linspace(0, 2 * math.pi, n)]


# Path length over straight-line distance, against the heading error that
# produced it. Measured on the plant rather than derived: a closed-loop
# controller with a rotated command frame reaches its target by a curve, and
# how much of a curve is a direct read on how rotated the frame is.
STRAIGHTNESS_TO_OFFSET = ((1.00, 0.0), (1.05, 20.0), (1.15, 30.0),
                          (1.44, 40.0), (2.2, 50.0), (3.68, 60.0))


def straightness(trail):
    """1.00 is a straight line. Higher is a curve. None if it barely moved."""
    pts = np.asarray(list(trail), dtype=float)
    if len(pts) < 5:
        return None
    net = float(np.linalg.norm(pts[-1] - pts[0]))
    if net < 8.0:
        return None
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) / net


def implied_heading_error(ratio):
    """Rough degrees of frame error behind a given curvature, or None.

    Rough on purpose: it exists to tell a trainer whether to go and calibrate,
    not to replace the calibration. A curved approach has other possible causes
    — an obstacle, a sticky wheel — so this suggests rather than concludes.
    """
    if ratio is None or ratio < 1.03:
        return None
    lo = STRAIGHTNESS_TO_OFFSET[0]
    for hi in STRAIGHTNESS_TO_OFFSET[1:]:
        if ratio <= hi[0]:
            span = max(hi[0] - lo[0], 1e-9)
            return lo[1] + (hi[1] - lo[1]) * (ratio - lo[0]) / span
        lo = hi
    return STRAIGHTNESS_TO_OFFSET[-1][1]


def tracking_error(path, trail, skip=0.0):
    """RMS and worst distance from a trail of positions to the path itself.

    Measured against the *shape*, not against where the setpoint happened to be
    — a robot lagging a quarter turn round a circle is still on the circle, and
    calling that an error would confuse a phase lag with a tracking failure.
    """
    pts = np.asarray([p for p in trail], dtype=float)
    if not len(pts):
        return None
    # Densely: the reference is a polyline, and its own discretisation shows up
    # as tracking error. On a 45cm circle, 96 segments carry 1.3cm of that —
    # comparable to what is being measured, which would flatter or damn a
    # controller for the sampling rate of the yardstick.
    shape = np.asarray(path.preview(512), dtype=float)
    d = np.linalg.norm(pts[:, None, :] - shape[None, :, :], axis=2).min(axis=1)
    if skip:
        d = d[int(len(d) * skip):]
    if not len(d):
        return None
    return {"rms_cm": round(float(np.sqrt((d ** 2).mean())), 2),
            "max_cm": round(float(d.max()), 2),
            "samples": int(len(d))}


# -- turn and go -------------------------------------------------------------

class TurnAndGo:
    """Point at the target, drive straight, stop when close. Nothing else.

    A PD loop assumes it can correct continuously. On this hardware it cannot:
    a drive command takes about 230ms to reach the ball, so roughly four
    commands a second get through and every one of them is acting on a picture
    that is already a fifth of a second old. Feeding that loop thirty new
    opinions a second does not make it more responsive, it makes it argue with
    itself — and the ball wanders because the commands disagree, not because
    the gains are wrong.

    So: aim once, commit, and only re-aim when the bearing has genuinely
    drifted. Straight legs, a handful of commands, and a stop. It gives up the
    PD's smooth deceleration into the target — it arrives at speed and stops —
    which is precisely what an arrival radius is for.
    """

    def __init__(self, speed=8.0, arrive_cm=6.0, retarget_deg=10.0,
                 min_interval_s=0.45, release=1.7, creep_frac=0.45,
                 creep_within=2.5):
        self.speed = float(speed)
        self.tol = float(arrive_cm)
        self.retarget = float(retarget_deg)
        self.min_interval = float(min_interval_s)
        self.release = float(release)
        # Inside a couple of arrival radii, halve the speed. Not a controller
        # so much as an admission: arriving at full tilt and stopping means
        # overshooting by the stopping distance, and the last few centimetres
        # are the ones that decide whether it lands in the circle.
        self.creep_frac = float(creep_frac)
        self.creep_within = float(creep_within)

        self.aim = None                 # the bearing currently committed to
        self.since = 1e9                # seconds since the last new command
        self.error = None
        self.holding = False
        self.arrived = False
        self.commands = 0               # how many were actually issued

    def step(self, pos, vel, setpoint, feedforward=None, dt=1.0 / 30.0):
        pos = np.asarray(pos, dtype=float)
        target = np.asarray(setpoint, dtype=float)
        to = target - pos
        d = float(np.linalg.norm(to))
        self.error = d
        self.since += dt

        inside = self.tol if not self.holding else self.tol * self.release
        if d <= inside:
            self.holding = True
            self.arrived = float(np.linalg.norm(vel)) < 4.0
            self.aim = None
            return np.zeros(2)
        self.holding = False
        self.arrived = False

        bearing = math.degrees(math.atan2(to[1], to[0]))
        drifted = self.aim is None or abs(
            (bearing - self.aim + 180.0) % 360.0 - 180.0) > self.retarget

        # Re-aim only when the bearing has really moved, and never faster than
        # the radio can keep up with. Anything more often is a command the ball
        # has not finished acting on being replaced by another one.
        if drifted and self.since >= self.min_interval:
            self.aim = bearing
            self.since = 0.0
            self.commands += 1

        if self.aim is None:
            self.aim = bearing
            self.commands += 1
        speed = self.speed
        if d < self.creep_within * self.tol:
            speed *= self.creep_frac
        rad = math.radians(self.aim)
        return np.array([math.cos(rad), math.sin(rad)]) * speed

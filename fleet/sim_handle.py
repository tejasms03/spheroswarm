"""A simulated robot.

Runs the same dynamics as `swarm.sim`: command latency, first-order motor lag,
a per-robot gain and a heading bias. A sim robot and a real robot should feel
close enough that a controller tuned on one is not surprised by the other.
"""

from collections import deque

import numpy as np

from .handle import MAX_SPEED, RobotHandle, now

# Guesses, used only for a robot nothing has measured. Pass `motion=` and the
# measured numbers take over — see `from_motion`.
DRIFT_DEG_PER_MIN = 8.0     # random-walk scale on the heading bias
DRIFT_LIMIT = 0.6           # rad, ~34 deg; a bounded walk, never a runaway
SLIP_MAX = 0.06             # fraction of a command that can be lost

BATTERY_DRAIN_PER_S = 1.0 / (90 * 60)     # a notional 90-minute run

# The command queue holds whole ticks, so a delay in seconds only becomes a
# number of slots once a tick rate is chosen. The battery converts its measured
# delay at 30Hz, and this is the same 30Hz — kept as a named constant so the
# assumption is visible rather than buried in a rounding.
SIM_TICK_HZ = 30.0


def from_motion(fit):
    """Measured plant constants out of a `calib/motion.json` entry, or {}.

    Only what the fit actually stood behind. A run whose stages errored has
    those numbers withheld from `recommend` by `fleet/characterize`, so an
    unmeasured constant simply is not here and the robot keeps its guess —
    which is the honest outcome, and better than a sim confidently wrong.
    """
    rec = (fit or {}).get("recommend") or {}
    out = {}
    if rec.get("sim_tau_s"):
        out["tau"] = float(rec["sim_tau_s"])
    if rec.get("sim_latency_steps"):
        # Back to seconds. The battery converted its measured delay at 30Hz,
        # and a queue counted in ticks means whatever the caller's dt happens
        # to be — 16 slots is 0.53s at 30Hz and 1.6s at 10Hz. Storing the
        # duration and sizing the queue per dt is what makes the number mean
        # the same thing at any tick rate.
        out["latency_s"] = max(1, int(rec["sim_latency_steps"])) / SIM_TICK_HZ
    if rec.get("sim_gain"):
        out["gain"] = float(rec["sim_gain"])
    if rec.get("stopping_distance_s_per_cm_s"):
        # The COAST, which until now never reached the simulator at all. The
        # brake test measures it and `fit` exports it; nothing read it, so a
        # fully characterised ball still shed speed on its ACCELERATION
        # constant -- a sim that stops better than the hardware, which is the
        # one direction of error a tracking controller cannot survive.
        out["coast_s"] = float(rec["stopping_distance_s_per_cm_s"])
    return out


class SimRobot(RobotHandle):
    kind = "sim"

    # The simulator rotates the commanded velocity vector directly, so its
    # offset runs the same way the estimator's residual does. Opposite to
    # `SpheroRobot`; see the note there.
    HEADING_SIGN = 1.0

    def __init__(self, name, code, color, workspace=None, pos=None, seed=None,
                 randomize=True, motion=None):
        """`motion` is measured plant constants from `from_motion`, or None.

        Passed in rather than loaded here on purpose. `calib/motion.json` is
        live state, and a handle that read it at construction would make every
        test's behaviour depend on whatever the last hardware session left on
        disk — which is exactly how twelve tests broke the day the real arena
        was measured. The caller that knows which robot this is loads it.
        """
        super().__init__(name, code, color, workspace)
        rng = np.random.default_rng(seed)
        self.rng = rng

        if randomize:
            self.latency = int(rng.integers(1, 4))
            self.tau = float(rng.uniform(0.25, 0.5))
            self.gain = float(rng.uniform(0.8, 1.2))
            self.bias = float(rng.uniform(-0.12, 0.12))
            # A real Sphero's heading is its own gyro estimate, and that
            # estimate wanders. A constant bias is a robot you can calibrate
            # once and forget; a drifting one is why the offset has to be
            # tracked continuously rather than measured at startup. Modelled as
            # a random walk on the bias, bounded so it cannot run away.
            self.drift_rate = float(rng.uniform(0.5, DRIFT_DEG_PER_MIN))
            self.slip = float(rng.uniform(0.0, SLIP_MAX))
        else:
            self.latency, self.tau, self.gain, self.bias = 2, 0.35, 1.0, 0.0
            self.drift_rate = 0.0
            self.slip = 0.0

        # No coast constant unless one was MEASURED. A guessed asymmetry would
        # change the dynamics of every uncharacterised robot in the suite for
        # no gain -- the same reasoning `_resize_queue` applies to a guessed
        # latency. Left at None the deceleration runs on `tau`, exactly as it
        # always has. `TrackEnv` randomises it explicitly for training, which
        # is where a spread of plausible coasts is wanted and is asked for.
        self.coast_s = None

        # Measurement beats a guess, per constant rather than all-or-nothing:
        # a run that established the motor lag but not the top speed should
        # hand over the lag and leave the gain alone.
        self.measured = sorted(motion or ())
        for k, v in (motion or {}).items():
            setattr(self, k, v)

        self._drift = 0.0
        self._rng = rng

        # A guessed latency has no units attached, so it stays a slot count and
        # behaves exactly as it always has. A MEASURED one is a real duration
        # and is held as one; `_resize_queue` turns it into slots against the
        # dt actually being stepped.
        self.latency_s = getattr(self, "latency_s", None)
        self.queue = deque([np.zeros(2)] * self.latency, maxlen=self.latency)
        self.battery = 1.0
        self.last_seen = now()
        # The simulation's own clock, advanced by dt. The estimator is fed from
        # this rather than from `now()`: a test loop runs thirty simulated
        # seconds in a few real milliseconds, so on a wall clock nothing is ever
        # far enough apart to measure and no interval ever elapses.
        self._t = 0.0

        if pos is not None:
            self.pos = np.asarray(pos, dtype=float).copy()
        elif workspace is not None:
            self.pos = np.asarray(workspace.random_valid_point(rng), dtype=float)
        else:
            self.pos = np.zeros(2)

    @property
    def connected(self):
        return True

    def set_velocity(self, v):
        v = np.asarray(v, dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            return
        speed = float(np.linalg.norm(v))
        if speed > MAX_SPEED:
            v = v / speed * MAX_SPEED
        self._desired = v
        speed = float(np.linalg.norm(v))
        if speed > 1e-6:
            # What was REQUESTED, before the offset is applied — math
            # convention, matching how the estimator measures travel.
            self._believed_course = float(
                np.degrees(np.arctan2(v[1], v[0])) % 360.0)

    def aim_zero(self, error_deg):
        """The simulated equivalent: take the frame error out of the ball.

        A real Sphero rotates its drive assembly and calls that zero. There is
        no assembly here, so the same thing is done to the bias the simulator
        rotates commands by — which is what the assembly's misalignment IS, in
        this model. The sign is not reasoned about; a test drives the robot
        afterwards and checks it goes where it was told.
        """
        # PLUS, not minus. The error arrives as a compass bearing and the bias
        # is a maths angle, and those run in opposite directions — so the
        # correction that looks wrong is the one that works. Verified by
        # driving afterwards rather than by reasoning about it.
        self.bias = float(self.bias) + np.radians(float(error_deg))
        self._drift = 0.0
        self.heading_offset = 0.0
        return True

    def set_led(self, rgb, blink=None):
        self.rgb = tuple(int(np.clip(c, 0, 255)) for c in rgb)
        self.blink = blink

    def stop(self):
        self._desired = np.zeros(2)
        self.queue.append(np.zeros(2))

    def _resize_queue(self, dt):
        """Hold a measured latency to its DURATION, whatever dt is being used.

        Only for a measured one. A guessed latency is a slot count with no
        seconds behind it, and reinterpreting it would change the dynamics of
        every uncharacterised robot in the suite for no gain.
        """
        if not self.latency_s:
            return
        want = max(1, int(round(self.latency_s / max(dt, 1e-6))))
        if want == self.queue.maxlen:
            return
        held = list(self.queue)[-want:]
        self.queue = deque(held, maxlen=want)
        while len(self.queue) < want:
            self.queue.appendleft(np.zeros(2))
        self.latency = want

    def step(self, dt):
        self._resize_queue(dt)
        # Heading drift: a bounded random walk, in radians. Bounded because an
        # unbounded walk eventually points a robot backwards, which teaches a
        # controller nothing except that the world is broken.
        if self.drift_rate:
            self._drift += float(self._rng.normal(
                0.0, np.radians(self.drift_rate) * np.sqrt(max(dt, 1e-6) / 60.0)))
            self._drift = float(np.clip(self._drift, -DRIFT_LIMIT, DRIFT_LIMIT))

        # The offset rotates the command before the bias rotates it back, which
        # is exactly what it does on a real ball: `heading + heading_offset` is
        # applied on the way out, and the world adds its own error after.
        total = self.bias + self._drift + np.radians(self.heading_offset)
        c, s = np.cos(total), np.sin(total)
        d = self._desired
        rotated = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])

        self.queue.append(rotated)
        cmd = self.queue[0] * self.gain

        if self.slip:
            # Loses a little of each command, the way a light ball does on a
            # smooth floor. Multiplicative, so it never adds energy.
            cmd = cmd * (1.0 - self.slip * float(self._rng.random()))
        # SPEEDING UP AND SLOWING DOWN ARE NOT THE SAME MOVE. A Sphero drives
        # by climbing the inside of its shell; cut the command and there is no
        # brake left, only a ball rolling on cloth. Running both directions on
        # `tau` gave a simulated ball that stopped on request, and a controller
        # tuned against it brakes late and overshoots every target on the rig.
        #
        # The coast enters as a TIME CONSTANT because that is the form the
        # measurement already takes. The brake test fits `coast_cm = k * v`
        # through the origin, and a first-order decay from `v` with time
        # constant `k` travels exactly `v * k` before it stops -- so the same
        # constant reproduces the measured stopping distance at every speed,
        # with nothing fitted twice. `coast_rest` in the bench runs the same
        # model forwards to say where the ball will end up.
        # THE DELAY IS ALREADY IN THERE, and adding it again was the bug in the
        # first version of this. `cut_pos` in the brake test is where the
        # CAMERA thought the ball was, which is where it was one loop delay
        # ago -- so the measured constant spans the delay and the roll-out
        # together, which is what `fit` says it does and what makes it the
        # right number for a controller. This simulator models the delay
        # separately, in the command queue. Decaying by the whole constant on
        # top of that stops the ball a full `v * delay` too late.
        #
        # Taken off the queue rather than off `latency_s`, because a guessed
        # latency has no seconds behind it and is still a real delay here.
        shedding = float(np.linalg.norm(cmd)) < float(np.linalg.norm(self.vel))
        lag = self.tau
        if shedding and self.coast_s:
            # TWO TICKS come back, and both are bookkeeping rather than
            # physics: `stop` appends to the queue itself and `step` appends
            # again before reading slot zero, so the modelled delay runs one
            # tick short; and position integrates the velocity AFTER the decay
            # is applied, which costs the roll-out another `v * dt`. Left
            # uncorrected the ball stops about 6% early at 30Hz -- small, but
            # always in the direction that teaches a controller it can brake
            # later than it can.
            #
            # Exact near `SIM_TICK_HZ`, which is the rate the bench and the
            # battery both run at. Well away from it the correction is only
            # approximate, and that is deliberate: the constant it corrects
            # carries +/-30% scatter between repeat stops on the real ball, so
            # chasing the last tick here would be precision the measurement
            # cannot support.
            #
            # Clamped to one tick. A coast shorter than the delay means the
            # brake test and the latency probe disagree, and a ball that stops
            # dead is the worst way to represent that.
            lag = max(self.coast_s - (self.queue.maxlen - 2) * dt, dt)
        self.vel += (cmd - self.vel) * min(dt / lag, 1.0)
        self.pos = self.pos + self.vel * dt

        if self.ws is not None and not self.ws.is_valid_point(self.pos):
            self.pos = np.asarray(self.ws.nearest_valid_point(self.pos), dtype=float)
            self.vel = np.zeros(2)

        self.battery = max(0.0, self.battery - BATTERY_DRAIN_PER_S * dt)
        self.last_seen = now()
        self._t += dt
        # Same path a real robot takes with no usable gyro, so the camera-only
        # fallback is exercised by the whole suite rather than only on the day
        # a sensor stream drops.
        self.observe_heading(self._t, None)

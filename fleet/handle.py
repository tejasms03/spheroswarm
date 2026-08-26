"""The one interface every robot presents, sim or real.

Above this line nothing knows or cares whether a robot is a process or a ball
on a floor. `set_velocity` in cm/s is the only motion API — headings, speed
bytes, and BLE writes live strictly below it.
"""

import time
from abc import ABC, abstractmethod

import numpy as np

MAX_SPEED = 60.0        # cm/s at full command, matches swarm.sim


class RobotHandle(ABC):
    kind = "abstract"

    def __init__(self, name, code, color, workspace=None):
        self.name = name
        self.code = code
        self.color = color
        self.ws = workspace
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.battery = None
        self.last_seen = 0.0
        # Degrees added to every commanded heading. On a real robot it
        # reconciles the camera frame with the ball's aim frame; on a sim robot
        # it cancels the modelled heading bias. Living on the base class is what
        # lets the whole calibration path be exercised with no hardware.
        self._heading_offset = 0.0
        # Fed every fix, from whichever source the hardware turned out to have.
        # Present on simulated robots too, so nothing above `fleet/` can come to
        # depend on a robot being real to have a heading.
        from .heading import HeadingEstimator
        self.estimator = HeadingEstimator()
        self._believed_course = None     # math-convention, offset included
        self._last_fold = None
        self._heading_offset = 0.0
        self._seed_offset = 0.0
        self.rgb = (255, 255, 255)
        self.blink = None
        # The aiming taillight. Off is what a Sphero powers up as, and it is
        # the wrong default for anything being watched: the main LED says WHICH
        # robot this is and says nothing about which way it is pointing, and
        # which way it is pointing is the thing every aim-frame bug in this
        # project comes down to. Held here rather than in each handle so a
        # renderer can draw it for a simulated robot too.
        self.back_led = 0
        self.target = None          # set by controllers, read by the renderer
        self._desired = np.zeros(2)

    # -- state ---------------------------------------------------------

    @property
    @abstractmethod
    def connected(self):
        ...

    @property
    def stale(self):
        return not self.connected

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    def state(self):
        """The uniform dict every layer above the fleet reads."""
        return {
            "code": self.code,
            "name": self.name,
            "kind": self.kind,
            "color": self.color,
            "pos": [round(float(self.pos[0]), 1), round(float(self.pos[1]), 1)],
            "vel": [round(float(self.vel[0]), 1), round(float(self.vel[1]), 1)],
            "speed": round(self.speed, 1),
            "connected": bool(self.connected),
            "battery": None if self.battery is None else round(float(self.battery), 2),
            "led": list(self.rgb),
            "back_led": (list(self.back_led)
                         if isinstance(self.back_led, tuple) else self.back_led),
            "blink": self.blink,
            "last_seen": round(float(self.last_seen), 2),
            "target": None if self.target is None
                      else [round(float(self.target[0]), 1), round(float(self.target[1]), 1)],
            # Uniform across kinds, deliberately. The renderer draws a heading
            # arrow from this and never asks whether the robot is real — which
            # is the same rule that keeps everything else above `fleet/` honest.
            "heading": self.heading_state(),
        }

    # -- motion --------------------------------------------------------

    @abstractmethod
    def set_velocity(self, v):
        """Desired velocity in cm/s, workspace frame."""

    # -- heading ---------------------------------------------------------
    #
    # `heading_offset` stays an ordinary stored number, and the estimator
    # corrects it rather than replacing it. Letting the property read the
    # estimator directly is circular: the residual is measured with the offset
    # already applied, so an offset defined as that residual defines itself.
    # Folding a fraction of it in and re-measuring is a closed loop, converges,
    # and is the same shape as the active calibration that already works.
    #
    # The two handles do not agree on the SENSE of the offset — a simulated
    # robot rotates its velocity anticlockwise, a real one adds to a clockwise
    # compass heading — so each declares its own sign and a test pins both by
    # asking whether the robot ends up going where it was told.

    @property
    def heading_offset(self):
        return self._heading_offset

    @heading_offset.setter
    def heading_offset(self, value):
        """Set by calibration. Remembered as the seed a drift is measured from."""
        self._heading_offset = float(value or 0.0) % 360.0
        self._seed_offset = self._heading_offset
        self.estimator.restart()

    HEADING_SIGN = 1.0
    FOLD_GAIN = 0.5             # of the measured residual, per fold
    FOLD_EVERY = 2.0            # s

    # Tracking corrects the offset while the robot drives. It is turned OFF
    # while the robot is being MEASURED, and that is not a detail: the
    # characterisation battery exists to describe the uncorrected plant, and a
    # loop quietly cancelling the very error being measured reports a ball with
    # no bias and no drift no matter how badly it wanders. `calib.py` disables
    # it for the duration of a run and puts it back afterwards.
    heading_tracking = True

    def heading_state(self):
        est = self.estimator
        d = est.state()
        # How far tracking has had to move the offset away from the value
        # calibration wrote. Not the estimator's residual against the seed —
        # that residual converges to zero by design, so it would report a
        # perfectly tracked robot as having drifted by its whole calibration.
        from .heading import wrap180
        drift = abs(wrap180(self._heading_offset - self._seed_offset))
        d.update(applied_deg=round(self._heading_offset, 1),
                 seed_deg=round(self._seed_offset, 1),
                 tracked=est.ready,
                 drifted_deg=round(drift, 1))
        return d

    def observe_heading(self, t, yaw_deg=None):
        """One fix. `yaw_deg` None means camera-only: use what we commanded.

        What is fed is the course that was REQUESTED, before the offset was
        applied — so the residual coming back is the total error still
        outstanding, which is the thing the offset exists to cancel.

        Feeding the post-offset course instead is the tempting mistake, and it
        does not work: the residual is then the raw hardware bias, a constant
        that no amount of offset changes, and the loop walks the offset round
        the compass forever without ever converging.
        """
        est = self.estimator
        if yaw_deg is None:
            yaw_deg = self._believed_course
        if yaw_deg is None:
            return
        est.update(t, float(self.pos[0]), float(self.pos[1]), float(yaw_deg))
        # Timestamps here are wall clock. Anchoring on the first observation
        # rather than on zero matters: a fold interval measured from the epoch
        # elapsed long ago, so every robot folded on its very first fix, before
        # it had travelled far enough for the measurement to mean anything.
        if self._last_fold is None:
            self._last_fold = t
        elif (self.heading_tracking and est.ready
              and (t - self._last_fold) >= self.FOLD_EVERY):
            self._fold_heading(t)

    def _fold_heading(self, t):
        """Move the applied offset toward cancelling the measured residual."""
        est = self.estimator
        residual = est.offset
        if residual is None or abs(residual) < 0.5:
            self._last_fold = t
            return
        self._heading_offset = (self._heading_offset
                                + self.HEADING_SIGN * self.FOLD_GAIN * -residual) % 360.0
        # Re-measure from scratch: everything in the trail was observed under
        # the OLD offset, and leaving it there would have the loop correcting
        # for an error it has already corrected.
        est.restart()
        self._last_fold = t
    def drive_raw(self, heading_deg, speed_byte):
        """Command the hardware's own units, bypassing calibration.

        Characterisation only. `set_velocity` is the API everything else uses,
        and it converts cm/s to a speed byte through MAX_SPEED — which is the
        very number a speed map is trying to measure, so calibrating through it
        would be circular. This is the escape hatch that breaks that circle;
        the base implementation converts back so the whole rig is exercisable
        against a sim robot with no hardware present.
        """
        byte = int(np.clip(speed_byte, 0, 255))
        rad = np.radians(float(heading_deg))
        # velocity_to_command's convention: heading 0 is +y, clockwise.
        v = np.array([np.sin(rad), np.cos(rad)]) * (byte / 255.0 * MAX_SPEED)
        self.set_velocity(v)

    @abstractmethod
    def set_led(self, rgb, blink=None):
        ...

    def set_back_led(self, value):
        """The taillight: an int brightness 0-255, or an (r, g, b) for a BOLT.

        Concrete rather than abstract, and it stores rather than sends. A
        simulated robot has no radio to send it to and a renderer still wants
        to draw the thing, so the state lives here and only the handles with
        hardware under them override this to also write it out.
        """
        if isinstance(value, (tuple, list)):
            self.back_led = tuple(int(np.clip(c, 0, 255)) for c in value)
        else:
            self.back_led = int(np.clip(value, 0, 255))
        return self.back_led

    def aim_zero(self, error_deg):
        """Make the robot's own forward match the arena's.

        The contract is behavioural, not arithmetic: the robot was told to
        travel a course and went `error_deg` away from it; after this, being
        told that course sends it there. A caller passes what it measured and
        gets a robot that no longer does it.

        This is a different mechanism from `heading_offset`, and a better one
        where it exists. An offset is a correction added to every command
        forever, and a Sphero establishes its heading reference when it
        connects — so a stored offset is stale the moment the link drops, which
        is why re-measuring it never stuck. Zeroing the aim puts the correction
        in the ROBOT. There is nothing left to apply, so there is no sign left
        to get wrong.

        The base class does nothing and reports so: a robot with no aim to
        reset is not a failure, it is a robot the caller should keep correcting
        the old way.
        """
        return False

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def step(self, dt):
        """Advance internal state / pull a fresh position."""

    def close(self):
        """Release any hardware. Safe to call on a robot that never connected."""

    def __repr__(self):
        return f"<{type(self).__name__} {self.code} {self.pos[0]:.0f},{self.pos[1]:.0f}>"


def velocity_to_command(v, max_speed=MAX_SPEED):
    """(vx, vy) cm/s -> (heading_deg, speed_byte), the Sphero convention.

    Heading 0 is +y and increases clockwise, matching `swarm.sim.sphero_command`
    and the camera frame (y down).
    """
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    if not np.isfinite(speed) or speed < 1e-6:
        return 0.0, 0
    heading = float(np.degrees(np.arctan2(v[0], v[1])) % 360.0)
    return heading, int(np.clip(speed / max_speed * 255.0, 0, 255))


def now():
    return time.time()

"""A real Sphero over Bluetooth.

Two independent facts decide whether this robot is usable:

  * the BLE link is up   — we can send it commands
  * the tracker has a recent fix — we know where it is

Only when both hold does `connected` go true. Reporting a stale camera fix as
if it were live is how a controller drives a robot into a wall it thinks it is
nowhere near, so a missing fix is treated as a disconnection.

All BLE traffic happens on a per-robot worker thread. `set_velocity` only ever
writes to a slot and returns, so a controller running at 10Hz is never blocked
by a radio that decided to take 400ms.
"""

import logging
import threading
import time

import numpy as np

from .handle import MAX_SPEED, RobotHandle, now, velocity_to_command

log = logging.getLogger("fleet.real")

STALE_AFTER = 0.5           # s without a tracker fix before we stop trusting the position
HEADING_DEADBAND = 8.0      # degrees
SPEED_DEADBAND = 10         # speed byte
CONNECT_STAGGER = 1.5       # s between connect attempts, fleet-wide
BACKOFF_START = 2.0
BACKOFF_MAX = 30.0
MAX_CONNECTIONS_HINT = (
    "CBErrorDomain Code=11 means the Bluetooth controller hit its maximum "
    "connection count — this machine will not hold any more robots at once. "
    "Disconnect one, or add a second BLE adapter."
)

# Connect attempts are serialised fleet-wide: simultaneous connects fail far
# more often than sequential ones, and the failure looks like a flaky robot
# rather than a busy radio.
_connect_lock = threading.Lock()
_last_connect_at = [0.0]


def default_connector(ble_name, timeout=8.0):
    """Find and open one Sphero. Returns an object with the SpheroEduAPI surface."""
    from spherov2 import scanner
    from spherov2.sphero_edu import SpheroEduAPI

    toy = scanner.find_toy(toy_name=ble_name, timeout=timeout)
    api = SpheroEduAPI(toy)
    api.__enter__()
    return api


def is_max_connections_error(exc):
    text = str(exc)
    return "CBErrorDomain" in text and "Code=11" in text


class SpheroRobot(RobotHandle):
    kind = "real"

    # A Sphero's heading is a compass bearing: zero is +y and it increases
    # CLOCKWISE. The estimator measures travel as an ordinary maths angle,
    # anticlockwise from +x. The two run in opposite directions, so a residual
    # measured in one is cancelled by moving the other the other way. Pinned by
    # a test that asks whether a ball told to go north actually goes north,
    # because that is a question a sign error cannot answer correctly.
    HEADING_SIGN = -1.0

    def __init__(self, name, code, color, ble_name, workspace=None, tracker=None,
                 connector=None, autostart=True, heading_offset=0.0):
        super().__init__(name, code, color, workspace)
        self.ble_name = ble_name
        self.heading_offset = float(heading_offset or 0.0) % 360.0
        self.tracker = tracker
        self._connector = connector or default_connector

        self._api = None
        self._link_up = False
        self._stop_flag = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()

        self._pending = None            # (heading, speed) awaiting a write
        self._pending_led = None
        self._last_sent = None          # (heading, speed) actually written
        self._last_write_at = 0.0

        # Set by the bench once the sensor probe says which call answers on
        # this toy. Until then the estimator runs camera-only, which works.
        self._yaw_source = None

        self._force_write = False       # set by drive_raw; skips the deadband once
        self.rtt = None                 # seconds, last BLE round trip
        self.rtt_mean = None
        self.attempts = 0
        self.last_error = None
        self.max_connections_hit = False

        self._thread = threading.Thread(target=self._run, name=f"ble-{code}", daemon=True)
        if autostart:
            self._thread.start()

    # -- state ---------------------------------------------------------

    @property
    def link_up(self):
        """BLE only. Says nothing about whether we know where the robot is."""
        return self._link_up

    @property
    def tracked(self):
        return (now() - self.last_seen) < STALE_AFTER if self.last_seen else False

    @property
    def connected(self):
        return self._link_up and self.tracked

    def state(self):
        d = super().state()
        d.update(link_up=self._link_up, tracked=self.tracked,
                 rtt_ms=None if self.rtt is None else round(self.rtt * 1000, 1))
        return d

    # -- motion (caller side; never blocks) ------------------------------

    def set_velocity(self, v):
        v = np.asarray(v, dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            return
        speed = float(np.linalg.norm(v))
        if speed > MAX_SPEED:
            v = v / speed * MAX_SPEED
        self._desired = v

        heading, byte = velocity_to_command(v)
        # The one place the camera frame is reconciled with the ball's own.
        # Applied here rather than in the controller so that nothing above the
        # fleet ever learns that heading exists.
        #
        # The PRE-offset heading is what the estimator is later fed as this
        # robot's own belief about its aim, which is what makes the residual it
        # measures the total correction rather than a correction to a
        # correction. Recorded before the offset is added, deliberately.
        if byte:
            # What was REQUESTED, before the offset — as a math-convention
            # course, which is the frame the estimator measures travel in.
            self._believed_course = (90.0 - heading) % 360.0
        heading = (heading + self.heading_offset) % 360.0
        with self._lock:
            self._pending = (heading, byte)
        self._wake.set()

    def drive_raw(self, heading_deg, speed_byte):
        """Straight to the radio: no heading offset, no deadband, no clamping.

        The deadband exists to protect radio airtime with six robots sharing an
        adapter. A speed sweep is one robot stepping through byte values, and
        suppressing a step because it resembles the last one would silently
        drop the very samples being measured.
        """
        byte = int(np.clip(speed_byte, 0, 255))
        heading = float(heading_deg) % 360.0
        self._desired = np.array([np.sin(np.radians(heading)),
                                  np.cos(np.radians(heading))]) * (byte / 255.0 * MAX_SPEED)
        with self._lock:
            self._pending = (heading, byte)
            self._force_write = True
        self._wake.set()

    def set_led(self, rgb, blink=None):
        self.rgb = tuple(int(np.clip(c, 0, 255)) for c in rgb)
        self.blink = blink
        with self._lock:
            self._pending_led = self.rgb
        self._wake.set()

    def stop(self):
        self._desired = np.zeros(2)
        with self._lock:
            self._pending = (self._last_sent[0] if self._last_sent else 0.0, 0)
        self._wake.set()

    def step(self, dt):
        """Pull the latest tracker fix. Velocity is differentiated from position."""
        if self.tracker is None:
            return
        try:
            fixes = self.tracker.read()
        except Exception as e:
            self.last_error = f"tracker read failed: {e}"
            return

        p = fixes.get(self.color)
        if p is None:
            p = fixes.get(self.name)
        if p is None:
            return                      # no fix; `tracked` decays on its own

        p = np.asarray(p, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            return

        t = now()
        prev_t, prev_p = self.last_seen, self.pos.copy()

        # The tracker's filtered velocity when it has one. Differencing two
        # camera positions is the fallback and it is a poor one: the noise in a
        # frame-to-frame difference is the position noise divided by the frame
        # interval, so a centimetre at 30fps is tens of cm/s — and the
        # controller's derivative term multiplies exactly that into the motors.
        # The filter has been computing a proper estimate all along; it was
        # simply never asked for it.
        filtered = None
        if self.tracker is not None and hasattr(self.tracker, "velocities"):
            try:
                vels = self.tracker.velocities()
                filtered = vels.get(self.color, vels.get(self.name))
            except Exception:
                filtered = None
        if filtered is not None and np.isfinite(filtered).all():
            self.vel = np.asarray(filtered, dtype=float)
        elif prev_t and t > prev_t:
            gap = t - prev_t
            if gap < STALE_AFTER:
                self.vel = (p - prev_p) / gap
        self.pos = p
        self.last_seen = t
        # A gyro reading if this toy turned out to have one, and the commanded
        # heading otherwise — see `fleet/sensors.py` for which branch applies.
        self.observe_heading(t, self.read_yaw())
        self.last_seen = t

    def read_yaw(self):
        """The ball's own idea of its aim, if it has one. None means camera-only.

        Deliberately not called on the worker thread: a blocking sensor read in
        the same loop that writes drive commands would spend the airtime the
        deadband exists to protect. `probe_sensors` in the bench measures what
        that would cost before anything switches it on.
        """
        if not self._yaw_source:
            return None
        try:
            return float(getattr(self._api, self._yaw_source)())
        except Exception:
            self._yaw_source = None        # stop asking; the fallback is fine
            return None

    def close(self):
        self._stop_flag.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._teardown()

    # -- worker thread ---------------------------------------------------

    def _teardown(self):
        api, self._api = self._api, None
        self._link_up = False
        if api is None:
            return
        try:
            api.__exit__(None, None, None)
        except Exception:
            pass

    def _connect_once(self):
        """Serialised, staggered connect. Returns True on success."""
        self.attempts += 1
        with _connect_lock:
            wait = CONNECT_STAGGER - (now() - _last_connect_at[0])
            if wait > 0:
                time.sleep(wait)
            _last_connect_at[0] = now()
            try:
                self._api = self._connector(self.ble_name)
                self._link_up = True
                self.last_error = None
                self.max_connections_hit = False
                log.info("%s connected as %s", self.code, self.ble_name)
                return True
            except Exception as e:
                self._api = None
                self._link_up = False
                self.last_error = str(e)
                if is_max_connections_error(e):
                    self.max_connections_hit = True
                    log.error("%s: %s", self.code, MAX_CONNECTIONS_HINT)
                else:
                    log.warning("%s connect failed: %s", self.code, e)
                return False

    def _write(self, heading, byte):
        """One packet where possible, and never while the library is mid-write.

        Three things learned the hard way from one traceback:

        `SpheroEduAPI.roll(h, v, duration)` is NOT a combined command. It sets
        the speed, sleeps for `duration`, and then calls `stop_roll()`. Passing
        duration 0 therefore starts the robot and stops it in the same breath —
        which is not slow or twitchy control, it is no control at all.

        `set_heading` already carries the current speed with it: it calls
        `roll_start(heading, speed)`. So heading-and-speed is one packet if the
        stored speed is updated first, and calling both setters sends two
        packets describing the same intent.

        And the library runs its OWN thread re-sending the speed every 0.8s,
        under a lock. Writing without that lock lets the two interleave on a
        link that takes 230ms per packet, and the loser raises TimeoutError
        from inside the library's thread.
        """
        t0 = now()
        heading = int(round(heading)) % 360
        byte = int(byte)
        lock = getattr(self._api, "_SpheroEduAPI__updating", None)
        try:
            if lock is not None:
                lock.acquire()
            last = self._last_sent
            speed_changed = last is None or last[1] != byte
            head_changed = last is None or int(round(last[0])) % 360 != heading

            if head_changed:
                # Update the stored speed without sending, then let set_heading
                # send both in a single roll_start.
                if speed_changed:
                    try:
                        setattr(self._api, "_SpheroEduAPI__speed", byte)
                    except Exception:
                        self._api.set_speed(byte)
                self._api.set_heading(heading)
            elif speed_changed:
                self._api.set_speed(byte)
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
        dt = now() - t0
        self.rtt = dt
        self.rtt_mean = dt if self.rtt_mean is None else 0.8 * self.rtt_mean + 0.2 * dt

    def _should_write(self, cmd):
        """Deadband. Radio airtime is the scarce resource with six robots on one adapter."""
        if self._force_write:
            self._force_write = False
            return True
        if self._last_sent is None:
            return True
        heading, byte = cmd
        last_h, last_b = self._last_sent
        if byte == 0 and last_b != 0:
            return True                       # a stop always goes out
        dh = abs((heading - last_h + 180.0) % 360.0 - 180.0)
        return dh > HEADING_DEADBAND or abs(byte - last_b) > SPEED_DEADBAND

    def _run(self):
        backoff = BACKOFF_START
        while not self._stop_flag.is_set():
            if not self._link_up:
                if self._connect_once():
                    backoff = BACKOFF_START
                else:
                    if self._stop_flag.wait(backoff):
                        break
                    backoff = min(backoff * 2, BACKOFF_MAX)
                    continue

            self._wake.wait(timeout=0.2)
            self._wake.clear()
            # Never faster than the link has been observed to manage. A command
            # issued while the previous one is still in flight is not a faster
            # loop, it is a queue — and the library's own keepalive thread is
            # sharing the same radio. Latest-wins means nothing is lost by
            # waiting: the newest intent is the one that goes.
            if self.rtt_mean:
                gap = (self.rtt_mean * 1.1) - (now() - self._last_write_at)
                if gap > 0 and self._stop_flag.wait(min(gap, 0.5)):
                    break
            if self._stop_flag.is_set():
                break

            with self._lock:
                cmd, led = self._pending, self._pending_led
                self._pending = self._pending_led = None

            try:
                if led is not None:
                    from spherov2.types import Color
                    self._api.set_main_led(Color(*led))
                if cmd is not None and self._should_write(cmd):
                    self._write(*cmd)
                    self._last_sent = cmd
                    self._last_write_at = now()
            except Exception as e:
                self.last_error = str(e)
                if is_max_connections_error(e):
                    self.max_connections_hit = True
                    log.error("%s: %s", self.code, MAX_CONNECTIONS_HINT)
                else:
                    log.warning("%s write failed, will reconnect: %s", self.code, e)
                self._teardown()
                self._last_sent = None

        self._teardown()

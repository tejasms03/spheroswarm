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
import os
import threading
import time

import numpy as np

from . import sphero_fast
from .handle import MAX_SPEED, RobotHandle, now, velocity_to_command
from .stillness import IMU_PERIOD_S, Stillness

log = logging.getLogger("fleet.real")

STALE_AFTER = 0.5           # s without a tracker fix before we stop trusting the position
HEADING_DEADBAND = 8.0      # degrees
SPEED_DEADBAND = 10         # speed byte
YAW_POLL_S = 0.25           # at most this often, and only on the worker thread
FAST_WRITES = os.environ.get("SPHERO_FAST_WRITES", "") not in ("", "0", "false")
"""Send drive commands without waiting for an acknowledgement.

Off by default, and it should stay off until a physical ball has confirmed
it. The change is sound on paper — see `fleet/sphero_fast` — but "the robot
ignores our packets" and "the robot obeys instantly" look identical from
here, and only a camera watching a ball can tell them apart.
"""

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
                 connector=None, autostart=True, heading_offset=0.0,
                 fast_writes=None):
        super().__init__(name, code, color, workspace)
        self.ble_name = ble_name
        self.heading_offset = float(heading_offset or 0.0) % 360.0
        self.tracker = tracker
        self._connector = connector or default_connector

        self._api = None
        self._link_up = False
        self.imu = Stillness()
        self._imu_read_at = 0.0
        self._stop_flag = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()

        self._pending = None            # (heading, speed) awaiting a write
        self._pending_led = None
        self._pending_back = None
        # Spinning in place on raw motor power, and re-zeroing where it points.
        # See `spin_raw`. `_raw_active` is True from the first spin until a
        # stop has actually been WRITTEN — not queued — because until then the
        # ball has stabilisation off and a roll command cannot run.
        self._pending_raw = None        # (left, right), "stop", or None
        self._raw_active = False
        self._back_led_works = True
        self._last_sent = None          # (heading, speed) actually written
        self._last_write_at = 0.0

        # Set by the bench once the sensor probe says which call answers on
        # this toy. Until then the estimator runs camera-only, which works.
        self._yaw_source = None
        self._yaw = None                # last value read, or None
        self._yaw_at = 0.0

        self._force_write = False       # set by drive_raw; skips the deadband once

        # Fire-and-forget drive packets. `_fast` is the writer once a link is
        # up and None whenever it is not, so one attribute answers both "is it
        # switched on" and "did it actually attach" — which are different
        # questions, and conflating them would report a speed-up that silently
        # fell back.
        self.fast_writes = FAST_WRITES if fast_writes is None else bool(fast_writes)
        self._fast = None

        # Seconds. With fast writes this is the enqueue, not a round trip, and
        # it collapses to microseconds — which is honest, and is why the pacing
        # in `_run` cannot be derived from it alone.
        self.rtt = None
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
                 fast_writes=self._fast is not None,
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

    def accel_quiet(self):
        """From spherov2's accelerometer stream, read at the rate it refreshes.

        `get_acceleration` returns a CACHE that the sensor stream `__enter__`
        started keeps up to date -- 150 ms a reading on this toy. That it
        answers instantly is how a streamed value is supposed to behave, not
        evidence that it is fake (`HANDOFF_CALIB.md` reasoned otherwise). Read
        on the stream's own clock rather than every tick: sampling one value
        thirty times a second would count a single reading thirty times.
        """
        now = time.monotonic()
        if now - self._imu_read_at >= IMU_PERIOD_S:
            self._imu_read_at = now
            api = self._api
            try:
                reading = api.get_acceleration() if api is not None else None
            except Exception:
                reading = None
            self.imu.add(now, reading)
        return self.imu.quiet(now)

    def accel_sigma(self):
        return self.imu.sigma(time.monotonic())

    def aim_zero(self, error_deg):
        """Move the BALL's zero, instead of carrying a correction forever.

        `heading_offset` is added to every command for as long as the roster
        holds it, and a Sphero establishes its heading reference when it
        connects — so a stored offset is stale the moment the link drops. That
        is why re-measuring it never stuck, and why the number kept coming back
        wrong after a reconnect.

        The v1.2 protocol can do better and this codebase has never used it.
        `reset_aim` takes whichever way the drive assembly is currently
        pointing and calls it zero. So: rotate the assembly by the error that
        was just measured, then declare that direction forward. The correction
        now lives in the robot, `heading_offset` goes to zero, and there is no
        longer a signed number applied on every command that can be applied the
        wrong way round.

        Rotating uses a speed of zero, which turns the assembly without driving
        the ball — that is how aiming works on a Sphero, and it is why this
        needs no floor. Stabilisation is briefly off inside `reset_aim`, so the
        ball will not self-right for that moment.
        """
        api = self._api
        if api is None:
            return False
        try:
            # Speed zero: the assembly turns, the ball stays put.
            setattr(api, "_SpheroEduAPI__speed", 0)
            api.set_heading(int(round(-float(error_deg))) % 360)
            api.reset_aim()
        except Exception as e:
            self.last_error = f"could not reset the aim: {e}"
            return False
        self.heading_offset = 0.0
        self._last_sent = None          # the frame changed under the deadband
        return True

    def spin_raw(self, power):
        """Turn on the spot at a motor POWER, not toward a heading.

        `roll(heading, 0)` hands the ball a final angle and its own controller
        slews there flat out, which overshoots and cannot be slowed from
        outside. `spin()` in the library is no better: it is a loop of
        `set_heading` calls, a heading ramp that busy-waits on the radio. This
        is the one command that sets a rate — left motor one way, right the
        other — and the caller decides when to stop.

        `power` is -255..255 and its sign picks the direction. Which way a
        positive power turns is a property of the drive and is not assumed
        here; the caller learns it by watching.

        Raw motors switch STABILISATION OFF, and `roll` does not work until it
        is back on. `stop_raw` restores it — and re-zeroes where the ball now
        points while doing so. See there for why those are one step. `stop`
        does the same, and so does a reconnect.
        """
        p = int(np.clip(int(power), -255, 255))
        with self._lock:
            # A queued roll must not be written after this: with stabilisation
            # off it cannot run, and if it arrived first it would leave the
            # library holding a speed its own thread keeps re-sending.
            self._pending = None
            self._pending_raw = (p, -p)
        self._wake.set()

    def stop_raw(self):
        """Stop turning, and STAY pointing where it stopped.

        Three things in one, because doing any of them alone is wrong:

        Motors off. Then ZERO HERE — `reset_aim`, which calls the current
        orientation zero. Then stabilisation on.

        The middle step is the one that matters. With stabilisation on, a
        Sphero holds the heading of its last ROLL command. After a raw spin it
        points somewhere new, and switching stabilisation straight back on can
        swing it back to that old heading and undo the turn completely. Zeroing
        first makes the held heading the current one, so there is nothing to
        swing back to. It also means driving straight on afterwards is simply
        `roll(0, speed)`: no stored offset, no frame to get the wrong way
        round, nothing that goes stale.

        Like `aim_zero`, this voids `heading_offset`, because the frame moved.
        """
        with self._lock:
            self._pending_raw = "stop"
        self._wake.set()

    @property
    def spinning(self):
        return self._raw_active or self._pending_raw not in (None, "stop")

    def _write_raw(self, raw):
        """Write a spin or a stop, under the library's own write lock."""
        api = self._api
        lock = getattr(api, "_SpheroEduAPI__updating", None)
        try:
            if lock is not None:
                lock.acquire()
            if raw == "stop":
                if not self._raw_active:
                    return
                # duration 0 sets the motors to zero and then OFF. It does not
                # restore stabilisation: the library reads its own flag at the
                # START of the call, and the spin already cleared it.
                api.raw_motor(0, 0, 0)
                # Zero HERE before stabilising, so the heading it then holds is
                # the one it points at now. See `stop_raw`.
                api.reset_aim()
                # `reset_aim` brings stabilisation back through the toy, not
                # through the API, so the API's own flag is still False. Left
                # that way, the next `raw_motor` believes it is already off,
                # does not switch it off, and the spin fights the stabiliser.
                api.set_stabilization(True)
                self._raw_active = False
                self.heading_offset = 0.0
                self._last_sent = None      # let the next roll actually go out
                return
            left, right = raw
            if not self._raw_active:
                # The library's background thread re-sends a roll every 0.8s
                # while its stored speed is non-zero. Leave one there and it
                # fights the spin, on a ball that cannot roll anyway.
                try:
                    setattr(api, "_SpheroEduAPI__speed", 0)
                except Exception:
                    pass
            # duration None: set the motors and RETURN. Any other value makes
            # the library sleep for it, on this thread, holding the radio.
            api.raw_motor(int(left), int(right), None)
            self._raw_active = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def set_led(self, rgb, blink=None):
        self.rgb = tuple(int(np.clip(c, 0, 255)) for c in rgb)
        self.blink = blink
        with self._lock:
            self._pending_led = self.rgb
        self._wake.set()

    def set_back_led(self, value):
        """Queue the taillight for the worker, like every other write.

        One packet, and only when the value CHANGES — see `_run`. An aiming
        light that is re-sent every tick is a light that costs as much airtime
        as driving, on a link where airtime is the binding constraint.
        """
        out = super().set_back_led(value)
        if not self._back_led_works:
            return out
        with self._lock:
            self._pending_back = out
        self._wake.set()
        return out

    def stop(self):
        self._desired = np.zeros(2)
        with self._lock:
            spinning = (self._raw_active
                        or self._pending_raw not in (None, "stop"))
            # A stop that leaves raw motors running is not a stop.
            if spinning:
                self._pending_raw = "stop"
            # After a spin the stop re-zeroes where the ball points, so the
            # roll that holds it must be heading 0. Reusing the last roll's
            # heading would ROTATE it back to where it was before the turn.
            held = 0.0 if spinning else (
                self._last_sent[0] if self._last_sent else 0.0)
            self._pending = (held, 0)
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

        t = self._fix_time()
        if self.last_seen and t <= self.last_seen:
            # The same fix we already have. A control loop running faster than
            # the camera reads the identical cache several times per frame, and
            # treating each read as a new observation feeds the heading
            # estimator duplicate samples it will happily average as evidence.
            return
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
        self.observe_heading(t, self._yaw)

    def _fix_time(self):
        """When the fix just read was MEASURED, not when we asked for it.

        The distinction is the whole of it. A threaded tracker caches its last
        result and answers every `read()` from that cache, so timing a fix by
        the read is how a camera that died keeps a robot looking freshly
        tracked: `last_seen` is refreshed by a position that never changes,
        `tracked` never decays, and `connected` stays true with no camera in
        the room at all. Every guard in the stack is built on `connected` —
        the drive loop's stop-on-lost-fix, the battery's watchdogs — so all of
        them end up protecting a ghost.

        Staleness has to be counted in seconds by whoever produced the fix.
        Trackers pumped synchronously by their caller have no gap to count and
        offer no timestamp; for those, asking really is seeing.
        """
        at = getattr(self.tracker, "fixes_at", None)
        if at is None:
            return now()
        try:
            at = float(at)
        except (TypeError, ValueError):
            return now()
        return at if at > 0 else now()

    def read_yaw(self):
        """The ball's own idea of its aim, if it has one. None means camera-only.

        A blocking radio read, so it belongs on the worker thread and nowhere
        else. It used to be called from `step()`, which runs on whichever
        thread ticks the controller — in the app, the one that also draws the
        window. Keeping it off the BLE worker was deliberate, to protect the
        airtime the deadband exists to save; the cost was a radio stall in the
        render loop instead, which is the worse of the two.

        Rate-limited rather than banished, so the airtime it spends is bounded
        and known: at most one read per `YAW_POLL_S`, taken after the drive
        command has already gone out. `step()` reads the cached value, which is
        at worst a quarter-second old — far fresher than the 25cm baseline the
        estimator needs before it will use anything.

        Dormant while `_yaw_source` is None, which is what the sensor probe
        concluded for this hardware.
        """
        if not self._yaw_source:
            return None
        try:
            return float(getattr(self._api, self._yaw_source)())
        except Exception:
            self._yaw_source = None        # stop asking; the fallback is fine
            return None

    def _poll_yaw(self):
        """Worker-side refresh of the cached yaw. Never raises."""
        if not self._yaw_source:
            return
        if now() - self._yaw_at < YAW_POLL_S:
            return
        self._yaw_at = now()
        self._yaw = self.read_yaw()

    def close(self):
        self._stop_flag.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._teardown()

    # -- worker thread ---------------------------------------------------

    def _teardown(self):
        api, self._api = self._api, None
        self._fast = None
        self._link_up = False
        # A reconnected ball comes up stabilised with its motors off.
        self._raw_active = False
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
                self._fast = sphero_fast.attach(self._api) if self.fast_writes else None
                # A ball that has just connected is a ball with its lights off,
                # whatever this handle last set them to. Re-assert both, or a
                # reconnect silently returns the robot to white-and-dark and
                # the roster's colours stop matching the floor -- which is the
                # tracker's entire input.
                with self._lock:
                    self._pending_led = self.rgb
                    self._pending_back = self.back_led or None
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

        if self._fast is not None:
            self._write_fast(heading, byte)
            dt = now() - t0
            self.rtt = dt
            self.rtt_mean = dt if self.rtt_mean is None else 0.8 * self.rtt_mean + 0.2 * dt
            return

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

    def _write_fast(self, heading, byte):
        """One unacknowledged roll packet, plus the bookkeeping that makes it safe.

        The library keeps its own idea of the current heading and speed, and a
        background thread re-sends them every 0.8s whenever the speed is
        non-zero. Bypassing `set_heading` never updates that cache, so without
        the two assignments below the keepalive would re-aim the ball at a
        stale heading roughly once a second — a slow twitch back toward an old
        direction, which is exactly the kind of fault that gets blamed on the
        controller.

        Kept in sync, the keepalive becomes useful instead: it restates the
        current intent on the acked path, so a fast path that silently stopped
        being delivered degrades to driving correctly at 1.25Hz rather than to
        a ball that ignores us. A keepalive packet can catch the pair
        half-updated and carry one stale value for one cycle; the next command
        overrides it well inside 0.8s.

        Deliberately WITHOUT the library's `__updating` lock, which the acked
        path above must take. That lock is held across an acked `roll_start` —
        230ms — so taking it here would reintroduce, about a third of the time,
        precisely the stall this path exists to remove. It was never needed for
        mutual exclusion on the radio: the hazard it guards is two callers each
        blocking on a reply and one timing out, and this path never waits for
        one. The queue underneath is a `SimpleQueue` and is thread-safe.
        """
        api = self._api
        try:
            setattr(api, "_SpheroEduAPI__speed", byte)
            setattr(api, "_SpheroEduAPI__heading", heading)
        except Exception:
            pass                        # the keepalive is a bonus, not a requirement
        self._fast.roll(byte, heading)

    def _pace_gap(self):
        """The minimum interval between writes, in seconds.

        On the acked path the round trip is its own pacing: a command issued
        while the last one is still in flight is not a faster loop, it is a
        queue.

        Fast writes remove the ack, so `rtt_mean` stops measuring the link and
        starts measuring an enqueue — microseconds — and pacing off it alone
        would queue commands far faster than the library's writer thread drains
        them. That is not a faster robot either; it is a growing backlog of
        stale intent, and latest-wins upstream cannot help once the packets are
        already on the queue. What the ack was standing in for is
        `cmd_safe_interval`, the writer's own pause between packets, so that
        becomes the floor.
        """
        floor = sphero_fast.MIN_WRITE_GAP if self._fast is not None else 0.0
        return max(floor, (self.rtt_mean or 0.0) * 1.1)

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

    def _drain_once(self):
        """Take everything queued and write it, in the order the ball needs.

        One iteration of `_run`'s work, pulled out so the ORDER of the writes
        can be tested without a thread and a clock — which is the whole of
        what can go wrong with a spin followed by a drive.
        """
        with self._lock:
            cmd, led = self._pending, self._pending_led
            back = self._pending_back
            raw = self._pending_raw
            self._pending = self._pending_led = None
            self._pending_back = None
            self._pending_raw = None

        try:
            if led is not None:
                from spherov2.types import Color
                self._api.set_main_led(Color(*led))
            if back is not None:
                # An int is the blue aiming light on every toy that has
                # one; a triple is a BOLT's addressable back LED. The
                # library branches on the type, so the type is the API.
                #
                # Guarded on its own, because everything else in this block
                # tears the link down and reconnects when it throws — which
                # is right for a drive command and absurd for an aiming
                # light. A toy that cannot do this says so once and then
                # stops being asked, exactly like the yaw source.
                try:
                    from spherov2.types import Color
                    self._api.set_back_led(
                        Color(*back) if isinstance(back, tuple) else int(back))
                except Exception as e:
                    self._back_led_works = False
                    self.last_error = f"no taillight on this toy: {e}"
                    log.info("%s: taillight unsupported, ignoring", self.code)
            # LATEST WINS, as for every other write here. `spin_raw` clears
            # the queued roll, so a roll still queued now was issued AFTER any
            # spin in this batch — it is the newer intent, and it means stop
            # spinning. The same holds for a roll arriving while a spin from an
            # earlier batch is still running. Dropping the roll instead would
            # leave a ball that was told to drive sitting and spinning.
            if cmd is not None and (isinstance(raw, tuple) or
                                    (raw is None and self._raw_active)):
                raw = "stop"
            # Order matters: a spin or its stop first, then the roll. Stopping
            # re-zeroes and restores stabilisation, and a roll can only run
            # after that.
            if raw is not None:
                self._write_raw(raw)
            if (cmd is not None and not self._raw_active
                    and self._should_write(cmd)):
                self._write(*cmd)
                self._last_sent = cmd
                self._last_write_at = now()
            # After the command, never before it: driving is what the loop
            # is for, and a sensor read that delays it buys a heading
            # estimate at the cost of the thing being estimated.
            self._poll_yaw()
        except Exception as e:
            self.last_error = str(e)
            if is_max_connections_error(e):
                self.max_connections_hit = True
                log.error("%s: %s", self.code, MAX_CONNECTIONS_HINT)
            else:
                log.warning("%s write failed, will reconnect: %s", self.code, e)
            self._teardown()
            self._last_sent = None

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
            pace = self._pace_gap()
            if pace:
                gap = pace - (now() - self._last_write_at)
                if gap > 0 and self._stop_flag.wait(min(gap, 0.5)):
                    break
            if self._stop_flag.is_set():
                break

            self._drain_once()

        self._teardown()

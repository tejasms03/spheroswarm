"""How old is a camera fix, and who is allowed to believe it?

The bug these guard against is the one the whole project keeps rediscovering:
the controller was fine and the measurement was lying. A threaded tracker
answers `read()` from a cache, so a consumer that times a fix by the moment it
asked cannot tell a live camera from one that died an hour ago — both reply
instantly, with a position.

Everything here asserts behaviour under a camera that has stopped, because
"it did not raise" is exactly what the broken version also did.
"""

import time

import numpy as np
import pytest

from fleet.manager import Fleet, FleetEnv
from fleet.real_handle import STALE_AFTER, SpheroRobot
from fleet.roster import RobotEntry
from fleet.sim_handle import SimRobot
from fleet.vision_link import RETRY_PAUSE, CameraTracker
from tests.conftest import FakeApi, FakeConnector


# -- fakes ---------------------------------------------------------------

class CachedTracker:
    """A threaded tracker: `read()` always answers, `fixes_at` says how stale."""

    def __init__(self, pos=(50.0, 50.0), live=True):
        self.pos = np.asarray(pos, dtype=float)
        self.live = live
        self.fixes_at = time.time()
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.live:
            self.fixes_at = time.time()
        return {"cyan": self.pos.copy()}

    def velocities(self):
        return {}


class PumpedTracker:
    """A synchronous tracker: no `fixes_at`, because asking really is seeing."""

    def __init__(self, pos=(50.0, 50.0)):
        self.pos = np.asarray(pos, dtype=float)

    def read(self):
        return {"cyan": self.pos.copy()}

    def velocities(self):
        return {}


def _robot(tracker, open_ws, **kw):
    return SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                       tracker=tracker, connector=FakeConnector(),
                       autostart=False, **kw)


# -- the freeze ----------------------------------------------------------

def test_a_frozen_tracker_stops_being_believed(open_ws):
    """The headline. A camera that died keeps answering `read()` forever; the
    robot must still notice it has not been SEEN."""
    t = CachedTracker(live=False)
    r = _robot(t, open_ws)
    r._link_up = True

    r.step(0.1)
    assert r.tracked is True, "a fresh fix is a fix"

    time.sleep(STALE_AFTER + 0.15)
    for _ in range(5):
        r.step(0.1)

    assert t.reads >= 5, "the tracker is still answering — that is the point"
    assert r.tracked is False
    assert r.connected is False


def test_a_live_tracker_is_still_believed(open_ws):
    """The fix must not achieve its result by disbelieving everything."""
    t = CachedTracker(live=True)
    r = _robot(t, open_ws)
    r._link_up = True

    for _ in range(6):
        r.step(0.05)
        time.sleep(0.05)

    assert r.tracked is True
    assert r.connected is True


def test_freshness_comes_from_the_tracker_not_the_clock(open_ws):
    t = CachedTracker(live=False)
    stamped = t.fixes_at
    r = _robot(t, open_ws)

    time.sleep(0.12)
    r.step(0.1)

    assert r.last_seen == pytest.approx(stamped, abs=1e-6), (
        "last_seen must be when the fix was measured, not when it was read")


def test_a_synchronous_tracker_is_unaffected(open_ws):
    """A tracker its caller pumps has no cache to go stale, offers no
    timestamp, and must keep working exactly as before."""
    r = _robot(PumpedTracker(), open_ws)
    r._link_up = True

    for _ in range(4):
        r.step(0.05)
        time.sleep(0.02)

    assert r.tracked is True
    assert r.connected is True


def test_a_nonsense_timestamp_falls_back_to_the_clock(open_ws):
    """A tracker whose `fixes_at` is unreadable must degrade to the old
    behaviour, not to an untracked robot."""
    class Broken(CachedTracker):
        fixes_at = "not a time"

    r = _robot(Broken(), open_ws)
    r._link_up = True
    r.step(0.1)
    assert r.tracked is True


def test_the_same_fix_is_not_counted_twice(open_ws):
    """A control loop faster than the camera reads one fix repeatedly. Each
    read is not an observation, and feeding them to the heading estimator as
    though they were is evidence manufactured from a single sample."""
    t = CachedTracker(live=False)
    r = _robot(t, open_ws)
    r._link_up = True

    r.step(0.05)
    first = r.last_seen
    for _ in range(5):
        r.step(0.05)

    assert r.last_seen == first, "no new fix means nothing new to record"


# -- the stumble ---------------------------------------------------------

def test_a_failing_camera_does_not_spin_the_thread(monkeypatch):
    """Both failure paths must pause. Without it a disconnected camera turns
    this thread into a busy core, starving the render loop and the BLE
    workers whose timing the calibration exists to measure."""
    ct = CameraTracker.__new__(CameraTracker)
    ct.error = None
    slept = []
    monkeypatch.setattr("fleet.vision_link.time.sleep", slept.append)

    ct._stumble("camera returned no frame")

    assert slept == [RETRY_PAUSE]
    assert ct.error == "camera returned no frame"


def test_a_stumble_keeps_the_last_fixes(open_ws):
    """A blink is not six robots vanishing. The timestamp already says how far
    to trust what is cached, which is the honest answer and one that recovers
    by itself."""
    ct = CameraTracker.__new__(CameraTracker)
    ct.error = None
    ct._lock = __import__("threading").Lock()
    ct._fixes = {"cyan": np.array([10.0, 10.0])}
    ct._fixed_at = time.time()

    import fleet.vision_link as vl
    real_sleep, vl.time.sleep = vl.time.sleep, lambda _: None
    try:
        ct._stumble("boom")
    finally:
        vl.time.sleep = real_sleep

    assert "cyan" in ct.read(), "the cache survives; its age is what changed"


def test_fix_age_before_any_frame_is_infinite():
    ct = CameraTracker.__new__(CameraTracker)
    ct._lock = __import__("threading").Lock()
    ct._fixed_at = 0.0
    assert ct.fix_age == float("inf")


# -- who gets driven -----------------------------------------------------

def _env(handles):
    fleet = Fleet.__new__(Fleet)
    fleet.handles = {h.code: h for h in handles}
    env = FleetEnv.__new__(FleetEnv)
    env.fleet = fleet
    env.codes = list(fleet.handles)
    return env


def test_a_lost_robot_is_stopped_not_skipped(open_ws):
    """Skipping and stopping are different instructions to a Sphero: it holds
    its last speed command until given another, so a loop that merely stops
    updating leaves the ball driving."""
    api = FakeApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=lambda n, timeout=8.0: api, autostart=False)
    h._link_up = True
    h.last_seen = 0.0                       # link up, camera has lost it

    assert h.connected is False
    h._pending = None
    _env([h]).apply(np.array([[1.0, 0.0]]), max_speed=30.0)

    assert h._pending is not None, "a lost robot must still be commanded"
    assert h._pending[1] == 0, "and the command must be stop"


def test_a_connected_robot_is_driven_normally(open_ws):
    api = FakeApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=lambda n, timeout=8.0: api, autostart=False)
    h._link_up = True
    h.last_seen = time.time()

    assert h.connected is True
    _env([h]).apply(np.array([[1.0, 0.0]]), max_speed=30.0)

    assert h._pending is not None
    assert h._pending[1] > 0


def test_simulated_robots_are_untouched_by_the_gate(open_ws):
    """Sim robots are always connected, so the branch must be invisible to
    them — every existing sim test depends on that."""
    s = SimRobot("S", "SSSS", "cyan", workspace=open_ws, pos=[50, 50],
                 randomize=False)
    _env([s]).apply(np.array([[1.0, 0.0]]), max_speed=30.0)
    assert np.linalg.norm(s._desired) == pytest.approx(30.0, rel=1e-3)

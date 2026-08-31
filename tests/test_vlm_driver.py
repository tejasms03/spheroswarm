"""Serving the framework's drive requests with the bench's own controller.

The driver has no control loop. What these pin is that a request becomes
`app.path` plus `app.arm()` -- the same path the GO button and the bench's own
agent tools take -- and that every way of NOT driving is reported rather than
swallowed.
"""

import numpy as np
import pytest

from vlm.bridge import ArenaFrame
from vlm.driver import Driver


ARENA_W, ARENA_H = 138.8, 110.8
A_PATH_PX = [[400, 300, 0.0, 0], [700, 550, 0.5, 0]]


class FakePath:
    """Stands in for `fleet_test.Path` so a window is never opened."""

    def __init__(self, pts, closed=False, kind="polyline"):
        self.pts = list(pts)
        self.closed = closed
        self.kind = kind
        self.length = 100.0

    @classmethod
    def point(cls, p):
        got = cls([p], kind="point")
        got.length = 0.0
        return got


class FakeHandle:
    def __init__(self, raises=None):
        self.raises = raises
        self.courses = []

    def drive_raw(self, heading_deg, speed_byte):
        if self.raises:
            raise self.raises
        self.courses.append((heading_deg, speed_byte))


class FakeFleet:
    def __init__(self, handle=None):
        self.handles = {"CRXS": handle} if handle else {}


class FakeApp:
    def __init__(self, arms=True, handle=None):
        self.armed = False
        self.path = None
        self.note = ""
        self.last_outcome = None
        self.last_note = ""
        self.code = "CRXS"
        self.fleet = FakeFleet(handle)
        self._arm_source = "button"
        self._arms = arms
        self.disarms = []

    def arm(self):
        self.armed = bool(self._arms)
        if not self._arms:
            self.note = "refused: the frame is mirrored — flip and re-probe first"

    def disarm(self, note=None):
        self.armed = False
        self.disarms.append(note)


class FakeRobot:
    def __init__(self, path=None):
        self.path = path if path is not None else []
        self.requests = []
        self.courses = []
        self.outcomes = []
        self.attached = False

    def attach(self):
        self.attached = True
        return True

    def detach(self):
        self.attached = False
        return True

    def get_path(self, rid):
        return self.path

    def take_request(self, rid):
        return self.requests.pop(0) if self.requests else None

    def take_course(self, rid):
        return self.courses.pop(0) if self.courses else None

    def set_outcome(self, rid, outcome, reason=None):
        self.outcomes.append({"outcome": outcome, "reason": reason})
        return True


class FakeClient:
    def __init__(self, path=None):
        self.Robot = FakeRobot(path)


def a_driver(app=None, path=None, arms=True, handle=None):
    app = app or FakeApp(arms=arms, handle=handle)
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2, path_cls=FakePath)
    return drv, app, FakeClient(path)


# -- conversion ---------------------------------------------------------------

def test_their_pixel_waypoints_become_the_bench_s_centimetres():
    drv, _app, _c = a_driver()
    got = drv.path_to_cm(A_PATH_PX)
    assert np.allclose(got[0], [40.0, 30.0])
    assert np.allclose(got[1], [70.0, 55.0])


def test_the_per_point_heading_is_dropped():
    """It is a property of the PATH, derived from consecutive waypoints, and
    pure pursuit works it out again from the lookahead. Carrying it would store
    an answer beside the question it comes from."""
    drv, _app, _c = a_driver()
    assert all(len(p) == 2 for p in drv.path_to_cm(A_PATH_PX))


def test_a_malformed_waypoint_is_skipped_rather_than_crashing_the_tick():
    drv, _app, _c = a_driver()
    got = drv.path_to_cm([[400, 300, 0.0, 0], None, [7], [700, 550, 0.5, 0]])
    assert len(got) == 2


def test_a_requested_DWELL_is_noticed_because_we_are_about_to_ignore_it():
    drv, _app, _c = a_driver()
    assert drv.wants_delay([[400, 300, 0.0, 0]]) is False
    assert drv.wants_delay([[400, 300, 0.0, 250]]) is True


# -- driving ------------------------------------------------------------------

def test_a_drive_request_arms_the_bench_s_own_controller():
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.armed is True
    assert app.path.kind == "polyline"
    assert drv.served == 1


def test_the_drive_is_attributed_to_the_framework_not_to_the_button():
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app._arm_source == "vlm"


def test_ONE_waypoint_is_a_goal_not_a_one_point_polyline():
    """A single-element polyline has no length for the lookahead to run along."""
    drv, app, client = a_driver(path=[[400, 300, 0.0, 0]])
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.path.kind == "point"


def test_an_EMPTY_path_is_refused_and_says_so():
    drv, app, client = a_driver(path=[])
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.armed is False
    assert client.Robot.outcomes[-1]["outcome"] == "refused"


def test_a_REFUSAL_TO_ARM_is_passed_back_verbatim():
    """`arm` refuses for a mirrored frame, no robot, no homography, no aim.

    Swallowing that and reporting a start is how an agent ends up planning on
    top of a rig that never moved.
    """
    drv, app, client = a_driver(path=A_PATH_PX, arms=False)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.armed is False
    assert client.Robot.outcomes[-1]["outcome"] == "refused"
    assert "mirrored" in client.Robot.outcomes[-1]["reason"]
    assert drv.refused == 1


def test_stopping_disarms_the_bench():
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "stop"})
    drv.serve(client)
    assert app.disarms == ["stopped by the framework"]


def test_an_unknown_request_is_noted_and_does_nothing():
    drv, app, client = a_driver()
    client.Robot.requests.append({"want": "pirouette"})
    drv.serve(client)
    assert app.armed is False
    assert "pirouette" in drv.last_note


# -- outcomes -----------------------------------------------------------------

def test_the_verdict_is_reported_on_the_TICK_THE_DRIVE_ENDS():
    """Watched as a transition, not polled as a state: the verdict is only
    written at disarm and the next drive clears it."""
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert client.Robot.outcomes == []

    app.armed = False
    app.last_outcome, app.last_note = "arrived", "arrived within 6cm"
    drv.serve(client)
    assert client.Robot.outcomes[-1] == {"outcome": "arrived",
                                         "reason": "arrived within 6cm"}


def test_STOPPED_is_not_reported_as_ARRIVED():
    """`armed` goes false for arriving, for giving up stuck, for losing the
    ball and for a person pressing escape. Only one of those is success."""
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    app.armed = False
    app.last_outcome, app.last_note = "stuck", "against a boundary"
    drv.serve(client)
    assert client.Robot.outcomes[-1]["outcome"] == "stuck"


def test_a_drive_that_ends_with_no_verdict_says_UNKNOWN_not_arrived():
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    app.armed = False
    drv.serve(client)
    assert client.Robot.outcomes[-1]["outcome"] == "unknown"


def test_the_verdict_is_reported_once_not_every_tick_afterwards():
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    app.armed = False
    app.last_outcome = "arrived"
    drv.serve(client)
    drv.serve(client)
    drv.serve(client)
    assert len(client.Robot.outcomes) == 1


# -- raw courses --------------------------------------------------------------

def test_a_course_reaches_the_handle():
    handle = FakeHandle()
    drv, _app, client = a_driver(handle=handle)
    client.Robot.courses.append({"heading_deg": 90.0, "speed": 12.0})
    drv.serve(client)
    assert handle.courses == [(90.0, 12)]


def test_a_course_SUPERSEDES_a_path_drive_rather_than_fighting_it():
    """Two things commanding one ball reads as a tuning problem for a week:
    the ball wanders, both sources look reasonable, neither log mentions the
    other."""
    handle = FakeHandle()
    drv, app, client = a_driver(handle=handle)
    app.armed = True
    client.Robot.courses.append({"heading_deg": 45.0, "speed": 8.0})
    drv.serve(client)
    assert app.armed is False
    assert "superseded" in app.disarms[-1]
    assert handle.courses == [(45.0, 8)]


def test_a_course_with_no_robot_connected_says_so():
    drv, _app, client = a_driver()
    client.Robot.courses.append({"heading_deg": 0.0, "speed": 5.0})
    drv.serve(client)
    assert "no robot connected" in drv.last_note


def test_a_handle_that_throws_is_reported_not_raised():
    handle = FakeHandle(raises=RuntimeError("link down"))
    drv, _app, client = a_driver(handle=handle)
    client.Robot.courses.append({"heading_deg": 0.0, "speed": 5.0})
    drv.serve(client)                      # must not raise
    assert "link down" in drv.last_note


# -- attachment ---------------------------------------------------------------

def test_serving_ATTACHES_so_the_service_stops_refusing_drives():
    drv, _app, client = a_driver()
    assert client.Robot.attached is False
    drv.serve(client)
    assert client.Robot.attached is True


def test_attaching_happens_once():
    drv, _app, client = a_driver()
    drv.serve(client)
    client.Robot.attached = False
    drv.serve(client)
    assert client.Robot.attached is False   # not re-attached every tick


def test_closing_DETACHES_so_a_late_request_is_refused_not_queued():
    drv, _app, client = a_driver()
    drv.serve(client)
    drv.close(client)
    assert client.Robot.attached is False
    assert drv.attached is False


def test_a_tick_with_nothing_waiting_does_nothing():
    drv, app, client = a_driver()
    assert drv.serve(client) is None
    assert app.armed is False
    assert client.Robot.outcomes == []


def test_arming_over_a_live_run_closes_the_old_one_first():
    """Otherwise the replaced run never reaches disarm and never gets its end
    row, which is where the verdict is written -- the reason `run_outcome` was
    blank on most logged runs. An agent re-aiming mid-drive is exactly the case
    that produces back-to-back arms."""
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.armed is True

    app.disarms.clear()
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    # The FakeApp's arm() is a stand-in, so this pins the driver's half: a
    # second request re-arms rather than being dropped, and the bench's own
    # arm() closes the previous run (tests/test_fleet.py covers that side).
    assert app.armed is True
    assert drv.served == 2

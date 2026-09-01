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
        self.anchor = False
        self.s = None

    def restart(self):
        self.s = 0.0 if self.anchor else None

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
        self.states = {}
        self.arena = None
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

    def set_arena(self, facts):
        self.arena = dict(facts)
        return True

    def set_state(self, rid, state):
        self.states[rid] = state
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


# -- calibrate = zero the aim, then probe the frame ---------------------------

class ProbingApp(FakeApp):
    """A bench whose `start_probe` can succeed or refuse."""

    def __init__(self, starts=True, mirrored=False):
        super().__init__()
        self.probe = None
        self.mirrored = mirrored
        self._starts = starts
        self.probes = 0

    def start_probe(self):
        self.probes += 1
        if self._starts:
            self.probe = {"i": 0, "rows": []}
        else:
            self.note = "needs a homography and a locked track"


def a_prober(**kw):
    app = ProbingApp(**kw)
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2, path_cls=FakePath)
    return drv, app, FakeClient()


def test_calibrate_runs_the_bench_s_own_probe():
    """`start_probe` zeroes the aim at rest before driving, so the sequence is
    one call rather than two."""
    drv, app, client = a_prober()
    client.Robot.requests.append({"want": "probe"})
    drv.serve(client)
    assert app.probes == 1
    assert "probing" in drv.last_note


def test_a_probe_the_bench_REFUSES_is_reported_not_swallowed():
    drv, app, client = a_prober(starts=False)
    client.Robot.requests.append({"want": "probe"})
    drv.serve(client)
    assert app.probe is None
    assert "homography" in drv.last_note


def test_a_MIRRORED_verdict_comes_back_when_the_probe_finishes():
    """The one thing that has to reach whoever pressed the button: nothing on
    this rig converges while the frame is a reflection."""
    drv, app, client = a_prober(mirrored=True)
    client.Robot.requests.append({"want": "probe"})
    drv.serve(client)
    app.probe = None                      # the bench finishes it in its own tick
    drv.serve(client)
    assert client.Robot.outcomes[-1]["outcome"] == "mirrored"


def test_a_CLEAN_probe_says_so_too():
    drv, app, client = a_prober(mirrored=False)
    client.Robot.requests.append({"want": "probe"})
    drv.serve(client)
    app.probe = None
    drv.serve(client)
    assert client.Robot.outcomes[-1]["outcome"] == "probed"


def test_the_probe_verdict_is_reported_once():
    drv, app, client = a_prober()
    client.Robot.requests.append({"want": "probe"})
    drv.serve(client)
    app.probe = None
    drv.serve(client); drv.serve(client); drv.serve(client)
    assert len(client.Robot.outcomes) == 1


# -- orbit: a real circle, not a polygon --------------------------------------

class CirclePath(FakePath):
    @classmethod
    def circle(cls, centre, radius, n=48):
        got = cls([centre], closed=True, kind="circle")
        got.centre, got.radius = centre, radius
        return got


def an_orbiter():
    app = FakeApp()
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2,
                 path_cls=CirclePath)
    return drv, app, FakeClient()


def test_orbit_builds_a_CIRCLE_not_a_string_of_waypoints():
    """`trace_targets` can only express waypoints, so an agent asked to orbit
    through it produces a polygon: corners the lookahead cuts, and it ends
    instead of repeating."""
    drv, app, client = an_orbiter()
    client.Robot.requests.append({"want": "orbit", "centre": [694, 554],
                                  "radius": 300})
    drv.serve(client)
    assert app.path.kind == "circle"
    assert app.path.closed is True
    assert app.armed is True


def test_the_orbit_is_converted_from_arena_pixels_to_centimetres():
    drv, app, client = an_orbiter()
    client.Robot.requests.append({"want": "orbit", "centre": [694, 554],
                                  "radius": 300})
    drv.serve(client)
    assert app.path.centre == pytest.approx([69.4, 55.4], abs=0.01)
    assert app.path.radius == pytest.approx(30.0, abs=0.01)


def test_a_refused_orbit_reports_like_any_other_refusal():
    app = FakeApp(arms=False)
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2,
                 path_cls=CirclePath)
    client = FakeClient()
    client.Robot.requests.append({"want": "orbit", "centre": [694, 554],
                                  "radius": 300})
    drv.serve(client)
    assert client.Robot.outcomes[-1]["outcome"] == "refused"
    assert "mirrored" in client.Robot.outcomes[-1]["reason"]


# -- flow: a curve described rather than enumerated ---------------------------

class FlowApp(FakeApp):
    def __init__(self, refuses=None):
        super().__init__()
        self._refuses = refuses

    def agent_check(self, points):
        return self._refuses


def a_flower(refuses=None):
    app = FlowApp(refuses)
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2, path_cls=FakePath)
    return drv, app, FakeClient()


EIGHT = ("points = [(cx + 400*sin(2*pi*i/64), cy + 200*sin(4*pi*i/64)) "
         "for i in range(64)]")


def test_a_figure_eight_becomes_a_closed_path_in_centimetres():
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "expression": EIGHT,
                                  "closed": True})
    drv.serve(client)
    assert app.path is not None
    assert app.path.closed is True
    assert len(app.path.pts) == 64
    xs = [p[0] for p in app.path.pts]
    # 400px either side of a 1388px-wide arena, converted at 10px/cm.
    assert max(xs) - min(xs) == pytest.approx(80.0, abs=1.0)


def test_an_OPEN_curve_drives_once_instead_of_looping():
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "expression": EIGHT,
                                  "closed": False})
    drv.serve(client)
    assert app.path.closed is False


def test_a_curve_that_leaves_the_ARENA_is_refused_by_the_bench_s_own_rule():
    """Not a second copy of the boundary rule -- `agent_check` refuses a goal
    the ball can only reach by shoving, and a curve is only as safe as its
    worst point."""
    drv, app, client = a_flower(refuses="refused: too close to the edge")
    client.Robot.requests.append({"want": "flow", "expression": EIGHT,
                                  "closed": True})
    drv.serve(client)
    assert app.armed is False
    assert "too close" in client.Robot.outcomes[-1]["reason"]


def test_a_BROKEN_expression_is_reported_not_raised():
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "closed": True,
                                  "expression": "points = [(1, 2), (oops, 4)]"})
    drv.serve(client)
    assert app.armed is False
    assert client.Robot.outcomes[-1]["outcome"] == "refused"


def test_an_expression_that_assigns_NOTHING_is_refused():
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "closed": True,
                                  "expression": "x = 1"})
    drv.serve(client)
    assert "never assigned `points`" in client.Robot.outcomes[-1]["reason"]


def test_a_single_point_is_not_a_curve():
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "closed": True,
                                  "expression": "points = [(694, 554)]"})
    drv.serve(client)
    assert "at least two points" in client.Robot.outcomes[-1]["reason"]


def test_a_refusal_is_restated_in_the_units_the_agent_WRITES_in():
    """`agent_check` speaks centimetres; every tool speaks arena pixels.

    An agent handed "aim inside x 10..129" after submitting a curve spanning
    294 to 1094 read it as a rig fault and stopped — the right call on the
    information it had, and the information was in the wrong units.
    """
    class Bounded(FlowApp):
        def agent_bounds(self):
            return [0.0, 0.0, ARENA_W, ARENA_H]

        def goal_margin_cm(self):
            return 10.0

    app = Bounded("refused: (104, 104) is too close to the edge. "
                  "Aim inside x 10..129, y 10..101")
    drv = Driver(app, ArenaFrame(ARENA_W, ARENA_H), robot_id=2, path_cls=FakePath)
    client = FakeClient()
    client.Robot.requests.append({"want": "flow", "expression": EIGHT,
                                  "closed": True})
    drv.serve(client)
    said = client.Robot.outcomes[-1]["reason"]
    assert "ARENA PIXELS" in said
    assert "x 100..1288" in said and "y 100..1008" in said


# -- a shape starts at its start ----------------------------------------------

def test_a_described_SHAPE_is_driven_from_its_beginning():
    """Pure pursuit locks on to the nearest point when a run starts, which is
    right for a goal and wrong for a shape. A ball parked near the middle of a
    figure eight would start halfway round, the first half would never be
    driven, and nothing would report a fault — the follower did as it was told.
    """
    drv, app, client = a_flower()
    client.Robot.requests.append({"want": "flow", "expression": EIGHT,
                                  "closed": True})
    drv.serve(client)
    assert app.path.anchor is True


def test_an_orbit_and_a_trajectory_are_anchored_too():
    drv, app, client = an_orbiter()
    client.Robot.requests.append({"want": "orbit", "centre": [694, 554],
                                  "radius": 300})
    drv.serve(client)
    assert app.path.anchor is True

    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.path.anchor is True


def test_a_single_GOAL_is_not_anchored():
    """A drive to one point should pick up from wherever the ball is."""
    drv, app, client = a_driver(path=[[400, 300, 0.0, 0]])
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.path.kind == "point"
    assert getattr(app.path, "anchor", False) is False


def test_the_path_class_comes_from_the_app_s_own_module():
    """The bench is forked. Handing a `fleet_test.Path` to a `coast_test`
    controller works only while the two happen to agree, and they are meant to
    diverge — that is what the fork is for."""
    import coast_test
    import fleet_test

    class OnCoast:
        pass
    OnCoast.__module__ = "coast_test"
    drv = Driver(OnCoast(), ArenaFrame(ARENA_W, ARENA_H), robot_id=2)
    assert drv.Path is coast_test.Path
    assert drv.Path is not fleet_test.Path


def test_an_app_whose_module_has_no_Path_falls_back():
    """An app is not always a bench: the end-to-end test drives a stand-in
    defined in the test module, which has no `Path` and does not need one."""
    import fleet_test
    drv = Driver(FakeApp(), ArenaFrame(ARENA_W, ARENA_H), robot_id=2)
    assert drv.Path is fleet_test.Path


# -- the arena is asked for, not memorised -------------------------------------

def test_the_arena_is_published_on_attach():
    """Anything that writes the arena into a prompt writes a number that goes
    stale silently. It already did: a re-click moved the centre from (694, 554)
    to (712, 613) and the agent kept orbiting the old one."""
    drv, app, client = a_driver()
    client.Robot.arena = None
    drv.serve(client)
    facts = client.Robot.arena
    assert facts["width"] == 1388 and facts["height"] == 1108
    assert facts["centre"] == [694.0, 554.0]


def test_the_published_centre_MOVES_when_the_arena_does():
    app = FakeApp()
    drv = Driver(app, ArenaFrame(142.5, 122.7, origin_cm=(-33.3, -31.8)),
                 robot_id=2, path_cls=FakePath)
    client = FakeClient()
    drv.serve(client)
    assert client.Robot.arena["centre"] == [712.5, 613.5]


def test_the_safe_box_is_inset_on_every_side():
    drv, _app, client = a_driver()
    drv.serve(client)
    f = client.Robot.arena
    x0, y0, x1, y1 = f["safe"]
    assert x0 > 0 and y0 > 0
    assert x1 < f["width"] and y1 < f["height"]


def test_the_arena_says_how_BIG_an_orbit_can_be():
    """Without it the agent has only "10 pixels is 1cm" to reason from and
    picks something timid — asked to orbit, it drew circles a tenth of the
    arena across, which is not what anybody means by orbiting."""
    drv, _app, client = a_driver()
    drv.serve(client)
    f = client.Robot.arena
    cx, cy = f["centre"]
    r = f["max_radius"]
    x0, y0, x1, y1 = f["safe"]
    assert r > 0
    # The circle it describes must fit inside the safe box on every side.
    assert cx - r >= x0 - 0.05 and cx + r <= x1 + 0.05
    assert cy - r >= y0 - 0.05 and cy + r <= y1 + 0.05
    # And it must be a real orbit, not a token one.
    assert 2 * r > 0.5 * min(f["width"], f["height"])


def test_an_ACCEPTED_request_marks_the_robot_moving():
    """The signal that lets a caller tell a request the bench took from one it
    then refused. Without it every drive tool returns a receipt, and the agent
    reported a robot 'looping continuously' while it sat still."""
    drv, app, client = a_driver(path=A_PATH_PX)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert app.armed is True
    assert client.Robot.states.get(2) == "moving"


def test_a_REFUSED_request_does_not():
    drv, app, client = a_driver(path=A_PATH_PX, arms=False)
    client.Robot.requests.append({"want": "drive"})
    drv.serve(client)
    assert client.Robot.states.get(2) != "moving"
    assert client.Robot.outcomes[-1]["outcome"] == "refused"

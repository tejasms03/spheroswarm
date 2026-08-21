"""Heading tracked continuously, rather than calibrated once and hoped over.

Every test here asks a behavioural question — does the robot end up going where
it was told — rather than inspecting the number the filter holds. A sign error
satisfies any assertion about a value and none about a direction of travel, and
this file exists because two such errors were made writing it.

Both handles are exercised. They apply the offset in OPPOSITE angular senses —
a simulator rotates a velocity anticlockwise, a Sphero adds to a clockwise
compass bearing — so a test that only covered one would pass with the other
silently inverted.
"""

import math

import numpy as np
import pytest

from fleet.heading import HeadingEstimator, wrap180
from fleet.real_handle import SpheroRobot
from fleet.sim_handle import SimRobot
from workspace.space import Workspace

BIG = Workspace(bounds_cm=[[0, 0], [400, 0], [400, 400], [0, 400]])


def shuttle(h, seconds=40.0, dt=0.05, leg=2.0, speed=30.0):
    """Drive back and forth. Short legs, so it never reaches a wall."""
    n = int(leg / dt)
    for i in range(int(seconds / dt)):
        a = 0.0 if (i // n) % 2 == 0 else math.pi
        h.set_velocity(np.array([math.cos(a), math.sin(a)]) * speed)
        h.step(dt)
    return h


def travel_error(h, dt=0.05, steps=24, speed=30.0):
    """Degrees between where it was told to go and where it went."""
    v = np.array([speed, 0.0])
    for _ in range(4):                    # clear the latency queue
        h.set_velocity(v)
        h.step(dt)
    start = h.pos.copy()
    for _ in range(steps):
        h.set_velocity(v)
        h.step(dt)
    d = h.pos - start
    return wrap180(math.degrees(math.atan2(d[1], d[0])))


def sim(bias_deg=0.0, drift=0.0, tracking=True, pos=(200.0, 200.0)):
    h = SimRobot("A", "AAAA", "cyan", workspace=BIG, pos=list(pos),
                 randomize=False)
    h.bias = math.radians(bias_deg)
    h.drift_rate = drift
    h.heading_tracking = tracking
    return h


# -- wrap180, the likeliest bug in the file ----------------------------------

@pytest.mark.parametrize("value,expect", [
    (0, 0), (179.9, 179.9), (180, -180), (-180, -180), (181, -179),
    (359, -1), (360, 0), (361, 1), (-359, 1), (720, 0), (-720, 0),
    (1080 + 45, 45), (-1080 - 45, -45),
])
def test_wrap180_across_every_boundary(value, expect):
    assert wrap180(value) == pytest.approx(expect, abs=1e-6)


# -- the loop converges ------------------------------------------------------

@pytest.mark.parametrize("bias", [0.0, 15.0, -25.0, 30.0, 60.0, -70.0])
def test_a_biased_sim_robot_learns_to_go_where_it_is_told(bias):
    h = shuttle(sim(bias_deg=bias))
    assert abs(travel_error(h)) < 6.0, (
        f"bias {bias}: still {travel_error(h):.0f}deg off, "
        f"offset {h.heading_offset:.0f}")


def test_without_tracking_the_bias_simply_stays():
    """The control: the same robot, the same drive, the loop switched off."""
    h = shuttle(sim(bias_deg=30.0, tracking=False))
    assert abs(travel_error(h)) > 20.0


def test_a_real_robot_learns_it_too_and_in_the_opposite_sense():
    """The two handles disagree on the sense of the offset. Both must converge."""
    class Api:
        def __init__(self):
            self.heading, self.speed = 0.0, 0

        def set_heading(self, h):
            self.heading = float(h) % 360.0

        def set_speed(self, v):
            self.speed = int(v)

        def set_main_led(self, c):
            pass

    class Ball:
        """Goes where it is sent, plus a constant compass error."""

        def __init__(self, api, err):
            self.api, self.err = api, err
            self.p = np.array([200.0, 200.0])

        def read(self):
            return {"cyan": self.p.copy()}

        def advance(self, dt, speed=30.0):
            if self.api.speed:
                r = math.radians((self.api.heading + self.err) % 360.0)
                self.p = self.p + np.array([math.sin(r), math.cos(r)]) * speed * dt

    api = Api()
    ball = Ball(api, 25.0)
    h = SpheroRobot("A", "AAAA", "cyan", "fake", tracker=ball,
                    connector=lambda n, timeout=8: api, autostart=False)
    h._api, h._link_up = api, True
    try:
        t, dt = 0.0, 0.05
        for i in range(int(45 / dt)):
            a = 0.0 if (i // 40) % 2 == 0 else math.pi
            h.set_velocity(np.array([math.sin(a), math.cos(a)]) * 30.0)
            if h._pending:
                api.set_heading(h._pending[0])
                api.set_speed(h._pending[1])
                h._pending = None
            ball.advance(dt)
            h.pos = ball.read()["cyan"]
            h.observe_heading(t, None)
            t += dt

        # tell it to go north, and see whether it does
        h.set_velocity(np.array([0.0, 30.0]))
        api.set_heading(h._pending[0])
        api.set_speed(h._pending[1])
        p0 = ball.p.copy()
        for _ in range(20):
            ball.advance(dt)
        d = ball.p - p0
        went = wrap180(math.degrees(math.atan2(d[0], d[1])))
        assert abs(went) < 6.0, f"told north, went {went:.0f}deg, offset {h.heading_offset:.0f}"
    finally:
        h.close()


# -- the things that must NOT move it ----------------------------------------

def test_a_stationary_robot_is_not_corrupted_by_camera_noise():
    """Below any useful baseline, travel direction is noise, not a heading."""
    h = sim()
    rng = np.random.default_rng(0)
    h.heading_offset = 12.0
    for i in range(600):
        h.pos = np.array([200.0, 200.0]) + rng.normal(0, 0.8, 2)
        h.observe_heading(i * 0.05, None)
    assert h.heading_offset == pytest.approx(12.0, abs=1.0), \
        "a parked robot moved its own calibration"


def test_a_crawling_robot_still_calibrates_eventually():
    """Baseline is a distance, not a number of frames — 15cm/s is still valid."""
    h = shuttle(sim(bias_deg=25.0), seconds=90.0, leg=4.0, speed=15.0)
    assert abs(travel_error(h, speed=15.0)) < 8.0


def test_reversing_does_not_slew_the_estimate_half_a_turn():
    h = shuttle(sim(bias_deg=20.0), seconds=30.0, leg=1.0)   # reverse constantly
    assert abs(travel_error(h)) < 10.0, h.heading_offset
    assert h.estimator.rejected["reversing"] >= 0


def test_a_settled_estimate_rejects_a_wild_sample():
    est = HeadingEstimator(min_baseline=10.0)
    for i in range(30):
        est.update(i * 0.1, i * 2.0, 0.0, 0.0)       # travelling due +x, yaw 0
    settled = est.offset
    for i in range(30, 40):
        est.update(i * 0.1, 60.0, (i - 29) * 4.0, 0.0)   # sudden 90deg turn
    assert abs(wrap180(est.offset - settled)) < 30.0
    assert est.rejected["outlier"] + est.rejected["turning"] > 0


# -- picking a robot up ------------------------------------------------------

def test_it_reconverges_after_the_frame_jumps():
    """A ball picked up and put down rotated. The commonest cause of drift."""
    h = shuttle(sim(bias_deg=20.0))
    assert abs(travel_error(h)) < 6.0
    h.bias = math.radians(-60.0)                    # picked up, put down turned
    shuttle(h, seconds=45.0)
    assert abs(travel_error(h)) < 8.0, h.heading_offset


def test_confidence_falls_when_nothing_is_being_observed():
    h = shuttle(sim(bias_deg=20.0))
    assert h.estimator.confidence > 0.3
    # time passes with the ball parked: no new corrections
    for i in range(400):
        h.observe_heading(h._t + i * 0.1, None)
    assert h.estimator.confidence < 0.2


def test_a_seeded_estimate_is_not_confident_until_it_has_seen_something():
    """A number from last session is a claim, not an observation."""
    est = HeadingEstimator().seed(30.0)
    assert est.offset == pytest.approx(30.0)
    assert est.confidence < 0.3


def test_drift_from_the_calibrated_seed_is_reported():
    h = sim(bias_deg=0.0)
    h.heading_offset = 10.0
    h.bias = math.radians(-50.0)
    shuttle(h, seconds=45.0)
    state = h.heading_state()
    assert state["drifted_deg"] is not None
    assert state["drifted_deg"] > 20.0, state


# -- the boundary ------------------------------------------------------------

def test_both_kinds_report_the_same_heading_surface():
    """Nothing above `fleet/` may need to know which kind it is holding."""
    s = sim()
    keys = set(s.heading_state())
    assert {"offset_deg", "confidence", "tracked", "applied_deg",
            "drifted_deg"} <= keys
    assert isinstance(s.heading_state()["confidence"], float)


def test_tracking_can_be_switched_off_for_measurement():
    """The battery measures the plant; a loop correcting it would hide it."""
    h = shuttle(sim(bias_deg=30.0, tracking=False))
    assert h.heading_offset == 0.0, "tracking was off; nothing should have moved"
    h.heading_tracking = True
    shuttle(h, seconds=40.0)
    assert h.heading_offset != 0.0


def test_setting_the_offset_by_hand_restarts_the_estimate():
    """Everything in the trail was seen under the old offset."""
    h = shuttle(sim(bias_deg=30.0))
    assert h.estimator.samples > 0
    h.heading_offset = 0.0
    assert h.estimator.samples == 0 and h.estimator.offset is None


# -- what the layers above see -----------------------------------------------

def test_the_state_dict_carries_heading_for_both_kinds():
    """The renderer draws an arrow from this and never asks which kind it is."""
    s = sim(bias_deg=15.0)
    d = s.state()
    assert "heading" in d
    assert {"confidence", "applied_deg", "drifted_deg", "tracked"} <= set(d["heading"])


def test_a_disturbed_robot_is_flagged_in_its_state():
    """Past 30 degrees from calibration it was picked up, not drifting."""
    h = sim()
    h.heading_offset = 0.0
    h.bias = math.radians(-55.0)
    shuttle(h, seconds=45.0)
    assert h.state()["heading"]["drifted_deg"] > 30.0


def test_the_ui_draws_a_heading_arrow_without_knowing_the_kind(monkeypatch):
    import app as appmod
    calls = []
    monkeypatch.setattr(appmod.pygame.draw, "line",
                        lambda *a, **k: calls.append(a))
    h = shuttle(sim(bias_deg=20.0))

    class Fake:
        screen = object()
        fs = None

        def to_px(self, p):
            return (100, 100)

    fake = Fake()
    appmod.App.draw_heading(fake, np.array([100.0, 100.0]), h.state(),
                            (255, 255, 255), True)
    assert calls, "a confident heading should draw an arrow"


def test_no_arrow_is_drawn_for_a_heading_nobody_has_observed(monkeypatch):
    import app as appmod
    calls = []
    monkeypatch.setattr(appmod.pygame.draw, "line",
                        lambda *a, **k: calls.append(a))
    h = sim()

    class Fake:
        screen = object()

    appmod.App.draw_heading(Fake(), np.array([100.0, 100.0]), h.state(),
                            (255, 255, 255), True)
    assert not calls, "an unobserved heading must not be drawn as if known"

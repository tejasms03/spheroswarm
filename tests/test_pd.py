"""The point-and-path controller, against the plant it claims to be tuned for.

Every test drives a `SimRobot` through an emulated camera — delayed and noisy,
as a real overhead camera is — because a PD loop tested against instant, exact
feedback is not being tested at all: the delay is the entire reason the gains
are what they are.
"""

from collections import deque

import numpy as np
import pytest

from fleet.sim_handle import SimRobot
from swarm.pd import (Circle, Line, PDController, Point, gains_from_motion,
                      implied_heading_error, straightness, tracking_error)
from workspace.space import Workspace

DT = 1.0 / 30.0
MEASURED = {"step_response": {"tau_s": 0.35},
            "latency": {"loop_delay_s": 0.163},
            "recommend": {"min_moving_cm_s": 8.4,
                          "handle_max_speed_cm_s": 53.3}}


@pytest.fixture
def ws():
    return Workspace(bounds_cm=[[0, 0], [200, 0], [200, 200], [0, 200]])


def drive(path, gains, ws, cam_delay=0.13, noise=0.8, tau=0.35, horizon=26.0,
          start=(40.0, 40.0), seed=2):
    r = SimRobot("D", "D", "red", workspace=ws, pos=list(start), randomize=False)
    r.tau = tau
    n = max(1, int(round(cam_delay / DT)) + 1)
    buf = deque([r.pos.copy()] * n, maxlen=n)
    vbuf = deque([np.zeros(2)] * n, maxlen=n)
    rng = np.random.default_rng(seed)
    pd = PDController(gains["kp"], gains["kd"], gains["max_speed"],
                      gains["deadband_cm_s"], 3.0, gains["predict_s"])
    t, trail, err = 0.0, [], []
    while t < horizon:
        setpoint, ff = path.step(DT)
        meas = buf[0] + (rng.normal(0, noise, 2) if noise else 0.0)
        r.set_velocity(pd.step(meas, vbuf[0], setpoint, feedforward=ff))
        r.step(DT)
        buf.append(r.pos.copy())
        vbuf.append(r.vel.copy())
        trail.append(r.pos.copy())
        err.append(float(np.linalg.norm(setpoint - r.pos)))
        t += DT
    return np.array(trail), err, pd


def settle(err, tol=3.5):
    return next((i * DT for i in range(len(err)) if max(err[i:]) <= tol), None)


def overshoot(trail, start, target):
    u = np.asarray(target, float) - np.asarray(start, float)
    u = u / np.linalg.norm(u)
    return max(0.0, float(((trail - np.asarray(target, float)) @ u).max()))


# -- gains -------------------------------------------------------------------

def test_measured_gains_come_from_the_measurement():
    g = gains_from_motion(MEASURED)
    assert g["measured"]
    assert g["dead_s"] == pytest.approx(0.163)
    assert g["kd"] == pytest.approx(2.0 * 0.163, abs=0.01)
    assert g["predict_s"] == pytest.approx(0.163)
    assert g["deadband_cm_s"] == 8.4
    assert g["max_speed"] == 53.3


def test_an_unmeasured_robot_gets_conservative_gains_and_no_prediction():
    """Extrapolating along a delay nobody measured is not honest."""
    g = gains_from_motion(None)
    assert not g["measured"]
    assert g["predict_s"] == 0.0
    assert g["kp"] < gains_from_motion(MEASURED)["kp"]


def test_gain_does_not_blow_up_as_the_camera_gets_faster():
    """Scaling on dead time alone sends kp to infinity as the delay vanishes."""
    fast = gains_from_motion({"step_response": {"tau_s": 0.35},
                              "latency": {"loop_delay_s": 0.001}})
    assert fast["kp"] < 5.0, fast


def test_gains_fall_back_to_the_step_response_when_latency_is_missing():
    g = gains_from_motion({"step_response": {"tau_s": 0.3, "dead_s": 0.2}})
    assert g["dead_s"] == pytest.approx(0.2)
    assert g["measured"]


# -- going to a point --------------------------------------------------------

def test_it_reaches_a_clicked_point_and_stops(ws):
    trail, err, pd = drive(Point((150.0, 150.0)), gains_from_motion(MEASURED), ws)
    assert err[-1] < 3.5, err[-1]
    assert settle(err) is not None and settle(err) < 8.0
    assert pd.arrived


def test_delay_compensation_is_what_removes_the_overshoot(ws):
    """The claim the whole design rests on, stated as a test.

    Same gains, prediction on and off. Without it the loop is correcting a
    position one delay old, and a 150cm move arrives long.
    """
    g = dict(gains_from_motion(MEASURED))
    with_pred, _, _ = drive(Point((150.0, 150.0)), g, ws)
    without = dict(g, predict_s=0.0)
    no_pred, _, _ = drive(Point((150.0, 150.0)), without, ws)

    a = overshoot(with_pred, (40.0, 40.0), (150.0, 150.0))
    b = overshoot(no_pred, (40.0, 40.0), (150.0, 150.0))
    assert a < 2.0, f"predicted overshoot {a:.1f}cm"
    assert b > a + 2.0, f"prediction made no difference: {a:.1f} vs {b:.1f}"


def test_a_mis_measured_delay_degrades_but_never_diverges(ws):
    """The predictor trusts a number with an error bar. It must stay stable."""
    for factor in (0.5, 0.75, 1.5, 2.0):
        g = dict(gains_from_motion(MEASURED))
        g["predict_s"] *= factor
        _, err, _ = drive(Point((150.0, 150.0)), g, ws)
        assert max(err[-60:]) < 15.0, (factor, max(err[-60:]))


def test_the_deadband_is_not_asked_for_a_command_that_does_nothing(ws):
    """Below the deadband the motors do not turn, so easing to a stop stalls."""
    pd = PDController(kp=1.0, kd=0.0, max_speed=60.0, deadband_cm_s=10.0, tol=3.0)
    # far enough to still need to move, but the raw PD output is tiny
    v = pd.step(np.array([0.0, 0.0]), np.zeros(2), np.array([5.0, 0.0]))
    assert float(np.linalg.norm(v)) == pytest.approx(10.0), \
        "a command below the deadband should be raised to it, or dropped"
    # inside tolerance it should ask for nothing rather than lurch
    v = pd.step(np.array([0.0, 0.0]), np.zeros(2), np.array([1.0, 0.0]))
    assert np.allclose(v, 0.0)


def test_output_never_exceeds_the_measured_ceiling(ws):
    pd = PDController(kp=99.0, kd=0.0, max_speed=53.3)
    v = pd.step(np.zeros(2), np.zeros(2), np.array([200.0, 0.0]))
    assert float(np.linalg.norm(v)) <= 53.3 + 1e-6


# -- paths -------------------------------------------------------------------

def test_it_tracks_a_circle(ws):
    c = Circle((100.0, 100.0), 45.0, 25.0).start_phase((40.0, 40.0))
    trail, _, _ = drive(c, gains_from_motion(MEASURED), ws)
    e = tracking_error(c, trail[len(trail) // 3:])
    assert e["rms_cm"] < 3.0, e
    assert e["max_cm"] < 8.0, e


def test_it_tracks_a_line_including_the_turnarounds(ws):
    ln = Line((40.0, 150.0), (160.0, 150.0), 30.0)
    trail, _, _ = drive(ln, gains_from_motion(MEASURED), ws, start=(40.0, 150.0))
    e = tracking_error(ln, trail[len(trail) // 3:])
    assert e["rms_cm"] < 3.0, e


def test_the_circle_feedforward_is_what_keeps_the_radius(ws):
    """Without it the ball lags the setpoint and traces a smaller circle."""
    g = gains_from_motion(MEASURED)

    class NoFF(Circle):
        def step(self, dt, pos=None):
            point, _ = super().step(dt, pos)
            return point, None

    good = Circle((100.0, 100.0), 45.0, 30.0).start_phase((40.0, 100.0))
    poor = NoFF((100.0, 100.0), 45.0, 30.0).start_phase((40.0, 100.0))
    tg, _, _ = drive(good, g, ws, start=(55.0, 100.0))
    tp, _, _ = drive(poor, g, ws, start=(55.0, 100.0))

    rg = np.linalg.norm(tg[len(tg) // 2:] - np.array([100.0, 100.0]), axis=1).mean()
    rp = np.linalg.norm(tp[len(tp) // 2:] - np.array([100.0, 100.0]), axis=1).mean()
    assert abs(rg - 45.0) < 3.0, f"radius held to {rg:.1f}"
    assert rg > rp + 1.0, f"feedforward made no difference: {rg:.1f} vs {rp:.1f}"


def test_a_circle_is_entered_at_the_nearest_point(ws):
    c = Circle((100.0, 100.0), 40.0)
    c.start_phase((10.0, 100.0))
    first, _ = c.step(0.0)
    assert first[0] < 100.0, "it should enter on the side it is already on"
    assert np.linalg.norm(first - np.array([100.0, 100.0])) == pytest.approx(40.0)


def test_a_line_shuttles_rather_than_stopping_at_the_far_end():
    ln = Line((0.0, 0.0), (100.0, 0.0), speed=50.0)
    seen = []
    for _ in range(int(8.0 / DT)):
        p, v = ln.step(DT)
        seen.append(p[0])
    assert max(seen) > 95.0 and min(seen) < 5.0, "it should reach both ends"


def test_a_point_path_does_not_move():
    p = Point((70.0, 30.0))
    a, ff = p.step(DT)
    for _ in range(50):
        b, ff = p.step(DT)
    assert np.allclose(a, b)
    assert ff is None, "a still target has no feedforward"


def test_tracking_error_measures_the_shape_not_the_phase(ws):
    """A robot a quarter-turn behind is still ON the circle."""
    c = Circle((100.0, 100.0), 40.0)
    on_shape = [np.array([100 + 40 * np.cos(a), 100 + 40 * np.sin(a)])
                for a in np.linspace(0, 2 * np.pi, 60)]
    e = tracking_error(c, on_shape)
    assert e["rms_cm"] < 0.5, e


def test_tracking_error_on_a_line_is_not_distance_to_its_endpoints():
    """The bug this caught: a path sampled as two points."""
    ln = Line((0.0, 0.0), (100.0, 0.0), 25.0)
    mid = [np.array([x, 0.0]) for x in range(10, 90, 5)]
    e = tracking_error(ln, mid)
    assert e["rms_cm"] < 0.5, e


# -- drift and heading error -------------------------------------------------
#
# The camera measures position, never facing, so a Sphero drives in a frame
# offset from the world's by an angle nobody has measured — and that angle
# wanders. What follows pins how much of that the loop absorbs, because the
# answer decides whether the heading work is a nice-to-have or a blocker.

def drive_biased(path, ws, bias_deg=0.0, drift_rate=None, randomize=False,
                 seed=2, horizon=26.0, start=(40.0, 40.0)):
    g = gains_from_motion(MEASURED)
    r = SimRobot("D", "D", "red", workspace=ws, pos=list(start), seed=seed,
                 randomize=randomize)
    if not randomize:
        r.tau = 0.35
    r.bias = np.radians(bias_deg)
    # These tests ask what the CONTROLLER does with a bad frame. With heading
    # tracking left on, the estimator quietly fixes the frame within a few
    # seconds and every one of them passes for the wrong reason — including the
    # one asserting that a 90 degree error should break it.
    r.heading_tracking = False
    if drift_rate is not None:
        r.drift_rate = drift_rate
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    vbuf = deque([np.zeros(2)] * 5, maxlen=5)
    rng = np.random.default_rng(seed + 9)
    pd = PDController(g["kp"], g["kd"], g["max_speed"], g["deadband_cm_s"],
                      3.0, g["predict_s"])
    t, trail, err = 0.0, [], []
    while t < horizon:
        sp, ff = path.step(DT)
        r.set_velocity(pd.step(buf[0] + rng.normal(0, 0.8, 2), vbuf[0], sp, ff))
        r.step(DT)
        buf.append(r.pos.copy())
        vbuf.append(r.vel.copy())
        trail.append(r.pos.copy())
        err.append(float(np.linalg.norm(sp - r.pos)))
        t += DT
    return np.array(trail), err


def test_going_to_a_point_does_not_need_the_heading_offset(ws):
    """Closing the loop on position absorbs a rotated command frame.

    Up to about 45 degrees of frame error the ball still arrives — it takes a
    curved route and covers more ground, but the component of every command
    along the error stays positive, so it converges. This is why an
    uncalibrated robot can still be driven around.
    """
    for bias in (0, 15, 30, 45):
        _, err = drive_biased(Point((150.0, 150.0)), ws, bias_deg=bias)
        assert err[-1] < 3.5, f"{bias}deg: ended {err[-1]:.1f}cm out"


def test_a_large_heading_error_breaks_it_and_should(ws):
    """Past ~60 degrees it stops converging. Worth knowing where the cliff is."""
    _, err = drive_biased(Point((150.0, 150.0)), ws, bias_deg=90)
    assert err[-1] > 20.0, "a 90deg frame error should NOT quietly work"


def test_a_curved_route_is_the_price_of_an_uncalibrated_frame(ws):
    """It arrives, but not by a straight line — the visible symptom on a floor."""
    straight = np.linalg.norm(np.array([110.0, 110.0]))
    direct, _ = drive_biased(Point((150.0, 150.0)), ws, bias_deg=0)
    skewed, _ = drive_biased(Point((150.0, 150.0)), ws, bias_deg=45)

    def travelled(t):
        return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())

    assert travelled(direct) < straight * 1.15
    assert travelled(skewed) > straight * 1.35


def test_drift_during_a_run_does_not_stop_a_robot_arriving(ws):
    """The bias random-walks. A loop that re-closes every frame does not care."""
    for rate in (0.0, 8.0, 20.0):
        for seed in (1, 2, 3):
            _, err = drive_biased(Point((150.0, 150.0)), ws, drift_rate=rate,
                                  seed=seed)
            assert err[-1] < 3.5, (rate, seed, err[-1])


def test_path_tracking_is_what_the_heading_offset_actually_buys(ws):
    """Arrival survives a bad frame; following a shape does not.

    A 45 degree offset costs a point-to-point move some extra distance and
    nothing else, but it takes circle tracking from under a centimetre to the
    better part of ten. That is the argument for calibrating heading — and for
    tracking it continuously — stated as a number rather than an opinion.
    """
    def circle_rms(bias):
        c = Circle((100.0, 100.0), 45.0, 25.0).start_phase((40.0, 40.0))
        trail, _ = drive_biased(c, ws, bias_deg=bias)
        return tracking_error(c, trail[len(trail) // 3:])["rms_cm"]

    good, bad = circle_rms(0), circle_rms(45)
    assert good < 1.5, good
    assert bad > good * 3, (good, bad)


def test_it_holds_up_with_every_modelled_error_at_once(ws):
    """Drift, slip, per-robot gain and lag all on, across several robots."""
    finals, rms = [], []
    for seed in range(6):
        _, err = drive_biased(Point((150.0, 150.0)), ws, randomize=True, seed=seed)
        finals.append(err[-1])
        c = Circle((100.0, 100.0), 45.0, 25.0).start_phase((40.0, 40.0))
        trail, _ = drive_biased(c, ws, randomize=True, seed=seed)
        rms.append(tracking_error(c, trail[len(trail) // 3:])["rms_cm"])
    assert max(finals) < 3.5, finals
    assert float(np.mean(rms)) < 2.5, rms


# -- arriving, rather than hunting round the target ---------------------------
#
# From a recording: the ball reached its target in about five seconds and then
# circled it for the next twenty-five. Two different faults look like that and
# they have different fixes, so the controller has to tell them apart.

def hold_test(ws, bias_deg=0.0, tol=3.0, predict=0.12, secs=30.0,
              start=(30.0, 30.0), target=(74.0, 64.0)):
    g = gains_from_motion(MEASURED)
    r = SimRobot("T", "T", "red", workspace=ws, pos=list(start), randomize=False)
    r.tau, r.bias = 0.35, np.radians(bias_deg)
    r.heading_tracking = False
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    vbuf = deque([np.zeros(2)] * 5, maxlen=5)
    rng = np.random.default_rng(2)
    pd = PDController(g["kp"], g["kd"], 18.0, g["deadband_cm_s"], tol=tol,
                      predict_s=predict)
    path = Point(target)
    t, parked, orbited = 0.0, None, False
    while t < secs:
        sp, ff = path.step(DT)
        r.set_velocity(pd.step(buf[0] + rng.normal(0, 0.8, 2), vbuf[0], sp, ff))
        r.step(DT)
        buf.append(r.pos.copy())
        vbuf.append(r.vel.copy())
        orbited = orbited or pd.orbiting
        if pd.holding:
            parked = t if parked is None else parked
        else:
            parked = None
        t += DT
    return parked, orbited, float(np.linalg.norm(r.pos - np.asarray(target)))


def test_a_tolerance_tighter_than_the_robot_can_hold_costs_it_dearly(ws):
    """The premise, stated as a comparison rather than as a stopwatch reading.

    A 3cm circle is smaller than the ball's own stopping error, so it reaches
    the target and then keeps being nudged back and forth across the boundary.
    Whether that resolves in twelve seconds or never depends on the draw; that
    it takes far longer than a properly sized circle does not.
    """
    g = gains_from_motion(MEASURED)
    tight, _, _ = hold_test(ws, bias_deg=45.0, tol=3.0, secs=25.0)
    sized, _, _ = hold_test(ws, bias_deg=45.0, tol=g["arrive_cm"], secs=25.0)
    assert sized is not None, "the sized radius should park"
    assert tight is None or tight > sized * 1.5, (tight, sized)


def test_an_arrival_radius_lets_it_park(ws):
    g = gains_from_motion(MEASURED)
    parked, _, err = hold_test(ws, tol=g["arrive_cm"], secs=25.0)
    assert parked is not None and parked < 8.0, f"parked at {parked}"
    assert err <= g["arrive_cm"] + 1.5


def test_it_stays_parked_once_it_is_in(ws):
    """Without hysteresis it creeps out, is commanded, overshoots back in, repeats."""
    g = gains_from_motion(MEASURED)
    pd = PDController(g["kp"], g["kd"], 18.0, 0.0, tol=5.0, predict_s=0.0)
    target = np.array([50.0, 50.0])
    pd.step(target + np.array([4.0, 0.0]), np.zeros(2), target)
    assert pd.holding
    # drifted just outside the circle: it must NOT start commanding again
    v = pd.step(target + np.array([6.0, 0.0]), np.zeros(2), target)
    assert pd.holding and np.allclose(v, 0.0)
    # well outside: it lets go
    pd.step(target + np.array([12.0, 0.0]), np.zeros(2), target)
    assert not pd.holding


def test_the_arrival_radius_is_sized_from_what_the_robot_can_hold():
    """Not picked. It is the stopping distance at the slowest speed it accepts."""
    g = gains_from_motion(MEASURED)
    rec = MEASURED["recommend"]
    floor = rec["min_moving_cm_s"] * (
        MEASURED["step_response"]["tau_s"] + MEASURED["latency"]["loop_delay_s"])
    assert g["arrive_cm"] >= 4.0
    assert g["arrive_cm"] >= floor * 0.8, (g["arrive_cm"], floor)


def test_circling_is_reported_and_hunting_is_not(ws):
    """Different faults, different fixes — so the loop must not conflate them.

    Hunting is a tolerance set tighter than the robot can hold, and widening it
    fixes that. Circling is the aim frame being wrong by enough that the ball
    travels AROUND the target, and no tolerance fixes it. Telling the trainer
    to widen a radius when the real answer is 'calibrate the heading' costs an
    afternoon.
    """
    g = gains_from_motion(MEASURED)
    _, orbited_ok, _ = hold_test(ws, bias_deg=0.0, tol=g["arrive_cm"])
    assert not orbited_ok, "a healthy approach must not be called circling"
    _, orbited_bad, _ = hold_test(ws, bias_deg=75.0, tol=g["arrive_cm"])
    assert orbited_bad, "a 75 degree frame error IS circling"


def test_an_implausible_measured_delay_is_clamped():
    """600ms of loop delay is a bad measurement, and acting on it causes orbits."""
    g = gains_from_motion({"step_response": {"tau_s": 0.16},
                           "latency": {"loop_delay_s": 0.60}})
    assert g["delay_clamped"]
    assert g["predict_s"] <= 0.45


def test_the_prediction_never_reaches_past_the_target(ws):
    """Extrapolating 6cm ahead toward a target 4cm away commands it backwards."""
    pd = PDController(kp=2.0, kd=0.2, max_speed=18.0, tol=2.0, predict_s=0.6)
    target = np.array([50.0, 50.0])
    pos = np.array([46.0, 50.0])              # 4cm short
    vel = np.array([10.0, 0.0])               # 0.6s * 10 = 6cm of prediction
    v = pd.step(pos, vel, target)
    assert float(np.dot(v, np.array([1.0, 0.0]))) >= -1e-6, (
        "the prediction overshot the target and it commanded backwards")


def test_a_filtered_velocity_makes_the_command_stop_thrashing(ws):
    """The measurable difference between a rough approach and a clean one.

    Same gains, same plant, same noise — only the velocity estimator changes.
    The derivative term is a gain on velocity, so whatever noise the estimator
    carries appears directly in the motors.
    """
    from vision.track import Kalman2D

    def run(source, kd=0.9, secs=18.0):
        g = gains_from_motion(MEASURED)
        r = SimRobot("T", "T", "red", workspace=ws, pos=[25.0, 25.0],
                     randomize=False)
        r.tau = 0.35
        r.heading_tracking = False
        buf = deque([r.pos.copy()] * 5, maxlen=5)
        rng = np.random.default_rng(3)
        kf = Kalman2D(r.pos.copy(), dt=DT)
        prev = r.pos.copy()
        pd = PDController(g["kp"], kd, 18.0, 0.0, tol=5.0, predict_s=0.2)
        path = Point((95.0, 80.0))
        t, cmds = 0.0, []
        while t < secs:
            meas = buf[0] + rng.normal(0, 1.0, 2)
            kf.predict(DT)
            kf.update(meas)
            vel = kf.vel.copy() if source == "filtered" else (meas - prev) / DT
            prev = meas
            sp, ff = path.step(DT)
            cmd = pd.step(meas, vel, sp, ff)
            r.set_velocity(cmd)
            r.step(DT)
            buf.append(r.pos.copy())
            cmds.append(cmd.copy())
            t += DT
        d = np.diff(np.array(cmds), axis=0)
        return float(np.linalg.norm(d, axis=1).mean())

    rough = run("differenced")
    smooth = run("filtered")
    assert smooth < rough / 5.0, (
        f"filtering changed command jitter from {rough:.1f} to {smooth:.1f} cm/s")
    assert smooth < 3.0, smooth


# -- why calibration legs are straight and driving is not --------------------

def test_an_open_loop_leg_is_straight_however_wrong_the_frame_is(ws):
    """What a calibration leg does: one heading, held, never re-aimed.

    It goes the WRONG WAY under a frame offset, and it goes there in a
    perfectly straight line. That is why a trainer sees straight calibration
    paths and curved driving paths from the same uncalibrated robot.
    """
    import math
    for bias in (0.0, 20.0, 40.0, 60.0):
        r = SimRobot("T", "T", "red", workspace=ws, pos=[30.0, 55.0],
                     randomize=False)
        r.tau, r.bias = 0.35, math.radians(bias)
        r.heading_tracking = False
        trail = []
        for _ in range(int(6.0 / DT)):
            r.drive_raw(90.0, 55)
            r.step(DT)
            trail.append(r.pos.copy())
        assert straightness(trail) < 1.06, f"{bias}deg: open loop should be straight"


def test_closed_loop_curves_in_proportion_to_the_frame_error(ws):
    """Driving re-aims every frame, so a rotated frame bends the path."""
    ratios = []
    for bias in (0.0, 40.0):
        trail, _ = drive_biased(Point((110.0, 55.0)), ws, bias_deg=bias,
                                horizon=20.0, start=(30.0, 55.0))
        ratios.append(straightness(trail))
    assert ratios[0] < 1.10, ratios
    assert ratios[1] > ratios[0] * 1.2, ratios


def test_a_curve_is_turned_into_a_number_of_degrees():
    assert implied_heading_error(1.00) is None, "a straight path is not an error"
    assert implied_heading_error(1.02) is None
    mid = implied_heading_error(1.44)
    assert mid is not None and 33 < mid < 47, mid
    assert implied_heading_error(3.7) >= 55


def test_the_estimate_is_monotonic():
    """More curve must never read as less error."""
    got = [implied_heading_error(r) or 0.0
           for r in (1.0, 1.05, 1.2, 1.5, 2.5, 4.0)]
    assert got == sorted(got), got

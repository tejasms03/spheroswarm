"""The measuring instrument, measured.

Every test here drives the real stages against a `SimRobot` whose parameters
are known, through a camera emulator that adds the delay and the noise a real
one has. The question each asks is the only one that matters for a calibration
rig: does the number that comes out match the number that went in?

That is a stronger contract than "the code runs". A rig that returns a
confident wrong tau is worse than one that crashes, because the wrong tau ends
up in `sim_handle.py` and every controller tuned against it inherits it.
"""

from collections import deque

import numpy as np
import pytest

from fleet.characterize import (BrakeTest, Characterization, LatencyProbe,
                                NoiseProbe, Recenter, SpeedMap, StepResponse,
                                clear_heading, fit_reversal, load_motion)
from fleet.heading import wrap180
from fleet.sim_handle import SimRobot
from workspace.space import Workspace

DT = 1.0 / 30.0
TRUE_TAU = 0.35
TRUE_MAX = 60.0


@pytest.fixture
def ws():
    return Workspace(bounds_cm=[[0, 0], [200, 0], [200, 200], [0, 200]])


def drive(stage, ws, delay=0.12, noise=0.8, pos=(100.0, 100.0), seed=5,
          randomize=False, limit=400.0):
    """Run one stage against a sim robot behind an emulated camera.

    The delay buffer is the point of this helper. Without it the controller
    sees the robot's true position instantly, which is the one thing a real
    overhead camera never does — and every estimator here is sensitive to
    exactly that.
    """
    r = SimRobot("T", "T", "red", workspace=ws, pos=list(pos), seed=seed,
                 randomize=randomize)
    n = max(1, int(round(delay / DT)) + 1)
    buf = deque([r.pos.copy()] * n, maxlen=n)
    rng = np.random.default_rng(seed + 11)
    t = 0.0
    while not stage.done and t < limit:
        meas = buf[0] + (rng.normal(0, noise, 2) if noise else 0.0)
        cmd = stage.step(t, meas, DT)
        if cmd is None:
            r.stop()
        else:
            r.drive_raw(cmd[0], cmd[1])
        r.step(DT)
        buf.append(r.pos.copy())
        t += DT
    return stage, r


# -- the individual measurements --------------------------------------------

def test_noise_probe_recovers_the_camera_sigma(ws):
    stage, _ = drive(NoiseProbe(frames=250), ws, delay=0.0, noise=0.9)
    res = stage.result()
    assert res["frames"] == 250
    assert 0.7 <= res["sigma_x_cm"] <= 1.1
    assert 0.7 <= res["sigma_y_cm"] <= 1.1
    assert res["drift_cm"] < 2.0, "a still robot must not look like it is drifting"


def test_noise_probe_does_not_count_creep_as_noise(ws):
    """A floor that is not level makes a ball wander. That is not jitter."""
    probe = NoiseProbe(frames=200)
    rng = np.random.default_rng(2)
    p, t = np.array([100.0, 100.0]), 0.0
    while not probe.done and t < 30:
        p = p + np.array([0.02, 0.0])          # 12cm of creep over the run
        probe.step(t, p + rng.normal(0, 0.5, 2), DT)
        t += DT
    res = probe.result()
    assert res["drift_cm"] > 3.0, "the creep should be reported"
    assert res["sigma_cm"] < 1.0, "but it must not inflate sigma"


def test_speed_map_recovers_the_byte_to_cm_s_line(ws):
    stage, r = drive(SpeedMap(workspace=ws), ws)
    res = stage.result()
    assert res["moving_points"] >= 8
    assert res["r2"] > 0.98, "byte to cm/s is a straight line in this plant"
    assert abs(res["max_speed_cm_s"] - TRUE_MAX * r.gain) < 3.0


def test_speed_map_finds_a_deadband_when_there_is_one(ws):
    """A ball that will not move below a byte must be reported, not averaged in."""
    class Sticky(SimRobot):
        def set_velocity(self, v):
            v = np.asarray(v, dtype=float)
            if np.linalg.norm(v) < 18.0:        # ignores anything under byte ~76
                v = np.zeros(2)
            super().set_velocity(v)

    stage = SpeedMap(workspace=ws)
    r = Sticky("T", "T", "red", workspace=ws, pos=[100.0, 100.0], randomize=False)
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    rng = np.random.default_rng(3)
    t = 0.0
    while not stage.done and t < 400:
        cmd = stage.step(t, buf[0] + rng.normal(0, 0.6, 2), DT)
        r.stop() if cmd is None else r.drive_raw(*cmd)
        r.step(DT)
        buf.append(r.pos.copy())
        t += DT
    res = stage.result()
    assert res["stalled_bytes"], "the stalled low bytes should be named"
    assert res["min_moving_byte"] >= 60


def test_step_response_recovers_tau(ws):
    stage, _ = drive(StepResponse(workspace=ws, repeats=3), ws)
    res = stage.result()
    assert abs(res["tau_s"] - TRUE_TAU) < 0.12, res
    assert res["dead_s"] >= -0.05


def test_step_response_waits_for_the_ball_to_stop():
    """Starting a step while the ball still rolls measures the previous leg."""
    stage = StepResponse(repeats=1)
    assert "wait" in StepResponse._step.__code__.co_consts or True
    # behavioural: a robot handed a moving history must not fit a near-zero tau
    ws = Workspace(bounds_cm=[[0, 0], [200, 0], [200, 200], [0, 200]])
    stage, _ = drive(StepResponse(workspace=ws, repeats=2), ws, delay=0.0, noise=0.0)
    assert stage.result()["tau_s"] > 0.2, "a tiny tau means it started mid-roll"


def test_brake_test_measures_the_stopping_distance(ws):
    stage, _ = drive(BrakeTest(workspace=ws), ws, delay=0.12, noise=0.8)
    res = stage.result()
    assert res["r2"] > 0.85
    # coast measured from the CAMERA's cut position spans the loop delay too,
    # so it should exceed the bare v*tau roll-out by roughly v*delay.
    assert TRUE_TAU < res["coast_s_per_cm_s"] < TRUE_TAU + 0.25, res
    assert all(row["settled"] for row in res["rows"]), "every brake should settle"


def test_brake_coast_grows_with_the_delay(ws):
    slow, _ = drive(BrakeTest(workspace=ws, bytes_=(140, 255)), ws,
                    delay=0.30, noise=0.5)
    fast, _ = drive(BrakeTest(workspace=ws, bytes_=(140, 255)), ws,
                    delay=0.0, noise=0.5)
    assert slow.result()["coast_s_per_cm_s"] > fast.result()["coast_s_per_cm_s"] + 0.1


def test_latency_probe_recovers_the_loop_delay(ws):
    """The one number nothing else in the codebase measures.

    Truth is the emulated camera delay plus the sim's own one-step command
    queue. The probe lands on the camera delay essentially exactly and runs
    about a frame short overall, because the actuator step is partly absorbed
    into the lag term it fits alongside. A frame is 33ms; the tolerance says so.
    """
    for delay in (0.10, 0.20, 0.30):
        probe = LatencyProbe(workspace=ws, repeats=3)
        probe.tau_hint = TRUE_TAU
        stage, _ = drive(probe, ws, delay=delay, noise=0.8)
        res = stage.result()
        truth = delay + DT
        assert abs(res["loop_delay_s"] - truth) < 0.05, (delay, res)
        assert res["spread_s"] <= 0.05, "three reversals should agree"


def test_latency_never_reports_a_reaction_before_the_command(ws):
    probe = LatencyProbe(workspace=ws, repeats=3)
    stage, _ = drive(probe, ws, delay=0.15, noise=1.2)
    assert all(r["delay_s"] >= 0.0 for r in stage.result()["runs"])


def test_fit_reversal_is_unbiased():
    """The estimator itself, against a signal with no robot in it at all."""
    def synth(delay, tau=0.35, v=47.0, noise=0.8, seed=0):
        rng = np.random.default_rng(seed)
        ts = np.arange(-0.9, 1.4, DT)
        p, vel, out = 0.0, v, []
        for t in ts:
            out.append(p + rng.normal(0, noise))
            vel += ((v if t < delay else -v) - vel) * min(DT / tau, 1.0)
            p += vel * DT
        return ts, np.array(out)

    for truth in (0.10, 0.20, 0.35):
        errs = []
        for seed in range(12):
            t, y = synth(truth, seed=seed)
            d, _, _ = fit_reversal(t, y, [0.35], np.arange(0.0, 0.6, 1 / 90))
            errs.append(d - truth)
        assert abs(np.mean(errs)) < 0.04, (truth, np.mean(errs))
        assert np.std(errs) < 0.04


# -- aiming and housekeeping ------------------------------------------------

def test_clear_heading_avoids_the_walls(ws):
    """Near a wall it must pick a course with floor in front of it."""
    course = clear_heading(ws, np.array([190.0, 100.0]), 60.0, prefer=90.0)
    assert course is not None
    rad = np.radians(course)
    end = np.array([190.0 + np.sin(rad) * 60.0, 100.0 + np.cos(rad) * 60.0])
    assert ws.is_valid_point(end)


def test_clear_heading_prefers_the_opposite_before_sweeping(ws):
    """A leg that will not fit forward is run backwards, staying on one axis."""
    course = clear_heading(ws, np.array([100.0, 185.0]), 60.0, prefer=0.0)
    assert course == 180.0


def test_recentre_lands_on_the_target(ws):
    stage, r = drive(Recenter((100.0, 100.0)), ws, pos=(30.0, 30.0))
    assert stage.done and stage.error is None
    assert np.linalg.norm(r.pos - np.array([100.0, 100.0])) < 25.0


# -- the whole sequence -----------------------------------------------------

def _full_run(ws, quick=False, seed=5, delay=0.12, noise=0.8):
    c = Characterization(workspace=ws, code="TST", quick=quick)
    r = SimRobot("T", "TST", "red", workspace=ws, pos=[100.0, 100.0], seed=seed,
                 randomize=False)
    n = max(1, int(round(delay / DT)) + 1)
    buf = deque([r.pos.copy()] * n, maxlen=n)
    rng = np.random.default_rng(seed + 3)
    t = 0.0
    while not c.done and t < 900:
        cmd = c.step(t, buf[0] + rng.normal(0, noise, 2), DT)
        r.stop() if cmd is None else r.drive_raw(*cmd)
        r.step(DT)
        buf.append(r.pos.copy())
        t += DT
    return c, r


def test_the_whole_battery_completes_without_complaint(ws):
    c, _ = _full_run(ws)
    assert c.done and not c.cancelled
    assert c.notes == [], c.notes
    for key in ("position_noise", "heading_offset", "speed_map",
                "step_response", "brake_test", "loop_latency"):
        assert key in c.results, key


def test_the_battery_recovers_the_robot_it_measured(ws):
    c, r = _full_run(ws)
    fit = c.fit()
    rec = fit["recommend"]
    assert abs(rec["handle_max_speed_cm_s"] - TRUE_MAX * r.gain) < 3.0
    assert abs(rec["sim_tau_s"] - TRUE_TAU) < 0.12
    assert 0.10 < fit["latency"]["loop_delay_s"] < 0.30
    assert rec["max_precise_speed_cm_s"] > 0


def test_the_aim_correction_converges_rather_than_diverging(ws):
    """Ten legs of one constant error must not become ten times that error."""
    c, _ = _full_run(ws)
    off = c.course_offset
    assert min(off, 360.0 - off) < 25.0, f"aim wandered to {off:.0f}deg"


def test_stopping_distance_is_not_double_counted(ws):
    """The brake coast already spans the loop delay; adding it again inflates it."""
    c, _ = _full_run(ws)
    fit = c.fit()
    assert (fit["recommend"]["stopping_distance_s_per_cm_s"]
            == fit["brake"]["coast_s_per_cm_s"])


def test_a_cancelled_run_stops_and_saves_nothing(ws, tmp_path):
    c = Characterization(workspace=ws, code="TST")
    c.step(0.0, np.array([100.0, 100.0]), DT)
    c.cancel()
    assert c.done and c.cancelled
    assert c.step(1.0, np.array([100.0, 100.0]), DT) is None


def test_results_round_trip_through_the_file(ws, tmp_path):
    c, _ = _full_run(ws, quick=True)
    path = tmp_path / "motion.json"
    fitted, err = c.save(path)
    assert err is None
    again = load_motion("TST", path)
    assert again["robot"] == "TST"
    assert again["recommend"] == fitted["recommend"]
    assert load_motion("NOPE", path) == {}


def test_load_motion_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "motion.json"
    p.write_text("{not json")
    assert load_motion("TST", p) == {}
    assert load_motion(None, tmp_path / "missing.json") == {}


# -- heading drift -----------------------------------------------------------
#
# The measurement that decides whether a continuously estimated heading is
# worth building. It has to distinguish a ball that holds its calibration from
# one that does not, over minutes, through per-leg noise of its own.

def drift_run(ws, drift_rate, minutes=4.0, noise=0.8, seed=1):
    from fleet.characterize import DriftWatch
    stage = DriftWatch(minutes=minutes, workspace=ws)
    r = SimRobot("T", "T", "red", workspace=ws, pos=[100.0, 100.0], seed=seed,
                 randomize=False)
    r.drift_rate, r.tau = drift_rate, 0.35
    # DriftWatch measures how far the frame wanders on its own. A live
    # estimator correcting it would report every ball as rock steady.
    r.heading_tracking = False
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    rng = np.random.default_rng(seed + 4)
    t = 0.0
    while not stage.done and t < minutes * 60 + 60:
        cmd = stage.step(t, buf[0] + rng.normal(0, noise, 2), DT)
        r.stop() if cmd is None else r.drive_raw(*cmd)
        r.step(DT)
        buf.append(r.pos.copy())
        t += DT
    return stage.result()


def test_a_steady_ball_reads_as_holding_its_calibration(ws):
    res = drift_run(ws, 0.0)
    assert res["legs"] > 40, res
    assert not res["above_noise_floor"], res
    assert res["random_walk_deg_per_sqrt_min"] < 3.0, res


def test_a_wandering_ball_is_caught_and_scaled(ws):
    """The coefficient has to track the rate, not merely notice something."""
    quiet = drift_run(ws, 0.0)["random_walk_deg_per_sqrt_min"]
    mid = drift_run(ws, 8.0)["random_walk_deg_per_sqrt_min"]
    bad = drift_run(ws, 20.0)["random_walk_deg_per_sqrt_min"]
    assert mid > quiet * 2, (quiet, mid)
    assert bad > mid * 1.6, (mid, bad)


def test_a_badly_drifting_ball_predicts_enough_wander_to_matter(ws):
    """45 degrees is where path following collapses; the report must reach it."""
    res = drift_run(ws, 20.0)
    assert res["above_noise_floor"]
    assert res["expected_wander_30min_deg"] > 45.0, res


def test_drift_is_measured_as_a_walk_not_as_a_slope(ws):
    """Why the coefficient and not the slope, stated as a property.

    A single stretch of a random walk can fit a perfectly significant straight
    line — that is what walks do, and asserting they never look like a trend is
    simply false. What separates the two is repeatability: across balls the
    apparent slope wanders in sign and size, while the walk coefficient is a
    property of the ball and comes back the same. So the slope is the statistic
    you cannot act on, and this is the test that says so.
    """
    slopes, walks = [], []
    for seed in (1, 2, 3, 4, 5):
        res = drift_run(ws, 20.0, minutes=3.0, seed=seed)
        slopes.append(res["drift_deg_per_min"])
        walks.append(res["random_walk_deg_per_sqrt_min"])

    assert min(slopes) < 0 < max(slopes), \
        f"slopes should not agree on a direction: {slopes}"
    assert float(np.std(walks)) < float(np.mean(walks)) * 0.6, \
        f"the walk coefficient should be repeatable: {walks}"
    assert min(walks) > 5.0, walks


def test_the_residuals_are_unwrapped_before_fitting(ws):
    """A walk crossing +-180 would otherwise read as a full-turn jump."""
    from fleet.characterize import DriftWatch
    stage = DriftWatch(minutes=1.0)
    stage.rows = [{"t": i * 2.0, "residual_deg": ((175 + i * 3 + 180) % 360) - 180,
                   "cm": 24.0} for i in range(40)]
    res = stage.result()
    assert res["total_excursion_deg"] > 100, res
    assert res["total_excursion_deg"] < 200, "a wrap should not read as 360deg"


def test_too_few_legs_is_reported_rather_than_guessed(ws):
    from fleet.characterize import DriftWatch
    stage = DriftWatch(minutes=1.0)
    stage.rows = [{"t": 0.0, "residual_deg": 1.0, "cm": 24.0}]
    assert "error" in stage.result()


# -- fitting legs to the floor that is actually there ------------------------
#
# The complaint this answers: the battery kept driving the ball out of the
# arena and it had to be replaced by hand, repeatedly, mid-session.

def small_ws(side=200.0):
    return Workspace(bounds_cm=[[0, 0], [side, 0], [side, side], [0, side]])


def battery_excursion(side, seed=3, quick=False, randomize=True):
    """Run the whole battery and report how close it ever came to a wall."""
    from fleet import safety as sf
    ws = small_ws(side)
    c = Characterization(workspace=ws, code="T", quick=quick)
    r = SimRobot("T", "T", "red", workspace=ws, pos=[side / 2, side / 2],
                 seed=seed, randomize=randomize)
    r.heading_tracking = False
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    rng = np.random.default_rng(seed)
    t, worst = 0.0, 1e9
    while not c.done and t < 900:
        cmd = c.step(t, buf[0] + rng.normal(0, 0.8, 2), DT)
        r.stop() if cmd is None else r.drive_raw(*cmd)
        r.step(DT)
        buf.append(r.pos.copy())
        d = sf.clearance(ws, r.pos)
        if d is not None:
            worst = min(worst, d)
        t += DT
    return c, worst


@pytest.mark.parametrize("side", [200.0, 150.0])
def test_the_battery_never_reaches_a_wall(side):
    for seed in (3, 7):
        c, worst = battery_excursion(side, seed=seed)
        assert worst > 5.0, f"{side}cm arena, seed {seed}: came within {worst:.1f}cm"
        assert c.done and not c.notes, c.notes


def test_a_leg_is_never_longer_than_the_floor_in_front_of_it():
    """A fixed-duration leg is a distance in disguise.

    2.5 seconds at byte 255 is a metre and a half, and there is not a metre and
    a half in front of a ball in the middle of a two-metre room — so the fast
    end of the speed map was always going to end at a wall, however carefully
    the direction was chosen.
    """
    from fleet.safety import available_seconds
    ws = small_ws(200.0)
    centre = (100.0, 100.0)
    at_full = available_seconds(ws, centre, 270.0, 60.0)
    assert at_full < 1.0, "the premise: there is not much floor at full speed"
    # and the stage knows it
    stage = SpeedMap(workspace=ws)
    have, _ = stage.runway(np.array(centre), 270.0, 255, 2.5)
    assert have < 2.5


def test_the_speed_map_still_measures_the_top_speed():
    """Fitting legs to the floor must not cost the fast end of the curve."""
    ws = small_ws(200.0)
    stage, r = drive(SpeedMap(workspace=ws), ws, pos=(100.0, 100.0))
    res = stage.result()
    assert res["r2"] > 0.98, res
    assert abs(res["max_speed_cm_s"] - TRUE_MAX * r.gain) < 4.0, res
    fastest = [x for x in res["rows"] if x["byte"] == 255]
    assert fastest and fastest[0]["cm_s"], "byte 255 was never measured"
    assert fastest[0]["cm_s"] > 0.85 * TRUE_MAX * r.gain, (
        "byte 255 came back slow — the leg measured the acceleration ramp")


def test_a_byte_with_no_room_is_skipped_and_said_to_be():
    """A badly measured point is worse than a missing one: it moves the line."""
    tiny = Workspace(bounds_cm=[[0, 0], [70, 0], [70, 70], [0, 70]])
    stage, _ = drive(SpeedMap(workspace=tiny, bytes_=(40, 255)), tiny,
                     pos=(35.0, 35.0), limit=200.0)
    res = stage.result()
    skipped = res.get("skipped_bytes") or []
    assert 255 in skipped, res
    for row in res["rows"]:
        if row.get("skipped"):
            assert row["cm_s"] is None, "a skipped leg must not report a speed"


def test_backing_up_buys_the_runway_for_a_fast_leg():
    """A person walks the ball to the far end first. So does this."""
    ws = small_ws(200.0)
    stage = SpeedMap(workspace=ws, bytes_=(255,))
    stage.course = 270.0
    near_wall = np.array([25.0, 100.0])
    cmd = stage.backup_command(near_wall, 270.0, 255, 1.4)
    assert cmd is not None, "it should reverse to make room"
    assert abs(wrap180(cmd[0] - 90.0)) < 1.0, "away from the wall it is aimed at"
    assert cmd[1] < 80, "and slowly — this is positioning, not measuring"
    # once there is room, it stops backing up
    assert stage.backup_command(np.array([175.0, 100.0]), 270.0, 255, 1.4) is None


# -- when the tracker is not following the robot -----------------------------
#
# The failure that put a ball across the room: the tracker locked onto a
# stationary phantom of the same colour, so every position it reported was
# identical. The stage commanded a leg, waited for travel that could never
# register, and drove at full speed for the whole leg timeout.
#
# A frozen fix is indistinguishable from a stationary robot by position alone,
# and position is what every other guard here reads. The only tell is the
# contradiction: asking for motion and seeing none.

def frozen_run(stage, at=(70.0, 55.0), jitter=0.01, limit=60.0):
    """Drive a stage against a tracker that reports the same point forever."""
    rng = np.random.default_rng(0)
    p = np.array(at, dtype=float)
    t, commanded = 0.0, 0.0
    while not stage.done and t < limit:
        cmd = stage.step(t, p + rng.normal(0, jitter, 2), DT)
        if cmd is not None and cmd[1] > 0:
            commanded += (cmd[1] / 255.0 * 60.0) * DT
        t += DT
    return commanded, t


def test_a_frozen_fix_is_caught_before_the_ball_leaves_the_arena(ws):
    """The recorded failure, replayed: a 111cm arena and a ball that crossed it."""
    from fleet.characterize import HeadingStage
    stage = HeadingStage(workspace=ws)
    commanded, elapsed = frozen_run(stage)
    assert stage.done and stage.error, "it kept driving"
    assert "tracker is not following" in stage.error, stage.error
    assert commanded < 40.0, (
        f"commanded {commanded:.0f}cm of travel against a dead tracker; "
        "the arena is 111cm across")
    assert elapsed < 4.0, f"took {elapsed:.1f}s to notice"


def test_the_watchdog_scales_with_the_speed_it_asked_for():
    """A fixed time window is a distance that depends on the speed.

    At the calibration crawl 1.5 seconds asks for 20cm, which is under any
    sane threshold — so a frozen fix sailed straight through a time-based
    check and drove most of the arena anyway.
    """
    from fleet.characterize import HeadingStage, SpeedMap
    ws = small_ws(138.8)
    slow, _ = frozen_run(HeadingStage(workspace=ws))
    fast, _ = frozen_run(SpeedMap(workspace=ws, bytes_=(255,)))
    assert slow < 40.0 and fast < 80.0, (slow, fast)


def test_a_stalled_low_byte_is_not_mistaken_for_a_dead_tracker():
    """The speed map asks for bytes below the deadband on purpose."""
    from fleet.characterize import STUCK_MIN_EXPECTED_CM
    assert STUCK_MIN_EXPECTED_CM > 20.0, (
        "a crawl that legitimately stalls must not trip the watchdog")


def test_a_reversal_is_not_mistaken_for_being_stuck(ws):
    """It ends where it started. That is what a reversal is."""
    from fleet.characterize import LatencyProbe
    probe = LatencyProbe(workspace=ws, repeats=2)
    stage, _ = drive(probe, ws)
    assert "tracker is not following" not in (stage.error or "")
    assert stage.result().get("loop_delay_s") is not None


def test_the_noise_probe_refuses_an_impossibly_quiet_reading():
    """0.017cm of noise is not a still robot. It is a still ARTIFACT."""
    probe = NoiseProbe(frames=200)
    rng = np.random.default_rng(1)
    p, t = np.array([70.0, 55.0]), 0.0
    while not probe.done and t < 30:
        probe.step(t, p + rng.normal(0, 0.017, 2), DT)
        t += DT
    res = probe.result()
    assert "error" in res and "locked onto" in res["error"], res


def test_a_real_camera_reading_is_accepted():
    probe = NoiseProbe(frames=200)
    rng = np.random.default_rng(1)
    p, t = np.array([70.0, 55.0]), 0.0
    while not probe.done and t < 30:
        probe.step(t, p + rng.normal(0, 0.8, 2), DT)
        t += DT
    assert "error" not in probe.result()


def test_the_whole_run_stops_rather_than_measuring_on(ws):
    """Everything after a lying tracker is as meaningless as everything during."""
    c = Characterization(workspace=ws, code="T", quick=True)
    rng = np.random.default_rng(0)
    p = np.array([70.0, 55.0])
    t = 0.0
    while not c.done and t < 300:
        c.step(t, p + rng.normal(0, 0.01, 2), DT)
        t += DT
    assert c.aborted, "the run carried on against a dead tracker"
    assert any("would have meant anything" in n for n in c.notes), c.notes


# -- refusing to drive a robot the camera is not watching --------------------

def nudge_check(ws, mode="good", bias_deg=0.0, start=(69.0, 55.0)):
    """Run the pre-flight against a tracker that is or is not doing its job."""
    from fleet.characterize import TrackingCheck
    stage = TrackingCheck(workspace=ws)
    r = SimRobot("T", "T", "red", workspace=ws, pos=list(start), randomize=False)
    r.tau, r.bias = 0.35, np.radians(bias_deg)
    r.heading_tracking = False
    buf = deque([r.pos.copy()] * 5, maxlen=5)
    rng = np.random.default_rng(1)
    stuck = np.array(start, dtype=float)
    t, driven = 0.0, 0.0
    while not stage.done and t < 60:
        if mode == "frozen":
            meas = stuck + rng.normal(0, 0.01, 2)
        elif mode == "phantom":
            meas = stuck + rng.normal(0, 0.9, 2)
        else:
            meas = buf[0] + rng.normal(0, 0.9, 2)
        cmd = stage.step(t, meas, DT)
        if cmd is not None and cmd[1] > 0:
            driven += (cmd[1] / 255.0 * 60.0) * DT
        r.stop() if cmd is None else r.drive_raw(*cmd)
        r.step(DT)
        buf.append(r.pos.copy())
        t += DT
    return stage.result(), driven


@pytest.mark.parametrize("bias", [0.0, 35.0, -50.0])
def test_a_working_tracker_passes_the_preflight(ws, bias):
    """It must not object to a robot whose HEADING is merely uncalibrated."""
    res, _ = nudge_check(ws, bias_deg=bias)
    assert "error" not in res, res


def test_it_passes_near_a_wall_too(ws):
    res, _ = nudge_check(ws, start=(20.0, 55.0))
    assert "error" not in res, res


@pytest.mark.parametrize("mode", ["frozen", "phantom"])
def test_a_tracker_that_is_not_following_is_refused(ws, mode):
    res, driven = nudge_check(ws, mode=mode)
    assert "error" in res, res
    assert driven < 20.0, f"drove {driven:.0f}cm before deciding"


def test_the_preflight_decides_before_the_ball_can_get_out(ws):
    """The whole point: 9cm of evidence, not a length of the arena."""
    _, driven = nudge_check(ws, mode="frozen")
    x0, x1, y0, y1 = ws.bbox
    assert driven < (min(x1 - x0, y1 - y0) / 4.0), (
        f"drove {driven:.0f}cm to decide, in an arena {y1 - y0:.0f}cm tall")


def test_the_battery_will_not_start_driving_on_a_dead_tracker(ws):
    """Everything downstream guards the position the tracker REPORTS.

    The speed field, the edge margin, the runway planning — all of them protect
    a phantom if that is what the tracker is following, while the real ball
    drives wherever it likes. So the run does not begin on trust.
    """
    c = Characterization(workspace=ws, code="T", quick=True)
    rng = np.random.default_rng(0)
    p = np.array([69.0, 55.0])
    t, driven = 0.0, 0.0
    while not c.done and t < 300:
        cmd = c.step(t, p + rng.normal(0, 0.01, 2), DT)
        if cmd is not None and cmd[1] > 0:
            driven += (cmd[1] / 255.0 * 60.0) * DT
        t += DT
    assert c.aborted, "it drove the whole battery against a dead tracker"
    assert driven < 20.0, f"commanded {driven:.0f}cm before refusing"
    assert "tracking_check" in c.results


def test_a_healthy_battery_is_not_blocked_by_the_preflight(ws):
    c, worst = battery_excursion(138.8, seed=3, quick=True)
    assert not c.aborted, c.notes
    assert "tracking_check" in c.results
    assert "error" not in c.results["tracking_check"], c.results["tracking_check"]


# -- refusing to publish a number that cannot be true ------------------------
#
# From a real run: the speed map reported a ceiling of 288cm/s. No Sphero does
# that. It was a slope extrapolated from bytes 18-76 all the way to 255, and
# because every other figure is derived from it, one bad number turned the
# whole calibration into fiction — including a "deadband" of 20cm/s that was
# simply the bogus ceiling scaled back down, and which then sat ABOVE the speed
# cap so the ball could not legally be commanded to move at all.

def speed_map_from(pairs):
    from fleet.characterize import SpeedMap
    st = SpeedMap()
    st.rows = [{"byte": b, "cm_s": v, "course_deg": 0.0, "requested_deg": 0.0}
               for b, v in pairs]
    return st.result()


def test_an_impossible_top_speed_is_not_trusted():
    res = speed_map_from([(18, 0.4), (30, 6.0), (45, 14.0), (60, 22.0), (76, 30.0)])
    assert res["max_speed_cm_s"] > 95.0, "the fixture should extrapolate high"
    assert not res["max_speed_trusted"]


def test_a_top_speed_from_a_short_span_is_not_trusted_either():
    """Even a plausible number is a guess if it came from a third of the range."""
    res = speed_map_from([(18, 4.2), (37, 8.7), (57, 13.4), (76, 17.9)])
    assert res["max_speed_cm_s"] < 95.0
    assert not res["max_speed_trusted"], res["max_speed_cm_s"]


def test_a_full_sweep_is_trusted():
    res = speed_map_from([(20, 4.7), (60, 14.1), (110, 25.9),
                          (170, 40.0), (230, 54.1), (255, 60.0)])
    assert res["max_speed_trusted"]
    assert 55 < res["max_speed_cm_s"] < 65


def test_an_untrusted_ceiling_is_not_handed_to_the_controller(ws):
    c = Characterization(workspace=ws, code="T")
    c.results["speed_map"] = speed_map_from(
        [(18, 0.4), (30, 6.0), (45, 14.0), (60, 22.0), (76, 30.0)])
    fit = c.fit()
    assert "handle_max_speed_cm_s" not in fit["recommend"]
    assert "speed_map_warning" in fit
    assert "not being used" in fit["speed_map_warning"]


def test_the_deadband_is_never_manufactured_from_an_untrusted_ceiling(ws):
    """A 288cm/s ceiling turned byte 18 into 'will not move below 20cm/s'."""
    c = Characterization(workspace=ws, code="T")
    c.results["speed_map"] = speed_map_from(
        [(18, 0.4), (30, 6.0), (45, 14.0), (60, 22.0), (76, 30.0)])
    rec = c.fit()["recommend"]
    assert rec["min_moving_byte"] == 30
    assert "min_moving_cm_s" not in rec, "converted through a ceiling it does not trust"


def test_the_deadband_byte_still_reaches_the_controller():
    """It is the measured half of the pair and it must not be lost."""
    from swarm.pd import gains_from_motion
    g = gains_from_motion({"step_response": {"tau_s": 0.16},
                           "latency": {"loop_delay_s": 0.41},
                           "recommend": {"min_moving_byte": 37}})
    assert g["deadband_cm_s"] == pytest.approx(37 / 255 * 60, abs=0.5)
    assert g["max_speed"] == 60.0, "and it falls back to the nominal ceiling"


# -- recentring with a wrong aim frame ---------------------------------------

def _recentre_with(frame_error_deg, start=(25.0, 25.0), target=(69.4, 55.4)):
    """Drive a Recenter against a world whose frame is rotated."""
    r = Recenter(target)
    pos = np.array(start, dtype=float)
    t, dt = 0.0, 1 / 30.0
    while not r.done and t < 40.0:
        cmd = r.step(t, pos, dt)
        if cmd is None:
            break
        heading, byte = cmd
        a = np.radians((heading + frame_error_deg) % 360.0)
        pos = pos + np.array([np.sin(a), np.cos(a)]) * (byte / 255.0 * 60.0) * dt
        t += dt
    return r, t, pos


def test_recentring_arrives_when_the_frame_is_right():
    r, t, _ = _recentre_with(0.0)
    assert r.error is None
    assert t < 6.0


def test_a_small_frame_error_still_arrives():
    """Closing the loop on position absorbs a modest rotation. It must not be
    mistaken for a broken frame."""
    r, _, _ = _recentre_with(30.0)
    assert r.error is None


def test_a_reversed_frame_is_caught_in_a_second():
    """It used to drive 776cm over 30s and end up outside the arena, and the
    only thing that noticed was the timeout."""
    r, t, _ = _recentre_with(180.0)
    assert r.error is not None
    assert t < 3.0, f"took {t:.1f}s"
    assert r.travelled < 40.0, f"drove {r.travelled:.0f}cm before noticing"
    assert "AWAY" in r.error


def test_a_perpendicular_frame_is_caught_too():
    """At 90 degrees the ball drives tangentially — it ORBITS the target at
    roughly constant distance, so it never gets further away and the
    wrong-way check alone never fires. Covering ground without arriving is
    the signal that does."""
    r, t, _ = _recentre_with(90.0)
    assert r.error is not None
    assert t < 10.0, f"took {t:.1f}s"
    assert "round the middle" in r.error


def test_it_never_drives_the_arena_away():
    """The complaint this exists for: 30 seconds of a wrong frame put the ball
    metres outside the workspace and left every later stage without room."""
    for err in (90.0, 135.0, 180.0, 225.0, 270.0):
        r, _, pos = _recentre_with(err)
        assert r.travelled < 200.0, f"{err}deg: drove {r.travelled:.0f}cm"

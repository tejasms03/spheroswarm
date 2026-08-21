"""Fusing gyro yaw with camera travel into a frame offset.

The estimator is fed a simulated robot whose true heading we know, so accuracy
is measured rather than asserted. The robot REFLECTS off walls: a clipped robot
slides along one while its heading says otherwise, and the harness would then
be measuring its own bug.
"""

import math

import numpy as np
import pytest

from fleet.heading import HeadingEstimator, circular_mean, wrap180


def drive(estimator, true_offset=37.0, cam_noise=1.0, gyro_drift=3.0,
          gyro_noise=0.5, speed=45.0, dt=1 / 30, seconds=40.0,
          turn_every=6.0, turn_rate=120.0, seed=7, teleport_at=None):
    rng = np.random.default_rng(seed)
    pos = np.array([120.0, 90.0])
    heading = target = 20.0
    bias = t = 0.0
    errors = []

    for i in range(int(seconds / dt)):
        if teleport_at is not None and i == int(teleport_at / dt):
            true_offset = wrap180(true_offset + 90.0)      # picked up, put down turned
        if i and i % int(turn_every / dt) == 0:
            target = wrap180(heading + rng.uniform(-120, 120))

        d = wrap180(target - heading)
        heading = wrap180(heading + max(-turn_rate * dt, min(turn_rate * dt, d)))

        rad = math.radians(heading)
        nxt = pos + np.array([math.cos(rad), math.sin(rad)]) * speed * dt
        for ax, (lo, hi) in enumerate(((10, 230), (10, 170))):
            if nxt[ax] < lo or nxt[ax] > hi:
                heading = wrap180(180.0 - heading) if ax == 0 else wrap180(-heading)
                target = heading
                rad = math.radians(heading)
                nxt = pos + np.array([math.cos(rad), math.sin(rad)]) * speed * dt
        pos = nxt

        bias += gyro_drift / 60.0 * dt
        yaw = wrap180(wrap180(heading - true_offset) + bias + rng.normal(0, gyro_noise))
        seen = pos + rng.normal(0, cam_noise, 2)
        estimator.update(t, seen[0], seen[1], yaw)
        t += dt
        if estimator.ready:
            errors.append(abs(wrap180(estimator.heading(yaw) - heading)))
    return np.array(errors)


def good():
    return HeadingEstimator(min_baseline=25.0, max_turn=5.0)


# -- the maths that is easy to get wrong -------------------------------------

def test_circular_mean_does_not_average_across_the_wrap():
    """A scalar mean of 359 and 1 is 180 — the exact opposite of the answer."""
    assert circular_mean([359.0, 1.0]) == pytest.approx(0.0, abs=0.1)
    assert circular_mean([170.0, -170.0]) == pytest.approx(180.0, abs=0.1)


def test_wrap180_folds_both_ways():
    assert wrap180(370) == pytest.approx(10)
    assert wrap180(-190) == pytest.approx(170)


# -- accuracy ----------------------------------------------------------------

def test_it_recovers_a_known_offset_to_within_a_few_degrees():
    err = drive(good())
    assert len(err), "never settled"
    assert np.median(err) < 3.0, f"median {np.median(err):.1f}deg"
    assert np.percentile(err, 90) < 6.0


def test_a_long_baseline_beats_frame_to_frame_by_an_order_of_magnitude():
    """Waiting for distance is the single biggest lever: noise goes as sigma/d."""
    naive = HeadingEstimator(min_baseline=0.01, max_turn=360.0, gate=360.0, alpha=1.0)
    assert np.median(drive(naive)) > 10 * np.median(drive(good()))


def test_gyro_drift_is_absorbed():
    """20x the drift must not degrade the answer — correcting it is the point."""
    assert np.median(drive(good(), gyro_drift=60.0)) < 3.0


def test_it_works_at_any_offset():
    for offset in (-140.0, -37.0, 0.0, 91.0, 179.0):
        err = drive(good(), true_offset=offset)
        assert len(err) and np.median(err) < 3.0, f"offset {offset}"


def test_a_slow_robot_still_calibrates():
    """The trail has to span enough TIME to reach the baseline at low speed.

    At 40 samples the trail was 1.3s — 20cm at 15cm/s — so a creeping robot
    never reached a 25cm baseline and never calibrated at all.
    """
    err = drive(good(), speed=15.0)
    assert len(err), "a slow robot never settled"
    assert np.median(err) < 4.0


def test_noise_degrades_it_gracefully():
    clean = np.median(drive(good(), cam_noise=1.0))
    noisy = np.median(drive(good(), cam_noise=3.0))
    assert noisy > clean
    assert noisy < 8.0, "3cm of camera noise should still be usable"


def test_it_recovers_after_the_ball_is_picked_up_and_turned():
    """Self-healing is why this beats a calibrate-once constant."""
    est = good()
    drive(est, seconds=60.0, teleport_at=30.0)
    assert est.ready
    assert abs(est.last_residual) < 20.0, "never re-converged after the change"


# -- refusing to answer ------------------------------------------------------

def test_it_says_nothing_before_it_knows():
    est = good()
    assert est.offset is None and not est.ready
    assert est.heading(12.0) is None


def test_a_stationary_robot_never_calibrates():
    """No travel, no information. It must not invent an answer."""
    est = good()
    for i in range(600):
        est.update(i / 30.0, 100.0, 100.0, 45.0)
    assert not est.ready
    assert est.rejected["short"] > 0


def test_turning_samples_are_rejected():
    """Travel across a turn is a chord, and a chord is nobody's heading."""
    est = good()
    drive(est, turn_every=0.7, turn_rate=360.0, seconds=20.0)
    assert est.rejected["turning"] > 0


def test_state_is_reportable():
    est = good()
    drive(est, seconds=20.0)
    st = est.state()
    assert st["ready"] and st["samples"] > 0
    assert -180 <= st["offset_deg"] <= 180
    assert set(st["rejected"]) == {"short", "turning", "outlier", "reversing"}
    assert 0.0 <= st["confidence"] <= 1.0


# -- active calibration ------------------------------------------------------
#
# Drive a known pattern, read the offset straight off it. No gyro involved:
# command a heading in the ball's frame, see where it goes in the camera's.

from fleet.heading import ActiveCalibration  # noqa: E402


def calibrate(true_offset=37.0, cam_noise=1.0, dt=1 / 30, workspace=None,
              slip=1.0, start=(120.0, 90.0), seed=3, cal=None, max_s=60.0):
    """Run a calibration against a robot with a known frame offset."""
    rng = np.random.default_rng(seed)
    cal = cal or ActiveCalibration(workspace=workspace)
    pos = np.array(start, dtype=float)

    for _ in range(int(max_s / dt)):
        seen = pos + rng.normal(0, cam_noise, 2)
        v = cal.step(seen, dt)
        if v is None:
            break
        # The ball moves in the CAMERA frame, offset from what we commanded.
        speed = float(np.linalg.norm(v))
        if speed > 1e-9:
            commanded = math.degrees(math.atan2(v[1], v[0]))
            actual = math.radians(wrap180(commanded + true_offset))
            pos = pos + np.array([math.cos(actual), math.sin(actual)]) * speed * dt * slip
    return cal, pos


def test_active_calibration_reads_the_offset_off_four_legs():
    cal, _ = calibrate(true_offset=37.0)
    assert cal.done and cal.error is None, cal.error
    assert abs(wrap180(cal.offset - 37.0)) < 4.0, f"got {cal.offset:.1f}"
    assert len(cal.residuals) == 4


def test_it_needs_no_gyro_at_all():
    """Nothing in the routine consumes a yaw reading."""
    import inspect
    src = inspect.getsource(ActiveCalibration)
    assert "yaw" not in src


def test_it_works_at_any_offset():
    for offset in (-140.0, -37.0, 0.0, 91.0, 179.0):
        cal, _ = calibrate(true_offset=offset)
        assert cal.done and cal.error is None
        assert abs(wrap180(cal.offset - offset)) < 5.0, f"offset {offset}"


def test_the_legs_agreeing_is_what_makes_it_trustworthy():
    cal, _ = calibrate()
    assert cal.spread is not None and cal.spread < 10.0


def test_a_slipping_robot_is_reported_rather_than_averaged():
    """Four legs that disagree are a diagnosis, not a number to average."""
    rng = np.random.default_rng(1)
    cal = ActiveCalibration()
    pos = np.array([120.0, 90.0])
    dt = 1 / 30
    # One fixed error PER LEG. A fresh error every tick would be a random walk,
    # which averages back out over the leg and proves nothing.
    per_leg = [55.0, -70.0, 20.0, -40.0]
    for _ in range(int(60 / dt)):
        leg = min(cal.leg, len(per_leg) - 1)
        v = cal.step(pos + rng.normal(0, 1.0, 2), dt)
        if v is None:
            break
        speed = float(np.linalg.norm(v))
        if speed > 1e-9:
            commanded = math.degrees(math.atan2(v[1], v[0]))
            actual = math.radians(wrap180(commanded + per_leg[leg]))
            pos = pos + np.array([math.cos(actual), math.sin(actual)]) * speed * dt
    assert cal.done
    assert cal.error is not None and "disagree" in cal.error
    assert cal.spread > 25.0


def test_a_stuck_robot_times_out_with_a_useful_message():
    cal = ActiveCalibration(timeout=2.0)
    pos = np.array([120.0, 90.0])
    for _ in range(int(30 / (1 / 30))):
        if cal.step(pos, 1 / 30) is None:      # never moves
            break
    assert cal.done and cal.error is not None
    assert "stuck" in cal.error or "camera" in cal.error


def test_it_refuses_when_there_is_no_room(open_ws):
    """Better to say so than to drive into a wall measuring a wall."""
    cal = ActiveCalibration(workspace=open_ws)
    corner = np.array([open_ws.bbox[0] + 2.0, open_ws.bbox[2] + 2.0])
    for _ in range(int(20 / (1 / 30))):
        if cal.step(corner, 1 / 30) is None:
            break
    assert cal.done
    assert cal.error is None or "space" in cal.error


def test_it_uses_the_opposite_direction_when_blocked(open_ws):
    """A leg away from the wall measures the same thing as one into it."""
    cal, _ = calibrate(true_offset=20.0, workspace=open_ws,
                       start=(open_ws.bbox[1] - 40.0, 90.0))
    assert cal.done
    if cal.error is None:
        assert abs(wrap180(cal.offset - 20.0)) < 6.0


def test_the_pattern_returns_roughly_to_where_it_started():
    """Out and back on both axes: it should not wander off across the arena."""
    _, end = calibrate(start=(120.0, 90.0))
    assert float(np.linalg.norm(end - np.array([120.0, 90.0]))) < 30.0


def test_progress_is_reportable_while_it_runs():
    cal = ActiveCalibration()
    assert "1/4" in cal.progress
    st = cal.state()
    assert st["done"] is False and st["legs"] == 0


# -- small arenas ------------------------------------------------------------

def _small_ws(w=138.8, h=110.8):
    from workspace.space import Workspace
    return Workspace(bounds_cm=[[0, 0], [w, 0], [w, h], [0, h]])


def test_it_calibrates_in_a_room_smaller_than_the_nominal_leg_needs():
    """A real arena, 138.8 by 110.8cm.

    The four cardinal legs wanted 57cm of clearance each and there are 55
    vertically, so the calibration refused outright — leaving the robot with an
    unknown frame, which costs far more accuracy than a shorter leg does. It
    now shortens the leg instead of giving up.
    """
    ws = _small_ws()
    for offset in (0.0, 37.0, -60.0):
        cal, _ = calibrate(true_offset=offset, workspace=ws,
                           start=(ws.bbox[1] / 2, ws.bbox[3] / 2), max_s=90.0)
        assert cal.error is None, f"offset {offset}: {cal.error}"
        assert len(cal.residuals) == 4
        assert abs(wrap180(cal.offset - offset)) < 8.0


def test_it_calibrates_from_a_corner_of_a_small_arena():
    ws = _small_ws()
    cal, _ = calibrate(true_offset=37.0, workspace=ws, start=(30.0, 30.0),
                       max_s=90.0)
    assert cal.error is None, cal.error
    assert abs(wrap180(cal.offset - 37.0)) < 10.0


def test_a_shortened_leg_is_still_long_enough_to_mean_something():
    """Baseline sets the accuracy, so there is a length below which it stops."""
    from fleet.heading import CAL_MIN_LEG_CM
    import math
    noise_cm = 1.0
    worst = math.degrees(math.atan2(noise_cm * math.sqrt(2), CAL_MIN_LEG_CM))
    assert worst < 10.0, f"a {CAL_MIN_LEG_CM}cm leg is worth {worst:.0f}deg"


def test_a_room_too_small_for_even_a_short_leg_is_refused_not_faked():
    tiny = _small_ws(40.0, 40.0)
    cal, _ = calibrate(true_offset=20.0, workspace=tiny, start=(20.0, 20.0),
                       max_s=40.0)
    assert cal.error is not None

"""The analysis of a calibration run, on synthetic legs with known answers."""

import math

import numpy as np
import pytest

from swarm import ball_calib as bc


def leg(bearing_a=30.0, turned=20.0, steer=20.0, speed=6.0, delay=0.4,
        noise=0.0, t_a=5.0, t_b=5.0, dt=1 / 30.0, start=(20.0, 20.0), seed=0):
    """A leg driven straight at `bearing_a`, then bending by `turned` degrees
    `delay` seconds after a `steer` command. Positions as a camera sees them."""
    rng = np.random.default_rng(seed)
    rows, t, p = [], 100.0, np.array(start, float)
    go_at = t
    steer_at = go_at + t_a
    b = bearing_a
    while t < steer_at + t_b:
        seg = "A" if t < steer_at else "B"
        if t >= steer_at + delay:
            b = bearing_a + turned
        r = math.radians(b)
        p = p + np.array([math.sin(r), math.cos(r)]) * speed * dt
        q = p + rng.normal(0, noise, 2)
        rows.append([t, float(q[0]), float(q[1]), seg])
        t += dt
    return {"samples": rows, "steer": steer, "steer_at": steer_at,
            "go_at": go_at, "stop_at": t, "stop_xy": p.tolist(),
            "end_xy": (p + 1.2).tolist(), "mid": start}


@pytest.mark.parametrize("turned,steer,sign", [(20, 20, 1), (-20, 20, -1),
                                               (20, -20, -1), (-20, -20, 1)])
def test_sign_follows_which_way_the_path_bent(turned, steer, sign):
    got = bc.analyse_leg(leg(turned=turned, steer=steer))
    assert got["sign"] == sign
    assert got["gain"] == pytest.approx(1.0, abs=0.05)


def test_speed_and_coast_are_recovered():
    got = bc.analyse_leg(leg(speed=7.5, noise=0.3))
    assert got["speed"] == pytest.approx(7.5, rel=0.05)
    assert got["coast"] == pytest.approx(math.hypot(1.2, 1.2), abs=1e-6)


@pytest.mark.parametrize("delay", [0.2, 0.6, 1.0])
def test_delay_is_recovered_from_where_the_path_bends(delay):
    got = bc.analyse_leg(leg(delay=delay, noise=0.2))
    assert got["delay"] == pytest.approx(delay, abs=0.15)


def test_bearing_wraps_through_north():
    got = bc.analyse_leg(leg(bearing_a=350.0, turned=20.0, steer=20.0))
    assert got["sign"] == 1 and got["turned"] == pytest.approx(20, abs=1)


def test_a_leg_too_short_says_why():
    got = bc.analyse_leg(leg(t_a=0.9))
    assert "why" in got and "sign" not in got


M = np.eye(3)


def test_a_good_run_summarises_and_round_trips(tmp_path, monkeypatch):
    from vision import config
    monkeypatch.setattr(config, "CALIB", tmp_path)
    legs = [leg(bearing_a=b, turned=-20 * s, steer=20 * s, noise=0.2, seed=i)
            for i, (b, s) in enumerate([(90, 1), (0, -1), (270, 1), (180, -1),
                                        (40, 1)])]
    rec, why = bc.summarise(legs, 26, M, "SK-A")
    assert why is None
    assert rec["sign"] == -1 and rec["byte"] == 26
    assert rec["speed_cm_s"] == pytest.approx(6.0, rel=0.05)
    bc.save(rec)
    back, why = bc.load("SK-A", M)
    assert why is None and back["sign"] == -1


def test_a_different_camera_mapping_refuses_the_saved_numbers(tmp_path,
                                                             monkeypatch):
    from vision import config
    monkeypatch.setattr(config, "CALIB", tmp_path)
    legs = [leg(seed=i) for i in range(4)]
    rec, _ = bc.summarise(legs, 26, M, "SK-A")
    bc.save(rec)
    moved = M.copy()
    moved[0, 2] = 3.0
    back, why = bc.load("SK-A", moved)
    assert back is None and "different corners" in why
    assert bc.load("SK-B", M)[0] is None


def test_legs_that_disagree_on_sign_are_not_saved():
    legs = [leg(turned=20, seed=i) for i in range(3)] + [leg(turned=-20)]
    rec, why = bc.summarise(legs, 26, M, "SK-A")
    assert rec is None and "disagree" in why


def test_too_few_measured_legs_are_not_saved():
    legs = [leg(seed=i) for i in range(3)] + [leg(t_a=0.9), leg(t_a=0.9)]
    rec, why = bc.summarise(legs, 26, M, "SK-A")
    assert rec is None and "3 of 5" in why


def test_speed_that_differs_leg_to_leg_is_saved_with_a_warning():
    legs = [leg(speed=s, seed=i) for i, s in enumerate([6, 6, 6, 8])]
    rec, why = bc.summarise(legs, 26, M, "SK-A")
    assert why is None and "speed varies" in rec["warnings"][0]


def test_a_ball_that_does_not_follow_the_steer_is_not_saved():
    legs = [leg(turned=3, seed=i) for i in range(4)]
    rec, why = bc.summarise(legs, 26, M, "SK-A")
    assert rec is None


def test_a_steer_that_bends_quickly_on_a_short_stretch_is_measured():
    """The bench's real legs: ~14cm/s, a 2s steered stretch, bending 0.25s
    after the steer. The first real run saved nothing because of this."""
    legs = [leg(speed=14.0, t_a=2.0, t_b=2.0, delay=0.25, turned=-20 * s,
                steer=20 * s, noise=0.2, seed=i)
            for i, s in enumerate([1, -1, 1, -1])]
    rec, why = bc.summarise(legs, 30, M, "SK-A")
    assert why is None and rec["sign"] == -1

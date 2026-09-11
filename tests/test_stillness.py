"""A ball's own accelerometer, as a veto on "this ball is still".

See `fleet/stillness.py` for why the accelerometer and not the gyro, and why
it may only ever say "moving" or "no idea" and never "still" on its own.
"""

import numpy as np
import pytest

from fleet.real_handle import SpheroRobot
from fleet.sim_handle import SimRobot
from fleet.stillness import IMU_PERIOD_S, QUIET_G, Stillness, as_vector


def _feed(s, seconds, spread, rng=None, t0=0.0):
    rng = rng or np.random.default_rng(0)
    t = t0
    while t < t0 + seconds:
        s.add(t, np.array([0.0, 0.0, 1.0]) + rng.normal(0.0, spread, 3))
        t += IMU_PERIOD_S
    return t


# -- readings ----------------------------------------------------------------

def test_a_spherov2_reading_is_a_dict_in_g():
    assert np.allclose(as_vector({"x": 0.1, "y": -0.2, "z": 0.98}), [0.1, -0.2, 0.98])


@pytest.mark.parametrize("junk", [None, {}, {"x": 1}, "abc", [1, 2],
                                  {"x": float("nan"), "y": 0, "z": 1}])
def test_a_reading_that_is_not_three_numbers_is_no_reading(junk):
    assert as_vector(junk) is None


# -- the verdict -------------------------------------------------------------

def test_a_parked_ball_is_quiet():
    s = Stillness()
    t = _feed(s, 1.2, spread=0.003)
    assert s.quiet(t) is True


def test_a_rolling_ball_is_not():
    s = Stillness()
    t = _feed(s, 1.2, spread=0.08)
    assert s.quiet(t) is False


def test_too_few_readings_is_no_verdict():
    s = Stillness()
    t = _feed(s, 0.3, spread=0.003)
    assert s.quiet(t) is None


def test_a_stream_gone_quiet_is_no_verdict_rather_than_still():
    """A dead stream usually leaves its last value frozen, and a frozen value
    looks perfectly still. It must read as "unknown"."""
    s = Stillness()
    t = _feed(s, 1.2, spread=0.003)
    assert s.quiet(t + 2.0) is None


def test_a_bump_ends_a_quiet_window():
    s = Stillness()
    t = _feed(s, 1.2, spread=0.003)
    s.add(t, [0.4, -0.3, 1.2])
    assert s.quiet(t) is False


# -- the handles -------------------------------------------------------------

def test_a_parked_sim_ball_reads_quiet_and_a_driven_one_does_not():
    parked = SimRobot("A", "AAAA", "cyan", randomize=False, seed=1)
    moving = SimRobot("B", "BBBB", "red", randomize=False, seed=2)
    moving.set_velocity(np.array([20.0, 0.0]))
    for _ in range(60):                            # two seconds
        parked.step(1 / 30)
        moving.step(1 / 30)
    assert parked.accel_quiet() is True, parked.accel_sigma()
    assert moving.accel_quiet() is False, moving.accel_sigma()
    assert moving.accel_sigma() > QUIET_G > parked.accel_sigma()


def test_the_sim_sensor_leaves_seeded_runs_where_they_were():
    """Its own generator, so every existing seeded test draws the same drift."""
    a = SimRobot("A", "AAAA", "cyan", seed=5)
    b = SimRobot("A", "AAAA", "cyan", seed=5)
    b.imu = Stillness()                        # a sensor that is never read
    for r in (a, b):
        r.set_velocity(np.array([15.0, 5.0]))
        for _ in range(40):
            r.step(1 / 30)
    assert np.allclose(a.pos, b.pos)


class FakeApi:
    def __init__(self):
        self.reads = 0
        self.value = {"x": 0.0, "y": 0.0, "z": 1.0}

    def get_acceleration(self):
        self.reads += 1
        return dict(self.value)


def _real():
    r = SpheroRobot("A", "AAAA", "cyan", "SK-1",
                    connector=lambda n, timeout=8: None, autostart=False)
    r._api = FakeApi()
    return r


def test_a_real_ball_is_read_on_the_stream_s_clock_not_every_tick():
    """The cache refreshes every 150 ms. Reading it thirty times a second would
    count one reading thirty times and make any ball look perfectly still."""
    r = _real()
    for _ in range(50):
        r.accel_quiet()
    assert r._api.reads == 1


def test_a_real_ball_with_no_link_has_no_verdict():
    r = _real()
    r._api = None
    assert r.accel_quiet() is None


def test_a_handle_with_no_sensor_says_nothing():
    from fleet.handle import RobotHandle
    assert RobotHandle.accel_quiet(object()) is None

"""Does a simulated robot behave like the ball it was measured from?

Until now it did not, and nothing said so. The battery computes `sim_tau_s`,
`sim_latency_steps` and `sim_gain`, and the only thing that ever read them was
a line in the bench that PRINTED them for somebody to copy by hand. `SimRobot`
drew `rng.integers(1, 4)` regardless — so the sim ran a plant several times
more responsive than the hardware, and a controller tuned against it was tuned
against fiction.

Two things have to hold. The measurement has to reach the robot, and it has to
mean the same thing at any tick rate: a delay held as a queue length is 0.53s
at 30Hz and 1.6s at 10Hz, which is not a measurement, it is a coincidence.
"""

import numpy as np
import pytest

from fleet.manager import Fleet
from fleet.roster import Roster
from fleet.sim_handle import SIM_TICK_HZ, SimRobot, from_motion


def _fit(**rec):
    return {"recommend": rec}


# -- reading a calibration ------------------------------------------------

def test_measured_constants_are_pulled_out_of_a_fit():
    m = from_motion(_fit(sim_tau_s=0.32, sim_latency_steps=16, sim_gain=0.8))
    assert m["tau"] == 0.32
    assert m["gain"] == 0.8
    assert m["latency_s"] == pytest.approx(16 / SIM_TICK_HZ)


def test_a_fit_that_measured_nothing_offers_nothing():
    assert from_motion({}) == {}
    assert from_motion(None) == {}
    assert from_motion(_fit()) == {}


def test_withheld_constants_are_simply_absent():
    """A run whose stages errored has those numbers withheld from `recommend`.
    The robot must keep its guess rather than inherit a zero."""
    m = from_motion(_fit(sim_tau_s=0.32))
    assert "tau" in m
    assert "gain" not in m and "latency_s" not in m


def test_measurement_beats_a_guess_per_constant():
    """A run that established the lag but not the top speed hands over the lag
    and leaves the gain alone."""
    r = SimRobot("A", "AAAA", "cyan", seed=1, motion={"tau": 0.32})
    assert r.tau == 0.32
    assert 0.8 <= r.gain <= 1.2, "the unmeasured constant keeps its spread"
    assert r.measured == ["tau"]


def test_an_unmeasured_robot_is_unchanged():
    a = SimRobot("A", "AAAA", "cyan", seed=7)
    b = SimRobot("A", "AAAA", "cyan", seed=7, motion={})
    assert (a.tau, a.gain, a.latency) == (b.tau, b.gain, b.latency)
    assert a.measured == []


# -- the delay has to be a duration ---------------------------------------

@pytest.mark.parametrize("dt", [1 / 30, 1 / 60, 0.1, 0.05])
def test_a_measured_delay_holds_its_duration_at_any_tick_rate(dt):
    r = SimRobot("A", "AAAA", "cyan", randomize=False,
                 motion={"latency_s": 16 / SIM_TICK_HZ})
    r.step(dt)
    held = r.queue.maxlen * dt
    assert held == pytest.approx(16 / SIM_TICK_HZ, abs=dt), (
        f"{r.queue.maxlen} slots at dt={dt} is {held:.3f}s")


def test_a_guessed_delay_is_left_alone():
    """A guess is a slot count with no seconds behind it. Reinterpreting it
    would change the dynamics of every uncharacterised robot in the suite."""
    r = SimRobot("A", "AAAA", "cyan", seed=3)
    before = r.queue.maxlen
    r.step(0.1)
    assert r.queue.maxlen == before


def test_resizing_does_not_lose_commands_in_flight():
    r = SimRobot("A", "AAAA", "cyan", randomize=False,
                 motion={"latency_s": 0.5})
    r.set_velocity(np.array([20.0, 0.0]))
    for _ in range(4):
        r.step(1 / 30)
    r.step(1 / 60)                       # the tick rate changes under it
    assert len(r.queue) == r.queue.maxlen
    assert all(np.isfinite(v).all() for v in r.queue)


def test_a_measured_robot_actually_responds_more_slowly():
    """The point of all of it. Same command, same clock, different plant."""
    guess = SimRobot("A", "AAAA", "cyan", randomize=False)
    real = SimRobot("B", "BBBB", "red", randomize=False,
                    motion={"tau": 0.32, "latency_s": 16 / SIM_TICK_HZ})
    for r in (guess, real):
        r.pos = np.array([50.0, 50.0])
        r.vel = np.zeros(2)
        r.set_velocity(np.array([30.0, 0.0]))
    for _ in range(15):                  # half a second
        guess.step(1 / 30)
        real.step(1 / 30)

    assert guess.speed > real.speed, (
        f"guessed plant {guess.speed:.1f}cm/s vs measured {real.speed:.1f}cm/s "
        "— the measured ball has not even started moving yet")
    assert real.speed < 1.0, "0.53s of dead time means nothing has arrived"


# -- where the calibration is allowed to come from ------------------------

def test_a_fleet_does_not_read_a_calibration_unless_asked(open_ws, sim_entries):
    """`calib/motion.json` is live state. A fleet that loaded it by default
    would make every test's dynamics depend on the last hardware session — and
    the fixtures here use the real robot codes, so CRXS would quietly inherit
    its own measured half-second of command latency."""
    f = Fleet.from_roster(Roster(entries=sim_entries[:4], path="/dev/null"),
                          workspace=open_ws)
    try:
        assert all(h.measured == [] for h in f.handles.values())
    finally:
        f.close()


def test_a_fleet_given_a_path_uses_it(open_ws, sim_entries, tmp_path):
    import json
    p = tmp_path / "motion.json"
    p.write_text(json.dumps({"CRXS": _fit(sim_tau_s=0.44, sim_latency_steps=9)}))

    f = Fleet.from_roster(Roster(entries=sim_entries[:4], path="/dev/null"),
                          workspace=open_ws, motion_path=p)
    try:
        crxs = f.handles["CRXS"]
        assert crxs.tau == 0.44
        assert crxs.latency_s == pytest.approx(9 / SIM_TICK_HZ)
        others = [h for c, h in f.handles.items() if c != "CRXS"]
        assert all(h.measured == [] for h in others), "only the one measured"
    finally:
        f.close()


def test_an_unreadable_calibration_does_not_stop_a_fleet(open_ws, sim_entries,
                                                         tmp_path):
    p = tmp_path / "motion.json"
    p.write_text("{ this is not json")
    f = Fleet.from_roster(Roster(entries=sim_entries[:2], path="/dev/null"),
                          workspace=open_ws, motion_path=p)
    try:
        assert len(f.handles) == 2, "a bad calibration costs fidelity, not a fleet"
    finally:
        f.close()


# -- speeding up and slowing down are not the same move -------------------

def _roll_out(entry, coast_s, latency_s, dt=1 / 30):
    """Drive up to speed, cut, and measure the stop — the brake test's shape.

    Driving first matters: the delay is modelled as a queue of commands in
    flight, so a ball whose velocity was assigned rather than COMMANDED has a
    queue full of zeros and feels no delay at all. Measuring from that is
    measuring a stop that never happens on the rig.
    """
    r = SimRobot("A", "AAAA", "cyan", randomize=False,
                 motion={"coast_s": coast_s, "latency_s": latency_s})
    r.pos = np.array([50.0, 50.0])
    r.set_velocity(np.array([entry, 0.0]))
    for _ in range(120):                      # four seconds, well past settled
        r.step(dt)
    cut = r.pos.copy()
    r.stop()
    for _ in range(400):
        r.step(dt)
        if r.speed < 0.3:
            break
    return float(np.linalg.norm(r.pos - cut))


def test_a_measured_coast_is_pulled_out_of_a_fit():
    m = from_motion(_fit(stopping_distance_s_per_cm_s=0.51))
    assert m["coast_s"] == 0.51


def test_a_run_that_never_braked_offers_no_coast():
    assert "coast_s" not in from_motion(_fit(sim_tau_s=0.32))


def test_an_uncharacterised_ball_decelerates_exactly_as_it_always_did():
    """The asymmetry is a MEASUREMENT, not a new default. A guessed coast
    would move the dynamics of every robot in the suite for no gain."""
    r = SimRobot("A", "AAAA", "cyan", randomize=False)
    assert r.coast_s is None
    r.vel = np.array([30.0, 0.0])
    r.stop()
    for _ in range(3):
        r.step(1 / 30)
    assert r.speed == pytest.approx(30.0 * (1 - min((1 / 30) / 0.35, 1.0)) ** 3,
                                    rel=1e-6)


def test_a_measured_ball_keeps_rolling_after_the_motors_cut():
    slack = _roll_out(30.0, None, 1 / 30)
    real = _roll_out(30.0, 0.6, 1 / 30)
    assert real > slack + 2.0, (
        f"coasted {real:.1f}cm vs {slack:.1f}cm — a sim that stops better than "
        "the ball is the one error tracking cannot survive")


@pytest.mark.parametrize("entry", [15.0, 30.0, 45.0])
def test_the_roll_out_matches_the_constant_that_was_measured(entry):
    """The brake test fits `coast_cm = k * v` through the origin. A first-order
    decay with time constant `k` travels exactly `v * k` — so the sim reproduces
    the measurement at every speed, with nothing fitted twice."""
    assert _roll_out(entry, 0.5, 1 / 30) == pytest.approx(entry * 0.5, rel=0.1)


def test_the_coast_does_not_touch_acceleration():
    """It is the deceleration constant. A ball asked to speed up still does so
    on `tau`, or the measurement would be doing two jobs."""
    plain = SimRobot("A", "AAAA", "cyan", randomize=False)
    coasty = SimRobot("B", "BBBB", "red", randomize=False, motion={"coast_s": 0.9})
    for r in (plain, coasty):
        r.vel = np.zeros(2)
        r.set_velocity(np.array([30.0, 0.0]))
        for _ in range(10):
            r.step(1 / 30)
    assert coasty.speed == pytest.approx(plain.speed, rel=1e-9)


# -- against the hardware, not against the algebra ------------------------

# Eight real stops, SYRX, 2026-08-24. Frozen here rather than read from
# `calib/motion.json` for the reason the fleet tests give: live calibration
# state must not decide what a test asserts. This is the only brake data this
# project has ever collected, and its own fit was WITHHELD as untrustworthy —
# 26% speed spread, r2 0.45, and two stops from the same command differing by
# 3.7cm. So it pins the magnitude and nothing finer.
SYRX_STOPS = [(24.69, 8.34), (25.32, 15.64), (25.97, 11.74), (24.71, 13.26),
              (26.37, 12.05), (28.45, 16.37), (33.26, 17.28), (28.91, 16.73)]
SYRX_COAST_S = 0.5142
SYRX_LATENCY_S = 1 / SIM_TICK_HZ


def test_the_sim_stops_where_the_real_ball_stopped():
    """Checked against measurements, not against the fit it was derived from."""
    err = [abs(_roll_out(v, SYRX_COAST_S, SYRX_LATENCY_S) - d)
           for v, d in SYRX_STOPS]
    # The fitted line's own residuals against these same stops average 1.8cm.
    # The sim is not allowed to be worse than the calibration it came from.
    assert np.mean(err) < 2.5, f"mean {np.mean(err):.2f}cm off: {np.round(err, 1)}"


@pytest.mark.parametrize("latency_s", [1 / 30, 0.2, 0.5])
def test_the_loop_delay_is_counted_once(latency_s):
    """The measured constant spans the delay AND the roll-out, because the cut
    position is a camera position. The sim already delays the command in its
    queue, so decaying by the whole constant on top stopped the ball a full
    `v * delay` too late — the bug this test exists to hold shut."""
    got = _roll_out(30.0, 0.6, latency_s)
    assert got == pytest.approx(30.0 * 0.6, rel=0.1), (
        f"{got:.1f}cm at {latency_s}s delay — should be 18cm whatever the split")


def test_a_coast_shorter_than_the_delay_still_rolls():
    """Two calibrations disagreeing must not produce a ball that stops dead."""
    assert _roll_out(30.0, 0.1, 0.5) > 12.0

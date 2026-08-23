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

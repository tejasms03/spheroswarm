"""Moving the robot's own zero, instead of carrying a correction forever.

`heading_offset` is added to every command for as long as the roster holds it,
and a Sphero establishes its heading reference when it CONNECTS — so a stored
offset is stale the moment the link drops. That is why re-measuring it never
stuck across a session.

`reset_aim` exists on the v1.2 protocol and this codebase had never used it. It
takes whichever way the drive assembly is currently pointing and calls that
zero, so the correction lives in the robot and there is no signed number left
to apply the wrong way round.
"""

import numpy as np
import pytest

from fleet.handle import velocity_to_command
from fleet.sim_handle import SimRobot
from workspace.space import Workspace

OPEN = Workspace(bounds_cm=[[0, 0], [4000, 0], [4000, 4000], [0, 4000]])


def _robot(bias_deg):
    r = SimRobot("A", "AAAA", "cyan", workspace=OPEN, randomize=False)
    r.slip = 0.0
    r.drift_rate = 0.0
    r.bias = np.radians(bias_deg)
    r.heading_offset = 0.0
    # The frame has to be measured as it IS. Leaving the live estimator on has
    # it folding a correction mid-measurement — which showed up here as a
    # steady direction that quietly changed at step 119.
    r.heading_tracking = False
    return r


def _travel(r, course_deg, settle=40, n=60):
    """Where it actually goes when told `course_deg`, in compass degrees."""
    rad = np.radians(course_deg)
    r.pos = np.array([2000.0, 2000.0])
    r.vel = np.zeros(2)
    r.set_velocity(np.array([np.sin(rad), np.cos(rad)]) * 20.0)
    for _ in range(settle):
        r.step(1 / 30)
    start = r.pos.copy()
    for _ in range(n):
        r.step(1 / 30)
    return velocity_to_command(r.pos - start)[0]


def _err(got, told):
    return abs((got - told + 180.0) % 360.0 - 180.0)


# -- the behaviour, which a sign error cannot satisfy ---------------------

@pytest.mark.parametrize("bias", [40.0, -70.0, 150.0, 5.0, -175.0, 179.0])
def test_after_zeroing_it_goes_where_it_is_told(bias):
    """One leg, then every course is right — including near a reversal, which
    is where wraparound and sign errors bite hardest and where every
    offset-based attempt in this project has failed."""
    r = _robot(bias)
    measured = (_travel(r, 0.0) - 0.0 + 180.0) % 360.0 - 180.0
    assert r.aim_zero(measured) is True

    for course in (0.0, 90.0, 200.0, 315.0):
        assert _err(_travel(r, course), course) < 1.0, (
            f"bias {bias}: told {course}, went elsewhere")


def test_it_leaves_no_offset_to_go_stale():
    """The whole point. A correction in the robot needs nothing applied on
    every command, so there is nothing to be stale or wrongly signed."""
    r = _robot(40.0)
    r.heading_offset = 144.3
    r.aim_zero(-40.0)
    assert r.heading_offset == 0.0


def test_one_leg_is_enough():
    """Contrast with the offset route, which needs legs on two axes before it
    can even tell a rotation from a mirror."""
    r = _robot(-70.0)
    measured = (_travel(r, 0.0) - 0.0 + 180.0) % 360.0 - 180.0
    r.aim_zero(measured)
    assert _err(_travel(r, 137.0), 137.0) < 1.0


# -- the real robot -------------------------------------------------------

class FakeApi:
    def __init__(self):
        self.calls = []
        self._SpheroEduAPI__speed = 200

    def set_heading(self, h):
        self.calls.append(("set_heading", h, self._SpheroEduAPI__speed))

    def reset_aim(self):
        self.calls.append(("reset_aim",))


def _sphero():
    from fleet.real_handle import SpheroRobot
    r = SpheroRobot("A", "AAAA", "cyan", "SK-1",
                    connector=lambda n, timeout=8: None, autostart=False)
    r._api = FakeApi()
    return r


def test_the_assembly_is_turned_at_zero_speed():
    """Speed zero turns the drive assembly without moving the ball — which is
    how aiming works on a Sphero, and why this needs no clear floor."""
    r = _sphero()
    r.aim_zero(-40.0)
    kind, heading, speed = r._api.calls[0]
    assert kind == "set_heading"
    assert speed == 0, "a non-zero speed would drive the ball across the room"
    assert heading == 40, "rotate by MINUS the measured error"
    assert r._api.calls[1] == ("reset_aim",), "then declare that forward"


def test_it_clears_the_stored_offset_and_the_deadband_memory():
    """The frame moved under the deadband: a command that looks identical to
    the last one now means something different, so the suppression has to go."""
    r = _sphero()
    r.heading_offset = 144.3
    r._last_sent = (10, 200)
    assert r.aim_zero(-40.0) is True
    assert r.heading_offset == 0.0
    assert r._last_sent is None


def test_a_robot_with_no_radio_says_so_rather_than_pretending():
    r = _sphero()
    r._api = None
    assert r.aim_zero(10.0) is False


def test_a_failing_radio_is_reported_not_swallowed():
    class Boom(FakeApi):
        def reset_aim(self):
            raise RuntimeError("link dropped")

    r = _sphero()
    r._api = Boom()
    r.heading_offset = 144.3          # something a failure must not clear
    assert r.aim_zero(10.0) is False
    assert "reset the aim" in (r.last_error or "")
    assert r.heading_offset == pytest.approx(144.3), (
        "a failed reset must leave the old correction in force, not zero it "
        "and drive with no correction at all")


def test_the_base_handle_declines_rather_than_erroring():
    """A robot with no aim to reset is not a failure — it is one the caller
    should keep correcting with an offset."""
    from fleet.handle import RobotHandle
    assert RobotHandle.aim_zero(object(), 30.0) is False

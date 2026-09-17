"""Turning on raw motor power, and driving on straight after.

`roll(heading, 0)` hands a Sphero a FINAL angle and its own controller slews
there flat out, which overshoots and cannot be slowed from outside. `spin()` in
spherov2 is no better — it is a loop of `set_heading` calls. `raw_motor` is the
one command that sets a rate, and it comes with a condition: it switches
stabilisation off, and a roll cannot run until it is back on.

So what can go wrong is almost entirely ORDER, and these tests are about order.
"""

from conftest import FakeApi
from fleet.real_handle import SpheroRobot


def handle(open_ws):
    api = FakeApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=lambda n, timeout=8.0: api, autostart=False)
    h._api = api
    h._link_up = True
    return h, api


def kinds(api):
    return [e[0] for e in api.events]


def test_a_spin_drives_the_motors_in_opposite_directions(open_ws):
    h, api = handle(open_ws)
    h.spin_raw(60)
    h._drain_once()
    assert ("raw", 60, -60, None) in api.events
    assert h.spinning and not api.stabilized


def test_a_spin_never_blocks_the_worker(open_ws):
    """duration None sets the motors and returns. Anything else makes the
    library sleep on this thread, holding the radio for every other write."""
    h, api = handle(open_ws)
    h.spin_raw(-40)
    h._drain_once()
    assert [e for e in api.events if e[0] == "raw"][-1][3] is None


def test_a_spin_zeroes_the_librarys_stored_speed_first(open_ws):
    """The library's own thread re-sends a roll every 0.8s while its stored
    speed is non-zero. Leave one there and it fights the spin."""
    h, api = handle(open_ws)
    api._SpheroEduAPI__speed = 90
    h.spin_raw(50)
    h._drain_once()
    assert api._SpheroEduAPI__speed == 0


def test_stopping_the_spin_restores_stabilisation(open_ws):
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    h.stop_raw()
    h._drain_once()
    assert api.stabilized, "a roll cannot run without it"
    assert api.motors == (0, 0)
    assert not h.spinning


def test_stop_also_stops_a_spin(open_ws):
    """A stop that leaves raw motors running is not a stop."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    h.stop()
    h._drain_once()
    assert api.motors == (0, 0) and api.stabilized
    assert not h.spinning


def test_stopping_zeroes_here_before_stabilising(open_ws):
    """With stabilisation on, a Sphero holds the heading of its last ROLL. Turn
    it back on straight after a spin and it can swing back to where it was,
    undoing the turn. Zeroing first leaves nothing to swing back to."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    api.events.clear()

    h.stop_raw()
    h._drain_once()
    order = kinds(api)
    assert "zero" in order, "it stopped without zeroing where it points"
    assert order.index("raw") < order.index("zero"), order
    assert h.heading_offset == 0.0, "the frame moved, so the old offset is void"


def test_the_api_flag_is_resynced_so_the_next_spin_is_not_fought(open_ws):
    """`reset_aim` restores stabilisation through the toy and leaves the API's
    flag reading off. The next `raw_motor` then thinks it is already off, does
    not switch it off, and drives the motors against the stabiliser."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    h.stop_raw()
    h._drain_once()
    assert api._SpheroEduAPI__stabilization is True

    api.fought = False
    h.spin_raw(50)
    h._drain_once()
    assert not api.fought, "the second spin ran against the stabiliser"
    assert not api.stabilized


def test_spin_stop_then_drive_goes_out_in_that_order(open_ws):
    """The turn-and-go handover: stop the spin — zeroing and stabilising —
    THEN roll straight ahead, which is heading 0 in the new frame."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    api.events.clear()

    h.stop_raw()
    h.drive_raw(0, 26)
    h._drain_once()
    order = kinds(api)
    assert order.index("zero") < order.index("heading"), order
    roll = [e for e in api.events if e[0] == "heading"][-1]
    assert roll[1] == 0, "straight ahead in the re-zeroed frame is heading 0"
    assert roll[3] is True, "the roll went out with stabilisation still off"


def test_a_stop_after_a_spin_holds_heading_zero_not_the_old_one(open_ws):
    """After a spin the stop re-zeroes, so the roll that holds the ball must be
    heading 0. The last roll's heading would rotate it back to before the
    turn."""
    h, api = handle(open_ws)
    h.drive_raw(200, 30)
    h._drain_once()
    h.spin_raw(50)
    h._drain_once()
    api.events.clear()
    h.stop()
    h._drain_once()
    rolls = [e for e in api.events if e[0] == "heading"]
    assert all(e[1] == 0 for e in rolls), rolls


def test_a_roll_queued_after_a_spin_wins(open_ws):
    """Latest wins, as for every other write. A roll issued while a spin is
    queued means stop spinning — dropping the roll would leave a ball that was
    told to drive sitting and spinning."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h.drive_raw(0, 26)
    h._drain_once()
    assert not h.spinning
    assert any(e[0] == "heading" for e in api.events), "the roll was dropped"
    assert not any(e[0] == "raw" and e[1] != 0 for e in api.events), \
        "it spun first even though the roll was the newer intent"


def test_a_roll_during_a_running_spin_stops_it(open_ws):
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    h.drive_raw(0, 26)
    h._drain_once()
    assert not h.spinning and api.stabilized
    assert api.events[-1][0] == "heading"


def test_a_spin_queued_after_a_roll_wins(open_ws):
    h, api = handle(open_ws)
    h.drive_raw(0, 26)
    h.spin_raw(50)
    h._drain_once()
    assert h.spinning
    assert not any(e[0] == "heading" for e in api.events)


def test_a_reconnect_forgets_the_spin(open_ws):
    """A reconnected ball comes up stabilised with its motors off."""
    h, api = handle(open_ws)
    h.spin_raw(50)
    h._drain_once()
    h._teardown()
    assert not h._raw_active


def test_stopping_a_spin_that_is_not_running_writes_nothing(open_ws):
    h, api = handle(open_ws)
    h.stop_raw()
    h._drain_once()
    assert not api.events

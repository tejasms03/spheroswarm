"""Fire-and-forget drive packets.

No physical ball has ever received one of these. What can be checked without
hardware is that the bytes are right and that the surrounding machinery behaves
— so the framing is pinned against spherov2's own builder, and every handle
test asserts what was *sent*, not merely that nothing raised.
"""

import threading

import numpy as np
import pytest

from fleet import sphero_fast
from fleet.real_handle import SpheroRobot
from fleet.sphero_fast import (SOP1, SOP2_ANSWER, SOP2_NO_ANSWER, FastWriter,
                               attach, build_request, checksum, roll_data)


# -- fakes ---------------------------------------------------------------

class FakePacketManager:
    """spherov2's own allocator behaviour: hand out a seq, then advance."""

    def __init__(self, start=0):
        self.seq = start

    def new_packet(self, did, cid, proc, data=None):
        class P:
            pass
        p = P()
        p.seq = self.seq
        self.seq = (self.seq + 1) % 0x100
        return p


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeToy:
    """Enough of a v1.2 `Toy` to frame packets against."""

    def __init__(self, sop=SOP1):
        class Packet:
            SOP = sop
        self._packet = Packet
        self._packet_manager = FakePacketManager()
        self._Toy__packet_queue = FakeQueue()

    @property
    def queue(self):
        return self._Toy__packet_queue


class FakeFastApi:
    """A SpheroEduAPI with a reachable toy, and the two cached values the
    library's keepalive thread re-sends."""

    def __init__(self, toy=None):
        self._SpheroEduAPI__toy = toy if toy is not None else FakeToy()
        self._SpheroEduAPI__speed = 0
        self._SpheroEduAPI__heading = 0
        self._SpheroEduAPI__updating = threading.RLock()
        self.set_heading_calls = []
        self.set_speed_calls = []
        self.closed = False

    # If the fast path is working these must never be reached.
    def set_heading(self, h):
        self.set_heading_calls.append(h)

    def set_speed(self, s):
        self._SpheroEduAPI__speed = s
        self.set_speed_calls.append(s)

    def set_main_led(self, color):
        pass

    def __exit__(self, *a):
        self.closed = True

    @property
    def toy(self):
        return self._SpheroEduAPI__toy


def _connector(api):
    return lambda ble_name, timeout=8.0: api


# -- framing -------------------------------------------------------------

@pytest.mark.parametrize("seq", [0, 1, 7, 128, 255])
@pytest.mark.parametrize("speed,heading", [(0, 0), (128, 90), (255, 359), (60, 180)])
def test_acked_framing_is_byte_identical_to_spherov2(seq, speed, heading):
    """The only available evidence that the no-answer packet is framed right.

    Both packets come off the same builder and differ in one byte, so agreeing
    with the library on the case it can produce is what stands behind the case
    it cannot.
    """
    from spherov2.controls.v1 import Packet

    data = roll_data(speed, heading)
    theirs = Packet.Request(did=2, cid=48, seq=seq, data=bytearray(data)).build()
    mine = build_request(2, 48, seq, data, answer=True)
    assert bytes(mine) == bytes(theirs)


def test_no_answer_packet_differs_only_in_sop2():
    data = roll_data(128, 90)
    ack = build_request(2, 48, 7, data, answer=True)
    fast = build_request(2, 48, 7, data, answer=False)

    assert ack[1] == SOP2_ANSWER
    assert fast[1] == SOP2_NO_ANSWER
    assert fast[0] == ack[0] == SOP1
    # The checksum covers the body only, so clearing the answer bit does not
    # change it. If that ever stops being true the packet is malformed.
    assert bytes(fast[2:]) == bytes(ack[2:])


def test_answer_bit_is_the_low_bit():
    assert SOP2_ANSWER & 0x01 == 1
    assert SOP2_NO_ANSWER & 0x01 == 0
    assert SOP2_NO_ANSWER & 0x02 == 2, "the inactivity-timeout bit must stay set"


def test_checksum_matches_the_library_rule():
    from spherov2.helper import packet_chk

    body = bytearray([2, 48, 7, 6, 128, 0, 90, 1, 0])
    assert checksum(body) == packet_chk(body)


def test_roll_data_encodes_heading_big_endian():
    assert list(roll_data(100, 0x0102 % 360)) == list(roll_data(100, 258))
    d = roll_data(100, 300)
    assert d[0] == 100
    assert (d[1] << 8) | d[2] == 300
    assert d[3] == sphero_fast.ROLL_GO
    assert d[4] == sphero_fast.REVERSE_OFF


def test_roll_data_clamps_and_wraps():
    assert roll_data(999, 0)[0] == 255
    assert roll_data(-5, 0)[0] == 0
    d = roll_data(10, 725)                       # 725 % 360 == 5
    assert (d[1] << 8) | d[2] == 5


# -- the writer ----------------------------------------------------------

def test_writer_enqueues_one_packet_per_roll():
    toy = FakeToy()
    w = FastWriter(toy, toy.queue, toy._packet_manager)
    w.roll(128, 90)
    assert len(toy.queue.items) == 1
    assert toy.queue.items[0][1] == SOP2_NO_ANSWER
    assert w.sent == 1


def test_writer_takes_sequence_numbers_from_the_shared_allocator():
    """Acked traffic waits on `(SOP, seq)`. A privately minted number would
    eventually collide with one in flight and steal that caller's reply."""
    toy = FakeToy()
    w = FastWriter(toy, toy.queue, toy._packet_manager)
    for _ in range(3):
        w.roll(100, 0)
    seqs = [p[4] for p in toy.queue.items]
    assert seqs == [0, 1, 2]
    assert toy._packet_manager.seq == 3


def test_writer_stop_sends_zero_speed_in_stop_mode():
    toy = FakeToy()
    w = FastWriter(toy, toy.queue, toy._packet_manager)
    w.stop(heading=45)
    p = toy.queue.items[0]
    assert p[6] == 0, "speed byte"
    assert p[9] == sphero_fast.ROLL_STOP


def test_writer_led_uses_the_led_command():
    toy = FakeToy()
    w = FastWriter(toy, toy.queue, toy._packet_manager)
    w.set_main_led(10, 20, 30)
    p = toy.queue.items[0]
    assert p[3] == sphero_fast.CID_SET_MAIN_LED
    assert list(p[6:9]) == [10, 20, 30]


# -- attach, and its refusals -------------------------------------------

def test_attach_returns_a_writer_for_a_v1_toy():
    api = FakeFastApi()
    assert attach(api) is not None


def test_attach_refuses_a_non_v1_toy():
    api = FakeFastApi(toy=FakeToy(sop=0x8D))     # v2 start byte
    assert attach(api) is None


def test_attach_refuses_a_toy_with_no_reachable_queue():
    toy = FakeToy()
    del toy._Toy__packet_queue
    assert attach(FakeFastApi(toy=toy)) is None


def test_attach_never_raises_on_a_surprising_object():
    """A library upgrade renaming a private attribute must cost latency, not
    the ability to drive."""
    assert attach(object()) is None
    assert attach(None) is None


# -- the handle ----------------------------------------------------------

def test_fast_writes_are_off_by_default(open_ws):
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False)
    assert h.fast_writes is False
    h._connect_once()
    assert h._fast is None


def test_acked_path_is_used_when_fast_writes_are_off(open_ws):
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False)
    h._connect_once()
    h._write(90, 128)
    assert api.set_heading_calls == [90]
    assert api.toy.queue.items == []


def test_fast_path_sends_one_unacknowledged_packet(open_ws):
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()
    assert h._fast is not None

    h._write(90, 128)

    assert api.set_heading_calls == [], "the blocking setter must not be reached"
    assert len(api.toy.queue.items) == 1
    p = api.toy.queue.items[0]
    assert p[1] == SOP2_NO_ANSWER
    assert p[3] == sphero_fast.CID_ROLL
    assert p[6] == 128
    assert (p[7] << 8) | p[8] == 90


def test_fast_path_keeps_the_keepalive_cache_in_sync(open_ws):
    """The bug this guards: spherov2 re-sends `roll_start(__heading, __speed)`
    every 0.8s whenever the speed is non-zero. Bypassing `set_heading` leaves
    those two values at whatever they were, so an unsynced fast path would have
    the library re-aiming the ball at a stale heading about once a second.
    """
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()

    h._write(270, 200)

    assert api._SpheroEduAPI__heading == 270
    assert api._SpheroEduAPI__speed == 200


def test_fast_path_does_not_take_the_library_lock(open_ws):
    """Holding `__updating` would reintroduce the 230ms stall this exists to
    remove, because the keepalive holds it across an acked write."""
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()

    held = threading.Event()
    done = threading.Event()

    def hold():
        with api._SpheroEduAPI__updating:
            held.set()
            done.wait(2.0)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert held.wait(1.0)
    try:
        h._write(90, 128)                        # must not block on the lock
        assert len(api.toy.queue.items) == 1
    finally:
        done.set()
        t.join(1.0)


def test_falling_back_still_drives(open_ws):
    """Fast writes asked for, but the link cannot carry them: the robot must
    still be driven, on the acked path."""
    api = FakeFastApi(toy=FakeToy(sop=0x8D))
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()

    assert h._fast is None
    h._write(90, 128)
    assert api.set_heading_calls == [90]


def test_state_reports_whether_fast_writes_attached(open_ws):
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    assert h.state()["fast_writes"] is False, "asked for, not yet attached"
    h._connect_once()
    assert h.state()["fast_writes"] is True


def test_writer_is_dropped_on_teardown(open_ws):
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()
    assert h._fast is not None
    h._teardown()
    assert h._fast is None, "a writer holding a closed toy must not survive"


def test_fast_writes_are_paced_to_the_writer_thread(open_ws):
    """With the ack gone the measured round trip collapses to an enqueue, and
    pacing off it alone would queue commands faster than the library drains
    them. `cmd_safe_interval` is what the ack was standing in for."""
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False, fast_writes=True)
    h._connect_once()

    for _ in range(5):
        h._write(90, 128)

    assert h.rtt_mean is not None
    assert h.rtt_mean < sphero_fast.MIN_WRITE_GAP, (
        "an enqueue that took as long as a round trip is not fire-and-forget")
    assert h._pace_gap() >= sphero_fast.MIN_WRITE_GAP


def test_acked_pacing_still_follows_the_round_trip(open_ws):
    """The floor must not become a ceiling: a link measured slower than
    `cmd_safe_interval` still has to be waited for."""
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False)
    h._connect_once()
    assert h._pace_gap() == 0.0, "nothing measured yet"

    h.rtt_mean = 0.23
    assert h._pace_gap() == pytest.approx(0.253)

    h.fast_writes = True
    h._fast = attach(api)
    assert h._pace_gap() == pytest.approx(0.253), "a slow link still paces itself"


def test_velocity_still_reaches_the_radio_with_fast_writes(open_ws):
    """End to end through the public API: a commanded velocity becomes a
    packet, with the heading offset applied exactly once."""
    api = FakeFastApi()
    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=_connector(api), autostart=False,
                    fast_writes=True, heading_offset=90.0)
    h._connect_once()

    h.set_velocity(np.array([0.0, 30.0]))        # +y is heading 0, +90 offset
    h._write(*h._pending)

    p = api.toy.queue.items[0]
    assert (p[7] << 8) | p[8] == 90


# -- the A/B a hardware session actually runs ---------------------------

def test_compare_write_paths_reports_both_and_the_ratio(open_ws):
    from fleet.sensors import compare_write_paths

    api = FakeFastApi()
    out = compare_write_paths(api, samples=5)

    assert out["acked"]["writes"] == 5
    assert out["fast"]["writes"] == 5
    assert "speedup" in out, "a near-zero fast write is a result, not a missing one"
    assert len(api.toy.queue.items) == 5, "the fast half must reach the queue"
    assert len(api.set_heading_calls) == 5, "the acked half must not"


def test_compare_write_paths_says_so_when_fast_cannot_attach():
    from fleet.sensors import compare_write_paths

    api = FakeFastApi(toy=FakeToy(sop=0x8D))
    out = compare_write_paths(api, samples=3)

    assert "error" in out["fast"]
    assert out["acked"]["writes"] == 3, "the acked number is still worth having"
    assert "speedup" not in out


def test_the_ab_carries_the_caveat_that_matters():
    """An enqueue is not obedience. The one way this change fails silently is
    a packet the robot ignores, which from here looks like a total success."""
    from fleet.sensors import compare_write_paths

    out = compare_write_paths(FakeFastApi(), samples=3)
    assert "camera" in out["note"]
    assert out["ceiling_hz"] == pytest.approx(1.0 / sphero_fast.MIN_WRITE_GAP, rel=0.01)

"""Fire-and-forget drive commands, for the Sphero API v1.2.

Every command `spherov2` sends asks the robot to acknowledge it, and then
blocks until it does. Measured on this hardware that round trip is ~230ms —
larger than the motor lag and the camera put together, and the single biggest
number in the control problem.

It is not a property of the robot, and not of Python. In the v1.2 protocol the
second start-of-packet byte is a bitfield, not a constant:

    SOP2 = 1111 11RA      R = reset the inactivity timeout
                          A = answer requested

With A clear the robot acts on the command and says nothing back, and the
sequence number is ignored. `spherov2` hardcodes both SOP bytes to 0xFF in
`Packet.Request.build`, so the option is there on the wire and simply not
reachable through the library. That is what this module reaches past.

**What this does not buy.** The ack is not the only cost. `Toy.__process_packet`
sleeps `toy_type.cmd_safe_interval` after every write — 60ms on a SPRK+ — so
the ceiling moves from ~4Hz to ~16Hz, not to infinity. Anything that paces
itself off the measured round trip has to grow a floor to match; see
`MIN_WRITE_GAP` and its use in `real_handle`.

**Nothing here has been verified against a physical ball.** It is written from
the protocol document and the library source. Every entry point degrades to the
acked path rather than raising, `attach()` returns None on anything unexpected,
and the whole thing is off unless switched on. The first lab session with a
robot should A/B the round trip with the bench's `sensors` probe before
trusting a single number that comes out of it.
"""

import logging
import threading

log = logging.getLogger("fleet.fast")

SOP1 = 0xFF
SOP2_ANSWER = 0xFF          # answer requested + reset inactivity timeout
SOP2_NO_ANSWER = 0xFE       # reset inactivity timeout only — the whole point

DID_SPHERO = 0x02           # the "Sphero" device, per commands/sphero.py
CID_SET_MAIN_LED = 32
CID_ROLL = 48

ROLL_STOP, ROLL_GO, ROLL_CALIBRATE = 0, 1, 2
REVERSE_OFF, REVERSE_ON = 0, 1

# The writer thread's own pause between packets. Fire-and-forget removes the
# ack, not this, so it is the real floor on command rate.
MIN_WRITE_GAP = 0.06


def checksum(body):
    """The v1.2 checksum: ones' complement of the sum, over everything but the
    two SOP bytes. Same rule `spherov2.helper.packet_chk` applies."""
    return 0xFF - (sum(body) & 0xFF)


def build_request(did, cid, seq, data=(), answer=False):
    """One v1.2 request packet.

    With `answer=True` this is byte-identical to what `spherov2` would have
    built for the same arguments — a test pins that, because agreeing on the
    ordinary case is the only evidence available that the unusual one is
    framed correctly too.
    """
    data = bytearray(data)
    body = bytearray([did, cid, seq & 0xFF, len(data) + 1, *data])
    sop2 = SOP2_ANSWER if answer else SOP2_NO_ANSWER
    return bytearray([SOP1, sop2, *body, checksum(body)])


def roll_data(speed, heading, mode=ROLL_GO, reverse=REVERSE_OFF):
    """Payload for CID 48: speed, heading big-endian, mode, reverse flag."""
    speed = max(0, min(255, int(speed)))
    heading = int(heading) % 360
    return bytearray([speed, (heading >> 8) & 0xFF, heading & 0xFF,
                      int(mode), int(reverse)])


class FastWriter:
    """Enqueues drive packets on a live toy without waiting for a reply.

    Deliberately narrow: roll and the main LED, the only two things the control
    loop sends often enough for the round trip to matter. Everything else keeps
    going through the library, acked, where a lost packet would be a silent
    misconfiguration rather than one late frame.
    """

    def __init__(self, toy, queue, manager):
        self._toy = toy
        self._queue = queue
        self._manager = manager
        self._lock = threading.Lock()
        self.sent = 0

    def _next_seq(self):
        """Borrow the library's own allocator.

        The robot ignores the sequence number on a no-answer packet, so any
        value would drive correctly. It still has to come from here: acked
        traffic — the keepalive, sensor reads, the LED — is keyed on
        `(SOP, seq)` while it waits, and a number minted independently would
        eventually collide with one in flight and hand that caller our reply.
        """
        return self._manager.new_packet(DID_SPHERO, 0, None, b"").seq

    def _send(self, cid, data):
        payload = build_request(DID_SPHERO, cid, self._next_seq(), data)
        with self._lock:
            self._queue.put(payload)
            self.sent += 1
        return payload

    def roll(self, speed, heading, mode=ROLL_GO, reverse=REVERSE_OFF):
        """Speed and heading in one packet, which is what `roll_start` sends."""
        return self._send(CID_ROLL, roll_data(speed, heading, mode, reverse))

    def stop(self, heading=0):
        return self._send(CID_ROLL, roll_data(0, heading, ROLL_STOP))

    def set_main_led(self, r, g, b):
        return self._send(CID_SET_MAIN_LED,
                          bytearray([int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF]))


def _unwrap(api):
    """Find the `Toy` under a `SpheroEduAPI`, or the toy itself if handed one."""
    for attr in ("_SpheroEduAPI__toy", "toy", "_toy"):
        toy = getattr(api, attr, None)
        if toy is not None:
            return toy
    return api


def attach(api):
    """A `FastWriter` for this connection, or None if the link is not one we
    know how to frame packets for.

    None is not an error — it is the answer "keep using the acked path". The
    private attributes reached for here are exactly the kind of thing a library
    upgrade renames, and a control loop that stops driving because a writer
    could not be built would be a far worse failure than a slow one.
    """
    try:
        toy = _unwrap(api)

        packet = getattr(toy, "_packet", None)
        if packet is None or getattr(packet, "SOP", None) != SOP1:
            log.warning("fast writes off: %s is not API v1.2", type(toy).__name__)
            return None

        queue = getattr(toy, "_Toy__packet_queue", None)
        manager = getattr(toy, "_packet_manager", None)
        if queue is None or not hasattr(queue, "put") or manager is None:
            log.warning("fast writes off: no reachable packet queue on %s",
                        type(toy).__name__)
            return None

        writer = FastWriter(toy, queue, manager)
        log.info("fast writes on for %s — commands will not be acknowledged", toy)
        return writer
    except Exception as e:                      # pragma: no cover - defensive
        log.warning("fast writes off: %s", e)
        return None

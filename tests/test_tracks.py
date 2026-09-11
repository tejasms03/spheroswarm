"""Identity handed over once, and every way it can quietly go wrong.

Colour identity is re-read each frame, so it cannot drift. This cannot say the
same, so the tests here are mostly about the moment it breaks being VISIBLE —
a swap that is merely unlikely is a swap that happens on the day it matters.
"""

import math

import pytest

from vision.tracks import FLIP_COOLDOWN_S, Tracks, wrap180


def cluster(cx, cy, deg=0.0, span=40.0):
    """A tail and a tag blob, `span` apart, pointing `deg`."""
    r = math.radians(deg)
    ux, uy = math.cos(r), math.sin(r)
    back = {"x": cx - ux * span / 2, "y": cy - uy * span / 2,
            "area": 30.0, "peak": 255.0}
    front = {"x": cx + ux * span / 2, "y": cy + uy * span / 2,
             "area": 60.0, "peak": 255.0}
    return {"centre": (cx, cy), "group": [back, front],
            "front": front, "back": back}


def bare(cx, cy, deg=0.0, span=40.0):
    """The same, with no front/back call — only a line."""
    c = cluster(cx, cy, deg, span)
    c["front"] = c["back"] = None
    return c


def test_a_click_sets_which_end_is_the_front():
    t = Tracks()
    c = cluster(100, 100, deg=0.0)
    # Click the RIGHT-hand light: the heading points right.
    got, why = t.assign("A", c, front_px=(120, 100))
    assert why is None
    assert abs(wrap180(got.heading - 0.0)) < 1.0

    # The same cluster, clicking the LEFT light: the heading reverses.
    got, why = t.assign("B", c, front_px=(80, 100))
    assert abs(wrap180(got.heading - 180.0)) < 1.0


def test_a_rough_click_still_gives_a_precise_heading():
    """The precision comes from the axis, not from the finger."""
    t = Tracks()
    c = cluster(100, 100, deg=37.0)
    got, _ = t.assign("A", c, front_px=(150, 160))     # nowhere near the light
    assert abs(wrap180(got.heading - 37.0)) < 1.0


def test_the_front_is_carried_through_a_turn():
    """The blobs move as the ball turns and are detected afresh each frame, so
    what is remembered is a DIRECTION, not a blob."""
    t = Tracks()
    t.assign("A", cluster(100, 100, deg=0.0), front_px=(120, 100))
    for deg in range(0, 200, 10):          # a slow half turn
        t.update([cluster(100, 100, deg=deg)], dt=1 / 30.0)
    got = t.by_name["A"]
    assert got.live
    assert abs(wrap180(got.heading - 190.0)) < 12.0, got.heading


def test_an_axis_with_no_front_call_is_still_carried():
    """The reader may refuse to say which end is which. Continuity does not
    need it to — a line plus last frame's heading is enough."""
    t = Tracks()
    t.assign("A", bare(100, 100, deg=0.0), front_px=(120, 100))
    for deg in (0, 15, 30, 45, 60):
        t.update([bare(100, 100, deg=deg)], dt=1 / 30.0)
    assert abs(wrap180(t.by_name["A"].heading - 60.0)) < 5.0


def test_the_heading_cannot_flip_while_tracking_holds():
    """A flip is 180 degrees; a ball turns about 12 between frames. Feeding
    the axis in reversed must NOT reverse the belief."""
    t = Tracks()
    t.assign("A", cluster(100, 100, deg=0.0), front_px=(120, 100))
    for _ in range(30):
        # The same line, handed over pointing the other way each time.
        t.update([bare(100, 100, deg=180.0)], dt=1 / 30.0)
    assert abs(wrap180(t.by_name["A"].heading)) < 5.0, "it turned round"


def test_a_track_that_leaves_the_gate_is_lost_not_moved():
    """A stale dot a controller cannot tell from a fresh one is the one
    failure that controller cannot defend against."""
    t = Tracks()
    t.assign("A", cluster(100, 100), front_px=(120, 100))
    t.update([cluster(900, 900)], dt=1 / 30.0)
    got = t.by_name["A"]
    assert got.lost and not got.live
    assert got.centre == (100.0, 100.0), "it must not have teleported"
    assert "gate" in got.why


def test_an_empty_frame_loses_the_track_out_loud():
    t = Tracks()
    t.assign("A", cluster(100, 100), front_px=(120, 100))
    t.update([], dt=1 / 30.0)
    assert t.by_name["A"].lost
    assert "nothing lit" in t.by_name["A"].why


def test_two_tracks_reaching_for_one_cluster_move_neither():
    """Picking the closer one is how a swap becomes permanent."""
    t = Tracks()
    t.assign("A", cluster(100, 100), front_px=(120, 100))
    t.assign("B", cluster(130, 100), front_px=(150, 100))
    t.update([cluster(115, 100)], dt=1 / 30.0)     # one blob between them
    a, b = t.by_name["A"], t.by_name["B"]
    assert not (a.live and b.live), "both cannot own the same cluster"
    contested = [x for x in (a, b) if x.contended]
    assert contested, "and the clash has to be named"
    assert "same cluster" in contested[0].why


def test_two_separated_robots_keep_their_own_clusters():
    t = Tracks()
    t.assign("A", cluster(100, 100), front_px=(120, 100))
    t.assign("B", cluster(500, 400), front_px=(520, 400))
    t.update([cluster(505, 402), cluster(104, 101)], dt=1 / 30.0)
    assert t.by_name["A"].live and t.by_name["B"].live
    assert t.by_name["A"].centre[0] == pytest.approx(104, abs=1)
    assert t.by_name["B"].centre[0] == pytest.approx(505, abs=1)


def test_association_does_not_depend_on_the_order_tracks_were_added():
    a_first, b_first = Tracks(), Tracks()
    for t, order in ((a_first, ("A", "B")), (b_first, ("B", "A"))):
        spots = {"A": (100, 100), "B": (400, 300)}
        for name in order:
            t.assign(name, cluster(*spots[name]), front_px=None)
        t.update([cluster(402, 301), cluster(101, 99)], dt=1 / 30.0)
    assert (a_first.by_name["A"].centre == b_first.by_name["A"].centre)
    assert (a_first.by_name["B"].centre == b_first.by_name["B"].centre)


def test_travel_turns_a_reversed_heading_back_round():
    """A Sphero rolls the way it points, so a ball that has covered ground has
    voted on which way it faces."""
    t = Tracks()
    # Seeded pointing LEFT, then driven steadily right.
    t.assign("A", cluster(100, 100, deg=180.0), front_px=(80, 100))
    assert abs(wrap180(t.by_name["A"].heading - 180.0)) < 1.0
    for x in range(110, 260, 10):
        t.update([bare(x, 100, deg=0.0)], dt=1 / 30.0)
    got = t.by_name["A"]
    assert abs(wrap180(got.heading - 0.0)) < 10.0, got.heading
    assert got.flips_caught >= 1, "and it said so"


def test_a_short_hop_does_not_get_a_vote():
    """A turn in place is movement that says nothing about facing."""
    t = Tracks()
    t.assign("A", cluster(100, 100, deg=0.0), front_px=(120, 100))
    t.update([bare(103, 100, deg=0.0)], dt=1 / 30.0)
    assert t.by_name["A"].confirmed_at is None


def test_the_gate_grows_with_the_time_between_frames():
    """A dropped frame is not a lost robot."""
    slow = Tracks()
    slow.assign("A", cluster(100, 100), front_px=(120, 100))
    slow.update([cluster(340, 100)], dt=0.5)          # half a second of travel
    assert slow.by_name["A"].live, "a long gap must widen the gate"

    quick = Tracks()
    quick.assign("A", cluster(100, 100), front_px=(120, 100))
    quick.update([cluster(340, 100)], dt=1 / 30.0)
    assert quick.by_name["A"].lost, "the same jump in one frame is not motion"


def test_dropping_one_track_leaves_the_others():
    t = Tracks()
    t.assign("A", cluster(100, 100), front_px=None)
    t.assign("B", cluster(400, 400), front_px=None)
    t.drop("A")
    assert t.names == ["B"]
    t.drop()
    assert t.names == []


def test_a_cluster_with_one_light_cannot_be_assigned():
    t = Tracks()
    lone = {"centre": (100, 100), "group": [{"x": 100, "y": 100,
                                             "area": 9.0, "peak": 255.0}]}
    got, why = t.assign("A", lone, front_px=(110, 100))
    assert got is None and "axis" in why


# -- movement that is not travel ------------------------------------------

def settle(x0, y0, amp=45.0, n=60, period=20, decay=60.0):
    """A ball coming to rest: overshoot, roll back, overshoot less.

    The LEDs ride the internal chassis, which stays put, so the FACING never
    changes while the centre swings back and forth across it.
    """
    return [(x0 + amp * math.sin(2 * math.pi * i / period) * math.exp(-i / decay),
             y0) for i in range(n)]


def test_a_ball_rocking_itself_to_a_standstill_is_not_a_flip():
    """The bug this guards: every rebound reads as "you are pointing the wrong
    way". Measured before the fix, a settling ball that never turned flipped
    the heading four times and sat 180 out for most of the settle."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    for x, y in settle(400, 300):
        t.update([bare(x, y, deg=0.0)], dt=1 / 30.0)
        assert abs(wrap180(t.by_name["A"].heading)) < 5.0, "it turned round"
    assert t.by_name["A"].flips_caught == 0


def test_a_rebound_never_builds_enough_votes_to_turn_the_ball_round():
    """The first outbound swing IS travel and may vote — it agrees, so it does
    no harm. What must never happen is the rebounds stacking up against."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    worst = 0
    for x, y in settle(400, 300):
        t.update([bare(x, y, deg=0.0)], dt=1 / 30.0)
        worst = max(worst, t.by_name["A"]._against)
    assert worst < 3, f"a settle reached {worst} votes against"
    assert t.by_name["A"].flips_caught == 0


def test_sustained_driving_still_turns_a_reversed_heading_round():
    """The guard must not cost the thing it guards. A heading that really is
    180 out does not go away, so it survives the repeat rule."""
    t = Tracks()
    t.assign("A", bare(100, 300, deg=180.0), front_px=(80, 300))
    assert abs(wrap180(t.by_name["A"].heading - 180.0)) < 1.0
    for x in range(110, 400, 8):
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0)
    got = t.by_name["A"]
    assert abs(wrap180(got.heading)) < 10.0, got.heading
    assert got.flips_caught == 1


def test_one_disagreeing_burst_is_not_enough():
    """A hand pushing the ball, or a bounce off a wall, moves it in a
    direction it is not facing — briefly."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    for x in range(395, 355, -8):           # shoved backwards, then stops
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0)
    for _ in range(10):
        t.update([bare(360, 300, deg=0.0)], dt=1 / 30.0)
    assert t.by_name["A"].flips_caught == 0, "a shove is not a heading"
    assert abs(wrap180(t.by_name["A"].heading)) < 5.0


def test_a_gentle_curve_still_counts_as_travel():
    """Pure pursuit drives arcs, not straight lines. The straightness rule has
    to pass a real path or motion never confirms anything."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    seen = []
    for i in range(40):
        a = math.radians(i * 2.0)           # a slow arc
        x, y = 400 + i * 7 * math.cos(a), 300 + i * 7 * math.sin(a)
        t.update([bare(x, y, deg=i * 2.0)], dt=1 / 30.0)
        seen.append(t.by_name["A"].travelling)
    assert any(seen), "a curve must still be able to vote"
    assert t.by_name["A"].flips_caught == 0


def test_standing_perfectly_still_never_votes():
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    for _ in range(30):
        t.update([bare(400, 300, deg=0.0)], dt=1 / 30.0)
    got = t.by_name["A"]
    assert not got.travelling and got.confirmed_at is None
    assert got.flips_caught == 0


# -- letting the drive settle it ------------------------------------------

def drive(t, name, frm, to, target, steps=40, deg=None, dt=1 / 30.0):
    """Walk a ball from `frm` toward `to` while being driven at `target`."""
    fx, fy = frm
    tx, ty = to
    for i in range(1, steps + 1):
        k = i / steps
        x, y = fx + (tx - fx) * k, fy + (ty - fy) * k
        heading = deg if deg is not None else 0.0
        t.update([bare(x, y, deg=heading)], dt=dt, targets={name: target})


def test_driving_away_from_the_setpoint_turns_the_heading_round():
    """The condition `calib.py` already recognises and refuses on — "drove
    12cm AWAY from the middle, that is the aim frame being wrong". The only
    new idea is repairing it instead of giving up."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    # Being driven toward x=700, and going the other way.
    drive(t, "A", (400, 300), (150, 300), target=(700, 300), steps=40)
    got = t.by_name["A"]
    assert got.flips_caught >= 1, "it never noticed"
    assert abs(wrap180(got.heading - 180.0)) < 5.0
    assert "AWAY" in (got.why or "") or "turned" in (got.why or "")


def test_closing_on_the_setpoint_never_turns_anything_round():
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    drive(t, "A", (400, 300), (650, 300), target=(700, 300), steps=40)
    got = t.by_name["A"]
    assert got.flips_caught == 0
    assert got.confirmed_at is not None, "closing counts as confirmation"


def test_moving_sideways_to_the_setpoint_is_not_a_reversal():
    """A ball crabbing past its target is not a ball pointing backwards, and
    a 180 would not fix it if it were."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=90.0), front_px=(400, 320))
    drive(t, "A", (400, 300), (400, 560), target=(700, 430), deg=90.0)
    assert t.by_name["A"].flips_caught == 0


def test_a_ball_stuck_against_something_is_not_turned_round():
    """Stalled is not reversed: the distance stops falling but does not grow."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    for _ in range(60):
        t.update([bare(401, 300, deg=0.0)], dt=1 / 30.0,
                 targets={"A": (700, 300)})
    assert t.by_name["A"].flips_caught == 0


def test_it_gives_up_rather_than_spinning_when_flipping_does_not_help():
    """If turning round twice has not fixed the closing distance the fault is
    not a reversal — a wrong scale or a wrong arena look identical, and
    neither is cured by aiming the other way."""
    t = Tracks()
    t.assign("A", bare(4000, 300, deg=0.0), front_px=(4020, 300))
    x, target = 4000.0, (7000, 300)
    said = []
    # Receding steadily and never teleporting, so the gate is never the reason
    # anything happens. Whatever it is told, it keeps going the wrong way.
    for _ in range(400):
        x -= 6.0
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0, targets={"A": target})
        if t.by_name["A"].why:
            said.append(t.by_name["A"].why)
    got = t.by_name["A"]
    assert got.flips_caught <= 2, f"it flipped {got.flips_caught} times"
    assert any("not a reversed heading" in w for w in said), said[-1:]


def test_a_flip_is_given_time_to_prove_itself():
    t = Tracks()
    t.assign("A", bare(4000, 300, deg=0.0), front_px=(4020, 300))
    x, target = 4000.0, (7000, 300)
    first, at = 0, None
    while first == 0 and x > 3000:
        x -= 6.0
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0, targets={"A": target})
        first, at = t.by_name["A"].flips_caught, t.by_name["A"]._last_flip_at
    assert first == 1 and at is not None, "it never turned round once"

    # Still receding, but inside the cooldown: it must hold its nerve.
    flipped_at = t.t
    while t.t - flipped_at < FLIP_COOLDOWN_S * 0.8:
        x -= 6.0
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0, targets={"A": target})
    assert t.by_name["A"].flips_caught == 1, "it flipped again too soon"


def test_the_setpoint_beats_the_travel_heuristic_when_both_are_there():
    """With a target present the straightness rule is not consulted at all: a
    ball that wobbles its way toward its setpoint is doing fine."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    for i in range(40):
        x = 400 + i * 6 + (8 if i % 2 else -8)      # closing, but jittering
        t.update([bare(x, 300, deg=0.0)], dt=1 / 30.0,
                 targets={"A": (700, 300)})
    assert t.by_name["A"].flips_caught == 0
    assert t.by_name["A"].confirmed_at is not None


def test_a_full_turn_in_place_is_followed_through_every_quadrant():
    """What a bench yaw is FOR. A reader is easy to fool with a ball that only
    ever sits at one of four angles; the thing to watch is whether it follows
    smoothly across the seams, and whether the ends change places on the way."""
    t = Tracks()
    t.assign("A", bare(400, 300, deg=0.0), front_px=(420, 300))
    deg, worst = 0.0, 0.0
    for _ in range(int(360 / 80 * 30) + 1):
        deg = (deg + 80.0 / 30.0) % 360.0
        t.update([bare(400, 300, deg=deg)], dt=1 / 30.0)
        worst = max(worst, abs(wrap180(t.by_name["A"].heading - deg)))
    assert worst < 2.0, f"drifted {worst:.1f}deg over a full turn"
    assert t.by_name["A"].flips_caught == 0, "a ball turning in place has not moved"


def test_turning_while_driving_is_followed_too():
    t = Tracks()
    t.assign("B", bare(200, 300, deg=0.0), front_px=(220, 300))
    x, y, deg, worst = 200.0, 300.0, 0.0, 0.0
    for _ in range(120):
        deg = (deg + 80.0 / 30.0) % 360.0
        x += 5 * math.cos(math.radians(deg))
        y += 5 * math.sin(math.radians(deg))
        t.update([bare(x, y, deg=deg)], dt=1 / 30.0)
        worst = max(worst, abs(wrap180(t.by_name["B"].heading - deg)))
    assert worst < 2.0, f"drifted {worst:.1f}deg while driving a circle"
    assert t.by_name["B"].flips_caught == 0

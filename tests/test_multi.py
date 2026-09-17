"""Several robots on one bench, each with its own drive.

The selected robot's drive lives on the app exactly where the single-robot
bench kept it; every other robot's is swapped in for its own `drive` call (see
`BlobTest.as_bot`). What these pin is that the swap is complete -- a piece of
drive state missed by it is not an error anywhere, it is two robots silently
sharing one thing -- and that one robot still takes the path it always did.
"""

import ast
import pathlib
import time

import numpy as np
import pygame
import pytest

import coast_test as F

THREE = "SYRX,VHGR,MLYS"


def _tick(app, seconds, until=None):
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.01)
        app.tick()
        if until is not None and until():
            return True
    return False


@pytest.fixture
def three():
    app = F.BlobTest("sim", code=THREE)
    try:
        _tick(app, 0.5)
        yield app
    finally:
        app.close()


def _identified(app):
    app.start_reid()
    assert _tick(app, 5.0, lambda: app.reid is None), "roll call never finished"
    _tick(app, 0.2)


def _arm_toward(app, code, dy=25.0):
    app.select(code)
    here = np.asarray(app.fleet.handles[code].pos, dtype=float)
    goal = here + np.array([0.0, dy if here[1] < 55 else -dy])
    app.path = F.Path.point(goal)
    app.arm()
    assert app.armed, app.note
    return goal


# -- the list of what a drive owns -------------------------------------------

# Bench-wide, and meant to be shared: the roster of robots, the roll call that
# blinks all of them, the message line, the frame's mirror flag, and the run
# counter that keeps ids unique ACROSS robots.
SHARED = {"bots", "reid", "note", "note_tone", "_idle_track", "mirrored",
          "_runs", "code", "blob"}


def test_the_drive_state_list_covers_everything_a_drive_writes():
    """Walk what `drive`, `arm`, `disarm` and the blob bookkeeping reach and
    what they write. Anything written there that is neither listed as drive
    state nor deliberately shared would be shared by accident."""
    src = pathlib.Path(F.__file__).read_text()
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "BlobTest")
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}

    def self_attr(node):
        return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self")

    seen, todo = set(), ["drive", "arm", "disarm", "record_blob"]
    while todo:
        m = todo.pop()
        if m in seen or m not in methods:
            continue
        seen.add(m)
        todo += [n.attr for n in ast.walk(methods[m])
                 if self_attr(n) and n.attr in methods]
    # The machinery itself is not a drive; what it moves is the list.
    seen -= {"as_bot", "_switch_to", "_save_drive", "_load_drive", "stop_all",
             "drive_all", "background", "driving_codes", "_add_bot"}

    written = set()
    for m in seen:
        for n in ast.walk(methods[m]):
            targets = (n.targets if isinstance(n, ast.Assign) else
                       [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign))
                       else [])
            for t in targets:
                written |= {e.attr for e in ast.walk(t) if self_attr(e)}
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("append", "clear", "extend", "pop",
                                        "popleft", "update")
                    and self_attr(n.func.value)):
                written.add(n.func.value.attr)
    missing = written - set(F.BlobTest.DRIVE_STATE) - SHARED
    assert not missing, f"written by a drive but not per-robot: {sorted(missing)}"


def test_a_new_robot_starts_from_the_same_state_the_bench_does():
    app = F.BlobTest("sim", with_fleet=False)
    try:
        fresh = F.BlobTest.fresh_drive()
        assert set(fresh) == set(F.BlobTest.DRIVE_STATE)
        for k, v in fresh.items():
            have = getattr(app, k)
            if hasattr(v, "maxlen"):
                assert type(have) is type(v) and have.maxlen == v.maxlen, k
                assert len(have) == 0, k
            else:
                assert have == v, f"{k}: bench starts at {have!r}, a robot at {v!r}"
    finally:
        app.close()


def test_fresh_state_is_never_shared_between_robots():
    """Two dicts holding one deque would be one trail for two balls."""
    a, b = F.BlobTest.fresh_drive(), F.BlobTest.fresh_drive()
    for k in ("history", "trail", "cmd_log"):
        assert a[k] is not b[k], k


# -- one robot is untouched --------------------------------------------------

def test_one_robot_still_takes_the_single_robot_path():
    """`--robot SYRX` stays outside `bots`, exactly as before. Only a list of
    several goes through the per-robot machinery."""
    app = F.BlobTest("sim", code="SYRX")
    try:
        assert app.bots == {}
        assert app.code == "SYRX"
    finally:
        app.close()


# -- three at once -----------------------------------------------------------

def test_a_list_of_robots_connects_all_of_them(three):
    assert list(three.bots) == THREE.split(",")
    assert all(b["track"].locked for b in three.bots.values())


def test_three_robots_each_reach_their_own_goal(three):
    """The whole point. Identified, then all armed at once, and each lands at
    ITS goal -- which a leak in the swap would break first, by steering one
    ball on another's position."""
    _identified(three)
    goals = {c: _arm_toward(three, c) for c in list(three.bots)}
    assert sorted(three.driving_codes()) == sorted(goals)

    assert _tick(three, 30.0, lambda: not three.driving_codes()), \
        f"still driving: {three.driving_codes()}"
    for code, goal in goals.items():
        with three.as_bot(code):
            assert three.last_outcome == "arrived", (code, three.last_note)
            rest = np.asarray(three.fleet.handles[code].pos, dtype=float)
            reach = three.goal_tol + three.blob_radius_cm() + 4.0
            assert np.linalg.norm(rest - goal) < reach, (code, rest, goal)


def test_selecting_another_robot_does_not_stop_the_one_driving(three):
    _identified(three)
    a, b = list(three.bots)[:2]
    _arm_toward(three, a)
    three.select(b)
    _tick(three, 0.3)
    assert a in three.driving_codes()
    assert three.code == b and not three.armed, "b was never armed"
    three.stop_all()


def test_each_robot_keeps_its_own_route(three):
    a, b, _ = list(three.bots)
    goal = _arm_toward(three, a)
    three.select(b)
    assert three.path is None, "b inherited a's route"
    three.select(a)
    assert three.path is not None and np.allclose(three.path.pts[0], goal)
    three.stop_all()


def test_run_ids_are_unique_across_robots(three):
    ids = []
    for c in list(three.bots):
        _arm_toward(three, c)
        ids.append(three.run_id)
    assert len(set(ids)) == 3, ids
    three.stop_all()


# -- stopping ----------------------------------------------------------------

def test_esc_stops_every_robot_not_just_the_selected_one(three):
    for c in list(three.bots):
        _arm_toward(three, c)
    three.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE,
                                 unicode="", mod=0))
    assert three.driving_codes() == []


def test_the_roll_call_refuses_while_a_background_robot_drives(three):
    """While it runs nothing else is stepped, so a robot left armed would roll
    on its last command for two seconds with nobody steering."""
    a, b, _ = list(three.bots)
    _arm_toward(three, a)
    three.select(b)
    three.start_reid()
    assert three.reid is None
    three.stop_all()


# -- the keyboard steers one robot -------------------------------------------

def test_held_keys_only_ever_steer_the_selected_robot(three, monkeypatch):
    three.tab = "robot"

    class Held:
        def __getitem__(self, k):
            return k == pygame.K_w

    monkeypatch.setattr(pygame.key, "get_pressed", lambda: Held())
    other = three.background()[0]
    with three.as_bot(other):
        assert three.manual_drive() is False
        assert three.manual is False


def test_leaving_a_robot_mid_manual_drive_stops_it(three):
    a, b, _ = list(three.bots)
    three.select(a)
    three.manual, three.cmd_v = True, np.array([20.0, 0.0])
    three.select(b)
    with three.as_bot(a):
        assert three.manual is False and three.cmd_v is None


def test_a_background_message_names_its_robot(three):
    other = three.background()[0]
    with three.as_bot(other):
        three.say("arrived — stopped")
    assert three.note == f"{other}: arrived — stopped"
    three.say("plain")
    assert three.note == "plain"


# -- resting balls say who they are ------------------------------------------

def _truly(app, code):
    """Is `code`'s name on `code`'s ball? Asked of the simulator, not the bench."""
    blob = app.bots[code].get("blob")
    if blob is None:
        return False
    cm = np.asarray(app.to_cm(blob["xy"]), dtype=float)
    return float(np.linalg.norm(cm - app.fleet.handles[code].pos)) < 3.0


def _swap(app, a, b):
    A, B = app.bots[a], app.bots[b]
    A["track"], B["track"] = B["track"], A["track"]
    A["blob"], B["blob"] = B["blob"], A["blob"]


def _cycle(app, holder_means, owner_still_for=10.0, drift_cm=0.0):
    """One finished beacon cycle, made by hand. `holder_means` maps a robot to
    the four slot brightnesses its CURRENT track showed."""
    start = 1000.0
    for bot in app.bots.values():
        bot["still_since"] = start - owner_still_for
    seen = {}
    for holder, means in holder_means.items():
        track = app.bots[holder]["track"]
        seen[id(track)] = {"track": track,
                           "pts": [np.zeros(2), np.array([drift_cm, 0.0])],
                           "slots": [[m] for m in means]}
    return {"cycle": 0, "start": start, "seen": seen}


def _bright(bits, lo=73.0, hi=245.0):
    return [hi if b else lo for b in bits]


def test_every_robot_gets_its_own_code():
    app = F.BlobTest("sim", code="SYRX,VHGR,MLYS,SNFR")
    try:
        codes = [tuple(b["code_bits"]) for b in app.bots.values()]
        assert len(set(codes)) == 4
        assert all(0 < sum(c) < len(c) for c in codes), "steady is never a code"
    finally:
        app.close()


def test_resting_balls_blink_and_their_names_come_right_without_a_roll_call(three):
    """Acquisition hands out names by guess; resting balls fix them."""
    assert _tick(three, 10.0, lambda: all(_truly(three, c) for c in three.bots)), \
        {c: _truly(three, c) for c in three.bots}
    dims = [b for b in three.bots.values() if b["rgb"] != b["base_rgb"]]
    _tick(three, three.ID_SLOT_S * three.ID_SLOTS)
    assert any(b["rgb"] != b["base_rgb"] for b in three.bots.values()) or dims, \
        "nothing ever dimmed"


def test_a_swap_at_rest_is_put_right_within_a_few_cycles(three):
    _tick(three, 10.0, lambda: all(_truly(three, c) for c in three.bots))
    a, b, _ = list(three.bots)
    _swap(three, a, b)
    assert not _truly(three, a)
    assert _tick(three, 10.0, lambda: all(_truly(three, c) for c in three.bots))


def test_a_driving_ball_holds_steady(three):
    code = list(three.bots)[0]
    _arm_toward(three, code)
    _tick(three, 1.5)
    bot = three.bots[code]
    assert bot["still_since"] is None
    assert bot["rgb"] == bot["base_rgb"], "a moving ball must not blink"
    three.stop_all()


def test_its_own_accelerometer_can_stop_a_ball_counting_as_still(three, monkeypatch):
    """A bumped ball the bench never told to move."""
    code = list(three.bots)[0]
    monkeypatch.setattr(three.fleet.handles[code], "accel_quiet", lambda: False)
    _tick(three, 1.0)
    assert three.bots[code]["still_since"] is None
    assert three.bots[code]["rgb"] == three.bots[code]["base_rgb"]


def test_a_silent_accelerometer_does_not_veto(three, monkeypatch):
    """No stream is "unknown", not "moving" -- or a ball with a dead sensor
    could never identify itself."""
    code = list(three.bots)[0]
    monkeypatch.setattr(three.fleet.handles[code], "accel_quiet", lambda: None)
    _tick(three, 1.0)
    assert three.bots[code]["still_since"] is not None


def test_one_robot_never_blinks():
    """One robot can only be one ball. A single-robot run is untouched."""
    app = F.BlobTest("sim", code="SYRX")
    try:
        _tick(app, 1.0)
        assert app._beacon is None
    finally:
        app.close()


# -- reading a cycle ---------------------------------------------------------

def test_a_clean_contradiction_moves_the_names(three):
    """Judged against the evidence given, not the simulator: the fixture's
    names start as acquisition's guesses, so a made-up cycle is the truth here."""
    a, b, _ = list(three.bots)
    ta, tb = three.bots[a]["track"], three.bots[b]["track"]
    three.decode_beacon(_cycle(three, {
        a: _bright(three.bots[b]["code_bits"]),
        b: _bright(three.bots[a]["code_bits"])}))
    assert three.bots[b]["track"] is ta and three.bots[a]["track"] is tb


def test_a_slot_caught_mid_change_is_not_read(three):
    """The read that turned SYRX's 1000 into MLYS's 1100: slot 0 bleeding into
    slot 1 at a low frame rate. Halfway is not a bit."""
    a, b, c = list(three.bots)
    three.bots[a]["code_bits"] = [1, 0, 0, 0]
    three.bots[c]["code_bits"] = [1, 1, 0, 0]
    before = {k: v["track"] for k, v in three.bots.items()}
    three.decode_beacon(_cycle(three, {a: [202.0, 159.0, 73.0, 73.0]}))
    assert {k: v["track"] for k, v in three.bots.items()} == before


def test_a_code_from_a_ball_that_stopped_mid_cycle_is_not_read(three):
    """Half a cycle of steady and half a code can spell another robot exactly."""
    a, b, _ = list(three.bots)
    before = three.bots[a]["track"]
    three.decode_beacon(_cycle(three, {a: _bright(three.bots[b]["code_bits"])},
                               owner_still_for=-1.0))
    assert three.bots[a]["track"] is before


def test_a_track_that_moved_during_the_cycle_is_not_read(three):
    a, b, _ = list(three.bots)
    before = three.bots[a]["track"]
    three.decode_beacon(_cycle(three, {a: _bright(three.bots[b]["code_bits"])},
                               drift_cm=5.0))
    assert three.bots[a]["track"] is before


def test_noise_on_a_steady_ball_is_never_a_code(three):
    a, _, _ = list(three.bots)
    before = three.bots[a]["track"]
    three.decode_beacon(_cycle(three, {a: [245.0, 239.0, 244.0, 241.0]}))
    assert three.bots[a]["track"] is before


def test_one_side_of_a_swap_is_enough_when_the_other_is_moving(three):
    """The unread ball is moving, so it is not blinking -- and the robot left
    holding the wrong track may be driving on another ball's position. The
    swap it must be is made now, and the inferred name marked unidentified."""
    a, b, _ = list(three.bots)
    ta, tb = three.bots[a]["track"], three.bots[b]["track"]
    three.decode_beacon(_cycle(three, {a: _bright(three.bots[b]["code_bits"])}))
    assert three.bots[b]["track"] is ta and three.bots[a]["track"] is tb
    assert three.bots[b]["identified"] is True
    assert three.bots[a]["identified"] is False


def test_the_roll_call_restores_the_colour_not_a_dim_blink(three):
    code = list(three.bots)[0]
    bot = three.bots[code]
    bot["rgb"] = [int(v * three.ID_DIM) for v in bot["base_rgb"]]
    three.start_reid()
    assert three.reid["base"][code] == bot["base_rgb"]


def test_evidence_stays_with_the_track_when_a_name_moves(three, monkeypatch):
    """The first bug this had. Evidence kept by NAME stitched half of one
    ball's code to half of another's when a name moved mid-cycle -- and
    SYRX's 1000 plus VHGR's 0100 read as MLYS's 1100. Kept by track, a ball's
    readings stay that ball's whatever it is called."""
    clock = [1000.0 * three.ID_SLOT_S * three.ID_SLOTS]
    monkeypatch.setattr(F.time, "perf_counter", lambda: clock[0])
    for bot in three.bots.values():
        bot["driving_at"] = None
    a, c, _ = list(three.bots)
    for i, bot in enumerate(three.bots.values()):
        bot["blob"] = dict(bot["blob"], peak=100.0 + i)
    ta = three.bots[a]["track"]
    mine = three.bots[a]["blob"]["peak"]

    clock[0] += 0.45                           # late in slot 0
    three.step_beacon()
    _swap(three, a, c)                         # the name moves, the ball does not
    clock[0] += 2 * three.ID_SLOT_S            # late in slot 2, same cycle
    three.step_beacon()

    entry = three._beacon["seen"][id(ta)]
    peaks = [p for slot in entry["slots"] for p in slot]
    assert len(peaks) == 2 and set(peaks) == {mine}, peaks

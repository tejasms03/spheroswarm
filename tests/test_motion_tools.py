"""The four motion tools, against a live sim fleet."""

import numpy as np
import pytest

from tools import call
from workspace.entities import EntitySet

ROT = "vx = -(y-cy)*0.5\nvy = (x-cx)*0.5"


@pytest.fixture
def ctx(sim_ctx):
    c = sim_ctx(n=6, seed=4, obstacles=[])
    c.fleet["SSMK"].pos = np.array([80.0, 90.0])
    c.fleet["CRXS"].pos = np.array([160.0, 90.0])
    c.fleet["SYRX"].pos = np.array([80.0, 130.0])
    c.fleet["VHGR"].pos = np.array([160.0, 130.0])
    c.fleet["MLYS"].pos = np.array([80.0, 50.0])
    c.fleet["SNFR"].pos = np.array([160.0, 50.0])
    return c


def with_wanderer(ctx, speed=25.0):
    ctx.ws.entities = EntitySet.from_data([
        {"id": "wanderer", "role": "target",
         "shape": {"type": "circle", "center": [60, 90], "radius": 12},
         "motion": {"kind": "path", "waypoints": [[60, 90], [180, 90]],
                     "mode": "pingpong", "speed": speed}}])
    return ctx


def run(ctx, seconds, dt=0.1):
    for _ in range(int(seconds / dt)):
        ctx.tick(dt)


# -- set_path -----------------------------------------------------------------

def test_set_path_walks_waypoints(ctx):
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[60, 40], [180, 40]]},
                                "mode": "once", "duration": 30})
    assert r["ok"], r["error"]
    run(ctx, 12)
    assert ctx.fleet["SSMK"].pos[0] > 120, "did not walk toward the far waypoint"


def test_set_path_loops(ctx):
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[60, 40], [180, 40]]},
                                "mode": "loop", "duration": 60})
    assert r["ok"], r["error"]
    seen_left = seen_right = False
    for _ in range(600):
        ctx.tick(0.1)
        x = ctx.fleet["SSMK"].pos[0]
        seen_right |= x > 165
        seen_left |= (seen_right and x < 80)
    assert seen_right and seen_left, "a looping path never came back"


def test_set_path_rejects_a_bad_mode(ctx):
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[60, 40]]}, "mode": "zigzag"})
    assert not r["ok"] and "mode" in r["error"]


def test_set_path_rejects_unknown_robots(ctx):
    r = call(ctx, "set_path", {"assignments": {"NOPE": [[60, 40]]}})
    assert not r["ok"] and "NOPE" in r["error"]


def test_set_path_clamps_waypoints_into_the_arena(ctx):
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[500, 40], [60, 40]]}})
    assert r["ok"], r["error"]
    for p in r["waypoints"]["SSMK"]:
        assert ctx.ws.point_in_bounds(np.array(p))


def test_set_path_append_extends(ctx):
    call(ctx, "set_path", {"assignments": {"SSMK": [[60, 40]]}, "duration": 30})
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[180, 40]]},
                                "append": True, "duration": 30})
    assert r["ok"], r["error"]
    layer = ctx.stack.get("SSMK", "path")
    assert len(layer.params["waypoints"]) == 2


def test_duration_is_clamped_and_reported(ctx):
    r = call(ctx, "set_path", {"assignments": {"SSMK": [[60, 40], [180, 40]]},
                                "mode": "loop", "duration": 9999})
    assert r["ok"], r["error"]
    assert r["duration_clamped"] is True
    assert r["duration_s"] == 300.0
    assert "clamped" in (r["note"] or "")


# -- set_flow ------------------------------------------------------------------

def test_set_flow_rotates_the_swarm(ctx):
    r = call(ctx, "set_flow", {"expr": ROT, "duration": 30})
    assert r["ok"], r["error"]
    c = np.asarray(ctx.ws.centroid())
    before = ctx.active_positions() - c
    a0 = np.arctan2(before[:, 1], before[:, 0])
    run(ctx, 15)
    after = ctx.active_positions() - c
    a1 = np.arctan2(after[:, 1], after[:, 0])
    advanced = ((a1 - a0) % (2 * np.pi))
    assert (advanced > 0.2).sum() >= 4, "most robots should have swung round"
    for p in ctx.active_positions():
        assert ctx.ws.point_in_bounds(p)


def test_set_flow_refuses_a_field_that_pins_robots_to_a_wall(ctx):
    r = call(ctx, "set_flow", {"expr": "vx = 1.0\nvy = 0.0", "duration": 30})
    assert not r["ok"]
    assert "wall" in r["error"]
    assert ctx.stack.active_layers("SSMK") == [], "a refused field must not run"


def test_set_flow_refuses_an_outward_field(ctx):
    r = call(ctx, "set_flow", {"expr": "vx = (x-cx)*0.5\nvy = (y-cy)*0.5"})
    assert not r["ok"] and "wall" in r["error"]


def test_set_flow_rejects_a_field_that_does_not_assign_both(ctx):
    r = call(ctx, "set_flow", {"expr": "vx = 1.0"})
    assert not r["ok"] and "vy" in r["error"]


def test_set_flow_sandbox_blocks_escapes(ctx):
    for src in ("import os\nvx=1\nvy=1",
                "vx = ().__class__.__bases__[0].__subclasses__()\nvy = 0",
                "vx = open('/etc/passwd')\nvy = 0",
                "while True:\n    pass\nvx=1\nvy=1",
                "vx = eval('1')\nvy = 0",
                "vx = __import__('os').system('ls')\nvy = 0"):
        r = call(ctx, "set_flow", {"expr": src})
        assert not r["ok"], f"escaped: {src!r}"


def test_set_flow_binds_entity_position(ctx):
    """ex/ey must track the entity, which is what makes orbiting it possible."""
    with_wanderer(ctx, speed=20.0)
    r = call(ctx, "set_flow", {"expr": "vx = -(y-ey)*0.5\nvy = (x-ex)*0.5",
                                "entity": "wanderer", "duration": 30})
    assert r["ok"], r["error"]

    layer = ctx.stack.get("SSMK", "flow")
    assert layer.params["entity"] == "wanderer"
    field = layer.params["field"]
    # same robot position, different entity position -> different velocity
    v1 = field(100.0, 100.0, 0.0, 60.0, 90.0)
    v2 = field(100.0, 100.0, 0.0, 180.0, 90.0)
    assert v1 != v2, "ex/ey were not bound"


def test_set_flow_unknown_entity_is_named(ctx):
    r = call(ctx, "set_flow", {"expr": ROT, "entity": "dragonfly"})
    assert not r["ok"] and "dragonfly" in r["error"]


# -- follow ---------------------------------------------------------------------

def test_follow_trails_a_moving_entity(ctx):
    with_wanderer(ctx)
    r = call(ctx, "follow", {"target_id": "wanderer", "mode": "trail",
                              "distance": 35, "duration": 60})
    assert r["ok"], r["error"]
    run(ctx, 20)
    e = ctx.ws.entities.by_id("wanderer")
    d = [float(np.linalg.norm(ctx.fleet[c].pos - e.pos)) for c in ctx.active_codes()]
    assert np.mean(d) < 90, f"followers lost the target: {np.round(d)}"


def test_follow_surround_holds_a_ring(ctx):
    with_wanderer(ctx, speed=15.0)
    r = call(ctx, "follow", {"target_id": "wanderer", "mode": "surround",
                              "distance": 40, "duration": 60})
    assert r["ok"], r["error"]
    run(ctx, 25)
    e = ctx.ws.entities.by_id("wanderer")
    radii = [float(np.linalg.norm(ctx.fleet[c].pos - e.pos))
             for c in ctx.active_codes()]
    assert 20 < np.mean(radii) < 70, f"ring radius drifted: {np.round(radii)}"


def test_follow_a_robot(ctx):
    r = call(ctx, "follow", {"target_id": "SSMK", "followers": ["CRXS"],
                              "mode": "trail", "distance": 30, "duration": 30})
    assert r["ok"], r["error"]
    assert "SSMK" not in r["codes"], "the target must not follow itself"


def test_follow_refuses_an_obstacle_role_entity(ctx):
    ctx.ws.entities = EntitySet.from_data([
        {"id": "rock", "role": "obstacle",
         "shape": {"type": "circle", "center": [120, 90], "radius": 15}}])
    r = call(ctx, "follow", {"target_id": "rock"})
    assert not r["ok"] and "role" in r["error"]


def test_follow_unknown_target_is_named(ctx):
    r = call(ctx, "follow", {"target_id": "ghost"})
    assert not r["ok"] and "ghost" in r["error"]


def test_followers_hold_position_when_the_target_disappears(ctx):
    """A stale target must not be chased toward its last guess."""
    with_wanderer(ctx)
    call(ctx, "follow", {"target_id": "wanderer", "mode": "trail",
                          "distance": 30, "duration": 60})
    run(ctx, 6)
    ctx.ws.entities.remove("wanderer")
    before = ctx.active_positions().copy()
    run(ctx, 8)
    moved = np.linalg.norm(ctx.active_positions() - before, axis=1).max()

    # Some drift is physics, not pursuit: robots were rolling at up to 60cm/s
    # and take ~0.35s to stop. What must not happen is continued tracking.
    assert moved < 50, f"kept chasing a target that no longer exists ({moved:.0f}cm)"
    assert ctx.stack.get("SSMK", "follow").state.get("stale") is True


def test_follow_predicts_ahead_of_the_target():
    from swarm.fields import LEAD_TIME, predict

    aim = predict([100.0, 90.0], [40.0, 0.0])
    assert aim[0] == pytest.approx(100.0 + 40.0 * LEAD_TIME)


# -- motion_control -----------------------------------------------------------------

def test_motion_control_lists_what_is_running(ctx):
    call(ctx, "set_flow", {"expr": ROT, "duration": 30})
    r = call(ctx, "motion_control", {"action": "list"})
    assert r["ok"], r["error"]
    assert "SSMK" in r["layers"]
    assert r["layers"]["SSMK"][0]["name"] == "flow"


def test_motion_control_removes_a_named_layer(ctx):
    call(ctx, "set_flow", {"expr": ROT, "duration": 30})
    r = call(ctx, "motion_control", {"action": "remove", "name": "flow"})
    assert r["ok"], r["error"]
    assert call(ctx, "motion_control", {"action": "list"})["layers"] == {}


def test_motion_control_reweights(ctx):
    call(ctx, "set_flow", {"expr": ROT, "duration": 30})
    r = call(ctx, "motion_control", {"action": "set_weight", "name": "flow",
                                      "weight": 0.2})
    assert r["ok"], r["error"]
    assert ctx.stack.get("SSMK", "flow").weight == 0.2


def test_motion_control_clear(ctx):
    call(ctx, "set_flow", {"expr": ROT, "duration": 30})
    r = call(ctx, "motion_control", {"action": "clear"})
    assert r["ok"], r["error"]
    assert ctx.stack.codes() == []


def test_motion_control_remove_without_a_name_says_so(ctx):
    r = call(ctx, "motion_control", {"action": "remove"})
    assert not r["ok"] and "name" in r["error"]


def test_motion_control_unknown_action(ctx):
    r = call(ctx, "motion_control", {"action": "explode"})
    assert not r["ok"] and "explode" in r["error"]


# -- composition ---------------------------------------------------------------------

def test_follow_plus_flow_orbits_a_moving_entity(ctx):
    """The composition case: radius held by follow, rotation supplied by flow."""
    with_wanderer(ctx, speed=12.0)
    a = call(ctx, "follow", {"target_id": "wanderer", "mode": "surround",
                              "distance": 40, "duration": 90})
    assert a["ok"], a["error"]
    b = call(ctx, "set_flow", {"expr": "vx = -(y-ey)*0.6\nvy = (x-ex)*0.6",
                                "entity": "wanderer", "weight": 0.5,
                                "duration": 90})
    assert b["ok"], b["error"]

    e = ctx.ws.entities.by_id("wanderer")
    run(ctx, 10)
    start_angles = np.array([
        np.arctan2(*(ctx.fleet[c].pos - e.pos)[::-1]) for c in ctx.active_codes()])
    radii = []
    for _ in range(250):
        ctx.tick(0.1)
        radii.append([float(np.linalg.norm(ctx.fleet[c].pos - e.pos))
                      for c in ctx.active_codes()])
    end_angles = np.array([
        np.arctan2(*(ctx.fleet[c].pos - e.pos)[::-1]) for c in ctx.active_codes()])

    mean_r = float(np.mean(radii))
    assert 20 < mean_r < 75, f"radius did not hold near 40cm: {mean_r:.0f}"
    advanced = np.abs((end_angles - start_angles + np.pi) % (2 * np.pi) - np.pi)
    assert (advanced > 0.3).sum() >= 3, "nothing actually orbited"
    assert len(ctx.stack.active_layers("SSMK")) == 2, "both layers should be live"


# -- §6: what the existing tools report and clear once motion exists ----------

def test_get_state_reports_layers_and_stall_per_robot(ctx):
    call(ctx, "set_flow", {"expr": ROT, "robots": ["SSMK"], "duration": 30})
    robots = {r["code"]: r for r in call(ctx, "get_state", {})["robots"]}

    assert [l["name"] for l in robots["SSMK"]["layers"]] == ["flow"]
    assert robots["SSMK"]["layers"][0]["kind"] == "flow"
    assert robots["SSMK"]["stalled"] is False
    assert robots["CRXS"]["layers"] == [], "a robot with no motion listed one"


def test_get_state_reports_entities_at_their_live_position(ctx):
    with_wanderer(ctx)
    start = call(ctx, "get_state", {})["entities"]
    assert [e["id"] for e in start] == ["wanderer"]
    assert start[0]["role"] == "target" and start[0]["followable"] is True
    assert start[0]["moving"] is True

    run(ctx, 3)
    later = call(ctx, "get_state", {})["entities"][0]
    assert later["pos"] != start[0]["pos"], "entity position was reported stale"


def test_describe_scene_mentions_what_is_running(ctx):
    with_wanderer(ctx)
    call(ctx, "follow", {"target_id": "wanderer", "followers": ["SSMK"],
                          "mode": "trail", "duration": 30})
    text = call(ctx, "describe_scene", {})["description"]
    assert "wanderer" in text
    assert "following" in text and "trail" in text
    assert "SSMK" in text


def test_describe_scene_says_so_when_nothing_is_moving(ctx):
    assert "Nothing is moving" in call(ctx, "describe_scene", {})["description"]


def test_stop_clears_every_layer_type(ctx):
    with_wanderer(ctx)
    call(ctx, "set_flow", {"expr": ROT, "robots": ["SSMK"], "duration": 60})
    call(ctx, "set_path", {"assignments": {"CRXS": [[60, 40], [180, 40]]},
                            "mode": "loop", "duration": 60})
    call(ctx, "follow", {"target_id": "wanderer", "followers": ["SYRX"],
                          "duration": 60})
    assert ctx.stack.codes()

    assert call(ctx, "stop", {})["ok"]
    assert ctx.stack.codes() == [], "a layer survived stop"

    # And it must stay stopped: a surviving layer would drive again next tick.
    before = {c: ctx.fleet[c].pos.copy() for c in ("SSMK", "CRXS", "SYRX")}
    run(ctx, 3)
    for code, p in before.items():
        assert np.linalg.norm(ctx.fleet[code].pos - p) < 5.0, f"{code} kept moving"


def test_stop_with_codes_clears_only_those_layers(ctx):
    call(ctx, "set_flow", {"expr": ROT, "robots": ["SSMK", "CRXS"], "duration": 60})
    assert call(ctx, "stop", {"codes": ["SSMK"]})["ok"]
    assert ctx.stack.layers("SSMK") == []
    assert [l.name for l in ctx.stack.layers("CRXS")] == ["flow"]


def test_wait_until_settled_refuses_a_continuous_layer(ctx):
    call(ctx, "set_flow", {"expr": ROT, "robots": ["SSMK"], "duration": 60})
    r = call(ctx, "wait_until_settled", {})
    assert not r["ok"]
    assert "never settle" in r["error"]
    assert "motion_control" in r["error"], "the error must say how to get unstuck"


def test_wait_until_settled_still_works_for_a_plain_move(ctx):
    call(ctx, "move_to", {"assign": {"SSMK": [100, 90]}})
    r = call(ctx, "wait_until_settled", {"timeout": 20})
    assert r["ok"] and r["settled"], r.get("note")


def test_a_once_path_is_not_continuous(ctx):
    """`once` terminates, so waiting on it is legitimate."""
    call(ctx, "set_path", {"assignments": {"SSMK": [[100, 90]]},
                            "mode": "once", "duration": 30})
    assert ctx.stack.has_continuous() is False


# -- conditional rules -------------------------------------------------------
#
# A layer already has a weight; `when` makes it conditional. That turns a
# standing *action* into a standing *rule*, re-decided every tick rather than
# once when the command was given.

def test_a_layer_only_applies_while_its_condition_holds(ctx):
    with_wanderer(ctx, speed=0.0)
    ctx.fleet["SSMK"].pos = np.array([60.0, 90.0])     # near the wanderer (60,90)
    ctx.fleet["CRXS"].pos = np.array([220.0, 40.0])    # far away
    ctx.env.sync()

    r = call(ctx, "follow", {"target_id": "wanderer", "when": "d_target < 70",
                              "duration": 120})
    assert r["ok"], r["error"]

    # Displacement, not instantaneous velocity: a robot that has reached its
    # slot is stationary again, and reads as "never moved".
    before = {c: ctx.fleet[c].pos.copy() for c in ("SSMK", "CRXS")}
    run(ctx, 4)
    moved = {c: float(np.linalg.norm(ctx.fleet[c].pos - p))
             for c, p in before.items()}
    assert moved["SSMK"] > 5.0, f"the near robot idled ({moved['SSMK']:.1f}cm)"
    assert moved["CRXS"] < 5.0, f"the far robot drove ({moved['CRXS']:.1f}cm)"


def test_a_condition_that_is_never_true_drives_nobody(ctx):
    with_wanderer(ctx, speed=0.0)
    before = {c: ctx.fleet[c].pos.copy() for c in ctx.fleet.codes}
    assert call(ctx, "follow", {"target_id": "wanderer", "when": "d_target < 1",
                                 "duration": 60})["ok"]
    run(ctx, 5)
    for code, p in before.items():
        assert float(np.linalg.norm(ctx.fleet[code].pos - p)) < 8.0


def test_no_condition_means_the_old_behaviour_exactly(ctx):
    with_wanderer(ctx, speed=0.0)
    assert call(ctx, "follow", {"target_id": "wanderer", "duration": 60})["ok"]
    before = {c: ctx.fleet[c].pos.copy() for c in ctx.fleet.codes}
    run(ctx, 4)
    assert any(float(np.linalg.norm(ctx.fleet[c].pos - before[c])) > 5.0
               for c in ctx.fleet.codes)


def test_a_condition_is_sandboxed_like_every_other_expression(ctx):
    with_wanderer(ctx)
    for hostile in ("import os", "__import__('os').system('ls')",
                    "open('/etc/passwd')", "while True: pass"):
        r = call(ctx, "follow", {"target_id": "wanderer", "when": hostile,
                                  "duration": 30})
        assert not r["ok"], f"{hostile!r} was accepted"
        assert "rejected" in r["error"]


def test_a_bad_condition_explains_what_one_looks_like(ctx):
    with_wanderer(ctx)
    r = call(ctx, "follow", {"target_id": "wanderer", "when": "d_target <<< 3"})
    assert not r["ok"]
    assert "d_target" in r["error"], "the error must name the usable variables"


def test_a_condition_that_raises_drops_the_layer_rather_than_driving_on(ctx):
    with_wanderer(ctx, speed=0.0)
    assert call(ctx, "follow", {"target_id": "wanderer", "when": "1 / 0 > 0",
                                 "duration": 60})["ok"]
    run(ctx, 2)
    layer = ctx.stack.get("SSMK", "follow")
    assert layer is not None
    assert layer.state.get("when_error"), "a raising condition was ignored"


def test_flow_takes_conditions_too(ctx):
    r = call(ctx, "set_flow", {"expr": ROT, "when": "speed < 60",
                                "duration": 60})
    assert r["ok"], r["error"]
    before = {c: ctx.fleet[c].pos.copy() for c in ctx.fleet.codes}
    run(ctx, 3)
    assert any(float(np.linalg.norm(ctx.fleet[c].pos - before[c])) > 5.0
               for c in ctx.fleet.codes)


def test_selection_can_be_re_decided_every_tick(ctx):
    """The special case: whichever robot is nearest, with hysteresis."""
    with_wanderer(ctx, speed=0.0)
    ctx.fleet["SSMK"].pos = np.array([70.0, 90.0])
    ctx.fleet["CRXS"].pos = np.array([200.0, 90.0])
    ctx.env.sync()
    assert call(ctx, "follow", {"target_id": "wanderer", "select": "nearest",
                                 "duration": 120})["ok"]
    before = {c: ctx.fleet[c].pos.copy() for c in ("SSMK", "CRXS")}
    run(ctx, 3)
    assert float(np.linalg.norm(ctx.fleet["SSMK"].pos - before["SSMK"])) > 5.0
    assert float(np.linalg.norm(ctx.fleet["CRXS"].pos - before["CRXS"])) < 5.0

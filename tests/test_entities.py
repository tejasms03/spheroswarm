"""Entities: shapes with a role and, optionally, motion."""

import numpy as np
import pytest

from workspace.entities import Entity, EntitySet, validate_entity
from workspace.flow import compile_flow
from workspace.space import Workspace

BOUNDS = (0.0, 240.0, 0.0, 180.0)


def circle(cx=100.0, cy=90.0, r=15.0):
    return {"type": "circle", "center": [cx, cy], "radius": r}


def square(cx=100.0, cy=90.0, h=15.0):
    return {"type": "poly", "points": [[cx - h, cy - h], [cx + h, cy - h],
                                        [cx + h, cy + h], [cx - h, cy + h]]}


# -- schema -----------------------------------------------------------------

def test_a_minimal_static_obstacle_is_valid():
    assert validate_entity({"id": "rock", "shape": circle()}) == []


def test_missing_id_and_shape_are_reported():
    assert any("id" in e for e in validate_entity({"shape": circle()}))
    assert any("shape" in e for e in validate_entity({"id": "x"}))


def test_bad_role_is_named():
    errs = validate_entity({"id": "x", "shape": circle(), "role": "enemy"})
    assert any("role" in e and "enemy" in e for e in errs)


def test_path_needs_two_waypoints_and_a_valid_mode():
    base = {"id": "w", "shape": circle()}
    errs = validate_entity({**base, "motion": {"kind": "path", "waypoints": [[10, 10]]}})
    assert any("2 waypoints" in e for e in errs)
    errs = validate_entity({**base, "motion": {"kind": "path",
                                                "waypoints": [[10, 10], [20, 20]],
                                                "mode": "zigzag"}})
    assert any("mode" in e for e in errs)


def test_absurd_speed_is_rejected():
    errs = validate_entity({"id": "w", "shape": circle(),
                            "motion": {"kind": "path", "speed": 5000,
                                        "waypoints": [[10, 10], [20, 20]]}})
    assert any("speed" in e for e in errs)


def test_flow_needs_an_expression():
    errs = validate_entity({"id": "f", "shape": circle(), "motion": {"kind": "flow"}})
    assert any("expr" in e for e in errs)


def test_one_bad_entity_does_not_lose_the_others():
    s = EntitySet.from_data([{"id": "good", "shape": circle()},
                              {"id": "bad", "shape": {"type": "blob"}}])
    assert [e.id for e in s] == ["good"]
    assert s.errors


def test_duplicate_ids_are_rejected():
    s = EntitySet.from_data([{"id": "a", "shape": circle()},
                              {"id": "a", "shape": circle()}])
    assert len(s) == 1
    assert any("duplicate" in e for e in s.errors)


# -- roles -------------------------------------------------------------------

def test_roles_filter_correctly():
    s = EntitySet.from_data([
        {"id": "rock", "shape": circle(), "role": "obstacle"},
        {"id": "prey", "shape": circle(60, 60), "role": "target"},
        {"id": "both", "shape": circle(30, 30), "role": "both"},
    ])
    assert {e.id for e in s.by_role("obstacle")} == {"rock", "both"}
    assert {e.id for e in s.by_role("target")} == {"prey", "both"}
    assert len(s.blocking_shapes()) == 2, "a target must not block"


def test_by_id():
    s = EntitySet.from_data([{"id": "rock", "shape": circle()}])
    assert s.by_id("rock").id == "rock"
    assert s.by_id("nope") is None


# -- motion -------------------------------------------------------------------

def test_static_entities_never_move():
    e = Entity("rock", circle(), "obstacle", None)
    start = e.pos.copy()
    for _ in range(50):
        e.step(0.1, BOUNDS)
    np.testing.assert_allclose(e.pos, start)


def test_a_path_walks_toward_its_waypoint():
    e = Entity("w", circle(60, 60), "obstacle",
               {"kind": "path", "waypoints": [[60, 60], [180, 60]], "speed": 30})
    for _ in range(10):                      # 1 second at 30cm/s
        e.step(0.1, BOUNDS)
    assert e.pos[0] == pytest.approx(90, abs=4)
    assert e.pos[1] == pytest.approx(60, abs=1)
    assert e.vel[0] > 0


def test_pingpong_reverses_at_the_end():
    e = Entity("w", circle(60, 90, 5), "obstacle",
               {"kind": "path", "waypoints": [[60, 90], [120, 90]],
                "mode": "pingpong", "speed": 60})
    xs = []
    for _ in range(60):
        e.step(0.1, BOUNDS)
        xs.append(float(e.pos[0]))
    assert max(xs) <= 121 and min(xs) >= 59
    assert any(b < a for a, b in zip(xs, xs[1:])), "never turned around"


def test_loop_returns_to_the_first_waypoint():
    e = Entity("w", circle(60, 90, 5), "obstacle",
               {"kind": "path", "waypoints": [[60, 90], [120, 90], [120, 150]],
                "mode": "loop", "speed": 90})
    seen_first_again = False
    for _ in range(120):
        e.step(0.1, BOUNDS)
        if seen_first_again is False and e.pos[1] < 95 and e.pos[0] < 65:
            seen_first_again = True
    assert seen_first_again


def test_once_stops_at_the_end():
    e = Entity("w", circle(60, 90, 5), "obstacle",
               {"kind": "path", "waypoints": [[60, 90], [120, 90]],
                "mode": "once", "speed": 60})
    for _ in range(60):
        e.step(0.1, BOUNDS)
    assert e.pos[0] == pytest.approx(120, abs=2)
    for _ in range(20):
        e.step(0.1, BOUNDS)
    assert e.pos[0] == pytest.approx(120, abs=2), "kept moving after 'once'"


def test_entities_reflect_rather_than_escape():
    """A path aimed off the floor must bounce, not leave."""
    e = Entity("w", circle(120, 90, 10), "obstacle",
               {"kind": "path", "waypoints": [[120, 90], [400, 90]],
                "mode": "once", "speed": 80})
    for _ in range(80):
        e.step(0.1, BOUNDS)
        assert e.pos[0] + e.radius <= BOUNDS[1] + 1e-6, "escaped through the wall"
        assert e.pos[0] - e.radius >= BOUNDS[0] - 1e-6


def test_a_flow_entity_circulates():
    expr = "vx = -(y - 90) * 0.5\nvy = (x - 120) * 0.5"
    s = EntitySet.from_data([{"id": "swirl", "shape": circle(180, 90, 8),
                               "role": "target",
                               "motion": {"kind": "flow", "expr": expr, "speed": 30}}],
                             compiler=compile_flow)
    assert s.errors == []
    e = s.by_id("swirl")
    start = e.pos.copy()
    for _ in range(40):
        e.step(0.1, BOUNDS)
    moved = float(np.linalg.norm(e.pos - start))
    assert moved > 10, "flow entity did not move"
    r0 = float(np.linalg.norm(start - np.array([120.0, 90.0])))
    r1 = float(np.linalg.norm(e.pos - np.array([120.0, 90.0])))
    assert abs(r1 - r0) < 25, "rotation field should roughly preserve radius"


def test_a_broken_flow_is_reported_and_the_entity_holds_still():
    s = EntitySet.from_data([{"id": "bad", "shape": circle(),
                               "motion": {"kind": "flow", "expr": "vx = 1"}}],
                             compiler=compile_flow)
    assert any("vy" in e for e in s.errors)
    e = s.by_id("bad")
    start = e.pos.copy()
    for _ in range(10):
        e.step(0.1, BOUNDS)
    np.testing.assert_allclose(e.pos, start)


def test_flow_cannot_escape_the_sandbox():
    for src in ("import os\nvx = 1\nvy = 1",
                "vx = ().__class__.__bases__[0].__subclasses__()\nvy = 0",
                "vx = open('/etc/passwd')\nvy = 0",
                "while True:\n    pass\nvx = 1\nvy = 1",
                "vx = eval('1')\nvy = 0"):
        fn, err = compile_flow(src)
        assert fn is None and err, f"escaped: {src!r}"


# -- backwards compatibility ---------------------------------------------------

def test_the_old_schema_still_loads_and_still_blocks(tmp_path):
    """A workspace written before entities existed must behave identically."""
    p = tmp_path / "workspace.json"
    p.write_text('{"bounds_cm": [[0,0],[240,0],[240,180],[0,180]],'
                 ' "obstacles": [{"type":"circle","center":[120,90],"radius":20}]}')
    ws = Workspace.load(p)
    assert ws.errors == []
    assert len(ws.obstacles) == 1
    assert ws.point_in_obstacle(np.array([120.0, 90.0]))
    assert not ws.is_valid_point(np.array([120.0, 90.0]))


def test_entities_block_exactly_like_static_obstacles(tmp_path):
    p = tmp_path / "workspace.json"
    p.write_text('{"bounds_cm": [[0,0],[240,0],[240,180],[0,180]],'
                 ' "entities": [{"id":"rock","role":"obstacle",'
                 '  "shape":{"type":"circle","center":[120,90],"radius":20}}]}')
    ws = Workspace.load(p)
    assert ws.errors == []
    assert ws.point_in_obstacle(np.array([120.0, 90.0]))
    assert not ws.is_valid_point(np.array([125.0, 90.0]))
    assert ws.segment_blocked([60, 90], [180, 90])


def test_a_target_role_entity_does_not_block(tmp_path):
    p = tmp_path / "workspace.json"
    p.write_text('{"bounds_cm": [[0,0],[240,0],[240,180],[0,180]],'
                 ' "entities": [{"id":"prey","role":"target",'
                 '  "shape":{"type":"circle","center":[120,90],"radius":20}}]}')
    ws = Workspace.load(p)
    assert not ws.point_in_obstacle(np.array([120.0, 90.0]))
    assert ws.is_valid_point(np.array([120.0, 90.0]))


def test_a_moving_obstacle_blocks_where_it_currently_is(tmp_path):
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    ws.entities = EntitySet.from_data([
        {"id": "w", "role": "obstacle",
         "shape": {"type": "circle", "center": [60, 90], "radius": 15},
         "motion": {"kind": "path", "waypoints": [[60, 90], [180, 90]],
                     "speed": 60, "mode": "once"}}])

    assert ws.point_in_obstacle(np.array([60.0, 90.0]))
    assert not ws.point_in_obstacle(np.array([180.0, 90.0]))

    for _ in range(25):                       # 2.5s at 60cm/s -> +150cm
        ws.step(0.1)

    assert not ws.point_in_obstacle(np.array([60.0, 90.0])), "still blocking where it was"
    assert ws.point_in_obstacle(np.array([180.0, 90.0])), "not blocking where it is"


def test_round_trip_through_json_keeps_entities(tmp_path):
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
                   path=tmp_path / "workspace.json")
    ws.entities = EntitySet.from_data([
        {"id": "w", "role": "both",
         "shape": {"type": "circle", "center": [60, 60], "radius": 12},
         "motion": {"kind": "path", "waypoints": [[60, 60], [180, 120]],
                     "mode": "pingpong", "speed": 25}}])
    assert ws.save() == []

    back = Workspace.load(tmp_path / "workspace.json")
    assert back.errors == []
    e = back.entities.by_id("w")
    assert e is not None and e.role == "both"
    assert e.motion["mode"] == "pingpong"

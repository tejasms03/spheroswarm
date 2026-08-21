import json
import pathlib

import numpy as np
import pytest

from workspace.space import Workspace, validate


@pytest.fixture
def ws():
    return Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[
            {"type": "poly", "points": [[80, 60], [110, 60], [110, 95], [80, 95]]},
            {"type": "circle", "center": [180, 40], "radius": 15},
        ],
    )


def test_default_workspace_file_loads():
    w = Workspace.load()
    assert w.errors == [], w.errors
    assert len(w.bounds_cm) >= 3


def test_point_in_bounds(ws):
    assert ws.point_in_bounds([120, 90])
    assert not ws.point_in_bounds([-5, 90])
    assert not ws.point_in_bounds([250, 90])
    assert not ws.point_in_bounds([120, 200])


def test_point_in_bounds_rejects_garbage(ws):
    assert not ws.point_in_bounds([float("nan"), 10])
    assert not ws.point_in_bounds(["a", "b"])
    assert not ws.point_in_bounds(None)


def test_point_in_obstacle_poly(ws):
    assert ws.point_in_obstacle([95, 75])
    assert not ws.point_in_obstacle([60, 75])


def test_point_in_obstacle_circle(ws):
    assert ws.point_in_obstacle([180, 40])
    assert ws.point_in_obstacle([190, 40])
    assert not ws.point_in_obstacle([200, 40])


def test_is_valid_point(ws):
    assert ws.is_valid_point([20, 20])
    assert not ws.is_valid_point([95, 75])
    assert not ws.is_valid_point([300, 300])


def test_nearest_valid_point_outside_bounds(ws):
    for p in ([-40, 90], [300, 90], [120, -20], [120, 400], [-50, -50]):
        q = ws.nearest_valid_point(p)
        assert ws.is_valid_point(q), f"{p} -> {q}"


def test_nearest_valid_point_inside_obstacle(ws):
    for p in ([95, 75], [180, 40], [185, 45]):
        q = ws.nearest_valid_point(p)
        assert ws.is_valid_point(q), f"{p} -> {q}"


def test_nearest_valid_point_is_identity_when_already_valid(ws):
    p = np.array([30.0, 40.0])
    assert np.allclose(ws.nearest_valid_point(p), p)


def test_nearest_valid_point_survives_nan(ws):
    q = ws.nearest_valid_point([float("nan"), 5])
    assert ws.is_valid_point(q)


def test_segment_blocked(ws):
    assert not ws.segment_blocked([10, 10], [10, 170])
    assert ws.segment_blocked([60, 75], [130, 75])       # straight through the poly
    assert ws.segment_blocked([160, 40], [200, 40])      # straight through the circle
    assert ws.segment_blocked([10, 10], [300, 10])       # leaves bounds


def test_random_valid_point(ws):
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert ws.is_valid_point(ws.random_valid_point(rng))


def test_save_and_reload(tmp_path, ws):
    p = tmp_path / "workspace.json"
    assert ws.save(p) == []
    back = Workspace.load(p)
    assert back.errors == []
    assert back.bounds_cm == ws.bounds_cm
    assert len(back.obstacles) == 2


def test_validate_catches_bad_shapes():
    assert validate({"bounds_cm": [[0, 0], [1, 1]]})
    assert validate({"bounds_cm": "nope"})
    assert validate({"bounds_cm": [[0, 0], [1, 0], [1, 1]],
                     "obstacles": [{"type": "blob"}]})
    assert validate({"bounds_cm": [[0, 0], [1, 0], [1, 1]],
                     "obstacles": [{"type": "circle", "center": [1, 1], "radius": -3}]})
    assert validate({"bounds_cm": [[0, 0], [1, 0], [float("nan"), 1]]})


def test_load_missing_file_reports_not_raises(tmp_path):
    w = Workspace.load(tmp_path / "nope.json")
    assert w.errors
    assert w.point_in_bounds([10, 10])       # still usable via the default square


def test_non_rectangular_bounds():
    tri = Workspace(bounds_cm=[[0, 0], [200, 0], [0, 200]])
    assert tri.point_in_bounds([20, 20])
    assert not tri.point_in_bounds([180, 180])
    q = tri.nearest_valid_point([180, 180])
    assert tri.is_valid_point(q)


# -- rebuilding the arena must not delete what lives in it -------------------

def test_make_preserves_entities_across_a_rebuild(tmp_path):
    """Recalibrating is a geometry job; the entities are not being deleted."""
    import subprocess
    import sys

    out = tmp_path / "ws.json"
    out.write_text(json.dumps({
        "bounds_cm": [[0, 0], [200, 0], [200, 150], [0, 150]],
        "obstacles": [],
        "entities": [{"id": "wanderer", "role": "target",
                       "shape": {"type": "circle", "center": [60, 90],
                                  "radius": 12},
                       "motion": {"kind": "path",
                                   "waypoints": [[60, 90], [180, 90]],
                                   "mode": "pingpong", "speed": 25}}]}))

    root = pathlib.Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "workspace.make",
                        "--width", "240", "--height", "180", "--out", str(out)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    ws = Workspace.load(out)
    assert round(ws.width) == 240 and round(ws.height) == 180
    assert [e.id for e in ws.entities] == ["wanderer"], "the entity was wiped"


def test_make_can_be_told_to_drop_entities(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "ws.json"
    out.write_text(json.dumps({
        "bounds_cm": [[0, 0], [200, 0], [200, 150], [0, 150]],
        "obstacles": [],
        "entities": [{"id": "wanderer", "role": "target",
                       "shape": {"type": "circle", "center": [60, 90],
                                  "radius": 12}}]}))

    root = pathlib.Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "workspace.make", "--reset-entities",
                        "--width", "240", "--height", "180", "--out", str(out)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(Workspace.load(out).entities) == 0

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


# -- arena orientation -------------------------------------------------------

def test_any_click_order_gives_the_declared_frame():
    """workspace.json declares "top-left, x right, y down". Four corners can be
    clicked starting anywhere and going either way; three of those eight
    readings give a rotated or MIRRORED frame.

    A rotation is survivable — one heading offset cancels it. A mirror is not:
    it turns a commanded heading into its reflection, so the error changes sign
    with direction and no single offset can cancel it. That is the bug this
    guards, and it cost a session of blaming the controller.
    """
    import cv2
    import numpy as np

    from vision.homography import Homography

    quad = [(100, 100), (500, 110), (510, 400), (90, 390)]
    orders = {
        "top-left, clockwise": quad,
        "bottom-left, anticlockwise": [quad[3], quad[2], quad[1], quad[0]],
        "top-right, clockwise": [quad[1], quad[2], quad[3], quad[0]],
        "bottom-right, clockwise": [quad[2], quad[3], quad[0], quad[1]],
    }
    for name, pts in orders.items():
        h = Homography()
        h.set_rect(pts, 138.8, 110.8)
        inv = np.linalg.inv(h.M)
        to_px = lambda p: cv2.perspectiveTransform(
            np.array([[list(map(float, p))]]), inv)[0, 0]
        o = to_px((69.4, 55.4))
        down = to_px((69.4, 75.4))
        assert down[1] > o[1], f"{name}: arena +y goes up the image — mirrored"
        right = to_px((89.4, 55.4))
        assert right[0] > o[0], f"{name}: arena +x goes left — mirrored"


def test_a_mirrored_frame_cannot_be_fixed_by_an_offset():
    """Why the guard above has to exist rather than being a warning."""
    import numpy as np

    from fleet.handle import velocity_to_command

    errors = []
    for deg in range(0, 360, 45):
        r = np.radians(deg)
        intended = np.array([np.sin(r), np.cos(r)])
        mirrored = np.array([intended[0], -intended[1]])
        got = velocity_to_command(mirrored)[0]
        errors.append((got - deg + 180) % 360 - 180)

    assert max(errors) - min(errors) > 180, (
        "a mirror must show as an error that changes with direction — "
        "if it were constant, a heading_offset would absorb it")


def test_a_mirrored_camera_cannot_be_seen_in_the_picture():
    """Why `flip y` has to exist even though `_orient` already runs.

    `_orient` makes the arena frame agree with the IMAGE. If the camera
    delivers a mirrored image, the image is already a reflection of the world,
    so a frame consistent with it is inconsistent with reality — and every
    check available inside the picture passes. Corners map cleanly, the grid
    sits on the floor, and every heading comes out reflected.
    """
    import numpy as np

    from vision.homography import Homography

    quad = [(100, 100), (500, 110), (510, 400), (90, 390)]
    h = Homography()
    h.set_rect(quad, 138.8, 110.8)
    back = h.to_cm(h.corners)
    want = np.array([[0, 0], [138.8, 0], [138.8, 110.8], [0, 110.8]])
    clean_before = float(np.abs(back - want).max())

    h.flip_y()
    back = h.to_cm(h.corners)
    clean_after = float(np.abs(back - want).max())

    assert clean_before < 1.0
    assert clean_after < 1.0, (
        "a mirrored arena maps its own corners just as cleanly — which is "
        "exactly why nothing in the image can detect it")


def test_flipping_twice_is_the_identity():
    """So a trainer who guesses wrong can press it again."""
    import numpy as np

    from vision.homography import Homography

    h = Homography()
    h.set_rect([(100, 100), (500, 110), (510, 400), (90, 390)], 138.8, 110.8)
    before = h.to_cm([[300, 250], [180, 300]])
    h.flip_y()
    h.flip_y()
    assert np.allclose(before, h.to_cm([[300, 250], [180, 300]]))


def test_flipping_discards_a_parallax_fit():
    """Its nadir was measured on the other side of the arena."""
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(100, 100), (500, 110), (510, 400), (90, 390)], 138.8, 110.8)
    h.parallax = {"nadir_cm": [70.0, 20.0], "scale": 1.05}
    h.flip_y()
    assert h.parallax is None

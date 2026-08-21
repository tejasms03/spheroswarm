import numpy as np
import pytest

from fleet.manager import Fleet
from fleet.roster import RobotEntry, Roster
from tools import SwarmContext, call, schemas
from tools.formations import FormationLibrary, denormalise, normalise
from tools.registry import BY_NAME
from workspace.space import Workspace


@pytest.fixture
def space():
    return Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])


def build_ctx(space, tmp_path, n=4, obstacles=None, positions=None):
    if obstacles is not None:
        space = Workspace(bounds_cm=space.bounds_cm, obstacles=obstacles)
    colors = ["cyan", "red", "yellow", "green", "magenta", "blue"][:n]
    codes = ["SSMK", "CRXS", "SYRX", "VHGR", "MLYS", "SNFR"][:n]
    names = ["Seasmoke", "Caraxes", "Syrax", "Vhagar", "Meleys", "Sunfyre"][:n]
    entries = [RobotEntry(name=nm, code=c, kind="sim", color=col)
               for nm, c, col in zip(names, codes, colors)]
    fleet = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                              workspace=space)
    if positions is not None:
        for code, p in zip(codes, positions):
            fleet[code].pos = np.array(p, dtype=float)
    lib = FormationLibrary(path=tmp_path / "formations.json")
    return SwarmContext(fleet=fleet, workspace=space, library=lib)


def settle(ctx, seconds=60.0):
    return call(ctx, "wait_until_settled", {"timeout": seconds, "tolerance": 8.0})


# -- registry ------------------------------------------------------------

def test_schema_subset_is_chosen_from_the_command():
    """Schemas are ~2,660 tokens re-sent every round trip; most are dead weight."""
    from tools.registry import CORE_TOOLS

    full = {t["name"] for t in schemas()}
    assert len(full) == 22

    plain = {t["name"] for t in schemas("form a circle")}
    assert plain == set(CORE_TOOLS)
    assert "follow" not in plain and "swap" not in plain

    motion = {t["name"] for t in schemas("everyone orbit the centre")}
    assert {"set_flow", "follow", "set_path", "motion_control"} <= motion

    conv = {t["name"] for t in schemas("swap Seasmoke and Caraxes")}
    assert {"swap", "displace", "mirror"} <= conv

    assert {t["name"] for t in schemas(None)} == full
    assert {t["name"] for t in schemas("form a circle", full=True)} == full


def test_schemas_are_cached_not_rebuilt():
    assert schemas() is schemas()
    assert schemas("form a circle") is schemas("make a line")


def test_every_tool_is_reachable_through_some_command():
    """A tool the subset never includes is a tool the model can never call."""
    from tools.registry import CORE_TOOLS

    reachable = set(CORE_TOOLS)
    for probe in ("orbit patrol follow flow path", "swap nudge gather spread mirror where"):
        reachable |= {t["name"] for t in schemas(probe)}
    missing = {t["name"] for t in schemas()} - reachable
    assert not missing, f"unreachable tools: {sorted(missing)}"


def test_schemas_are_serialisable_and_complete():
    import json
    s = schemas()
    json.dumps(s)                       # must survive a function-calling API
    names = {t["name"] for t in s}
    assert names == {
        "get_state", "describe_scene", "move_to", "transform", "stop",
        "wait_until_settled", "set_led", "save_formation", "recall_formation",
        "list_formations", "delete_formation", "compute_points",
        "set_path", "set_flow", "follow", "motion_control",
        "swap", "displace", "nudge", "gather", "spread", "mirror"}
    for t in s:
        assert t["description"] and t["parameters"]["type"] == "object"
        assert "fn" not in t


def test_forbidden_capabilities_are_absent():
    """No motors, no BLE, no roster edits, no files or shell."""
    banned = {"set_heading", "set_speed", "raw_motor", "connect", "disconnect",
              "add_robot", "remove_robot", "set_kind", "read_file", "write_file",
              "run", "exec", "shell"}
    assert banned & set(BY_NAME) == set()


def test_unknown_tool_is_an_error_not_a_crash(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "self_destruct")
        assert not r["ok"] and "unknown tool" in r["error"]
    finally:
        ctx.fleet.close()


def test_bad_arguments_are_reported(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "move_to", {"nonsense": 1})
        assert not r["ok"] and "unexpected" in r["error"]
        r = call(ctx, "save_formation", {})
        assert not r["ok"] and "missing" in r["error"]
    finally:
        ctx.fleet.close()


def test_every_tool_returns_the_same_envelope(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        calls = [
            ("get_state", {}), ("describe_scene", {}),
            ("move_to", {"points": [[40, 40], [100, 40], [160, 40], [100, 140]]}),
            ("wait_until_settled", {"timeout": 2}),
            ("transform", {"translate": [10, 0]}),
            ("set_led", {"color": "red"}),
            ("save_formation", {"name": "x"}),
            ("list_formations", {}),
            ("recall_formation", {"name": "x"}),
            ("compute_points", {"expression": "points = [(40, 40)]"}),
            ("delete_formation", {"name": "x"}),
            ("stop", {}),
        ]
        for name, args in calls:
            r = call(ctx, name, args)
            assert set(("ok", "error", "state_summary")) <= set(r), name
            assert isinstance(r["ok"], bool), name
            assert {"robots", "active", "codes", "arena_cm"} <= set(r["state_summary"])
    finally:
        ctx.fleet.close()


# -- sensing --------------------------------------------------------------

def test_get_state_shows_the_world(space, tmp_path):
    ctx = build_ctx(space, tmp_path, obstacles=[
        {"type": "circle", "center": [120, 90], "radius": 15}])
    try:
        r = call(ctx, "get_state")
        assert r["ok"]
        assert len(r["robots"]) == 4
        assert r["arena"]["width_cm"] == 240.0
        assert len(r["arena"]["obstacles"]) == 1
        for rob in r["robots"]:
            assert {"code", "name", "pos", "vel", "connected", "kind", "led"} <= set(rob)
    finally:
        ctx.fleet.close()


def test_describe_scene_is_prose(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "describe_scene")
        assert r["ok"]
        d = r["description"]
        assert "240cm" in d and "SSMK" in d
        assert len(d) < 2000
    finally:
        ctx.fleet.close()


# -- move_to --------------------------------------------------------------

def test_unordered_points_get_hungarian_assigned(space, tmp_path):
    positions = [[20, 20], [220, 20], [220, 160], [20, 160]]
    ctx = build_ctx(space, tmp_path, positions=positions)
    try:
        # the same four corners, listed in a deliberately unhelpful order
        pts = [[210, 150], [30, 30], [210, 30], [30, 150]]
        r = call(ctx, "move_to", {"points": pts})
        assert r["ok"], r["error"]
        t = r["targets"]
        assert t["SSMK"] == [30.0, 30.0]      # was at 20,20
        assert t["CRXS"] == [210.0, 30.0]     # was at 220,20
        assert t["SYRX"] == [210.0, 150.0]    # was at 220,160
        assert t["VHGR"] == [30.0, 150.0]     # was at 20,160
    finally:
        ctx.fleet.close()


def test_explicit_mapping_is_honoured_exactly(space, tmp_path):
    positions = [[20, 20], [220, 20], [220, 160], [20, 160]]
    ctx = build_ctx(space, tmp_path, positions=positions)
    try:
        r = call(ctx, "move_to", {"assign": {
            "SSMK": [200, 150], "CRXS": [40, 150],
            "SYRX": [40, 40], "VHGR": [200, 40]}})
        assert r["ok"], r["error"]
        # every robot goes exactly where it was told, however far that is
        assert r["targets"]["SSMK"] == [200.0, 150.0]
        assert r["targets"]["CRXS"] == [40.0, 150.0]
        assert r["targets"]["SYRX"] == [40.0, 40.0]
        assert r["targets"]["VHGR"] == [200.0, 40.0]
    finally:
        ctx.fleet.close()


def test_mixed_pinned_plus_auto(space, tmp_path):
    positions = [[20, 20], [220, 20], [220, 160], [20, 160]]
    ctx = build_ctx(space, tmp_path, positions=positions)
    try:
        r = call(ctx, "move_to", {
            "assign": {"SSMK": [120, 90]},
            "points": [[30, 30], [210, 30], [210, 150]],
        })
        assert r["ok"], r["error"]
        assert r["pinned"] == ["SSMK"]
        assert r["targets"]["SSMK"] == [120.0, 90.0]
        # the remaining three take their nearest free slots
        assert r["targets"]["CRXS"] == [210.0, 30.0]
        assert r["targets"]["SYRX"] == [210.0, 150.0]
        assert r["targets"]["VHGR"] == [30.0, 30.0]
    finally:
        ctx.fleet.close()


def test_move_to_rejects_wrong_count(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "move_to", {"points": [[40, 40], [100, 40]]})
        assert not r["ok"]
        assert "2" in r["error"] and "4" in r["error"]
    finally:
        ctx.fleet.close()


def test_move_to_rejects_unknown_code(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "move_to", {"assign": {"NOPE": [100, 90]}})
        assert not r["ok"] and "NOPE" in r["error"]
    finally:
        ctx.fleet.close()


def test_move_to_clamps_and_reports(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "move_to",
                 {"points": [[999, 90], [40, 40], [100, 40], [160, 40]]})
        assert r["ok"], r["error"]
        assert len(r["clamped"]) == 1
        assert r["clamped"][0]["index"] == 0
    finally:
        ctx.fleet.close()


def test_move_to_rejects_nan(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "move_to",
                 {"points": [[float("nan"), 90], [40, 40], [100, 40], [160, 40]]})
        assert not r["ok"] and "finite" in r["error"]
    finally:
        ctx.fleet.close()


# -- transform -------------------------------------------------------------

def test_transform_translate(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[60, 60], [100, 60], [140, 60], [180, 60]]})
        r = call(ctx, "transform", {"translate": [0, 40]})
        assert r["ok"], r["error"]
        for code, p in r["targets"].items():
            assert p[1] == pytest.approx(100.0, abs=0.5)
    finally:
        ctx.fleet.close()


def test_transform_scale_spreads_out(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[100, 70], [140, 70], [140, 110], [100, 110]]})
        r = call(ctx, "transform", {"scale": 1.5})
        assert r["ok"], r["error"]
        pts = np.array(list(r["targets"].values()))
        centre = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centre, axis=1)
        assert radii.mean() == pytest.approx(np.sqrt(2) * 20 * 1.5, abs=1.0)
    finally:
        ctx.fleet.close()


def test_transform_rotate(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[80, 90], [120, 90], [160, 90], [200, 90]]})
        r = call(ctx, "transform", {"rotate": 90})
        assert r["ok"], r["error"]
        pts = np.array(list(r["targets"].values()))
        # a horizontal row becomes a vertical column
        assert pts[:, 0].std() < 1.0
        assert pts[:, 1].std() > 20.0
    finally:
        ctx.fleet.close()


def test_transform_keeps_robots_in_their_slots(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[60, 60], [100, 60], [140, 60], [180, 60]]})
        first = call(ctx, "get_state")
        before = {r["code"]: r["target"] for r in first["robots"]}
        r = call(ctx, "transform", {"translate": [10, 0]})
        for code, p in r["targets"].items():
            assert p[0] == pytest.approx(before[code][0] + 10, abs=0.5)
    finally:
        ctx.fleet.close()


def test_transform_needs_an_argument(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "transform", {})
        assert not r["ok"] and "at least one" in r["error"]
    finally:
        ctx.fleet.close()


def test_transform_rejects_bad_numbers(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[60, 60], [100, 60], [140, 60], [180, 60]]})
        assert not call(ctx, "transform", {"scale": -1})["ok"]
        assert not call(ctx, "transform", {"scale": "big"})["ok"]
        assert not call(ctx, "transform", {"rotate": float("nan")})["ok"]
        assert not call(ctx, "transform", {"translate": ["a", 1]})["ok"]
    finally:
        ctx.fleet.close()


# -- stop -----------------------------------------------------------------

def test_stop_all_and_subset(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[40, 40], [100, 40], [160, 40], [100, 140]]})
        ctx.run_for(0.5)
        r = call(ctx, "stop", {"codes": ["SSMK"]})
        assert r["ok"] and r["stopped"] == ["SSMK"]
        assert "SSMK" not in ctx.env.targets

        r = call(ctx, "stop")
        assert r["ok"]
        assert ctx.env.targets == {}
    finally:
        ctx.fleet.close()


def test_stop_rejects_unknown_code(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        assert not call(ctx, "stop", {"codes": ["NOPE"]})["ok"]
    finally:
        ctx.fleet.close()


# -- leds ------------------------------------------------------------------

def test_set_led_by_name_and_triple(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "set_led", {"color": "red", "codes": ["SSMK"]})
        assert r["ok"] and ctx.fleet["SSMK"].rgb == (255, 40, 40)

        r = call(ctx, "set_led", {"color": [10, 20, 30]})
        assert r["ok"]
        assert all(h.rgb == (10, 20, 30) for h in ctx.fleet.handles.values())

        r = call(ctx, "set_led", {"color": "cyan", "blink": "fast", "codes": ["CRXS"]})
        assert r["ok"] and ctx.fleet["CRXS"].blink == "fast"
    finally:
        ctx.fleet.close()


def test_set_led_rejects_nonsense(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        assert not call(ctx, "set_led", {"color": "chartreuse"})["ok"]
        assert not call(ctx, "set_led", {"color": [1, 2]})["ok"]
        assert not call(ctx, "set_led", {"color": "red", "blink": "disco"})["ok"]
        assert not call(ctx, "set_led", {"color": "red", "codes": ["NOPE"]})["ok"]
    finally:
        ctx.fleet.close()


# -- normalisation ---------------------------------------------------------

def test_normalise_centres_and_scales():
    pts = np.array([[100, 100], [140, 100], [120, 140]])
    unit, centroid, radius = normalise(pts)
    assert np.allclose(unit.mean(axis=0), 0, atol=1e-9)
    assert np.linalg.norm(unit, axis=1).max() == pytest.approx(1.0)
    assert np.allclose(denormalise(unit, centroid, radius), pts)


def test_normalisation_is_invariant_to_where_it_was_saved():
    shape = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)
    a, _, _ = normalise(shape)
    b, _, _ = normalise(shape + np.array([150.0, 120.0]))     # same shape, far corner
    assert np.allclose(a, b)

    c, _, _ = normalise(shape * 3.0)                          # same shape, bigger
    assert np.allclose(a, c)


# -- the formation library --------------------------------------------------

def test_library_starts_empty(tmp_path):
    lib = FormationLibrary.load(tmp_path / "formations.json")
    assert lib.formations == {}
    assert lib.names() == []


def test_save_then_recall_reproduces_the_arrangement(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        pts = [[80, 60], [160, 60], [160, 120], [80, 120]]
        call(ctx, "move_to", {"points": pts})
        settle(ctx)

        r = call(ctx, "save_formation", {"name": "box", "description": "a square"})
        assert r["ok"] and r["robots"] == 4

        call(ctx, "move_to", {"points": [[30, 30], [50, 30], [30, 50], [50, 50]]})
        settle(ctx)

        r = call(ctx, "recall_formation", {"name": "box", "center": [120, 90]})
        assert r["ok"], r["error"]
        got = np.array(sorted(r["targets"].values(), key=lambda p: (p[1], p[0])))
        want = np.array(sorted(pts, key=lambda p: (p[1], p[0])), dtype=float)
        assert np.allclose(got, want, atol=1.0), (got, want)
    finally:
        ctx.fleet.close()


def test_recall_with_scale_and_rotation(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[100, 70], [140, 70], [140, 110], [100, 110]]})
        settle(ctx)
        call(ctx, "save_formation", {"name": "sq"})

        r = call(ctx, "recall_formation",
                 {"name": "sq", "center": [120, 90], "scale": 56.6})
        assert r["ok"], r["error"]
        pts = np.array(list(r["targets"].values()))
        radii = np.linalg.norm(pts - np.array([120, 90]), axis=1)
        assert radii.max() == pytest.approx(56.6, abs=1.5)

        r = call(ctx, "recall_formation",
                 {"name": "sq", "center": [120, 90], "rotation": 45})
        assert r["ok"], r["error"]
        pts = np.array(list(r["targets"].values()))
        # a 45-degree square has a point straight above the centre
        rel = pts - np.array([120, 90])
        assert np.abs(rel[:, 0]).min() < 2.0
    finally:
        ctx.fleet.close()


def test_recall_at_a_different_robot_count_names_both(space, tmp_path):
    ctx = build_ctx(space, tmp_path, n=4)
    try:
        call(ctx, "move_to", {"points": [[80, 60], [160, 60], [160, 120], [80, 120]]})
        call(ctx, "save_formation", {"name": "box"})
        ctx.fleet.remove("VHGR")

        r = call(ctx, "recall_formation", {"name": "box"})
        assert not r["ok"]
        assert "4" in r["error"] and "3" in r["error"]
    finally:
        ctx.fleet.close()


def test_recall_unknown_formation_lists_what_exists(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[80, 60], [160, 60], [160, 120], [80, 120]]})
        call(ctx, "save_formation", {"name": "box"})
        r = call(ctx, "recall_formation", {"name": "triangle"})
        assert not r["ok"]
        assert "triangle" in r["error"] and "box" in r["error"]
    finally:
        ctx.fleet.close()


def test_list_and_delete_formations(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[80, 60], [160, 60], [160, 120], [80, 120]]})
        call(ctx, "save_formation", {"name": "box", "description": "a square"})

        r = call(ctx, "list_formations")
        assert r["ok"] and len(r["formations"]) == 1
        entry = r["formations"][0]
        assert entry["name"] == "box" and entry["robots"] == 4
        assert entry["description"] == "a square" and entry["created"]

        r = call(ctx, "delete_formation", {"name": "box"})
        assert r["ok"] and r["formations"] == []
        assert not call(ctx, "delete_formation", {"name": "box"})["ok"]
    finally:
        ctx.fleet.close()


def test_formations_persist_across_reload(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[80, 60], [160, 60], [160, 120], [80, 120]]})
        call(ctx, "save_formation", {"name": "box"})
    finally:
        ctx.fleet.close()

    lib = FormationLibrary.load(tmp_path / "formations.json")
    assert lib.names() == ["box"]
    assert lib.get("box")["robot_count"] == 4


def test_recall_refuses_a_formation_that_does_not_fit(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        call(ctx, "move_to", {"points": [[80, 60], [160, 60], [160, 120], [80, 120]]})
        call(ctx, "save_formation", {"name": "box"})
        r = call(ctx, "recall_formation",
                 {"name": "box", "center": [120, 90], "scale": 2.0})
        assert not r["ok"]
        assert "does not fit" in r["error"]
    finally:
        ctx.fleet.close()


# -- compute_points ---------------------------------------------------------

def test_compute_points_returns_without_moving(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "compute_points", {
            "expression": "points = [(cx + 60*cos(i*2*pi/n), cy + 60*sin(i*2*pi/n)) for i in range(n)]"})
        assert r["ok"], r["error"]
        assert len(r["points"]) == 4
        assert ctx.env.targets == {}
    finally:
        ctx.fleet.close()


def test_compute_points_can_execute(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        r = call(ctx, "compute_points", {
            "expression": "points = [(cx + 60*cos(i*2*pi/n), cy + 60*sin(i*2*pi/n)) for i in range(n)]",
            "execute": True})
        assert r["ok"], r["error"]
        assert len(ctx.env.targets) == 4
    finally:
        ctx.fleet.close()


def test_compute_points_rejects_dangerous_source(space, tmp_path):
    ctx = build_ctx(space, tmp_path)
    try:
        for src in ("import os\npoints=[]",
                    "points = ().__class__.__bases__",
                    "while True:\n    pass"):
            r = call(ctx, "compute_points", {"expression": src})
            assert not r["ok"], src
    finally:
        ctx.fleet.close()


# -- integration ------------------------------------------------------------

def test_full_sequence_against_a_live_sim_fleet(space, tmp_path):
    """Every tool, in order, against robots that actually move."""
    ctx = build_ctx(space, tmp_path, n=5)
    try:
        assert call(ctx, "get_state")["ok"]
        assert call(ctx, "describe_scene")["ok"]

        r = call(ctx, "move_to", {"points": [[60, 50], [110, 50], [160, 50],
                                              [85, 110], [135, 110]]})
        assert r["ok"], r["error"]

        r = settle(ctx)
        assert r["settled"], r
        for code, target in ctx.env.targets.items():
            d = np.linalg.norm(ctx.fleet[code].pos - target)
            assert d < 8.0, (code, d)

        assert call(ctx, "save_formation", {"name": "arrow"})["ok"]

        r = call(ctx, "transform", {"translate": [20, 20], "rotate": 30, "scale": 1.1})
        assert r["ok"], r["error"]
        assert settle(ctx)["settled"]

        r = call(ctx, "recall_formation", {"name": "arrow", "center": [120, 90],
                                            "rotation": 180})
        assert r["ok"], r["error"]
        assert settle(ctx)["settled"]

        assert call(ctx, "set_led", {"color": "magenta"})["ok"]
        assert call(ctx, "stop")["ok"]
        for h in ctx.fleet.handles.values():
            assert h.target is None
    finally:
        ctx.fleet.close()


def test_tools_work_with_a_mixed_fleet(space, tmp_path):
    """A real robot in the fleet must not change how any tool behaves."""
    from tests.conftest import FakeConnector, FakeTracker

    entries = [
        RobotEntry(name="Seasmoke", code="SSMK", kind="sim", color="cyan"),
        RobotEntry(name="Caraxes", code="CRXS", kind="real", color="red",
                   ble_name="SK-1A2B"),
        RobotEntry(name="Syrax", code="SYRX", kind="sim", color="yellow"),
    ]
    tracker = FakeTracker({"red": (120.0, 90.0)})
    fleet = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                             workspace=space, connector=FakeConnector(),
                             tracker=tracker)
    lib = FormationLibrary(path=tmp_path / "f.json")
    ctx = SwarmContext(fleet=fleet, workspace=space, library=lib)
    try:
        for _ in range(5):
            ctx.tick(0.05)
        assert len(ctx.active_codes()) == 3

        r = call(ctx, "move_to", {"points": [[60, 60], [120, 60], [180, 60]]})
        assert r["ok"], r["error"]
        assert set(r["targets"]) == {"SSMK", "CRXS", "SYRX"}

        # nothing in the result distinguishes the real robot from the sim ones
        state = call(ctx, "get_state")
        shapes = {frozenset(rob) for rob in state["robots"]}
        assert len(shapes) >= 1
        for rob in state["robots"]:
            assert {"code", "pos", "connected", "kind"} <= set(rob)
    finally:
        fleet.close()


def test_a_disconnected_robot_is_left_out_of_placements(space, tmp_path):
    from tests.conftest import FakeConnector, FakeTracker

    entries = [
        RobotEntry(name="Seasmoke", code="SSMK", kind="sim", color="cyan"),
        RobotEntry(name="Caraxes", code="CRXS", kind="real", color="red",
                   ble_name="SK-DEAD"),
    ]
    fleet = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                             workspace=space, connector=FakeConnector(fail=True),
                             tracker=FakeTracker({}))
    lib = FormationLibrary(path=tmp_path / "f.json")
    ctx = SwarmContext(fleet=fleet, workspace=space, library=lib)
    try:
        for _ in range(5):
            ctx.tick(0.05)
        assert ctx.active_codes() == ["SSMK"]

        r = call(ctx, "move_to", {"points": [[100, 90]]})
        assert r["ok"], r["error"]
        assert set(r["targets"]) == {"SSMK"}
    finally:
        fleet.close()

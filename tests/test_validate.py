from pathlib import Path

import numpy as np
import pytest

from tools.validate import (check_source, compute_points, run_sandboxed,
                            validate_targets)
from workspace.space import Workspace

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def space():
    return Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[{"type": "poly", "points": [[80, 60], [110, 60], [110, 95], [80, 95]]},
                   {"type": "circle", "center": [180, 40], "radius": 15}],
    )


# -- shape of the input ---------------------------------------------------

def test_accepts_a_clean_set(space):
    r = validate_targets([[40, 40], [140, 40], [40, 140]], space, expected_count=3)
    assert r["ok"] and r["error"] is None
    assert r["clamped"] == []
    assert len(r["points"]) == 3


def test_rejects_wrong_count(space):
    r = validate_targets([[40, 40], [140, 40]], space, expected_count=3)
    assert not r["ok"]
    assert "2" in r["error"] and "3" in r["error"]


def test_rejects_nan(space):
    r = validate_targets([[40, 40], [float("nan"), 90]], space)
    assert not r["ok"]
    assert "finite" in r["error"]
    assert r["problems"][0]["index"] == 1


def test_rejects_infinity(space):
    assert not validate_targets([[float("inf"), 10]], space)["ok"]
    assert not validate_targets([[10, float("-inf")]], space)["ok"]


def test_rejects_strings(space):
    r = validate_targets([[40, 40], ["120", "90"]], space)
    assert not r["ok"]
    assert "finite" in r["error"]


def test_rejects_nulls(space):
    assert not validate_targets([[40, 40], None], space)["ok"]
    assert not validate_targets([[40, 40], [None, 90]], space)["ok"]


def test_rejects_wrong_arity(space):
    assert not validate_targets([[40, 40, 40]], space)["ok"]
    assert not validate_targets([[40]], space)["ok"]


def test_rejects_a_bare_string_or_dict(space):
    assert not validate_targets("60,40 120,40", space)["ok"]
    assert not validate_targets({"a": 1}, space)["ok"]


def test_rejects_booleans_masquerading_as_numbers(space):
    assert not validate_targets([[True, False]], space)["ok"]


def test_accepts_numpy_and_tuples(space):
    r = validate_targets([np.array([40.0, 40.0]), (140.0, 40.0)], space)
    assert r["ok"], r["error"]


def test_empty_set_is_ok(space):
    r = validate_targets([], space)
    assert r["ok"]
    assert r["points"] == []


# -- geometry --------------------------------------------------------------

def test_clamps_out_of_bounds_and_says_which(space):
    r = validate_targets([[40, 40], [500, 90], [-30, 20]], space)
    assert r["ok"], r["error"]
    assert len(r["clamped"]) == 2
    assert {c["index"] for c in r["clamped"]} == {1, 2}
    for c in r["clamped"]:
        assert c["reason"] == "outside the arena"
        assert space.is_valid_point(np.array(c["to"]))
    for p in r["points"]:
        assert space.is_valid_point(np.array(p))


def test_clamps_points_inside_an_obstacle(space):
    r = validate_targets([[95, 75], [20, 20], [180, 40]], space)
    assert r["ok"], r["error"]
    assert {c["index"] for c in r["clamped"]} == {0, 2}
    assert all(c["reason"] == "inside an obstacle" for c in r["clamped"])
    for p in r["points"]:
        assert space.is_valid_point(np.array(p))


def test_can_refuse_instead_of_clamping(space):
    r = validate_targets([[500, 90]], space, clamp=False)
    assert not r["ok"]
    assert "outside the arena" in r["error"]

    r = validate_targets([[95, 75]], space, clamp=False)
    assert not r["ok"]
    assert "inside an obstacle" in r["error"]


def test_rejects_overlapping_targets_naming_the_pair(space):
    r = validate_targets([[40, 40], [45, 45], [140, 140]], space)
    assert not r["ok"]
    assert "20cm apart" in r["error"]
    assert r["problems"][0]["pair"] == [0, 1]
    assert r["problems"][0]["distance"] == pytest.approx(7.1, abs=0.2)


def test_separation_is_checked_after_clamping(space):
    # two points that only collide once both are pulled back inside
    r = validate_targets([[300, 90], [305, 92]], space)
    assert not r["ok"]
    assert "apart" in r["error"]


def test_identical_points_rejected(space):
    r = validate_targets([[60, 60], [60, 60]], space)
    assert not r["ok"]


def test_separation_threshold_is_configurable(space):
    pts = [[40, 40], [55, 40]]
    assert not validate_targets(pts, space)["ok"]
    assert validate_targets(pts, space, min_separation=10.0)["ok"]


def test_never_raises_on_hostile_input(space):
    for bad in (None, 42, object(), [[{}, {}]], [[[1], [2]]], b"xy"):
        r = validate_targets(bad, space)
        assert isinstance(r, dict) and "ok" in r
        assert r["ok"] is False or r["points"] == []


# -- the sandbox: static gate ---------------------------------------------

def test_rejects_imports():
    assert "import" in check_source("import os\npoints = []")
    assert "import" in check_source("from os import system\npoints = []")


def test_rejects_dunders():
    assert check_source("points = [].__class__") is not None
    assert check_source("points = __builtins__") is not None
    assert check_source("points = ().__class__.__bases__") is not None


def test_rejects_while_loops():
    assert "while" in check_source("while True:\n    pass\npoints = []")


def test_rejects_lambdas_and_defs():
    assert check_source("f = lambda x: x\npoints = []") is not None
    assert check_source("def f():\n    return 1\npoints = []") is not None


def test_rejects_attribute_access_outside_math():
    assert check_source("points = 'abc'.upper()") is not None
    assert check_source("points = [].pop()") is not None
    assert check_source("points = ''.join([])") is not None


def test_allows_building_a_list_with_append():
    assert check_source("points = []\npoints.append((1, 2))") is None


def test_allows_math_allowlist():
    assert check_source("points = [(math.cos(0), math.sin(0))]") is None
    assert check_source("points = [(cos(0), sin(0))]") is None


def test_rejects_math_outside_allowlist():
    assert check_source("points = math.factorial(5)") is not None


def test_rejects_open_and_eval():
    assert check_source("points = open('/etc/passwd')") is not None
    assert check_source("points = eval('1')") is not None
    assert check_source("points = exec('x=1')") is not None


# -- the sandbox: execution ------------------------------------------------

def test_accepts_a_parametric_spiral(space):
    src = """
points = []
for i in range(6):
    a = i * 2 * pi / 6
    r = 30 + i * 8
    points.append((120 + r * cos(a), 90 + r * sin(a)))
"""
    r = compute_points(src, space, expected_count=6)
    assert r["ok"], r["error"]
    assert len(r["points"]) == 6
    for p in r["points"]:
        assert space.is_valid_point(np.array(p))


def test_accepts_a_comprehension(space):
    src = "points = [(40 + i * 35, 40) for i in range(5)]"
    r = compute_points(src, space, expected_count=5)
    assert r["ok"], r["error"]
    assert len(r["points"]) == 5


def test_timeout_is_enforced(space):
    # no while loop, but a range big enough to run for minutes
    src = "points = [(i, i) for i in range(200000000)]"
    r = compute_points(src, space, timeout=1.0)
    assert not r["ok"]
    assert "did not finish" in r["error"] or "MemoryError" in r["error"]


def test_timeout_message_names_the_limit(space):
    src = "total = 0\nfor i in range(100000000):\n    total = total + i\npoints = [(1, 1)]"
    r = compute_points(src, space, timeout=0.5)
    assert not r["ok"]
    assert "0.5" in r["error"]


def test_result_must_be_points(space):
    assert not compute_points("points = 5", space)["ok"]
    assert not compute_points("points = 'hello'", space)["ok"]
    assert not compute_points("x = 1", space)["ok"]
    assert not compute_points("points = [(1,)]", space)["ok"]


def test_sandbox_output_is_validated_like_everything_else(space):
    # generates points that land on top of each other
    r = compute_points("points = [(60, 60), (61, 61)]", space)
    assert not r["ok"]
    assert "apart" in r["error"]

    # and points outside the arena get clamped, not accepted blindly
    r = compute_points("points = [(999, 999), (20, 20)]", space)
    assert r["ok"], r["error"]
    assert r["clamped"][0]["index"] == 0


def test_runtime_errors_are_reported_not_raised(space):
    r = compute_points("points = [(1 / 0, 0)]", space)
    assert not r["ok"]
    assert "ZeroDivisionError" in r["error"]


def test_variables_are_injected(space):
    r = compute_points("points = [(cx, cy)]", space,
                       variables={"cx": 120.0, "cy": 90.0})
    assert r["ok"], r["error"]
    assert r["points"][0] == [120.0, 90.0]


def test_sandbox_cannot_reach_the_filesystem():
    r = run_sandboxed("points = open('/etc/passwd').read()")
    assert not r["ok"]


def test_sandbox_cannot_escape_via_subscript_tricks():
    for src in ("points = ().__class__.__bases__[0].__subclasses__()",
                "points = [].__getattribute__('append')",
                "points = type(1).__mro__"):
        assert not run_sandboxed(src)["ok"]


# -- clearance: targets must be reachable, not merely legal --------------

def test_clamped_targets_keep_a_standoff_from_obstacles(space):
    """A point 0.5cm off an obstacle is legal geometry and a useless target.

    The navigation controller refuses to enter an obstacle's repulsion field,
    so a target buried in one leaves a robot orbiting it forever. Clamping has
    to land somewhere a robot can actually sit.
    """
    from tools.validate import TARGET_CLEARANCE

    r = validate_targets([[180, 40]], space)          # dead centre of the circle
    assert r["ok"], r["error"]
    got = np.array(r["points"][0])
    centre = np.array([180.0, 40.0])
    assert np.linalg.norm(got - centre) >= 15 + TARGET_CLEARANCE - 1.0


def test_clamped_targets_keep_a_standoff_from_walls(space):
    from tools.validate import TARGET_CLEARANCE

    r = validate_targets([[-50, 90]], space)
    assert r["ok"], r["error"]
    assert r["points"][0][0] >= TARGET_CLEARANCE - 1.0


def test_clearance_can_be_switched_off(space):
    r = validate_targets([[-50, 90]], space, clearance=0.0)
    assert r["ok"], r["error"]
    assert r["points"][0][0] < 2.0


def test_a_robot_actually_reaches_a_clamped_target():
    """The regression this clearance exists for: settle, do not orbit."""
    import numpy as np

    from swarm.navigate import Navigate
    from swarm.sim import SwarmEnv

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
                   obstacles=[{"type": "circle", "center": [180, 40], "radius": 15}])
    report = validate_targets([[190, 40]], ws, expected_count=1)
    assert report["ok"] and report["clamped"]

    env = SwarmEnv(1, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[40.0, 140.0]])
    env.vel = np.zeros((1, 2))
    nav = Navigate(targets=np.array(report["points"], dtype=float))
    for _ in range(600):
        env.step(nav.act(env))

    gap = float(np.linalg.norm(env.pos[0] - np.array(report["points"][0])))
    assert gap < 8.0, f"robot stalled {gap:.1f}cm from a clamped target"
    assert ws.is_valid_point(env.pos[0])


# -- the sandbox must not pay for the parent's imports ---------------------

def test_sandbox_does_not_reimport_the_parent_process(space):
    """The bug that made compute_points fail 100% of the time inside app.py.

    multiprocessing's spawn start method re-imports the parent's `__main__` in
    the child. Launched from the pygame app that meant importing torch and
    pygame before a single line of the expression ran, which consumed the whole
    one-second budget. Under pytest `__main__` is cheap, so the suite never saw
    it — hence this test runs a *real* subprocess with a heavy __main__.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, %r)
        import numpy, torch          # a deliberately expensive __main__
        from tools.validate import run_sandboxed
        t0 = time.time()
        r = run_sandboxed("points = [(i*10.0, 20.0) for i in range(6)]")
        print(r["ok"], round(time.time() - t0, 3))
    """ % str(ROOT))

    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-500:]
    ok, elapsed = out.stdout.strip().split()[-2:]
    assert ok == "True", f"sandbox failed under a heavy parent: {out.stdout}"
    assert float(elapsed) < 0.5, (
        f"sandbox took {elapsed}s under a heavy parent — it is paying for the "
        "parent's imports again")


def test_sandbox_startup_is_fast_enough_to_leave_the_budget_for_the_expression(space):
    import time

    t0 = time.time()
    r = run_sandboxed("points = [(i*10.0, 20.0) for i in range(6)]")
    assert r["ok"], r["error"]
    assert time.time() - t0 < 0.5


# -- double-escaped source: what models actually send ----------------------

def test_repairs_a_double_escaped_newline(space):
    """The failure seen in the app: `\\n` arrives as a literal backslash-n.

    Python reads that as a line continuation followed by junk — "unexpected
    character after line continuation character" — even though the code is
    otherwise fine. Unescape and run it rather than making the model guess.
    """
    src = ("corners = [[0,0],[0,180],[240,180],[240,0]]\\n"
           "points = [(float(x),float(y)) for x,y in corners]")
    r = run_sandboxed(src)
    assert r["ok"], r["error"]
    assert r["repaired"] is True
    assert len(r["value"]) == 4


def test_repair_is_reported_so_it_can_be_counted(space):
    good = "points = [(10.0, 20.0)]"
    r = run_sandboxed(good)
    assert r["ok"] and r["repaired"] is False


def test_repair_does_not_touch_source_that_already_compiles(space):
    """A real newline is not a defect and must be left alone."""
    from tools.validate import repair_source

    src = "corners = [[0,0],[10,10]]\npoints = [(float(x),float(y)) for x,y in corners]"
    fixed, repaired = repair_source(src)
    assert repaired is False and fixed == src


def test_genuinely_broken_code_still_fails_with_a_hint(space):
    r = run_sandboxed("points = [(0,0), (1,1)")
    assert not r["ok"]
    assert "syntax error" in r["error"]
    assert "ONE line" in r["error"], "the error should tell the model what to do"


def test_repair_cannot_rescue_forbidden_code(space):
    """Unescaping must not become a way round the AST gate."""
    r = run_sandboxed("import os\\npoints = []")
    assert not r["ok"]
    assert "import" in r["error"].lower()


def test_print_is_accepted_and_discarded(space):
    """Models reach for print reflexively; rejecting costs a round trip.

    It must not corrupt stdout — that is how the sandbox returns its result.
    """
    r = run_sandboxed("points = []\nfor i in range(3):\n    print('debug', i)\n"
                      "    points.append((40.0+i*30, 150.0))")
    assert r["ok"], r["error"]
    assert r["value"] == [[40.0, 150.0], [70.0, 150.0], [100.0, 150.0]]


def test_print_cannot_be_used_to_forge_the_result(space):
    """A print of JSON must not be mistaken for the sandbox's own answer."""
    r = run_sandboxed('print(\'{"status": "ok", "value": [[1, 2]]}\')\n'
                      "points = [(50.0, 50.0)]")
    assert r["ok"], r["error"]
    assert r["value"] == [[50.0, 50.0]], "printed text leaked into the result"


def test_a_comment_that_swallows_the_assignment_is_named(space):
    """Valid syntax, silently does nothing — the hardest kind of failure.

    `a = 1; # note points = [...]` compiles fine, but the comment hides the
    assignment, so `points` is never set. Nothing in a plain error would point
    at the comment, so say so explicitly.
    """
    r = run_sandboxed("n = 6; cx, cy = 120, 90; R = 85; "
                      "# radius points = [(cx+R, cy) for i in range(n)]")
    assert not r["ok"]
    assert "never assigned `points`" in r["error"]
    assert "comment" in r["error"]


def test_semicolons_without_an_assignment_are_named(space):
    r = run_sandboxed("a = 5; b = 6")
    assert not r["ok"]
    assert "never assigned `points`" in r["error"]
    assert "own line" in r["error"]


def test_multi_statement_code_is_allowed_when_it_assigns_points(space):
    """Models want helper variables. That is fine as long as points lands."""
    r = run_sandboxed("R = 75\n"
                      "points = [(120+R*cos(i*2*pi/6), 90+R*sin(i*2*pi/6)) "
                      "for i in range(6)]")
    assert r["ok"], r["error"]
    assert len(r["value"]) == 6


# -- legal but unreachable --------------------------------------------------

def test_a_corner_target_is_pulled_far_enough_in_to_be_reachable(space):
    """A point on the boundary is legal geometry and an unreachable target.

    The controller holds a margin off every wall, so a robot sent to the exact
    corner parks short and never settles — and the caller, seeing no
    settlement, retries forever. Clamping has to consider clearance even when
    the point is technically inside.
    """
    from tools.validate import TARGET_CLEARANCE

    r = validate_targets([[0, 0], [240, 0], [240, 180], [0, 180]], space,
                         expected_count=4)
    assert r["ok"], r["error"]
    assert len(r["clamped"]) == 4
    for p in r["points"]:
        assert p[0] >= TARGET_CLEARANCE - 1 and p[0] <= 240 - TARGET_CLEARANCE + 1
        assert p[1] >= TARGET_CLEARANCE - 1 and p[1] <= 180 - TARGET_CLEARANCE + 1


def test_a_target_just_inside_the_wall_is_reported_as_unreachable(space):
    r = validate_targets([[2, 90]], space, expected_count=1)
    assert r["ok"], r["error"]
    assert r["clamped"], "a target 2cm off the wall should have been moved"
    assert "reachable" in r["clamped"][0]["reason"]


def test_robots_actually_settle_on_clamped_corner_targets():
    """The regression this exists for: settle, do not park short forever."""
    import numpy as np

    from swarm.navigate import Navigate
    from swarm.sim import SwarmEnv

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    report = validate_targets([[0, 0], [240, 0], [240, 180], [0, 180]], ws,
                              expected_count=4)
    targets = np.array(report["points"], dtype=float)

    env = SwarmEnv(4, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[120., 90.], [110., 80.], [130., 100.], [100., 95.]])
    env.vel = np.zeros((4, 2))
    nav = Navigate(targets=targets)
    for _ in range(600):
        env.step(nav.act(env))

    gaps = np.linalg.norm(env.pos - targets, axis=1)
    assert (gaps <= 8.0).all(), f"did not settle: {gaps.round(1)}"


def test_clearance_zero_still_allows_boundary_targets(space):
    r = validate_targets([[0, 0]], space, expected_count=1, clearance=0.0)
    assert r["ok"], r["error"]
    assert r["points"][0] == [0.0, 0.0]


# -- a count mismatch has to say what to do about it -------------------------

def test_too_few_targets_says_how_many_to_add(open_ws):
    r = validate_targets([[60, 60], [100, 60], [140, 60], [60, 120], [100, 120]],
                          open_ws, expected_count=6)
    assert not r["ok"]
    assert "got 5 target(s) but there are 6 robot(s)" in r["error"]
    assert "add 1 more point" in r["error"], \
        "naming the numbers is not enough; a small model re-sends the same set"
    assert "20cm" in r["error"]


def test_too_many_targets_says_how_many_to_remove(open_ws):
    pts = [[40 + i * 25, 60] for i in range(8)]
    r = validate_targets(pts, open_ws, expected_count=6)
    assert not r["ok"]
    assert "remove 2 point(s)" in r["error"]
    assert "exactly 6" in r["error"]


def test_a_shape_drawn_too_small_is_told_to_scale_up(open_ws):
    """A near-miss on several pairs is a too-small shape, not a stray point."""
    pts = [[100, 80], [118, 80], [136, 80], [100, 98], [118, 98], [136, 98]]
    r = validate_targets(pts, open_ws, expected_count=6)
    assert not r["ok"]
    assert "at least 20cm apart" in r["error"]
    assert "too small" in r["error"] and "larger" in r["error"]
    assert "x larger" in r["error"], "the hint must say by how much"


def test_one_stray_point_is_not_told_to_redraw_everything(open_ws):
    """A single badly-off point should be nudged, not the whole shape rescaled."""
    pts = [[40, 40], [100, 40], [160, 40], [40, 140], [100, 140], [102, 141]]
    r = validate_targets(pts, open_ws, expected_count=6)
    assert not r["ok"]
    assert "20cm apart" in r["error"]
    assert "larger" not in r["error"]

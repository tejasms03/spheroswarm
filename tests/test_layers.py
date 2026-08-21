"""The layer stack: naming, weighting, caps, expiry, stalls."""

import time

import numpy as np
import pytest

from swarm import fields
from swarm import layers as layers_mod
from swarm.layers import (DEFAULT_DURATION, MAX_DURATION, MAX_LAYERS,
                          STALL_SECONDS, LayerStack, clamp_duration)
from swarm.navigate import Navigate
from swarm.sim import SwarmEnv
from workspace.space import Workspace


@pytest.fixture
def stack():
    return LayerStack()


def open_ws():
    return Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])


def env_with(codes, positions, ws=None):
    ws = ws or open_ws()
    env = SwarmEnv(len(codes), randomize=False, seed=0, workspace=ws)
    env.pos = np.asarray(positions, dtype=float)
    env.vel = np.zeros((len(codes), 2))
    env.codes = list(codes)
    return env


# -- naming and replacement --------------------------------------------------

def test_push_and_list(stack):
    layer, errs = stack.push_layer("SSMK", "orbit", "flow", {"field": None}, 0.6)
    assert errs == [] and layer.name == "orbit"
    assert [l["name"] for l in stack.active_layers("SSMK")] == ["orbit"]


def test_a_repeated_name_replaces_rather_than_stacks(stack):
    stack.push_layer("SSMK", "orbit", "flow", {}, 0.6)
    stack.push_layer("SSMK", "orbit", "flow", {}, 0.9)
    layers = stack.active_layers("SSMK")
    assert len(layers) == 1, "same name should replace, not duplicate"
    assert layers[0]["weight"] == 0.9


def test_named_layers_are_individually_removable(stack):
    stack.push_layer("SSMK", "orbit", "flow", {}, 0.6)
    stack.push_layer("SSMK", "patrol", "path", {"waypoints": [[10, 10]]}, 1.0)
    assert stack.pop_layer("SSMK", "orbit") == []
    assert [l["name"] for l in stack.active_layers("SSMK")] == ["patrol"]


def test_popping_something_absent_is_an_error_not_a_crash(stack):
    errs = stack.pop_layer("SSMK", "ghost")
    assert errs and "ghost" in errs[0]


def test_unknown_kind_is_rejected(stack):
    layer, errs = stack.push_layer("SSMK", "x", "teleport", {}, 1.0)
    assert layer is None and errs


def test_clear_removes_everything_for_one_robot(stack):
    stack.push_layer("SSMK", "a", "flow", {}, 1.0)
    stack.push_layer("CRXS", "b", "flow", {}, 1.0)
    stack.clear_layers("SSMK")
    assert stack.active_layers("SSMK") == []
    assert stack.active_layers("CRXS")


def test_clear_all(stack):
    stack.push_layer("SSMK", "a", "flow", {}, 1.0)
    stack.push_layer("CRXS", "b", "flow", {}, 1.0)
    stack.clear_layers()
    assert stack.codes() == []


# -- the cap ------------------------------------------------------------------

def test_cap_is_enforced(stack):
    for i in range(MAX_LAYERS):
        _, errs = stack.push_layer("SSMK", f"l{i}", "flow", {}, 1.0)
        assert errs == []
    layer, errs = stack.push_layer("SSMK", "one_too_many", "flow", {}, 1.0)
    assert layer is None
    assert errs and str(MAX_LAYERS) in errs[0]


def test_replacing_an_existing_layer_at_the_cap_is_allowed(stack):
    for i in range(MAX_LAYERS):
        stack.push_layer("SSMK", f"l{i}", "flow", {}, 1.0)
    layer, errs = stack.push_layer("SSMK", "l0", "flow", {}, 0.2)
    assert errs == [] and layer is not None
    assert len(stack.active_layers("SSMK")) == MAX_LAYERS


# -- weights ------------------------------------------------------------------

def test_set_weight(stack):
    stack.push_layer("SSMK", "orbit", "flow", {}, 0.6)
    assert stack.set_weight("SSMK", "orbit", 0.25) == []
    assert stack.active_layers("SSMK")[0]["weight"] == 0.25


def test_set_weight_rejects_rubbish(stack):
    stack.push_layer("SSMK", "orbit", "flow", {}, 0.6)
    assert stack.set_weight("SSMK", "orbit", "heavy")
    assert stack.set_weight("SSMK", "orbit", float("nan"))
    assert stack.set_weight("SSMK", "missing", 1.0)


# -- durations -----------------------------------------------------------------

def test_duration_defaults_and_clamps():
    assert clamp_duration(None) == (DEFAULT_DURATION, False)
    assert clamp_duration(30) == (30.0, False)
    assert clamp_duration(9999) == (MAX_DURATION, True)
    assert clamp_duration(-5) == (DEFAULT_DURATION, True)
    assert clamp_duration("forever") == (DEFAULT_DURATION, True)


def test_a_layer_expires_and_pops_itself(stack):
    stack.push_layer("SSMK", "brief", "flow", {}, 1.0, duration=0.05)
    assert stack.active_layers("SSMK")
    time.sleep(0.08)
    dropped = stack.expire()
    assert ("SSMK", "brief") in dropped
    assert stack.active_layers("SSMK") == []


def test_remaining_counts_down(stack):
    stack.push_layer("SSMK", "l", "flow", {}, 1.0, duration=5.0)
    r1 = stack.active_layers("SSMK")[0]["remaining"]
    time.sleep(0.05)
    r2 = stack.active_layers("SSMK")[0]["remaining"]
    assert r2 < r1


# -- stall detection -------------------------------------------------------------

def test_stall_fires_after_the_grace_period_and_clears(stack, monkeypatch):
    import swarm.layers as L

    clock = {"t": 1000.0}
    monkeypatch.setattr(L, "now", lambda: clock["t"])

    stack.push_layer("SSMK", "l", "flow", {}, 1.0)
    stack.note_speed("SSMK", 0.5)
    assert not stack.stalled("SSMK")

    clock["t"] += 3.0
    stack.note_speed("SSMK", 0.5)
    assert stack.stalled("SSMK"), "should be flagged after 2s of near-zero speed"

    stack.note_speed("SSMK", 40.0)
    assert not stack.stalled("SSMK"), "should clear once it moves again"


def test_a_robot_with_no_layers_is_never_stalled(stack):
    stack.note_speed("SSMK", 0.0)
    assert not stack.stalled("SSMK")


# -- the blend ---------------------------------------------------------------------

def test_with_no_stack_behaviour_is_the_old_controller():
    ws = open_ws()
    tg = np.array([[180.0, 90.0]])
    a = env_with(["SSMK"], [[60.0, 90.0]], ws)
    b = env_with(["SSMK"], [[60.0, 90.0]], ws)
    plain = Navigate(targets=tg)
    layered = Navigate(targets=tg, stack=LayerStack())
    np.testing.assert_allclose(plain.act(a), layered.act(b))


def test_an_empty_stack_changes_nothing_over_a_whole_run():
    """The regression that matters: positional behaviour must be untouched."""
    ws = open_ws()
    tg = np.array([[180.0, 120.0], [70.0, 40.0]])
    out = []
    for stack in (None, LayerStack()):
        env = env_with(["SSMK", "CRXS"], [[60.0, 90.0], [150.0, 60.0]], ws)
        nav = Navigate(targets=tg, stack=stack)
        for _ in range(300):
            env.step(nav.act(env))
        out.append(env.pos.copy())
    np.testing.assert_allclose(out[0], out[1], atol=1e-9)


def test_weights_scale_a_layers_contribution():
    ws = open_ws()
    field = lambda x, y, t, ex, ey: (1.0, 0.0)

    strong = LayerStack()
    strong.push_layer("SSMK", "push", "flow", {"field": field}, 1.0)
    weak = LayerStack()
    weak.push_layer("SSMK", "push", "flow", {"field": field}, 0.2)

    moved = []
    for stack in (weak, strong):
        env = env_with(["SSMK"], [[120.0, 90.0]], ws)
        nav = Navigate(targets=np.array([[120.0, 90.0]]), stack=stack)
        for _ in range(60):
            env.step(nav.act(env))
        moved.append(float(env.pos[0][0] - 120.0))
    assert moved[1] > moved[0] > 0, f"weights had no effect: {moved}"


def test_avoidance_is_not_a_layer_and_cannot_be_weighted_down():
    """Nothing the model can reach may turn collision avoidance off."""
    stack = LayerStack()
    assert "avoid" not in [l["name"] for l in stack.active_layers("SSMK")]
    assert stack.set_weight("SSMK", "avoid", 0.0), "there is no avoid layer to weaken"

    # and it still works with a full stack of layers pushing into an obstacle
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
                   obstacles=[{"type": "circle", "center": [120, 90], "radius": 25}])
    stack.push_layer("SSMK", "push", "flow",
                     {"field": lambda x, y, t, ex, ey: (1.0, 0.0)}, 1.0)
    env = env_with(["SSMK"], [[40.0, 90.0]], ws)
    nav = Navigate(targets=np.array([[40.0, 90.0]]), stack=stack)
    worst = 0.0
    for _ in range(400):
        env.step(nav.act(env))
        d = float(np.linalg.norm(env.pos[0] - np.array([120.0, 90.0])))
        worst = max(worst, 25.0 - d)
    assert worst < 10.0, f"a flow layer drove {worst:.0f}cm into an obstacle"


def test_continuous_layers_are_detectable(stack):
    assert not stack.has_continuous()
    stack.push_layer("SSMK", "p", "path", {"mode": "once"}, 1.0)
    assert not stack.has_continuous(), "a one-shot path does settle"
    stack.push_layer("SSMK", "q", "path", {"mode": "loop"}, 1.0)
    assert stack.has_continuous()


# -- field helpers -------------------------------------------------------------------

def test_prediction_leads_a_moving_target():
    aim = fields.predict([100.0, 90.0], [50.0, 0.0], lead_time=0.4)
    assert aim[0] == pytest.approx(120.0)


def test_surround_slots_ring_the_target_evenly():
    pts = [fields.follow_point("surround", i, 6, [120, 90], [0, 0], 40)
           for i in range(6)]
    r = [float(np.linalg.norm(np.asarray(p) - [120, 90])) for p in pts]
    assert all(abs(x - 40) < 1e-6 for x in r)
    d = [float(np.linalg.norm(np.asarray(pts[i]) - np.asarray(pts[(i + 1) % 6])))
         for i in range(6)]
    assert max(d) - min(d) < 1e-6, "slots are not evenly spaced"


def test_trail_sits_behind_the_target():
    p = fields.follow_point("trail", 0, 1, [120, 90], [10, 0], 30)
    assert p[0] < 120, "trail should be behind, not ahead"


def test_slot_assignment_has_hysteresis():
    codes = ["A", "B"]
    pos = [[0.0, 0.0], [10.0, 0.0]]
    slots = [[0.0, 0.0], [11.0, 0.0]]
    previous = {"A": 1, "B": 0}
    got = fields.assign_slots(codes, pos, slots, previous=previous)
    assert got == previous, "a small advantage should not steal a slot"

    fresh = fields.assign_slots(codes, pos, slots, previous=None)
    assert fresh == {"A": 0, "B": 1}


# -- import order must not decide whether the package works ------------------

def test_workspace_imports_without_tools_being_imported_first():
    """`workspace` sits below `tools`; importing upward at module load made the
    whole package depend on import order, and only the app's order worked."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-c",
         "from workspace.space import Workspace; "
         "from workspace.flow import compile_flow; "
         "fn, err = compile_flow('vx = 1.0\\nvy = 0.0'); "
         "assert err is None, err; print('ok')"],
        cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, f"importing workspace first failed:\n{r.stderr}"
    assert "ok" in r.stdout


# -- "stalled" must mean blocked, not satisfied ------------------------------

def test_a_satisfied_layer_is_not_a_stall():
    """A follow holding station on a stationary target is doing its job."""
    s = LayerStack()
    s.push_layer("SSMK", "follow", "follow", {"target": "CRXS"}, duration=60)
    for _ in range(60):                     # well past STALL_SECONDS
        s.note_speed("SSMK", 0.0, wanted_cms=0.0)
    assert s.stalled("SSMK") is False


def test_asking_to_move_and_going_nowhere_is_a_stall():
    """Layers cancelling, or a wall in the way: asked for speed, got none."""
    s = LayerStack()
    s.push_layer("SSMK", "flow", "flow", {}, duration=60)
    start = layers_mod.now()
    while layers_mod.now() - start < STALL_SECONDS + 0.3:
        s.note_speed("SSMK", 0.0, wanted_cms=30.0)
        time.sleep(0.02)
    assert s.stalled("SSMK") is True


def test_a_stall_clears_once_it_moves_again():
    s = LayerStack()
    s.push_layer("SSMK", "flow", "flow", {}, duration=60)
    start = layers_mod.now()
    while layers_mod.now() - start < STALL_SECONDS + 0.3:
        s.note_speed("SSMK", 0.0, wanted_cms=30.0)
        time.sleep(0.02)
    assert s.stalled("SSMK") is True

    s.note_speed("SSMK", 25.0, wanted_cms=30.0)
    assert s.stalled("SSMK") is False


def test_a_satisfied_layer_clears_an_existing_stall():
    """Blocked, then the obstruction goes away and the layer is content."""
    s = LayerStack()
    s.push_layer("SSMK", "flow", "flow", {}, duration=60)
    start = layers_mod.now()
    while layers_mod.now() - start < STALL_SECONDS + 0.3:
        s.note_speed("SSMK", 0.0, wanted_cms=30.0)
        time.sleep(0.02)
    assert s.stalled("SSMK") is True

    s.note_speed("SSMK", 0.0, wanted_cms=0.0)
    assert s.stalled("SSMK") is False


def test_omitting_wanted_keeps_the_old_behaviour():
    """Callers that cannot supply it still get a stall on low commanded speed."""
    s = LayerStack()
    s.push_layer("SSMK", "flow", "flow", {}, duration=60)
    start = layers_mod.now()
    while layers_mod.now() - start < STALL_SECONDS + 0.3:
        s.note_speed("SSMK", 0.0)
        time.sleep(0.02)
    assert s.stalled("SSMK") is True


def test_a_stall_reports_what_it_asked_for():
    """"stalled" alone sends you hunting; the numbers point at the cause."""
    s = LayerStack()
    s.push_layer("SSMK", "flow", "flow", {}, duration=60)
    start = layers_mod.now()
    while layers_mod.now() - start < STALL_SECONDS + 0.3:
        s.note_speed("SSMK", 0.4, wanted_cms=31.0)
        time.sleep(0.02)
    assert s.stalled("SSMK")
    asked, got = s.stall_reason("SSMK")
    assert asked == pytest.approx(31.0) and got == pytest.approx(0.4)

    s.note_speed("SSMK", 25.0, wanted_cms=31.0)
    assert s.stall_reason("SSMK") is None, "the reason outlived the stall"

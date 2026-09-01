"""A whole routine from one tool call, and the guards that make that safe.

`MAX_TOOL_CALLS` is 8 and a five-phase routine costs about fourteen, so without
this the fleet stops mid-formation. The cap is worth keeping — it stops a model
reasoning in circles — so the fix is to spend fewer calls on steps that were
already decided, not to raise it.

Everything here runs on a simulated fleet.
"""

import numpy as np
import pytest

from tools import registry
from tools.sequence import MAX_STEPS, run_sequence


@pytest.fixture
def ctx(sim_ctx):
    return sim_ctx(n=3)


def codes(ctx):
    return [h.code for h in ctx.fleet.handles.values()]


# -- the point of it -----------------------------------------------------

def test_a_routine_that_would_blow_the_cap_fits_in_one_call(ctx):
    """Fourteen actions, one tool call. That is the whole argument."""
    all_of = codes(ctx)
    steps = [{"tool": "move_to", "args": {"points": [[40, 40], [70, 40], [100, 40]]}},
             {"wait": {"timeout": 2}},
             {"tool": "set_led", "args": {"codes": all_of, "color": "cyan"}},
             {"tool": "spread", "args": {"min_distance": 25}},
             {"wait": {"timeout": 2}},
             {"tool": "gather", "args": {"codes": all_of[1:], "around": all_of[0]}},
             {"wait": {"timeout": 2}},
             {"tool": "set_led", "args": {"codes": all_of, "color": "magenta"}},
             {"tool": "move_to", "args": {"points": [[40, 70], [70, 70], [100, 70]]}},
             {"wait": {"timeout": 2}},
             {"tool": "stop", "args": {}}]
    out = run_sequence(ctx, steps)
    assert not out.get("error"), out
    assert len(out["steps"]) == len(steps)
    from llm.agent import MAX_TOOL_CALLS
    assert len(steps) <= MAX_STEPS
    assert len(steps) > MAX_TOOL_CALLS - 2, \
        "if this routine now fits inside the cap, the argument has weakened"


def test_it_is_in_the_live_registry():
    """Adopted. `tools/registry_seq.py` was the trial copy and is gone."""
    assert "run_sequence" in registry.BY_NAME
    import importlib
    with pytest.raises(ImportError):
        importlib.import_module("tools.registry_seq")


def test_it_dispatches_through_the_registry_like_any_other_tool(ctx):
    out = registry.call(ctx, "run_sequence",
                        {"steps": [{"tool": "stop", "args": {}}]})
    assert not out.get("error"), out
    assert len(out["steps"]) == 1


def test_the_AGENT_puts_its_cancel_event_on_the_context(ctx):
    """The wiring that makes `test_cancelling_halts_between_steps` mean
    anything in production rather than only under a hand-made flag.

    `Agent.cancel()` sets an Event and `_check_cancel` reads it BETWEEN tool
    calls, which suffices for every tool that returns promptly. `run_sequence`
    runs a whole routine inside one call, so without this it sees no flag,
    decides nothing is driving, and finishes a sixteen-step routine after STOP
    was pressed.
    """
    from llm.agent import SwarmAgent
    agent = SwarmAgent(client=None, ctx=ctx)
    assert getattr(ctx, "cancel_event", None) is agent._cancel
    assert not ctx.cancel_event.is_set()

    agent.cancel()
    out = run_sequence(ctx, [{"tool": "stop", "args": {}},
                             {"tool": "stop", "args": {}}])
    assert out.get("cancelled") is True
    assert out["steps"] == []


# -- nothing runs until everything validates -----------------------------

def test_a_bad_step_moves_no_robots(ctx):
    """The property that makes this better than calling the tools by hand.

    Validating as you go means finding out step 4 is malformed with steps 1-3
    already on the floor, and the swarm in a shape nobody asked for.
    """
    before = [np.array(h.pos) for h in ctx.fleet.handles.values()]
    out = run_sequence(ctx, [
        {"tool": "move_to", "args": {"points": [[40, 40], [70, 40], [100, 40]]}},
        {"tool": "move_to", "args": {"points": [[50, 50]], "nonsense": 1}}])
    assert out.get("error") and "not run" in out["error"]
    after = [np.array(h.pos) for h in ctx.fleet.handles.values()]
    for a, b in zip(before, after):
        assert np.allclose(a, b), "a rejected sequence must not have moved anything"


def test_the_error_names_the_step_and_the_tool(ctx):
    out = run_sequence(ctx, [{"tool": "stop", "args": {}},
                             {"tool": "nope", "args": {}}])
    assert "step 2" in out["error"] and "nope" in out["error"]


def test_an_over_long_routine_is_refused(ctx):
    out = run_sequence(ctx, [{"tool": "stop", "args": {}}] * (MAX_STEPS + 1))
    assert out.get("error") and str(MAX_STEPS) in out["error"]


def test_a_sequence_cannot_run_a_sequence(ctx):
    """Otherwise the bound is nominal: nest twice and it is gone."""
    out = run_sequence(ctx, [{"tool": "run_sequence", "args": {"steps": []}}])
    assert out.get("error") and "cannot run another sequence" in out["error"]


# -- it stops when told --------------------------------------------------

def test_cancelling_halts_between_steps(ctx):
    """Without this, a sequence IS the runaway the cap was written to stop."""
    class Flag:
        def is_set(self):
            return True
    ctx.cancel_event = Flag()
    out = run_sequence(ctx, [{"tool": "stop", "args": {}},
                             {"tool": "stop", "args": {}}])
    assert out.get("cancelled") is True
    assert out["steps"] == [], "cancelled before the first step, so none ran"


def test_a_context_with_no_cancel_flag_simply_runs(ctx):
    assert not hasattr(ctx, "cancel_event")
    out = run_sequence(ctx, [{"tool": "stop", "args": {}}])
    assert not out.get("error") and len(out["steps"]) == 1


# -- failing partway -----------------------------------------------------

def test_a_step_that_fails_at_run_time_stops_the_rest(ctx):
    """Validation cannot catch everything — a point can be legal in shape and
    still be off the floor once the workspace is consulted."""
    out = run_sequence(ctx, [
        {"tool": "move_to", "args": {"points": [[9999, 9999]] * 3}},
        {"tool": "set_led", "args": {"codes": codes(ctx), "color": "red"}}])
    assert out.get("error")
    assert len(out["steps"]) == 1, "the rest of the routine must not have run"


def test_stop_on_error_false_carries_on(ctx):
    out = run_sequence(ctx, [
        {"tool": "move_to", "args": {"points": [[9999, 9999]] * 3}},
        {"tool": "stop", "args": {}}], stop_on_error=False)
    assert len(out["steps"]) == 2

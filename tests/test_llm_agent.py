import numpy as np
import pytest

from llm.agent import SwarmAgent, _compact
from llm.client import ModelError, ModelUnreachable, ToolCall
from llm.stub import StubClient


@pytest.fixture
def ctx(sim_ctx):
    return sim_ctx(n=6, seed=5)


def agent(ctx, script, **kw):
    return SwarmAgent(StubClient(script=script), ctx, **kw)


# -- the happy path ---------------------------------------------------------

def test_a_single_tool_call_then_a_reply(ctx):
    a = agent(ctx, [[("move_to", {"points": [[40, 40], [90, 40], [140, 40],
                                              [190, 90], [90, 140], [140, 140]]})],
                     "Placed them in a row."])
    r = a.command("line them up")
    assert r.ok and not r.hit_cap
    assert r.reply == "Placed them in a row."
    assert r.tool_names == ["move_to"]
    assert r.tool_calls[0].ok
    assert r.total_latency_s >= 0


def test_a_plain_answer_with_no_tools(ctx):
    a = agent(ctx, ["I know no formations yet."])
    r = a.command("what do you know?")
    assert r.ok and r.tool_calls == []


def test_several_calls_in_one_turn(ctx):
    a = agent(ctx, [[("compute_points", {"expression": "points = [(60+i*25, 60) for i in range(n)]",
                                          "execute": True}),
                      ("wait_until_settled", {"timeout": 2})],
                     "Done."])
    r = a.command("row")
    assert r.ok
    assert r.tool_names == ["compute_points", "wait_until_settled"]


# -- the retry loop: the part that actually earns its keep ------------------

def test_validation_error_is_fed_back_and_the_retry_succeeds(ctx):
    """Two targets 7cm apart, then a corrected set. The model must see why."""
    a = agent(ctx, [
        [("move_to", {"points": [[40, 40], [45, 45], [140, 40],
                                  [190, 90], [90, 140], [140, 140]]})],
        [("move_to", {"points": [[40, 40], [90, 40], [140, 40],
                                  [190, 90], [90, 140], [140, 140]]})],
        "Fixed the spacing.",
    ])
    r = a.command("line up")

    assert r.ok
    assert r.retries == 1
    assert len(r.tool_calls) == 2
    assert r.tool_calls[0].ok is False
    assert r.tool_calls[1].ok is True

    # the model was told exactly what was wrong, verbatim
    tool_msgs = [m for m in a.client.calls[-1] if m.get("role") == "tool"]
    assert "20cm apart" in tool_msgs[0]["content"]


def test_the_error_text_reaches_the_model_unabridged(ctx):
    a = agent(ctx, [[("recall_formation", {"name": "nope"})], "No such shape."])
    a.command("do the nope")
    tool_msgs = [m for m in a.client.calls[-1] if m.get("role") == "tool"]
    assert "nope" in tool_msgs[0]["content"]


def test_count_mismatch_is_reported_with_both_numbers(ctx):
    a = agent(ctx, [[("move_to", {"points": [[40, 40], [90, 40]]})], "Sorry."])
    r = a.command("two only")
    assert r.tool_calls[0].ok is False
    content = [m for m in a.client.calls[-1] if m.get("role") == "tool"][0]["content"]
    assert "2" in content and "6" in content


def test_unknown_tool_is_an_error_not_a_crash(ctx):
    a = agent(ctx, [[("teleport", {"x": 1})], "That tool does not exist."])
    r = a.command("teleport")
    assert r.ok
    assert r.tool_calls[0].ok is False
    assert "unknown tool" in r.tool_calls[0].result["error"]


# -- the cap ----------------------------------------------------------------

def test_cap_is_enforced_and_stops_the_fleet(ctx):
    """A model that never stops calling tools must be cut off and the fleet halted."""
    forever = [[("get_state", {})] for _ in range(50)]
    a = agent(ctx, forever, max_tool_calls=4)

    ctx.apply_targets({c: np.array([120.0, 90.0]) for c in ctx.active_codes()})
    r = a.command("go forever")

    assert r.ok is False
    assert r.hit_cap is True
    assert len(r.tool_calls) == 4
    assert "halted" in r.reply
    assert ctx.env.targets == {}, "hitting the cap must clear targets"
    for h in ctx.fleet.handles.values():
        assert float(np.linalg.norm(h.vel)) < 1e-6


def test_cap_counts_across_turns_not_within_one(ctx):
    a = agent(ctx, [[("get_state", {}), ("get_state", {})],
                     [("get_state", {}), ("get_state", {})],
                     [("get_state", {})],
                     "done"], max_tool_calls=3)
    r = a.command("x")
    assert r.hit_cap
    assert len(r.tool_calls) == 3


def test_cap_of_one_still_works(ctx):
    a = agent(ctx, [[("get_state", {})] for _ in range(5)], max_tool_calls=1)
    r = a.command("x")
    assert r.hit_cap and len(r.tool_calls) == 1


# -- history ----------------------------------------------------------------

def test_history_is_kept_so_follow_ups_resolve(ctx):
    a = agent(ctx, [[("compute_points", {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
                                          "execute": True})],
                     "Circle formed.",
                     [("transform", {"scale": 1.5})],
                     "Made it bigger."])
    a.command("form a circle")
    a.command("now make it bigger")

    sent = a.client.calls[-1]
    joined = " ".join(m.get("content") or "" for m in sent)
    assert "form a circle" in joined
    assert "Circle formed." in joined


def test_history_is_trimmed_to_the_window(ctx):
    a = agent(ctx, ["ok"] * 40, history_turns=2)
    for i in range(6):
        a.command(f"command {i}")
    assert len(a.history) == 4                      # 2 exchanges
    joined = " ".join(m["content"] for m in a.history)
    assert "command 5" in joined and "command 0" not in joined


def test_history_stores_summaries_not_tool_payloads(ctx):
    a = agent(ctx, [[("get_state", {})], "Six robots, all idle."])
    a.command("status?")
    assert all("state_summary" not in m["content"] for m in a.history)


def test_reset_clears_history(ctx):
    a = agent(ctx, ["ok", "ok"])
    a.command("one")
    a.reset()
    assert a.history == []


# -- nothing escapes ---------------------------------------------------------

def test_unreachable_model_returns_a_result_not_an_exception(ctx):
    a = SwarmAgent(StubClient(fail_with=ModelUnreachable("ollama serve is not running")),
                   ctx)
    r = a.command("hello")
    assert r.ok is False
    assert "ollama serve" in r.error
    assert r.reply


def test_model_error_returns_a_result(ctx):
    a = SwarmAgent(StubClient(fail_with=ModelError("context overflow")), ctx)
    r = a.command("hello")
    assert r.ok is False and "context overflow" in r.error


def test_an_arbitrary_exception_is_contained(ctx):
    a = SwarmAgent(StubClient(fail_with=ValueError("kaboom")), ctx)
    r = a.command("hello")
    assert r.ok is False
    assert "ValueError" in r.error and "kaboom" in r.error


def test_a_tool_that_raises_is_contained(ctx, monkeypatch):
    import tools.movement as movement

    def explode(*a, **k):
        raise RuntimeError("tool blew up")

    monkeypatch.setattr(movement, "move_to", explode)
    a = agent(ctx, [[("move_to", {"points": [[50, 50]]})], "Sorry, that failed."])
    r = a.command("move")
    assert r.ok
    assert r.tool_calls[0].ok is False
    assert "tool blew up" in r.tool_calls[0].result["error"]


def test_a_broken_event_callback_does_not_kill_the_run(ctx):
    def bad_callback(event):
        raise RuntimeError("UI exploded")

    a = SwarmAgent(StubClient(script=["fine"]), ctx, on_event=bad_callback)
    assert a.command("hello").ok


# -- cancellation ------------------------------------------------------------

def test_cancel_aborts_and_stops_the_fleet(ctx):
    a = agent(ctx, [[("get_state", {})] for _ in range(20)], max_tool_calls=50)
    ctx.apply_targets({c: np.array([120.0, 90.0]) for c in ctx.active_codes()})

    events = []

    def watch(e):
        events.append(e["type"])
        if e["type"] == "tool_result":
            a.cancel()

    a.on_event = watch
    r = a.command("go")

    assert r.ok is False and r.cancelled is True
    assert ctx.env.targets == {}
    assert "cancelled" in events


def test_a_fresh_command_clears_a_previous_cancel(ctx):
    a = agent(ctx, ["first", "second"])
    a.cancel()
    r = a.command("hello")
    assert r.ok and r.reply == "first"


# -- events -------------------------------------------------------------------

def test_events_stream_in_order(ctx):
    seen = []
    a = SwarmAgent(StubClient(script=[[("get_state", {})], "All good."]), ctx,
                   on_event=lambda e: seen.append(e["type"]))
    a.command("status")
    assert seen[0] == "start"
    assert "thinking" in seen
    assert "tool_start" in seen and "tool_result" in seen
    assert seen[-1] == "done"


def test_tool_events_carry_name_latency_and_result(ctx):
    seen = []
    a = SwarmAgent(StubClient(script=[[("get_state", {})], "ok"]), ctx,
                   on_event=lambda e: seen.append(e))
    a.command("status")
    ev = next(e for e in seen if e["type"] == "tool_result")
    assert ev["name"] == "get_state"
    assert ev["ok"] is True
    assert ev["latency_s"] >= 0


# -- recovered-from-text accounting ------------------------------------------

def test_recovered_calls_are_counted(ctx):
    recovered = ToolCall(name="get_state", arguments={}, recovered_from_text=True)
    a = SwarmAgent(StubClient(script=[[recovered], "ok"]), ctx)
    r = a.command("status")
    assert r.recovered_from_text_count == 1
    assert r.tool_calls[0].recovered_from_text is True


# -- transcript compaction -----------------------------------------------------

def test_compact_keeps_errors_whole():
    big = {"ok": False, "error": "targets 0 and 1 are 7.1cm apart, need 20cm",
           "problems": [{"pair": [0, 1], "distance": 7.1}],
           "robots": [{"junk": "x" * 5000}]}
    out = _compact(big)
    assert out["error"] == big["error"]
    assert out["problems"] == big["problems"]
    assert len(repr(out)) < 2000


def test_compact_survives_a_non_dict():
    assert _compact("not a dict")["ok"] is False


# -- latency behaviours -------------------------------------------------------

def test_trailing_sensing_ends_the_turn(ctx):
    """Re-sensing after a successful action means the model is done.

    Measured: a plain formation recall went 3 calls -> 8 because the model kept
    calling describe_scene after it had already worked. Skipping the execution
    does not help — the cost is the round trip — so the loop ends instead.
    """
    a = agent(ctx, [
        [("compute_points", {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
                              "execute": True})],
        [("describe_scene", {})],
        [("describe_scene", {})],
        "never reached",
    ])
    r = a.command("form a circle")
    assert r.ok
    assert len(a.client.calls) == 2, "should have stopped at the first re-sense"
    assert r.skipped_sensing >= 1
    assert r.tool_names == ["compute_points"]


def test_sensing_first_is_still_allowed(ctx):
    """'What formations do you know?' must still work — nothing has acted yet."""
    a = agent(ctx, [[("list_formations", {})], "I know wedge."])
    r = a.command("what formations do you know?")
    assert r.ok and r.tool_names == ["list_formations"]
    assert r.skipped_sensing == 0


def test_a_failed_action_does_not_trigger_the_early_exit(ctx):
    """After a failure, re-reading state is a reasonable recovery move."""
    a = agent(ctx, [
        [("move_to", {"points": [[40, 40], [45, 45]]})],      # too close, fails
        [("describe_scene", {})],
        [("compute_points", {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
                              "execute": True})],
        "Fixed.",
    ])
    r = a.command("line up")
    assert r.ok
    assert "describe_scene" in r.tool_names, "recovery sensing was cut"
    assert "compute_points" in r.tool_names


def test_timings_split_the_wall_clock(ctx):
    a = agent(ctx, [[("wait_until_settled", {"timeout": 2})], "done"])
    r = a.command("wait")
    t = r.timings
    assert t.total_s > 0
    assert t.wait_s > 0, "waiting must be attributed to robots, not the model"
    assert t.tool_exec_s == 0.0, "a wait is not tool execution"
    assert t.overhead_s >= 0
    assert "wait (robots moving)" in t.table()


def test_model_calls_record_tokens_and_thinking(ctx):
    a = agent(ctx, ["hello"])
    r = a.command("hi")
    assert len(r.model_calls) == 1
    assert r.model_calls[0].latency_s >= 0


# -- a reply that claims an action nobody performed --------------------------
#
# Observed against Sonnet: "make the letter A" intermittently returns the single
# word "Done." with no tool call at all. ok=True, nothing moved, and nobody goes
# looking because it reported success.

def test_a_completion_claim_with_no_tool_call_is_nudged(ctx):
    a = agent(ctx, ["Done.",
                     [("move_to", {"points": [[60, 40], [120, 40], [180, 40],
                                               [60, 140], [120, 140], [180, 140]]})],
                     "Formed the shape."])
    r = a.command("make the letter A")
    assert r.nudged is True
    assert r.tool_names == ["move_to"], "the nudge did not produce an action"
    assert len(ctx.env.targets) == 6


def test_a_plain_answer_is_not_nudged(ctx):
    """"hi" -> "hello" is a perfectly good tool-free turn."""
    a = agent(ctx, ["hello"])
    r = a.command("hi")
    assert r.nudged is False
    assert len(r.model_calls) == 1, "a conversational reply cost an extra call"


def test_a_refusal_is_not_turned_into_an_action(ctx):
    """Refusing well is a pass condition, not a failure to act."""
    a = agent(ctx, ["I can't do that — six robots cannot occupy one point."])
    r = a.command("put every robot on exactly the same spot")
    assert r.nudged is False
    assert r.tool_names == []
    assert r.ok


def test_the_nudge_fires_at_most_once(ctx):
    a = agent(ctx, ["Done.", "Done again."])
    r = a.command("make the letter A")
    assert r.nudged is True
    assert r.tool_names == []
    assert len(r.model_calls) == 2, "the nudge must not loop"


def test_a_nudged_refusal_is_accepted(ctx):
    a = agent(ctx, ["Done.", "Nothing to do — they are already in that shape."])
    r = a.command("make the letter A")
    assert r.nudged is True
    assert r.ok and "already" in r.reply


def test_an_empty_reply_is_nudged_and_then_reported_honestly(ctx):
    """An empty reply with no tool call is not a turn at all."""
    a = agent(ctx, ["", ""])
    r = a.command("make the letter A")
    assert r.nudged is True
    assert r.tool_names == []
    assert "nothing moved" in r.reply.lower()


def test_a_repeated_completion_claim_is_contradicted(ctx):
    """Given a second chance and still calling nothing, do not pass the claim on."""
    a = agent(ctx, ["Done.", "Done."])
    r = a.command("make the letter A")
    assert r.nudged is True and r.tool_names == []
    assert "nothing moved" in r.reply.lower(), r.reply


def test_a_refusal_after_a_nudge_is_left_alone(ctx):
    a = agent(ctx, ["Done.", "Six robots cannot occupy one point."])
    r = a.command("stack them all on one spot")
    assert r.nudged is True
    assert "nothing moved" not in r.reply.lower()


# -- the agent files what worked ---------------------------------------------

def test_a_successful_command_is_remembered(ctx):
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    a = agent(ctx, [[("move_to", {"points": [[40, 40], [90, 40], [140, 40],
                                              [190, 90], [90, 140], [140, 140]]})],
                     "Lined them up."])
    a.command("line them up")
    assert len(ctx.memory) == 1
    assert ctx.memory.entries[0]["tools"] == ["move_to"]


def test_a_failed_command_is_not_remembered(ctx):
    """A failure recalled later is a suggestion to fail the same way."""
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    a = agent(ctx, [[("move_to", {"points": [[10, 10]]})], "Could not."])
    a.command("put them all on one spot")
    assert len(ctx.memory) == 0


def test_a_turn_that_only_sensed_is_not_a_precedent(ctx):
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    a = agent(ctx, [[("describe_scene", {})], "Six robots, all idle."])
    a.command("what is going on?")
    assert len(ctx.memory) == 0


def test_memory_failing_never_breaks_a_run(ctx):
    class Exploding:
        identifiers = set()

        def record(self, *a, **k):
            raise RuntimeError("disk on fire")

        def lines(self, *a, **k):
            return []

    ctx.memory = Exploding()
    a = agent(ctx, [[("move_to", {"points": [[40, 40], [90, 40], [140, 40],
                                              [190, 90], [90, 140], [140, 140]]})],
                     "Done."])
    r = a.command("line them up")
    assert r.ok, "a broken memory took the whole command down"

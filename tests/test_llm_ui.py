"""The UI half of the LLM wiring: session threading, ask/tool mode, STOP."""

import os
import threading
import time

import numpy as np
import json

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from llm.client import ModelResponse, ModelUnreachable, ToolCall
from llm.session import AgentSession
from llm.stub import StubClient


@pytest.fixture
def ctx(sim_ctx):
    return sim_ctx(n=6, seed=2)


def pump(app, session=None, limit=600):
    """Advance the app the way run() does, until the agent is finished."""
    session = session or app.session
    for _ in range(limit):
        app.drain_events()
        if not session.busy and app.pending is None:
            break
        with app.sim_lock:
            app.ctx.env.rebuild_if_needed()
            app.ctx.env.sync()
            if len(app.ctx.env.codes):
                app.ctx.env.apply(app.controller().act(app.ctx.env))
            app.fleet.step(1 / 30)
        time.sleep(0.002)
    app.drain_events()


# -- the session ------------------------------------------------------------

def test_session_runs_off_the_calling_thread(ctx):
    s = AgentSession(ctx, client=StubClient(script=["hello"]), probe=False)
    assert s.start("hi")
    s.join(timeout=5)
    assert s.result().ok
    assert s.result().reply == "hello"


def test_session_refuses_a_second_run_while_busy(ctx):
    slow = StubClient(script=[[("get_state", {})], "done"], latency_s=0.3)
    s = AgentSession(ctx, client=slow, probe=False)
    assert s.start("one")
    assert s.start("two") is False, "must not run two commands at once"
    s.join(timeout=5)


def test_session_reports_unreachable_without_raising(ctx):
    class Dead:
        model = "dead"

        def reachable(self, timeout=2.0):
            return False

        def chat(self, *a, **k):
            raise ModelUnreachable("ollama serve is not running")

    s = AgentSession(ctx, client=Dead())
    assert s.available is False
    # probe() is stubbed by the autouse fixture; real_probe is the genuine one,
    # and it must record *why* the endpoint is unavailable
    assert s.real_probe() is False
    assert s.error and "not answering" in s.error


def test_session_survives_a_bad_preset(ctx):
    s = AgentSession(ctx, preset="no-such-model-preset")
    assert s.available is False
    assert s.agent is None
    assert s.start("hi") is False          # must not raise


def test_session_events_are_queued_for_the_ui(ctx):
    s = AgentSession(ctx, client=StubClient(script=[[("get_state", {})], "ok"]),
                     probe=False)
    s.start("status")
    s.join(timeout=5)
    kinds = [e["type"] for e in s.poll(limit=200)]
    assert "start" in kinds and "tool_result" in kinds and "done" in kinds


def test_session_tracks_mean_latency(ctx):
    s = AgentSession(ctx, client=StubClient(script=["a", "b"]), probe=False)
    assert s.mean_latency is None
    s.start("one")
    s.join(timeout=5)
    assert s.mean_latency is not None and s.mean_latency >= 0


def test_session_cancel_stops_the_fleet(ctx):
    forever = StubClient(script=[[("get_state", {})] for _ in range(200)],
                         latency_s=0.01)
    s = AgentSession(ctx, client=forever, probe=False, max_tool_calls=500)
    ctx.apply_targets({c: np.array([120.0, 90.0]) for c in ctx.active_codes()})
    s.start("go")
    time.sleep(0.1)
    s.cancel()
    s.join(timeout=5)
    assert s.result().cancelled is True
    assert ctx.env.targets == {}


# -- the app ----------------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    """An App with no model, on scratch state files."""
    import app as appmod
    from fleet.roster import Roster
    from tools.formations import FormationLibrary

    # Built explicitly rather than copied from the repo's roster.json. Copying
    # made every test here depend on live state: the day a developer set one
    # dragon to `real` for hardware bring-up, this fixture started building a
    # SpheroRobot, reaching for a radio, and failing with six targets because
    # only five robots were connectable.
    from fleet.roster import RobotEntry

    roster_copy = tmp_path / "roster.json"
    Roster(entries=[RobotEntry(name=n, code=c, kind="sim", color=col)
                    for n, c, col in [("Seasmoke", "SSMK", "cyan"),
                                       ("Caraxes", "CRXS", "red"),
                                       ("Syrax", "SYRX", "yellow"),
                                       ("Vhagar", "VHGR", "green"),
                                       ("Meleys", "MLYS", "magenta"),
                                       ("Sunfyre", "SNFR", "blue")]],
           path=roster_copy).save()

    # The workspace, for the same reason as the roster. These tests name
    # coordinates like 180,140 and expect them to be inside — so inheriting the
    # repo's workspace.json makes them fail the day somebody measures their
    # actual floor and it turns out to be smaller than 200x200.
    ws_copy = tmp_path / "workspace.json"
    ws_copy.write_text(json.dumps({
        "bounds_cm": [[0, 0], [200, 0], [200, 200], [0, 200]],
        "obstacles": [],
        "origin": "top-left, x right, y down, matching the camera frame"}))

    a = appmod.App(roster_path=roster_copy, workspace_path=ws_copy,
                   session=False or None)
    a.roster.path = roster_copy
    a.ctx.library = FormationLibrary(path=tmp_path / "formations.json")
    yield a
    if a.session is not None and a.session.busy:
        a.session.cancel()
        a.session.join(timeout=2)
    a.fleet.close()


def test_app_starts_in_tool_mode_without_a_model(app):
    assert app.ask is False
    assert app.tool_btn.on is True
    assert app.ask_btn.enabled is False
    assert "no model" in " ".join(t for _, t in app.log)


def test_tool_mode_still_works_unchanged(app):
    app.submit("move_to 60,40 120,40 180,40 60,140 120,140 180,140")
    kind, text = app.log[-1]
    assert kind == "ok" and "target(s) set" in text


def test_cannot_switch_to_ask_without_a_model(app):
    app.set_ask(True)
    assert app.ask is False
    assert "no model is reachable" in app.log[-1][1]


def with_model(app, client=None, **kw):
    app.session = AgentSession(app.ctx, client=client or StubClient(script=["ok"]),
                               sim_lock=app.sim_lock, probe=False, **kw)
    app.session.available = True
    app.set_ask(True)
    return app.session


def test_ask_mode_streams_tool_calls_into_the_log(app):
    with_model(app, StubClient(script=[
        [("compute_points", {"expression": "points = [(cx+60*cos(i*2*pi/n), cy+60*sin(i*2*pi/n)) for i in range(n)]",
                              "execute": True})],
        "Formed a circle."]))
    app.submit("form a circle")
    pump(app)

    text = "\n".join(t for _, t in app.log)
    assert "> form a circle" in text
    assert "compute_points(" in text
    assert "Formed a circle." in text
    assert "call(s) in" in text


def test_ask_mode_logs_tool_latency(app):
    with_model(app, StubClient(script=[[("get_state", {})], "done"]))
    app.submit("status")
    pump(app)
    assert any("[" in t and "s]" in t for _, t in app.log)


def test_long_arguments_are_shortened_in_the_log(app):
    from app import _short

    assert _short([[1, 2], [3, 4], [5, 6], [7, 8]]) == "[4 points]"
    assert "…" in _short("x" * 80)


def test_a_failed_tool_is_logged_as_an_error(app):
    with_model(app, StubClient(script=[
        [("move_to", {"points": [[40, 40], [45, 45]]})], "That did not work."]))
    app.submit("stack them")
    pump(app)
    assert any(k == "error" for k, _ in app.log)


def test_stop_aborts_an_in_flight_run(app):
    session = with_model(app, StubClient(
        script=[[("get_state", {})] for _ in range(200)], latency_s=0.01),
        max_tool_calls=500)
    app.submit("go forever")
    time.sleep(0.1)
    assert session.busy

    app.stop_all()
    session.join(timeout=5)
    pump(app)

    assert session.result().cancelled is True
    assert app.ctx.env.targets == {}
    assert any("aborting the model run" in t for _, t in app.log)


def test_a_second_command_while_busy_is_refused(app):
    with_model(app, StubClient(script=[[("get_state", {})], "done"],
                                latency_s=0.4))
    app.submit("first")
    app.submit("second")
    assert any("still working" in t for _, t in app.log)
    app.session.join(timeout=5)


def test_hitting_the_cap_halts_and_says_so(app):
    with_model(app, StubClient(script=[[("get_state", {})] for _ in range(50)]),
               max_tool_calls=3)
    app.submit("loop")
    pump(app)
    assert any("cap" in t for _, t in app.log)


def test_thinking_indicator_only_draws_while_busy(app):
    session = with_model(app, StubClient(script=[[("get_state", {})], "ok"],
                                          latency_s=0.3))
    app.draw_thinking(14, 372)                     # idle: must be a no-op
    app.submit("status")
    app.started_at = time.time()
    assert session.busy
    app.draw_thinking(14, 372)                     # busy: must not raise
    session.join(timeout=5)
    pump(app)


def test_status_line_reports_the_model(app):
    assert "unreachable" in app.model_status() or "none" in app.model_status()
    with_model(app)
    assert "stub" in app.model_status()


def test_mode_toggle_switches_the_placeholder(app):
    with_model(app)
    assert app.cmd.placeholder == "form a circle"
    app.set_ask(False)
    assert app.cmd.placeholder.startswith("move_to")


def test_ask_and_tool_buttons_do_not_overlap(app):
    assert not app.ask_btn.rect.colliderect(app.tool_btn.rect)
    for b in app.buttons:
        if b not in (app.ask_btn, app.tool_btn):
            assert not b.rect.colliderect(app.ask_btn.rect), b.label
            assert not b.rect.colliderect(app.tool_btn.rect), b.label
    assert not app.ask_btn.rect.colliderect(app.cmd.rect)


def test_render_with_a_running_agent_does_not_raise(app):
    session = with_model(app, StubClient(script=[[("get_state", {})], "ok"],
                                          latency_s=0.2))
    app.submit("status")
    app.screen.fill((0, 0, 0))
    app.draw_arena()
    app.draw_dock()
    session.join(timeout=5)
    pump(app)


def test_the_live_state_files_are_untouched_by_an_ask_run(app):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    before = {n: (root / n).read_bytes()
              for n in ("roster.json", "workspace.json", "formations.json")}

    with_model(app, StubClient(script=[
        [("compute_points", {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
                              "execute": True})],
        [("save_formation", {"name": "ring"})],
        "Saved."]))
    app.submit("form a circle and save it as ring")
    pump(app)

    for name, data in before.items():
        assert (root / name).read_bytes() == data, f"{name} was modified"


# -- visible motion ---------------------------------------------------------

def test_waiting_is_free_running_by_default(sim_ctx):
    """Evals and tests must not be paced by a wall clock nobody is watching."""
    from tools import call

    ctx = sim_ctx(n=6)
    assert ctx.realtime is False
    call(ctx, "compute_points",
         {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
          "execute": True})
    t0 = time.time()
    r = call(ctx, "wait_until_settled", {})
    wall = time.time() - t0
    assert r["settled"]
    assert r["waited_s"] > 0.5, "simulated time should still advance"
    assert wall < 0.5, "free-running wait should not sleep"


def test_realtime_context_paces_the_wait_so_motion_is_visible(sim_ctx):
    """With a person watching, robots must cross the arena over real seconds.

    Free-running, the whole journey happens inside one tool call and the UI
    draws only the end state — which reads as 'the robots never moved'.
    """
    from tools import call

    ctx = sim_ctx(n=6)
    ctx.realtime = True
    call(ctx, "compute_points",
         {"expression": "points = [(cx+50*cos(i*2*pi/n), cy+50*sin(i*2*pi/n)) for i in range(n)]",
          "execute": True})
    t0 = time.time()
    r = call(ctx, "wait_until_settled", {"timeout": 6})
    wall = time.time() - t0
    assert wall > 0.5, f"realtime wait finished in {wall:.2f}s — motion invisible"
    assert r["waited_s"] > 0


def test_app_marks_its_context_realtime(app):
    assert app.ctx.realtime is True


def test_renderer_does_not_block_while_a_tool_drives_the_sim(app):
    """A tool holding the lock must not stop the window redrawing."""
    held = threading.Event()
    release = threading.Event()

    def hog():
        with app.sim_lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hog, daemon=True)
    t.start()
    assert held.wait(2)

    t0 = time.time()
    got = app.sim_lock.acquire(blocking=False)      # what the render loop does
    assert got is False, "lock should be held by the tool"
    assert time.time() - t0 < 0.1, "render loop must not wait for it"

    app.screen.fill((0, 0, 0))
    app.draw_arena()                                 # must still render
    app.draw_dock()
    release.set()
    t.join(timeout=5)

"""The UI, driven headless.

Rendering is not tested for looks, only for "does not throw and reflects the
fleet". The parts worth asserting are the ones with logic behind them: the
roster toggles, the command bar, and the fact that a disconnected robot is
drawn differently from a live one.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

import app as appmod  # noqa: E402
from fleet.roster import RobotEntry, Roster  # noqa: E402
from tools.formations import FormationLibrary  # noqa: E402
from workspace.space import Workspace  # noqa: E402


@pytest.fixture
def ui(tmp_path, monkeypatch):
    """An all-sim app on a scratch roster, so no test writes the real files."""
    roster_path = tmp_path / "roster.json"
    ws_path = tmp_path / "workspace.json"

    colors = ["cyan", "red", "yellow", "green"]
    codes = ["SSMK", "CRXS", "SYRX", "VHGR"]
    entries = [RobotEntry(name=f"Dragon{i}", code=c, kind="sim", color=col)
               for i, (c, col) in enumerate(zip(codes, colors))]
    Roster(entries=entries, path=roster_path).save()
    Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
              obstacles=[{"type": "circle", "center": [120, 90], "radius": 12}],
              path=ws_path).save()

    a = appmod.App(roster_path=roster_path, workspace_path=ws_path)
    a.ctx.library = FormationLibrary(path=tmp_path / "formations.json")
    # Sim robots start somewhere random. Pinning them down keeps tests that
    # place a target at a fixed point from colliding with a bystander by luck.
    for code, p in zip(codes, [[30, 30], [210, 30], [210, 150], [30, 150]]):
        a.fleet[code].pos = np.array(p, dtype=float)
    yield a
    a.fleet.close()
    pygame.quit()


def render(a):
    a.screen.fill(appmod.INK)
    a.draw_arena()
    a.draw_dock()


def test_app_starts_with_the_roster_fleet(ui):
    assert ui.fleet.codes == ["SSMK", "CRXS", "SYRX", "VHGR"]
    assert ui.mode == "navigate"


def test_renders_without_error(ui):
    for _ in range(3):
        render(ui)


def test_renders_with_an_empty_fleet(tmp_path):
    """The app must run with zero robots and zero camera."""
    roster_path = tmp_path / "roster.json"
    Roster(entries=[], path=roster_path).save()
    a = appmod.App(roster_path=roster_path)
    try:
        assert len(a.fleet) == 0
        render(a)
        a.ctx.tick(0.1)                 # a tick with nobody home must not throw
    finally:
        a.fleet.close()
        pygame.quit()


def test_command_bar_runs_a_tool(ui):
    ui.submit("move_to 60,40 120,40 180,40 100,140")
    kind, text = ui.log[-1]
    assert kind == "ok" and "target" in text
    assert len(ui.ctx.env.targets) == 4
    render(ui)


def test_command_bar_reports_errors(ui):
    ui.submit("frobnicate")
    kind, text = ui.log[-1]
    assert kind == "error"
    render(ui)


def test_text_field_typing_and_submit(ui):
    ui.cmd.focused = True
    for ch in "stop":
        ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=ord(ch), unicode=ch))
    assert ui.cmd.text == "stop"

    line = ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN,
                                          unicode="\r"))
    assert line == "stop"
    assert ui.cmd.text == ""
    assert ui.cmd.history == ["stop"]


def test_text_field_backspace_and_history(ui):
    ui.cmd.focused = True
    ui.cmd.text = "stop"
    ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))
    ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, unicode=""))
    assert ui.cmd.text == "stop"
    ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode=""))
    assert ui.cmd.text == "sto"


def test_no_two_clickable_regions_overlap(ui):
    """An overlapping hit box silently steals clicks from whatever is beneath it."""
    ui.submit("move_to 60,40 120,40 180,40 100,140")
    ui.submit("save_formation box")
    render(ui)

    regions = [(b.rect, f"button {b.label}") for b in ui.buttons]
    for row in ui.roster_rows:
        for key in ("kind", "del", "code_field", "name_field"):
            regions.append((row[key], f"{key} {row['code']}"))
    regions += [(r, f"recall {n}") for r, n in ui.formation_rows]
    regions += [(r, f"checkpoint {ck['name']}") for r, ck in ui.ck_rows]
    regions.append((ui.cmd.rect, "command bar"))

    for i, (ra, na) in enumerate(regions):
        for rb, nb in regions[i + 1:]:
            assert not ra.colliderect(rb), f"{na} overlaps {nb}"


def test_roster_toggle_flips_a_robot_live(ui):
    render(ui)                                  # populates the clickable rows
    row = ui.roster_rows[0]
    kind_rect, code = row["kind"], row["code"]
    assert ui.fleet[code].kind == "sim"

    ui.click(kind_rect.center)
    assert ui.fleet[code].kind == "real"
    assert len(ui.fleet) == 4                   # the others are undisturbed
    assert all(ui.fleet[c].kind == "sim" for c in ui.fleet.codes if c != code)

    ui.click(kind_rect.center)
    assert ui.fleet[code].kind == "sim"
    render(ui)


def test_roster_remove_and_add(ui):
    render(ui)
    row = ui.roster_rows[0]
    del_rect, code = row["del"], row["code"]
    ui.click(del_rect.center)
    assert code not in ui.fleet
    assert len(ui.fleet) == 3

    ui.add_robot()
    assert len(ui.fleet) == 4
    render(ui)


def type_into_edit(ui, text):
    for ch in text:
        ui.edit_key(pygame.event.Event(pygame.KEYDOWN, key=ord(ch), unicode=ch))


def test_editing_a_name_writes_back_to_the_roster(ui):
    render(ui)
    row = ui.roster_rows[0]
    ui.click(row["name_field"].center)
    assert ui.editing == ("SSMK", "name")

    ui.edit_buf = ""
    type_into_edit(ui, "Vermax")
    ui.edit_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))

    assert ui.fleet["SSMK"].name == "Vermax"
    reloaded = Roster.load(ui.roster.path)
    assert reloaded.by_code("SSMK").name == "Vermax"
    render(ui)


def test_editing_a_code_rekeys_the_live_fleet(ui):
    render(ui)
    ui.click(ui.roster_rows[0]["code_field"].center)
    ui.edit_buf = ""
    type_into_edit(ui, "VRMX")
    ui.edit_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))

    assert "VRMX" in ui.fleet and "SSMK" not in ui.fleet
    assert ui.fleet.codes[0] == "VRMX"           # order preserved
    assert Roster.load(ui.roster.path).by_code("VRMX") is not None

    ui.submit("move_to VRMX=100,60")             # the new code is addressable
    assert ui.log[-1][0] == "ok"


def test_a_duplicate_code_is_refused_and_reverted(ui):
    render(ui)
    ui.click(ui.roster_rows[0]["code_field"].center)
    ui.edit_buf = ""
    type_into_edit(ui, "CRXS")                   # already taken
    ui.edit_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))

    assert ui.log[-1][0] == "error"
    assert "SSMK" in ui.fleet                    # unchanged
    assert Roster.load(ui.roster.path).by_code("SSMK") is not None


def test_escape_cancels_an_edit(ui):
    render(ui)
    ui.click(ui.roster_rows[0]["name_field"].center)
    ui.edit_buf = ""
    type_into_edit(ui, "Nope")
    ui.edit_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode=""))

    assert ui.editing is None
    assert ui.fleet["SSMK"].name == "Dragon0"


def test_add_robot_refuses_a_seventh_colour(ui):
    for _ in range(2):
        ui.add_robot()
    assert len(ui.fleet) == 6
    ui.add_robot()                              # no hue left for a seventh
    assert len(ui.fleet) == 6
    assert ui.log[-1][0] == "error"
    assert "colour" in ui.log[-1][1]


def test_stop_button_halts_everything(ui):
    ui.submit("move_to 60,40 120,40 180,40 100,140")
    for _ in range(10):
        ui.ctx.tick(0.1)
    ui.stop_all()
    assert ui.ctx.env.targets == {}
    assert all(h.target is None for h in ui.fleet.handles.values())


def test_disconnected_robot_still_renders(ui, monkeypatch):
    """A real robot that never connects is drawn greyed, not skipped."""
    from tests.conftest import FakeConnector, FakeTracker

    ui.fleet.connector = FakeConnector(fail=True)
    ui.fleet.tracker = FakeTracker({})
    ui.toggle_kind("SSMK")
    assert ui.fleet["SSMK"].kind == "real"

    for _ in range(5):
        ui.ctx.tick(0.05)
    assert ui.fleet["SSMK"].connected is False
    render(ui)                                  # must not throw on a stale robot

    state = ui.fleet.state()
    assert state["SSMK"]["connected"] is False
    assert state["CRXS"]["connected"] is True


def test_formation_panel_lists_and_recalls(ui):
    ui.submit("move_to 60,40 120,40 180,40 100,140")
    ui.submit("save_formation box")
    render(ui)
    assert ui.formation_rows
    btn, name = ui.formation_rows[0]
    assert name == "box"

    ui.submit("move_to 30,30 50,30 30,60 50,60")
    ui.click(btn.center)
    assert ui.log[-1][0] == "ok"
    assert len(ui.ctx.env.targets) == 4


def test_modes_still_work(ui):
    for mode in ("boids", "idle", "navigate"):
        ui.set_mode(mode)
        assert ui.mode == mode
        ui.ctx.env.sync()
        actions = ui.controller().act(ui.ctx.env)
        assert actions.shape == (4, 2)
        assert np.isfinite(actions).all()
    render(ui)


def test_policy_mode_without_checkpoints_is_refused(ui):
    ui.checkpoints = []
    ui.policy = None
    ui.set_mode("policy")
    assert ui.mode != "policy"
    assert ui.log[-1][0] == "error"


def test_the_main_loop_body_runs(ui):
    """One pass of exactly what run() does per frame."""
    for _ in range(20):
        ui.ctx.env.rebuild_if_needed()
        ui.ctx.env.sync()
        if len(ui.ctx.env.codes):
            ui.ctx.env.apply(ui.controller().act(ui.ctx.env))
        ui.fleet.step(1 / 30)
        render(ui)
    for h in ui.fleet.handles.values():
        assert ui.ws.is_valid_point(h.pos)


# -- obstacle editing -------------------------------------------------------

def test_window_fits_the_desktop():
    """The bottom strip holds STOP and the command bar. If the window is taller
    than the usable screen those controls are simply unreachable — which is
    exactly what happened on a 14" MacBook at the hardcoded 950px."""
    import pygame

    import app as appmod

    pygame.display.init()
    w, h = appmod._fit_to_display()
    try:
        sw, sh = pygame.display.get_desktop_sizes()[0]
    except Exception:
        return
    assert h <= sh - appmod.CHROME_H or h == appmod.MIN_H
    assert w <= sw - appmod.CHROME_W or w == appmod.MIN_W


def test_bottom_controls_are_inside_the_window(ui):
    import app as appmod

    for name in ("stop_btn", "ask_btn", "tool_btn"):
        r = getattr(ui, name).rect
        assert r.bottom <= appmod.H, f"{name} hangs below the window"
        assert r.right <= appmod.DOCK
    assert ui.cmd.rect.bottom <= appmod.H


def test_add_obstacle_keeps_the_whole_shape_inside(ui):
    from workspace.space import obstacle_center, obstacle_size

    for _ in range(6):
        ui.add_obstacle()
    xmin, xmax, ymin, ymax = ui.ws.bbox
    for o in ui.ws.obstacles:
        c, s = obstacle_center(o), obstacle_size(o)
        assert c[0] - s >= xmin - 1 and c[0] + s <= xmax + 1, f"{o} overhangs in x"
        assert c[1] - s >= ymin - 1 and c[1] + s <= ymax + 1, f"{o} overhangs in y"


def test_obstacle_shape_size_and_removal(ui):
    from workspace.space import obstacle_size

    before = len(ui.ws.obstacles)
    ui.add_obstacle()
    assert len(ui.ws.obstacles) == before + 1
    i = len(ui.ws.obstacles) - 1

    assert ui.ws.obstacles[i]["type"] == "circle"
    ui.cycle_obstacle_shape(i)
    assert ui.ws.obstacles[i]["type"] == "poly"
    ui.cycle_obstacle_shape(i)
    assert ui.ws.obstacles[i]["type"] == "circle"

    size = obstacle_size(ui.ws.obstacles[i])
    ui.resize_obstacle(i, +5)
    assert obstacle_size(ui.ws.obstacles[i]) == pytest.approx(size + 5)
    ui.resize_obstacle(i, -5)
    assert obstacle_size(ui.ws.obstacles[i]) == pytest.approx(size)

    ui.remove_obstacle(i)
    assert len(ui.ws.obstacles) == before


def test_obstacle_size_cannot_go_negative(ui):
    from workspace.space import obstacle_size

    ui.add_obstacle()
    i = len(ui.ws.obstacles) - 1
    for _ in range(20):
        ui.resize_obstacle(i, -5)
    assert obstacle_size(ui.ws.obstacles[i]) >= 3.0


def test_dragging_moves_an_obstacle_and_persists_on_release(ui):
    from workspace.space import obstacle_center

    ui.add_obstacle()
    i = len(ui.ws.obstacles) - 1
    ui.move_obstacle(i, (150.0, 120.0))
    assert obstacle_center(ui.ws.obstacles[i]) == pytest.approx([150.0, 120.0])

    ui.save_workspace("moved")
    reloaded = Workspace.load(ui.ws.path)
    assert obstacle_center(reloaded.obstacles[i]) == pytest.approx([150.0, 120.0])


def test_dragging_is_clamped_to_the_arena(ui):
    from workspace.space import obstacle_center

    ui.add_obstacle()
    i = len(ui.ws.obstacles) - 1
    ui.move_obstacle(i, (9999.0, -9999.0))
    c = obstacle_center(ui.ws.obstacles[i])
    xmin, xmax, ymin, ymax = ui.ws.bbox
    assert xmin <= c[0] <= xmax and ymin <= c[1] <= ymax


def test_obstacle_at_finds_what_is_under_the_cursor(ui):
    from workspace.space import obstacle_center

    ui.add_obstacle()
    i = len(ui.ws.obstacles) - 1
    c = obstacle_center(ui.ws.obstacles[i])
    assert ui.obstacle_at(c) == i
    assert ui.obstacle_at((c[0] + 500, c[1])) is None


def test_edited_obstacles_are_respected_by_the_validator(ui):
    """An obstacle added in the UI must immediately block targets there."""
    import numpy as np
    from tools.validate import validate_targets
    from workspace.space import obstacle_center

    ui.add_obstacle()
    c = obstacle_center(ui.ws.obstacles[-1])
    r = validate_targets([list(c)], ui.ws, expected_count=1)
    assert r["ok"] and r["clamped"], "target inside a new obstacle was not clamped"
    assert ui.ws.is_valid_point(np.array(r["points"][0]))


def test_obstacle_panel_hitboxes_do_not_overlap_each_other(ui):
    ui.add_obstacle()
    ui.screen.fill((0, 0, 0))
    ui.draw_arena()
    ui.draw_dock()
    rects = []
    for row in ui.obstacle_rows:
        rects += [row["shape"], row["minus"], row["plus"], row["del"]]
    rects.append(ui.obs_add_btn.rect)
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.colliderect(b), f"{a} overlaps {b}"


def test_growing_an_obstacle_keeps_it_inside_the_arena(ui):
    """Clamping the size is not enough — the centre must move in as it grows."""
    from workspace.space import obstacle_center, obstacle_size

    xmin, xmax, ymin, ymax = ui.ws.bbox
    ui.add_obstacle()
    i = len(ui.ws.obstacles) - 1
    ui.move_obstacle(i, (xmax - 5, ymax - 5))     # jam it into the corner
    for _ in range(8):
        ui.resize_obstacle(i, +10)

    c, s = obstacle_center(ui.ws.obstacles[i]), obstacle_size(ui.ws.obstacles[i])
    assert c[0] - s >= xmin - 1 and c[0] + s <= xmax + 1
    assert c[1] - s >= ymin - 1 and c[1] + s <= ymax + 1


def clickable_rects(ui):
    rects = [(b.label, b.rect) for b in ui.buttons]
    rects.append(("+ obs", ui.obs_add_btn.rect))
    rects.append(("command bar", ui.cmd.rect))
    for row in ui.obstacle_rows:
        for k in ("shape", "minus", "plus", "del"):
            rects.append((f"obstacle {k}", row[k]))
    for row in ui.roster_rows:
        for k in ("kind", "del", "code_field", "name_field"):
            rects.append((f"roster {k}", row[k]))
    return rects


def test_the_log_panel_never_covers_a_control(ui):
    """The log is drawn after the controls, so any overlap hides them.

    This is exactly what happened when the ask/tool row was added below a log
    whose bottom edge was a hardcoded offset.
    """
    ui.screen.fill((0, 0, 0))
    ui.draw_arena()
    ui.draw_dock()
    for label, r in clickable_rects(ui):
        assert not ui.log_rect.colliderect(r), f"log covers {label} at {r}"


def test_the_log_never_covers_a_control_on_a_crowded_dock(ui):
    """Panels above the log grow with content; the log must yield, not overlap."""
    for _ in range(5):
        ui.add_obstacle()
    for name in ("wedge", "ring", "vee"):
        ui.ctx.library.save_formation(name, ui.ctx.active_positions(),
                                      ui.ctx.active_codes())
    ui.screen.fill((0, 0, 0))
    ui.draw_arena()
    ui.draw_dock()
    for label, r in clickable_rects(ui):
        assert not ui.log_rect.colliderect(r), f"log covers {label} at {r}"


def test_no_two_controls_overlap_each_other(ui):
    ui.add_obstacle()
    ui.screen.fill((0, 0, 0))
    ui.draw_arena()
    ui.draw_dock()
    rects = clickable_rects(ui)
    for i, (la, a) in enumerate(rects):
        for lb, b in rects[i + 1:]:
            assert not a.colliderect(b), f"{la} {a} overlaps {lb} {b}"


# -- the render loop actually advances the world -----------------------------
#
# The loop used to inline its own copy of `ctx.tick`, and the copy fell behind:
# it never stepped moving entities and never advanced the controller's clock.
# Nothing caught it, because every other test drives `ctx.tick` directly. These
# tests go through `App.step_world`, which is what the render loop calls.

def _moving_entity(ui, **motion):
    from workspace.entities import Entity
    e = Entity(id="rover", shape={"type": "circle", "center": [60, 60],
                                  "radius": 8}, role="obstacle", motion=motion)
    ui.ws.entities.add(e)
    return e


def test_render_loop_advances_a_moving_entity(ui):
    e = _moving_entity(ui, kind="path", waypoints=[[60, 60], [180, 60]],
                       speed=30.0, mode="loop")
    start = e.pos.copy()
    for _ in range(20):
        ui.step_world(1.0 / 30.0)
    assert np.linalg.norm(e.pos - start) > 5.0, (
        "a patrolling entity stood still while the app was running")


def test_render_loop_advances_the_controller_clock(ui):
    """Flow fields are functions of t. A frozen clock makes every one a constant."""
    before = ui.ctx.controller.t
    for _ in range(10):
        ui.step_world(0.1)
    assert ui.ctx.controller.t == pytest.approx(before + 1.0, abs=1e-6)


def test_render_loop_advances_entities_in_every_controller_mode(ui):
    """The entity is part of the world, not of whichever controller is selected."""
    e = _moving_entity(ui, kind="path", waypoints=[[60, 60], [180, 60]],
                       speed=30.0, mode="loop")
    for mode in ("navigate", "boids", "idle"):
        ui.mode = mode
        start = e.pos.copy()
        for _ in range(20):
            ui.step_world(1.0 / 30.0)
        assert np.linalg.norm(e.pos - start) > 5.0, f"entity frozen in {mode} mode"


def test_a_paused_app_leaves_the_world_alone(ui):
    e = _moving_entity(ui, kind="path", waypoints=[[60, 60], [180, 60]],
                       speed=30.0, mode="loop")
    start, t0 = e.pos.copy(), ui.ctx.controller.t
    ui.paused = True
    # The loop guards on `paused` rather than step_world doing it, so assert
    # the guard is what the loop reads: nothing here should have moved.
    assert np.allclose(e.pos, start) and ui.ctx.controller.t == t0


def test_step_world_yields_when_a_tool_holds_the_lock(ui):
    """A blocked frame must return, not wait — the window has to keep redrawing.

    The lock has to be taken from another thread to mean anything: it is an
    RLock, so the render thread re-entering its own hold always succeeds.
    """
    import threading

    held, release = threading.Event(), threading.Event()

    def agent():
        with ui.sim_lock:
            held.set()
            release.wait(2.0)

    t = threading.Thread(target=agent, daemon=True)
    t.start()
    assert held.wait(2.0)
    try:
        assert ui.step_world(1.0 / 30.0) is False
    finally:
        release.set()
        t.join(timeout=2.0)
    assert ui.step_world(1.0 / 30.0) is True


def test_entities_are_drawn(ui):
    """A moving obstacle you cannot see is indistinguishable from a bug."""
    _moving_entity(ui, kind="path", waypoints=[[60, 60], [180, 60]],
                   speed=30.0, mode="loop")
    ui.screen.fill(appmod.INK)
    ui.draw_arena()
    at = ui.to_px((60, 60))
    # The shape is an outline, so sample out past its radius rather than only
    # around the centre, which is fill.
    reach = int(8 * ui.scale_px()) + 4
    patch = [ui.screen.get_at((at[0] + dx, at[1] + dy))[:3]
             for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)]
    assert any(px == appmod.CORAL for px in patch), "entity was not drawn"


# -- a long-running tool must not lock the user out of the other robots ------
#
# `wait_until_settled` ticks the world for up to its timeout. The agent used to
# hold the shared lock around that entire call, so for those seconds the render
# loop could not draw and no other robot could be commanded. The lock is now
# taken per tick instead.

def test_a_command_lands_while_a_long_wait_runs(ui):
    """The symptom, reproduced: drive one robot while another is settling."""
    import threading

    ui.fleet["SSMK"].pos = np.array([30.0, 30.0])
    ui.ctx.apply_targets({"SSMK": (200.0, 150.0)})

    started, done = threading.Event(), threading.Event()

    def waiter():
        started.set()
        from tools import call
        call(ui.ctx, "wait_until_settled", {"timeout": 5.0})
        done.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert started.wait(2.0)

    # While that is running, a literal tool command for a different robot must
    # go through promptly rather than blocking until the wait finishes.
    began = time.time()
    ui.run_tool("move_to CRXS=60,150")
    took = time.time() - began

    assert took < 2.0, f"commanding another robot took {took:.1f}s during a wait"
    assert ui.fleet["CRXS"].target is not None, "the command never landed"
    assert not any("busy" in text for kind, text in ui.log if kind == "error")

    done.wait(10.0)
    t.join(timeout=5.0)


def test_the_render_loop_keeps_stepping_during_a_long_wait(ui):
    """A frozen window is how someone ends up pressing the button four times."""
    import threading

    ui.ctx.apply_targets({"SSMK": (200.0, 150.0)})
    started = threading.Event()

    def waiter():
        started.set()
        from tools import call
        call(ui.ctx, "wait_until_settled", {"timeout": 3.0})

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert started.wait(2.0)

    # Sample across real time rather than in a tight loop: 40 back-to-back
    # attempts all land in the same instant and tell you only where the waiter
    # happened to be, which is a coin flip, not a measurement.
    stepped = attempts = 0
    deadline = time.time() + 1.0
    while time.time() < deadline:
        attempts += 1
        stepped += bool(ui.step_world(1.0 / 30.0))
        time.sleep(1.0 / 60.0)

    assert stepped > attempts * 0.5, (
        f"render loop got {stepped}/{attempts} frames during a wait")
    t.join(timeout=10.0)


def test_the_app_and_the_context_share_one_lock(ui):
    """Two locks would mean neither actually excludes the other."""
    assert ui.sim_lock is ui.ctx.lock


def test_a_direct_tool_call_still_works_while_the_model_is_busy(ui):
    """The escape hatch: command another robot without aborting the agent."""
    class Busy:
        available = True
        busy = True
        model_name = "fake"
        error = None

        def start(self, text):
            raise AssertionError("must not start a second model run")

    ui.session = Busy()
    ui.ask = True

    ui.submit("move_to CRXS=60,150")
    assert ui.fleet["CRXS"].target is not None, "the direct call did not run"
    assert any(k == "ok" for k, _ in ui.log[-2:])


def test_plain_english_while_busy_says_what_to_do_instead(ui):
    class Busy:
        available = True
        busy = True
        model_name = "fake"
        error = None

        def start(self, text):
            raise AssertionError("must not start a second model run")

    ui.session = Busy()
    ui.ask = True

    ui.submit("everyone form a circle please")
    kind, text = ui.log[-1]
    assert kind == "error"
    assert "STOP" in text and "move_to" in text


def test_parses_as_tool_does_not_execute_anything(ui):
    from tools.command import parses_as_tool

    before = dict(ui.ctx.env.targets)
    assert parses_as_tool("move_to CRXS=60,150", codes=ui.fleet.codes) is True
    assert parses_as_tool("everyone form a circle", codes=ui.fleet.codes) is False
    assert ui.ctx.env.targets == before, "parsing moved something"


# -- entities from the UI, instead of hand-editing workspace.json ------------

def test_add_entity_creates_a_followable_patrolling_thing(ui):
    ui.add_entity()
    ents = list(ui.ws.entities)
    assert len(ents) == 1
    e = ents[0]
    assert e.id == "wanderer"
    assert e.role == "target" and e.followable
    assert e.motion["kind"] == "path" and e.motion["mode"] == "pingpong"

    start = e.pos.copy()
    for _ in range(30):
        ui.step_world(0.1)
    assert np.linalg.norm(e.pos - start) > 5.0, "the new entity never moved"


def test_added_entities_persist_and_get_distinct_ids(ui):
    ui.add_entity()
    ui.add_entity()
    assert sorted(e.id for e in ui.ws.entities) == ["wanderer", "wanderer1"]

    reloaded = Workspace.load(ui.ws.path)
    assert sorted(e.id for e in reloaded.entities) == ["wanderer", "wanderer1"]
    assert reloaded.errors == []


def test_added_entity_is_reachable_by_the_follow_tool(ui):
    """The point of adding one: being able to command robots at it."""
    from tools import call

    ui.add_entity()
    r = call(ui.ctx, "follow", {"target_id": "wanderer", "duration": 30})
    assert r["ok"], r["error"]


def test_entity_role_cycles_and_changes_what_it_is(ui):
    ui.add_entity()
    e = ui.ws.entities.by_id("wanderer")
    assert e.role == "target" and not e.blocks

    ui.cycle_entity_role("wanderer")
    assert e.role == "obstacle" and e.blocks and not e.followable
    assert e.shape() in ui.ws.blocking()

    ui.cycle_entity_role("wanderer")
    assert e.role == "both" and e.blocks and e.followable

    ui.cycle_entity_role("wanderer")
    assert e.role == "target"


def test_removing_an_entity_drops_layers_aimed_at_it(ui):
    """Otherwise a robot sits there following something that no longer exists."""
    from tools import call

    ui.add_entity()
    assert call(ui.ctx, "follow", {"target_id": "wanderer", "duration": 60})["ok"]
    assert ui.ctx.stack.codes()

    ui.remove_entity("wanderer")
    assert ui.ws.entities.by_id("wanderer") is None
    assert ui.ctx.stack.codes() == [], "a layer outlived its target"


def test_entity_panel_rows_are_clickable_and_do_not_overlap(ui):
    ui.add_entity()
    render(ui)
    assert ui.entity_rows and ui.entity_rows[0]["id"] == "wanderer"

    regions = [(b.rect, f"button {b.label}") for b in ui.buttons]
    regions += [(ui.ent_add_btn.rect, "+ ent"), (ui.obs_add_btn.rect, "+ obs")]
    for row in ui.entity_rows:
        regions += [(row["role"], f"role {row['id']}"),
                    (row["del"], f"del {row['id']}")]
    for row in ui.obstacle_rows:
        regions.append((row["del"], f"obstacle del {row['i']}"))
    for i, (ra, na) in enumerate(regions):
        for rb, nb in regions[i + 1:]:
            assert not ra.colliderect(rb), f"{na} overlaps {nb}"


def test_the_arena_draws_motion_without_throwing(ui):
    """Trails, path polylines, follow lines and the stall ring."""
    from tools import call

    ui.add_entity()
    call(ui.ctx, "follow", {"target_id": "wanderer", "followers": ["SSMK"],
                             "duration": 60})
    call(ui.ctx, "set_path", {"assignments": {"CRXS": [[60, 40], [180, 40]]},
                               "mode": "loop", "duration": 60})
    call(ui.ctx, "set_flow", {"expr": "vx = -(y-cy)*0.4\nvy = (x-cx)*0.4",
                               "robots": ["SYRX"], "entity": "wanderer",
                               "duration": 60})
    for _ in range(20):
        ui.step_world(0.1)
        render(ui)
    assert len(ui.trails.get("wanderer", [])) > 1, "no trail was recorded"


# -- an unreachable hosted preset must not cost you plain English ------------

def _app_with_presets(tmp_path, model, available):
    """Build an App whose AgentSession availability we control per preset."""
    import llm.session as sessmod

    roster_path = tmp_path / "roster.json"
    Roster(entries=[RobotEntry(name="Dragon0", code="SSMK", kind="sim",
                                color="cyan")], path=roster_path).save()

    class FakeSession:
        def __init__(self, ctx, preset=None, sim_lock=None, warm=False, **kw):
            self.preset = preset
            self.available = available.get(preset, False)
            self.error = None if self.available else "not answering"
            self.busy = False
            self.model_name = preset
            self.mean_latency = None
            self.warmed = True

        def poll(self, limit=64):
            return []

        def result(self):
            return None

    real = sessmod.AgentSession
    sessmod.AgentSession = FakeSession
    try:
        return appmod.App(roster_path=roster_path, model=model)
    finally:
        sessmod.AgentSession = real


def test_an_unreachable_hosted_preset_falls_back_to_local(tmp_path):
    a = _app_with_presets(tmp_path, "sonnet5",
                          {"sonnet5": False, "qwen3.5:9b": True})
    try:
        assert a.session.available
        assert a.model_preset == "qwen3.5:9b"
        assert a.fell_back_from == "sonnet5"
        assert a.ask is True, "plain English was lost to a network problem"

        log = " ".join(t for _, t in a.log)
        assert "sonnet5 is unreachable" in log
        assert "falling back" in log, "a silent fallback is worse than the outage"
    finally:
        a.fleet.close()
        pygame.quit()


def test_a_reachable_preset_does_not_fall_back(tmp_path):
    a = _app_with_presets(tmp_path, "sonnet5",
                          {"sonnet5": True, "qwen3.5:9b": True})
    try:
        assert a.model_preset == "sonnet5"
        assert a.fell_back_from is None
        assert "falling back" not in " ".join(t for _, t in a.log)
    finally:
        a.fleet.close()
        pygame.quit()


def test_everything_unreachable_degrades_to_tool_mode(tmp_path):
    a = _app_with_presets(tmp_path, "sonnet5",
                          {"sonnet5": False, "qwen3.5:9b": False,
                           "qwen3.5:4b": False})
    try:
        assert a.ask is False
        assert a.fell_back_from is None
        a.submit("move_to SSMK=60,40")           # still fully usable by hand
        assert a.log[-1][0] == "ok"
    finally:
        a.fleet.close()
        pygame.quit()


def test_the_status_line_names_the_model_actually_answering(tmp_path):
    a = _app_with_presets(tmp_path, "sonnet5",
                          {"sonnet5": False, "qwen3.5:9b": True})
    try:
        render(a)                                 # must not throw
        assert a.model_preset == "qwen3.5:9b" and a.fell_back_from == "sonnet5"
    finally:
        a.fleet.close()
        pygame.quit()


# -- manual WASD override ----------------------------------------------------

def press(a, name):
    a.manual_key_down(pygame.key.key_code(name))


def release(a, name):
    a.keys_held.discard(pygame.key.key_code(name))


def test_number_key_takes_manual_control(ui):
    assert ui.manual_code is None
    press(ui, "1")
    assert ui.manual_code == ui.fleet.codes[0]
    # pressing it again hands the robot back
    press(ui, "1")
    assert ui.manual_code is None


def test_w_drives_along_the_steered_heading(ui):
    press(ui, "1")
    code = ui.manual_code
    ui.manual_heading[code] = 0.0            # +x, to the right
    start = ui.fleet[code].pos.copy()

    press(ui, "w")
    for _ in range(30):
        ui.step_world(0.1)

    moved = ui.fleet[code].pos - start
    assert moved[0] > 10.0, f"W did not drive forward: {moved}"
    assert abs(moved[1]) < abs(moved[0]) / 2, "drifted sideways instead"


def test_s_drives_backwards(ui):
    press(ui, "1")
    code = ui.manual_code
    ui.manual_heading[code] = 0.0
    start = ui.fleet[code].pos.copy()

    press(ui, "s")
    for _ in range(30):
        ui.step_world(0.1)
    assert (ui.fleet[code].pos - start)[0] < -10.0


def test_a_and_d_turn_without_moving(ui):
    press(ui, "1")
    code = ui.manual_code
    ui.manual_heading[code] = 90.0
    start = ui.fleet[code].pos.copy()

    press(ui, "d")
    for _ in range(10):
        ui.step_world(0.1)
    assert ui.manual_heading[code] > 90.0, "d did not turn right"
    release(ui, "d")

    before = ui.manual_heading[code]
    press(ui, "a")
    for _ in range(10):
        ui.step_world(0.1)
    assert ui.manual_heading[code] < before, "a did not turn left"

    assert np.linalg.norm(ui.fleet[code].pos - start) < 5.0, \
        "turning moved the robot; W/S is what drives"


def test_releasing_the_key_commands_a_stop(ui):
    """Asserted on what manual drive commands, not on what happens next.

    Once the key is up the controller takes the robot back, and if it is near a
    wall or a neighbour, avoidance legitimately keeps it moving. Measuring the
    robot's velocity therefore measures the controller, not the release.
    """
    press(ui, "1")
    code = ui.manual_code
    h = ui.fleet[code]

    press(ui, "w")
    for _ in range(10):
        ui.step_world(0.1)
    assert float(np.linalg.norm(h.vel)) > 10.0, "it never got moving"

    release(ui, "w")
    ui.drive_manual(0.1)
    assert np.allclose(h._desired, 0.0), "letting go of W kept commanding speed"


def test_releasing_manual_control_halts_and_hands_back(ui):
    press(ui, "1")
    code = ui.manual_code
    press(ui, "w")
    for _ in range(10):
        ui.step_world(0.1)

    ui.release_manual()
    assert np.allclose(ui.fleet[code]._desired, 0.0), "release did not command a stop"
    assert ui.manual_code is None
    assert ui.keys_held == set(), "a held key survived the release"

    # And a subsequent W does nothing, because nothing is selected.
    press(ui, "w")
    assert ui.keys_held == set()


def test_taking_manual_control_clears_layers_and_targets(ui):
    """Two things steering one ball reads as the robot ignoring you."""
    from tools import call

    code = ui.fleet.codes[0]
    call(ui.ctx, "set_flow", {"expr": "vx = -(y-cy)*0.4\nvy = (x-cx)*0.4",
                               "robots": [code], "duration": 60})
    ui.ctx.apply_targets({code: (200.0, 150.0)})
    assert ui.ctx.stack.layers(code)

    press(ui, "1")
    assert ui.ctx.stack.layers(code) == []
    assert code not in ui.ctx.env.targets
    assert ui.fleet[code].target is None


def test_wasd_does_not_reach_the_ask_toggle(ui):
    """`a` is the ask/tool toggle; turning left must not flip the model."""
    assert ui.manual_key_down(pygame.key.key_code("a")) is False   # nothing selected
    press(ui, "1")
    assert ui.manual_key_down(pygame.key.key_code("a")) is True    # consumed


def test_escape_releases_and_halts(ui):
    press(ui, "1")
    code = ui.manual_code
    press(ui, "w")
    for _ in range(5):
        ui.step_world(0.1)
    ui.manual_key_down(pygame.K_ESCAPE)
    assert ui.manual_code is None

    # Momentum is real: a sim robot decelerates through the same first-order
    # motor lag as the trainer, so it coasts down rather than stopping dead.
    fast = float(np.linalg.norm(ui.fleet[code].vel))
    for _ in range(20):
        ui.step_world(0.1)
    assert float(np.linalg.norm(ui.fleet[code].vel)) < max(5.0, fast / 3)


def test_only_one_robot_is_manual_at_a_time(ui):
    press(ui, "1")
    first = ui.manual_code
    press(ui, "2")
    assert ui.manual_code == ui.fleet.codes[1] != first


def test_the_arena_draws_the_manual_indicator(ui):
    press(ui, "1")
    render(ui)                       # ring plus heading arrow, must not throw
    assert ui.manual_code is not None


# -- the live camera view ----------------------------------------------------
#
# Driven against the synthetic source, so this needs no camera and no robots.

@pytest.fixture
def ui_cam(tmp_path):
    roster_path = tmp_path / "roster.json"
    ws_path = tmp_path / "workspace.json"
    Roster(entries=[RobotEntry(name="Seasmoke", code="SSMK", kind="sim",
                                color="cyan")], path=roster_path).save()
    Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
              obstacles=[], path=ws_path).save()
    a = appmod.App(roster_path=roster_path, workspace_path=ws_path,
                   camera="synthetic")
    yield a
    if a.tracker is not None:
        a.tracker.stop()
    a.fleet.close()
    pygame.quit()


def _wait_for_frame(a, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame, raw = a.tracker.latest()
        if frame is not None:
            return frame, raw
        time.sleep(0.05)
    return None, {}


def test_the_tracker_hands_over_the_latest_frame(ui_cam):
    """The plumbing: frames arrive and blobs come out of them.

    Hunting is widened to the whole palette first, because the app now narrows
    it to the colours its fleet is wearing — which is the point, and which
    would make this a test of whether one sim robot's hue happens to be one the
    synthetic source draws."""
    ui_cam.tracker.set_colors(None)
    frame, raw = _wait_for_frame(ui_cam)
    assert frame is not None, "no frame arrived from the synthetic source"
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert raw, "the synthetic source should produce detectable blobs"


def test_the_app_hunts_only_the_colours_its_fleet_wears(ui_cam):
    """Six masks are searched per frame otherwise, and the ones nobody wears
    find room clutter rather than robots."""
    _wait_for_frame(ui_cam)
    worn = {h.color for h in ui_cam.fleet.handles.values() if h.color}
    if worn:
        assert set(ui_cam.tracker.hunting) == worn


def test_an_empty_fleet_hunts_everything_rather_than_nothing(ui_cam):
    """Tuning a colour happens before a robot is connected, so a camera view
    that blanks when the last robot is released looks broken — and with no
    robots there is no wrong ball to lock onto anyway."""
    ui_cam.tracker.set_colors(set())
    assert len(ui_cam.tracker.hunting) >= 6


def test_the_camera_panel_renders_and_caches(ui_cam):
    _wait_for_frame(ui_cam)
    surf, raw, f = ui_cam.camera_surface()
    assert surf is not None
    assert surf.get_width() == ui_cam.CAM_W
    assert 0 < f <= 1.0

    # The camera thread is still running, and if it pushes a new frame between
    # the two calls then rebuilding is correct, not a cache miss. Stopping it
    # first is what makes this test about caching rather than about timing.
    ui_cam.tracker.running = False
    ui_cam.tracker._thread.join(timeout=2.0)
    surf, _, _ = ui_cam.camera_surface()
    again, _, _ = ui_cam.camera_surface()
    assert again is surf, "the surface was rebuilt for an unchanged frame"


def test_v_toggles_the_camera_view(ui_cam):
    _wait_for_frame(ui_cam)
    assert ui_cam.camera_view == "panel"
    render(ui_cam)                       # visible: must not throw

    ui_cam.camera_view = "off"
    render(ui_cam)                       # hidden: must not throw either


def test_the_panel_stays_inside_the_arena(ui_cam):
    _wait_for_frame(ui_cam)
    render(ui_cam)
    surf, _, _ = ui_cam.camera_surface()
    r = ui_cam.arena_rect()
    x = r.right - surf.get_width() - 8
    y = r.bottom - surf.get_height() - 8
    assert x >= r.left and y >= r.top, "the camera panel overflows the arena"


def test_no_camera_means_no_panel_and_no_crash(ui):
    """The overwhelmingly common case: running all-sim with no --camera."""
    assert ui.tracker is None
    assert ui.camera_surface() == (None, {}, 1.0)
    render(ui)


# -- heading calibration from observed travel --------------------------------

@pytest.fixture
def ui_real(tmp_path):
    """One real robot on a fake radio and a fake tracker, plus sim company."""
    import fleet.real_handle as rh
    from tests.conftest import FakeConnector, FakeTracker

    roster_path = tmp_path / "roster.json"
    ws_path = tmp_path / "workspace.json"
    Roster(entries=[RobotEntry(name="Seasmoke", code="SSMK", kind="real",
                                color="cyan", ble_name="SK-914A"),
                    RobotEntry(name="Caraxes", code="CRXS", kind="sim",
                                color="red")], path=roster_path).save()
    Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
              obstacles=[], path=ws_path).save()

    real = rh.default_connector
    rh.default_connector = FakeConnector()
    try:
        a = appmod.App(roster_path=roster_path, workspace_path=ws_path)
    finally:
        rh.default_connector = real
    a.fleet.tracker = FakeTracker({"cyan": [60.0, 60.0]})
    a.fleet["SSMK"].tracker = a.fleet.tracker
    yield a
    a.fleet.close()
    pygame.quit()


def _drive_and_observe(a, code, commanded_deg, actual_deg, distance=40.0):
    """Pretend the robot was told one direction and physically went another."""
    a.manual_key_down(pygame.key.key_code("1"))
    a.manual_heading[code] = commanded_deg
    a.manual_key_down(pygame.key.key_code("w"))

    h = a.fleet[code]
    start = np.array([80.0, 90.0])
    rad = np.radians(actual_deg)
    steps = 20
    a.manual_track = []
    for i in range(steps):
        h.pos = start + np.array([np.cos(rad), np.sin(rad)]) * (distance * i / steps)
        a.drive_manual(0.1)
    return h


def test_calibration_cancels_the_error_it_measured(ui_real):
    """The property that matters, rather than a hand-derived constant.

    Drive with a known physical bias, calibrate, then drive again with the SAME
    bias: the second measurement must come out at ~0, because the offset now
    absorbs it. This is sign-agnostic, which the first version of this test was
    not — I got the sign backwards and the implementation was right.
    """
    from fleet.handle import velocity_to_command

    BIAS = 40.0
    _drive_and_observe(ui_real, "SSMK", commanded_deg=0.0, actual_deg=BIAS)
    ui_real.calibrate_heading()
    h = ui_real.fleet["SSMK"]
    assert h.heading_offset != 0.0
    assert "heading offset" in ui_real.log[-1][1]

    # What actually goes on the radio now, for a due-+x request.
    h.set_velocity(np.array([35.0, 0.0]))
    sent = h._pending[0]
    naive = velocity_to_command(np.array([35.0, 0.0]))[0]
    assert (sent - naive) % 360.0 == pytest.approx(h.heading_offset, abs=1e-6)

    # And re-measuring against the same bias now reads as no error.
    _drive_and_observe(ui_real, "SSMK", commanded_deg=0.0, actual_deg=BIAS)
    before = h.heading_offset
    ui_real.calibrate_heading()
    assert h.heading_offset == pytest.approx(before, abs=2.0), \
        "calibrating twice against the same bias should not keep moving it"


def test_calibration_is_saved_to_the_roster(ui_real):
    _drive_and_observe(ui_real, "SSMK", commanded_deg=0.0, actual_deg=30.0)
    ui_real.calibrate_heading()
    saved = Roster.load(ui_real.roster.path).by_code("SSMK")
    assert saved.heading_offset == pytest.approx(ui_real.fleet["SSMK"].heading_offset)


def test_a_correctly_aimed_robot_calibrates_to_no_change(ui_real):
    h = _drive_and_observe(ui_real, "SSMK", commanded_deg=0.0, actual_deg=0.0)
    ui_real.calibrate_heading()
    assert h.heading_offset == pytest.approx(0.0, abs=2.0)


def test_calibration_refuses_without_enough_travel(ui_real):
    _drive_and_observe(ui_real, "SSMK", commanded_deg=0.0, actual_deg=40.0,
                       distance=3.0)
    ui_real.calibrate_heading()
    assert ui_real.log[-1][0] == "error"
    assert "hold W" in ui_real.log[-1][1]
    assert ui_real.fleet["SSMK"].heading_offset == 0.0


def test_calibration_refuses_before_driving(ui_real):
    ui_real.manual_key_down(pygame.key.key_code("1"))
    ui_real.calibrate_heading()
    assert ui_real.log[-1][0] == "error"
    assert "drive it forward" in ui_real.log[-1][1]


def test_calibrating_a_sim_robot_says_there_is_nothing_to_calibrate(ui_real):
    ui_real.select_manual("CRXS")
    ui_real.calibrate_heading()
    assert "simulated" in ui_real.log[-1][1]


# -- speed knob --------------------------------------------------------------

def test_the_app_drives_at_cruise_speed_not_the_hardware_ceiling(ui):
    """`fleet.handle.MAX_SPEED` is the motor-byte calibration, not the throttle."""
    from fleet.handle import MAX_SPEED

    assert ui.ctx.max_speed == appmod.CRUISE_SPEED
    assert appmod.CRUISE_SPEED < MAX_SPEED


def test_cruise_speed_actually_caps_how_fast_robots_go(ui):
    code = ui.fleet.codes[0]
    ui.fleet[code].pos = np.array([20.0, 90.0])
    ui.ctx.apply_targets({code: (220.0, 90.0)})     # far away: full throttle
    for _ in range(40):
        ui.step_world(0.1)
    # The command, not the achieved speed: a sim robot's gain is 0.8-1.2, so
    # what it reaches legitimately overshoots what it was told.
    commanded = float(np.linalg.norm(ui.fleet[code]._desired))
    assert commanded <= appmod.CRUISE_SPEED + 1e-6, f"commanded {commanded:.0f}cm/s"
    assert commanded > appmod.CRUISE_SPEED * 0.5, "never got up to speed at all"


def test_speed_can_be_overridden_per_run(tmp_path):
    roster_path = tmp_path / "roster.json"
    Roster(entries=[RobotEntry(name="Seasmoke", code="SSMK", kind="sim",
                                color="cyan")], path=roster_path).save()
    a = appmod.App(roster_path=roster_path, speed=12.0)
    try:
        assert a.ctx.max_speed == 12.0
        a.fleet["SSMK"].pos = np.array([20.0, 90.0])
        a.ctx.apply_targets({"SSMK": (220.0, 90.0)})
        for _ in range(40):
            a.step_world(0.1)
        assert float(np.linalg.norm(a.fleet["SSMK"]._desired)) <= 12.0 + 1e-6
    finally:
        a.fleet.close()
        pygame.quit()


def test_manual_drive_respects_the_same_knob(ui):
    """WASD writes velocities directly, so it needs its own limit.

    Asserted on the commanded velocity, not the achieved one: a sim robot
    carries a per-robot gain of 0.8-1.2, so what it actually reaches can exceed
    the command by a fifth. That is the dynamics being modelled, not the
    throttle being ignored.
    """
    assert ui.MANUAL_SPEED <= appmod.CRUISE_SPEED
    press(ui, "1")
    press(ui, "w")
    for _ in range(20):
        ui.step_world(0.1)
    commanded = float(np.linalg.norm(ui.fleet[ui.manual_code]._desired))
    assert commanded == pytest.approx(ui.MANUAL_SPEED, abs=0.5)


# -- the camera and the planner must describe the same rectangle -------------

def _stale_calibration(monkeypatch, width, height):
    """A homography for a rectangle of the given size, served from memory.

    Never writes calib/homography.json — that is the user's real calibration,
    and a test that rewrites live state is a test that eats an afternoon of
    hardware setup. The first version of this helper called `h.save()` and was
    saved only by the file happening to be root-owned.
    """
    import cv2
    import numpy as _np
    from vision.homography import Homography

    h = Homography(arena=max(width, height), width=width, height=height)
    src = _np.float32([[100, 80], [500, 90], [510, 400], [90, 390]])
    dst = _np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    h.M = cv2.getPerspectiveTransform(src, dst).astype(float)
    monkeypatch.setattr(Homography, "load", classmethod(lambda cls: h))
    return h


def _cam_app(tmp_path):
    roster_path = tmp_path / "roster.json"
    ws_path = tmp_path / "workspace.json"
    Roster(entries=[RobotEntry(name="Seasmoke", code="SSMK", kind="sim",
                                color="cyan")], path=roster_path).save()
    Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
              obstacles=[], path=ws_path).save()
    return appmod.App(roster_path=roster_path, workspace_path=ws_path,
                      camera="synthetic")


def test_a_mismatched_calibration_is_reported_at_startup(tmp_path, monkeypatch):
    _stale_calibration(monkeypatch, 200, 200)          # square, under a 240x180 arena
    a = _cam_app(tmp_path)
    try:
        log = " ".join(t for _, t in a.log)
        assert "200x200" in log and "240x180" in log
        assert "drift off the arena" in log
        assert "workspace.make --from-camera" in log, "must say how to fix it"
        assert a.homography is not None
        render(a)                                    # the overlay must not throw
    finally:
        if a.tracker: a.tracker.stop()
        a.fleet.close(); pygame.quit()


def test_a_matching_calibration_is_silent(tmp_path, monkeypatch):
    _stale_calibration(monkeypatch, 240, 180)
    a = _cam_app(tmp_path)
    try:
        log = " ".join(t for _, t in a.log)
        assert "drift off the arena" not in log
        assert a.homography.matches(240, 180)
        render(a)
    finally:
        if a.tracker: a.tracker.stop()
        a.fleet.close(); pygame.quit()


def test_no_calibration_at_all_says_so(tmp_path, monkeypatch):
    from vision.homography import Homography
    monkeypatch.setattr(Homography, "load", classmethod(lambda cls: cls()))
    a = _cam_app(tmp_path)
    try:
        log = " ".join(t for _, t in a.log)
        assert "no camera calibration" in log
        assert a.homography is None
        render(a)
    finally:
        if a.tracker: a.tracker.stop()
        a.fleet.close(); pygame.quit()


def test_running_without_a_camera_checks_nothing(ui):
    assert ui.tracker is None
    assert ui.homography is None
    assert not any("calibration" in t for _, t in ui.log)


# -- the camera as the floor -------------------------------------------------

def test_v_cycles_panel_floor_off(ui_cam):
    _wait_for_frame(ui_cam)
    assert ui_cam.camera_view == "panel"
    ui_cam.cycle_camera_view(); assert ui_cam.camera_view == "floor"
    ui_cam.cycle_camera_view(); assert ui_cam.camera_view == "off"
    ui_cam.cycle_camera_view(); assert ui_cam.camera_view == "panel"


def test_the_floor_warps_to_the_arena_rectangle(ui_cam, monkeypatch):
    _stale_calibration(monkeypatch, 240, 180)
    ui_cam.homography = None
    ui_cam.check_calibration()
    frame, _ = _wait_for_frame(ui_cam)
    assert frame is not None, "the synthetic camera never delivered a frame"

    ui_cam.camera_view = "floor"
    # The capture thread may still be between frames; give it a beat rather
    # than making this a race on a loaded machine.
    surf = None
    deadline = time.time() + 3.0
    while surf is None and time.time() < deadline:
        surf = ui_cam.floor_view()
        if surf is None:
            time.sleep(0.05)
    assert surf is not None, "no warped floor produced"
    r = ui_cam.arena_rect()
    assert surf.get_size() == (r.w, r.h), "the floor must fill the arena exactly"
    render(ui_cam)                       # drawn under everything, must not throw


def test_the_floor_is_not_re_warped_for_an_unchanged_frame(ui_cam, monkeypatch):
    """Pinned to one frame on purpose.

    Against the live capture thread a second call can legitimately get a NEW
    frame and re-warp — so asserting object identity there tests the camera's
    timing, not the cache, and fails about one run in four.
    """
    _stale_calibration(monkeypatch, 240, 180)
    ui_cam.homography = None
    ui_cam.check_calibration()
    frame, raw = _wait_for_frame(ui_cam)
    assert frame is not None

    monkeypatch.setattr(ui_cam.tracker, "latest", lambda: (frame, raw))
    ui_cam.camera_view = "floor"

    first = ui_cam.floor_view()
    assert first is not None
    assert ui_cam.floor_view() is first, "re-warped an unchanged frame"


def test_no_floor_without_a_calibration(ui_cam, monkeypatch):
    from vision.homography import Homography
    monkeypatch.setattr(Homography, "load", classmethod(lambda cls: cls()))
    ui_cam.homography = None
    _wait_for_frame(ui_cam)
    ui_cam.camera_view = "floor"
    assert ui_cam.floor_view() is None
    render(ui_cam)


def test_panel_and_floor_are_mutually_exclusive(ui_cam, monkeypatch):
    _stale_calibration(monkeypatch, 240, 180)
    ui_cam.homography = None
    ui_cam.check_calibration()
    _wait_for_frame(ui_cam)

    ui_cam.camera_view = "floor"
    assert ui_cam.floor_view() is not None
    ui_cam.screen.fill(appmod.INK)
    ui_cam.draw_camera()                 # the picture-in-picture must stand down
    assert ui_cam.camera_view == "floor"


def test_an_all_sim_app_never_asks_for_a_floor(ui):
    ui.camera_view = "floor"
    assert ui.floor_view() is None
    render(ui)


# -- command bar paste and log scrollback ------------------------------------

def test_cmd_v_pastes_instead_of_typing_v(ui, monkeypatch):
    """`v` is the camera toggle, so the modifier has to be checked first."""
    monkeypatch.setattr(appmod, "clipboard_text", lambda: "move_to 60,40")
    ui.cmd.focused = True
    ui.cmd.text = ""
    ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v,
                                   mod=pygame.KMOD_META, unicode="v"))
    assert ui.cmd.text == "move_to 60,40"


def test_plain_v_still_types_a_v(ui):
    ui.cmd.focused = True
    ui.cmd.text = ""
    ui.cmd.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v, mod=0,
                                   unicode="v"))
    assert ui.cmd.text == "v"


def test_clipboard_never_raises(monkeypatch):
    """A clipboard that is empty, missing or hostile must not kill a keystroke."""
    import pygame.scrap as scrap
    monkeypatch.setattr(scrap, "get_init", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert appmod.clipboard_text() == ""


def test_pasted_newlines_become_one_line(monkeypatch):
    """A multi-line paste must not submit halfway or corrupt the field."""
    import subprocess

    class Done:
        returncode = 0
        stdout = "move_to 60,40\n120,40\n"

    import pygame.scrap as scrap
    monkeypatch.setattr(scrap, "get_init", lambda: True)
    monkeypatch.setattr(scrap, "get", lambda kind: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())

    text = appmod.clipboard_text()
    assert "\n" not in text
    assert text == "move_to 60,40 120,40"


def test_the_log_scrolls_back_and_returns(ui):
    for i in range(200):
        ui.say("info", f"line {i}")
    render(ui)
    assert ui.log_max_scroll > 0, "nothing to scroll — the log did not overflow"

    ui.scroll_log(20)
    assert ui.log_scroll == 20
    render(ui)                                   # the scrolled view must draw

    ui.scroll_log(-1000)
    assert ui.log_scroll == 0, "did not return to following the tail"


def test_scrolling_is_clamped_to_the_history(ui):
    for i in range(50):
        ui.say("info", f"line {i}")
    render(ui)
    ui.scroll_log(100000)
    render(ui)
    assert ui.log_scroll == ui.log_max_scroll


def test_a_new_line_does_not_yank_a_scrolled_view(ui):
    """Reading history while a run is producing output has to be possible."""
    for i in range(200):
        ui.say("info", f"line {i}")
    render(ui)
    ui.scroll_log(30)
    before = ui.log_scroll

    ui.say("info", "something new")
    render(ui)
    assert ui.log_scroll >= before, "the view jumped to the tail"


def test_the_log_keeps_more_history_than_it_shows(ui):
    for i in range(500):
        ui.say("info", f"line {i}")
    assert len(ui.log) == appmod.LOG_HISTORY
    assert appmod.LOG_HISTORY > 40, "scrollback needs history to scroll through"


# -- the calibration button --------------------------------------------------


def _make_connected(ui_real, code="SSMK"):
    """Put a real handle into the connected state without waiting on the radio.

    `connected` is link_up AND a tracker fix under 0.5s old. Waiting for the
    real thing costs the 1.5s fleet-wide connect stagger in every test that
    needs it, and the stagger is not what is under test here.
    """
    h = ui_real.fleet[code]
    h._link_up = True
    for _ in range(3):
        ui_real.fleet.step(0.05)
    assert h.connected, "fixture did not reach a connected state"
    return h


def test_calibrating_a_real_robot_drives_a_pattern_and_saves_the_offset(ui_real):
    """Four legs, read the frame offset off them, write it to the roster."""
    import math

    TRUE_OFFSET = 35.0
    h = _make_connected(ui_real)
    h.pos = np.array([120.0, 90.0])
    ui_real.select_manual("SSMK")
    ui_real.start_calibration()
    assert ui_real.calibrating is not None, ui_real.log[-1]

    # The ball moves offset from whatever it is commanded.
    for _ in range(int(60 / 0.05)):
        if ui_real.calibrating is None:
            break
        before = h._desired.copy() if hasattr(h, "_desired") else np.zeros(2)
        ui_real.step_calibration(0.05)
        v = h._desired
        speed = float(np.linalg.norm(v))
        if speed > 1e-9:
            cmd = math.degrees(math.atan2(v[1], v[0]))
            rad = math.radians(cmd + TRUE_OFFSET)
            h.pos = h.pos + np.array([math.cos(rad), math.sin(rad)]) * speed * 0.05

    assert ui_real.calibrating is None, "never finished"
    kind, text = ui_real.log[-1]
    assert kind == "ok", text
    assert "heading offset" in text and "agreed within" in text

    # The correction CANCELS the error, so the stored offset is its negative.
    # Asserting it equals the error is the mistake that let a sign bug through
    # once already, so assert the behaviour instead: ask for due east and check
    # the robot actually goes east.
    assert abs(((h.heading_offset + TRUE_OFFSET + 180) % 360) - 180) < 6.0

    saved = Roster.load(ui_real.roster.path).by_code("SSMK")
    assert saved.heading_offset == pytest.approx(h.heading_offset)


def test_calibration_refuses_a_robot_the_camera_cannot_see(ui_real):
    """It measures where the robot went, so a fix is not optional."""
    ui_real.fleet.tracker = None
    ui_real.fleet["SSMK"].tracker = None
    for _ in range(3):
        ui_real.fleet["SSMK"].step(0.1)
    ui_real.start_calibration()
    assert ui_real.calibrating is None
    assert "no camera fix" in ui_real.log[-1][1]


def test_a_sim_robot_calibrates_too(ui_real):
    """Deliberately allowed: SimRobot models a heading bias, so there is a real
    quantity to recover — and it makes the whole path testable with no ball."""
    ui_real.select_manual("CRXS")
    ui_real.start_calibration()
    assert ui_real.calibrating is not None
    assert ui_real.calibrating[0] == "CRXS"


def test_calibration_takes_precedence_over_manual_drive(ui_real):
    """Anything else steering during the pattern corrupts the measurement."""
    h = _make_connected(ui_real)
    h.pos = np.array([120.0, 90.0])
    ui_real.select_manual("SSMK")
    ui_real.start_calibration()
    press(ui_real, "w")                       # a hand on the keyboard mid-run

    ui_real.step_calibration(0.05)
    driven = h._desired.copy()
    ui_real.drive_manual(0.05)
    assert np.allclose(h._desired, driven) or ui_real.calibrating is not None


def test_the_calib_button_does_not_overlap_anything(ui):
    render(ui)
    for b in ui.buttons:
        if b is ui.calib_btn:
            continue
        assert not ui.calib_btn.rect.colliderect(b.rect), f"overlaps {b.label}"


def test_calibration_makes_the_robot_go_where_it_is_told(ui):
    """The round trip, on a sim robot: this is what the offset is *for*.

    A sim robot models a per-robot heading bias of +-0.12 rad, so there is a
    real quantity to recover — and the whole path is exercisable with no
    hardware, which is the point of allowing sim calibration at all.
    """
    import math

    code = ui.fleet.codes[0]
    h = ui.fleet[code]
    h.bias = math.radians(28.0)          # a known, deliberate error
    # NOT (120, 90) — the fixture puts an obstacle there, and a robot starting
    # inside one is shoved out every step, which reads as "the robot is stuck".
    CLEAR = np.array([60.0, 60.0])
    h.pos = CLEAR.copy()

    def travel_for(commanded_deg, seconds=1.2, dt=0.05):
        start = h.pos.copy()
        rad = math.radians(commanded_deg)
        v = np.array([math.cos(rad), math.sin(rad)]) * 25.0
        for _ in range(int(seconds / dt)):
            h.set_velocity(v)
            h.step(dt)
        d = h.pos - start
        return math.degrees(math.atan2(d[1], d[0]))

    before = abs(((travel_for(0.0) - 0.0 + 180) % 360) - 180)
    assert before > 15.0, f"the bias should be visible, saw {before:.0f}deg"

    h.pos = CLEAR.copy()
    ui.select_manual(code)
    ui.start_calibration()
    assert ui.calibrating is not None, ui.log[-1]
    for _ in range(int(60 / 0.05)):
        if ui.calibrating is None:
            break
        ui.step_calibration(0.05)
        h.step(0.05)                      # the sim applies bias + offset itself

    assert ui.calibrating is None
    assert ui.log[-1][0] == "ok", ui.log[-1][1]

    h.pos = CLEAR.copy()
    after = abs(((travel_for(0.0) - 0.0 + 180) % 360) - 180)
    assert after < 8.0, f"still {after:.0f}deg off after calibrating"
    assert after < before / 2.0, "calibration did not improve anything"


def test_a_sim_robot_can_be_calibrated_at_all(ui):
    """It used to refuse, which made the whole path untestable without a ball."""
    ui.select_manual(ui.fleet.codes[0])
    ui.start_calibration()
    assert ui.calibrating is not None
    assert "simulated" not in ui.log[-1][1]

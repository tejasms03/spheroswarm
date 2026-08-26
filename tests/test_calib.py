"""The calibration bench: colour learning, the dock, and the battery it drives.

Headless throughout. The colour tests build frames rather than reading them
from a camera, because the property that matters — "does the signature this
learns actually detect the ball?" — is checkable exactly when the ball is
synthetic and cannot be checked at all when it is a photograph.
"""

import json
import math
import time
import pathlib

import cv2
import numpy as np
import pytest

from vision import config as vconfig
from vision.detect import Detector, autotune_signature, locate_change

pygame = pytest.importorskip("pygame")


# -- colour signatures -------------------------------------------------------

def frames(lit_bgr=None, n=6, seed=0, size=(240, 320), centre=(160, 120),
           radius=16, floor=70, core=True):
    """A grey floor, optionally with a lit translucent ball on it."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img = np.full((size[0], size[1], 3), floor, np.uint8)
        noise = rng.normal(0, 3, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if lit_bgr is not None:
            cv2.circle(img, centre, radius, lit_bgr, -1)
            if core:
                # A lit shell blows out to near-white in the middle. This is
                # the feature that breaks naive hue sampling, so it belongs in
                # the fixture rather than being tidied away.
                cv2.circle(img, centre, radius // 2, (245, 235, 245), -1)
        out.append(img)
    return out


@pytest.mark.parametrize("name,bgr", [
    ("red", (50, 50, 210)), ("green", (70, 200, 70)), ("blue", (215, 110, 60)),
    ("magenta", (190, 40, 190)), ("cyan", (210, 190, 60)), ("yellow", (60, 200, 220)),
])
def test_autotune_learns_a_signature_that_then_detects_the_ball(name, bgr):
    off, on = frames(), frames(bgr, seed=1)
    r = autotune_signature(off, on)
    assert r["ok"], r
    assert r["centroid"] == pytest.approx([160.0, 120.0], abs=3.0)

    d = Detector()
    d.colors[name].update({k: r[k] for k in vconfig.SIGNATURE_KEYS if k in r})
    found = d.detect(on[0], only=[name])
    assert name in found, f"{name}: learned a signature that finds nothing"
    x, y, _ = found[name]
    assert (x, y) == pytest.approx((160.0, 120.0), abs=4.0)


def test_autotune_ignores_the_blown_out_core():
    """Hue in the middle of a lit shell is white noise, not the ball's colour."""
    on = frames((190, 40, 190), seed=2)
    r = autotune_signature(frames(), on)
    assert r["ok"]
    assert 140 <= r["hue"] <= 160, f"magenta should land near 150, got {r['hue']}"


def test_autotune_handles_red_across_the_hue_wrap():
    """Red straddles 0/179. A scalar mean of those puts it at cyan."""
    r = autotune_signature(frames(), frames((50, 50, 215), seed=3))
    assert r["ok"]
    assert r["hue"] <= 12 or r["hue"] >= 168, r["hue"]


def test_autotune_refuses_when_nothing_lit_up():
    r = autotune_signature(frames(), frames(seed=9))
    assert not r["ok"]
    assert "nothing changed" in r["error"]


def test_autotune_refuses_a_signature_that_matches_the_whole_frame():
    """The failure that matters: a robot the camera cannot actually see.

    Drive a simulated robot's LED and the frame still changes — the arena has
    other things moving in it — so `locate_change` finds *something*. Learning
    a hue from that produces a signature which selects half the picture, and
    without this check it is returned as a confident success and saved over a
    working calibration.
    """
    rng = np.random.default_rng(7)
    off = [np.clip(np.full((240, 320, 3), 70, np.int16)
                   + rng.normal(0, 22, (240, 320, 3)), 0, 255).astype(np.uint8)
           for _ in range(6)]
    on = [np.clip(np.full((240, 320, 3), 70, np.int16)
                  + rng.normal(0, 22, (240, 320, 3)), 0, 255).astype(np.uint8)
          for _ in range(6)]
    r = autotune_signature(off, on)
    assert not r["ok"], r


def test_autotune_rejects_a_change_that_is_not_ball_shaped():
    """A wide bar of light is a change. It is not a Sphero."""
    off = frames()
    on = [f.copy() for f in frames(seed=5)]
    for f in on:
        cv2.rectangle(f, (10, 100), (310, 118), (60, 200, 220), -1)
    assert locate_change(off, on) is None


def test_autotune_is_not_fooled_by_a_changing_background():
    """A brightening floor is not a ball. It has no compact blob."""
    off = frames(floor=60)
    on = frames(floor=64, seed=4)
    assert locate_change(off, on) is None


def test_signatures_round_trip_through_the_calibration_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    colors = {n: dict(s) for n, s in vconfig.COLORS.items()}
    colors["green"].update(hue=64, tol=11, s_min=120, v_min=95, min_area=180)
    vconfig.save_signatures(colors)

    back = vconfig.load_signatures()
    assert back["green"]["hue"] == 64
    assert back["green"]["s_min"] == 120
    assert back["green"]["draw"] == vconfig.COLORS["green"]["draw"], \
        "draw colours are presentation, not calibration"


def test_a_stale_signature_file_cannot_invent_a_colour(tmp_path, monkeypatch):
    """The roster validates against COLORS; a calib file must not widen it."""
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    vconfig.save("colors", {"chartreuse": {"hue": 45}, "red": {"hue": 3}})
    back = vconfig.load_signatures()
    assert "chartreuse" not in back
    assert back["red"]["hue"] == 3


def test_a_corrupt_signature_file_falls_back_to_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    (tmp_path / "colors.json").write_text("{{{")
    assert vconfig.load_signatures()["red"]["hue"] == vconfig.COLORS["red"]["hue"]


def test_per_colour_floors_override_the_global_ones():
    # Explicit colours, not whatever is on disk: this is about the override
    # rule, and reading the live calibration would make it depend on the last
    # session someone ran.
    d = Detector(colors={n: dict(s) for n, s in vconfig.COLORS.items()})
    d.thresh.update(s_min=90, v_min=70, min_area=60, max_area=20000)
    assert d.limits("red")["s_min"] == 90
    d.colors["red"]["s_min"] = 140
    assert d.limits("red")["s_min"] == 140
    assert d.limits("blue")["s_min"] == 90, "one colour must not move the others"


# -- the bench ---------------------------------------------------------------

@pytest.fixture
def bench(monkeypatch, tmp_path):
    """A bench that cannot touch the live calibration.

    `calib/` is live state, exactly like roster.json and workspace.json: the
    bench writes signatures and motion fits into it by design. A test that
    saves a signature learned from the SYNTHETIC camera would leave the real
    red hue pointing at cyan, and the next real session would fail to see a
    ball for reasons nobody would connect to a test run.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import fleet.characterize as characterize
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    monkeypatch.setattr(characterize, "MOTION_PATH", tmp_path / "motion.json")
    import calib
    globals()["calib"] = calib
    roster = tmp_path / "roster.json"
    roster.write_text((tmp_path.parent / "x").parent.joinpath().as_posix() and
                      __import__("json").dumps([
                          {"name": "One", "code": "ONE", "kind": "sim",
                           "color": "red", "enabled": True},
                          {"name": "Two", "code": "TWO", "kind": "sim",
                           "color": "cyan", "enabled": True}]))
    # Its own arena, not the live one. These tests place robots at fixed
    # coordinates, so inheriting whatever workspace.json currently holds means
    # they break the day somebody measures their real floor — which is exactly
    # what happened.
    ws = tmp_path / "workspace.json"
    ws.write_text(__import__("json").dumps({
        "bounds_cm": [[0, 0], [200, 0], [200, 200], [0, 200]],
        "obstacles": [],
        "origin": "top-left, x right, y down, matching the camera frame"}))
    app = calib.CalibApp(camera="synthetic", roster_path=str(roster),
                         workspace_path=str(ws))
    yield app
    app.fleet.close()
    if app.tracker:
        app.tracker.stop()


def render(app):
    app.screen.fill((0, 0, 0))
    app.draw_dock()
    {"colour": app.draw_colour, "motion": app.draw_motion,
     "drive": app.draw_drive}[app.tab]()
    return app.screen


def test_the_bench_opens_and_draws_both_tabs(bench):
    render(bench)
    bench.tab = "motion"
    bench._build()
    render(bench)


def test_clicking_the_tab_buttons_switches_and_redraws(bench):
    """Clicking, not assigning. The two are not the same thing.

    The tabs do not share a layout: each branch of `_build` creates the
    attributes its own tab's draw reads. Every test here set `bench.tab` and
    then called `_build()` by hand, so none of them exercised the path a person
    actually takes — and on that path the button flipped the tab without
    rebuilding, so the first draw died on a missing attribute.
    """
    def controls():
        return {b.label for b in bench.buttons} | {s.label for s in bench.sliders}

    for target in ("motion", "drive", "colour", "motion", "drive"):
        bench.click(bench.tab_rects[target].rect.center)
        assert bench.tab == target
        render(bench)
        # Not merely "it did not raise": the controls on screen have to be the
        # ones this tab owns. A defensive default can stop a stale layout from
        # crashing, and should — but it must not be able to hide the fact that
        # the layout was never rebuilt.
        have = controls()
        if target == "motion":
            assert {"run full", "run quick", "STOP"} <= have, have
            assert "hue" not in have, "the colour sliders are still on screen"
            assert bench.palette_rects == {}, "the palette outlived its tab"
        elif target == "drive":
            assert {"point", "line", "circle"} <= have, have
            assert "run full" not in have, "the run buttons outlived their tab"
            assert bench.palette_rects == {}
        else:
            assert {"hue", "tol", "s_min"} <= have, have
            assert "run full" not in have, "the run buttons are still on screen"
            assert bench.palette_rects, "the palette did not come back"


def test_every_button_survives_being_clicked_from_either_tab(bench):
    """A blunt sweep: no control may raise, whatever tab it was built in.

    Cheap, and it catches the whole family the tab bug belongs to — a callback
    that changes something the layout depends on and does not rebuild.
    """
    bench.connect("ONE", "sim")
    for tab in ("colour", "motion", "drive"):
        bench.click(bench.tab_rects[tab].rect.center)
        for b in list(bench.buttons):
            if b.label == "scan":
                continue                  # spawns a BLE thread; covered elsewhere
            bench.click(b.rect.center)
            render(bench)


def test_every_clickable_region_survives_a_click(bench):
    bench.connect("ONE", "sim")
    for tab in ("colour", "motion", "drive"):
        bench.click(bench.tab_rects[tab].rect.center)
        rects = [r.center for r in bench.row_rects.values()]
        rects += [r.center for r in bench.palette_rects.values()]
        if getattr(bench, "cam_rect", None):
            rects.append(bench.cam_rect.center)
        for point in rects:
            bench.click(point)
            render(bench)


def test_connecting_a_sim_robot_joins_the_fleet(bench):
    assert bench.fleet.handles == {}
    bench.connect("ONE", "sim")
    assert "ONE" in bench.fleet.handles
    assert bench.fleet.handles["ONE"].kind == "sim"
    bench.connect("ONE", "sim")             # the same button releases it
    assert "ONE" not in bench.fleet.handles


def test_connecting_real_without_a_ble_name_is_refused_not_attempted(bench):
    """Building a SpheroRobot starts a BLE thread. Refusing must come first."""
    bench.connect("ONE", "real")
    assert bench.fleet.handles == {}
    assert any("ble_name" in t for _, t in bench.log)


def test_a_run_will_not_start_without_a_robot(bench):
    bench.start_run()
    assert bench.run is None
    assert any("connect a robot" in t for _, t in bench.log)


def test_the_battery_runs_to_completion_through_the_app(bench):
    bench.connect("ONE", "sim")
    bench.handle.pos[:] = [100.0, 100.0]
    bench.tab = "motion"
    bench._build()
    bench.start_run(quick=True)
    assert bench.run is not None
    for _ in range(30 * 400):
        bench.step(1 / 30.0)
        if bench.run is None:
            break
    assert bench.last_fit is not None, "the run never finished"
    # Assert on what was MEASURED. Under a speed cap the top of the curve is an
    # extrapolation and is deliberately not published, so requiring it here
    # would be asserting that the bench lies confidently.
    sm = bench.last_fit["speed_map"]
    assert sm["r2"] > 0.95, sm
    assert sm["cm_s_per_byte"] > 0.05, sm
    assert bench.last_fit["recommend"]["min_moving_byte"] is not None
    render(bench)                            # the results page must draw


def test_stopping_a_run_halts_the_robot(bench):
    bench.connect("ONE", "sim")
    bench.handle.pos[:] = [100.0, 100.0]
    bench.start_run(quick=True)
    for _ in range(200):
        bench.step(1 / 30.0)
    bench.stop_run()
    assert bench.run is None
    assert np.allclose(bench.handle._desired, 0.0)


def test_a_measured_heading_offset_is_written_back_to_the_roster(bench):
    bench.connect("ONE", "sim")
    bench.handle.pos[:] = [100.0, 100.0]
    bench.start_run(quick=True)
    for _ in range(30 * 400):
        bench.step(1 / 30.0)
        if bench.run is None:
            break
    assert bench.roster.by_code("ONE").heading_offset == \
        pytest.approx(bench.fleet.handles["ONE"].heading_offset)


# -- assigning hues to robots ------------------------------------------------

def test_assigning_a_taken_hue_swaps_rather_than_refusing(bench):
    """Six robots hold six hues, so every reassignment is a clash."""
    bench.selected = "ONE"                       # red
    bench.set_color("cyan")                      # TWO holds cyan
    assert bench.roster.by_code("ONE").color == "cyan"
    assert bench.roster.by_code("TWO").color == "red", "the swap must complete"
    assert not bench.roster.errors


def test_reassigning_follows_through_to_the_live_robot(bench):
    """The tracker looks a robot up BY colour, so the handle has to move too."""
    bench.connect("ONE", "sim")
    assert bench.fleet.handles["ONE"].color == "red"
    bench.selected = "ONE"
    bench.set_color("green")
    assert bench.fleet.handles["ONE"].color == "green"
    # Derived from the hue the detector is looking for, not from a second
    # table beside it: that is how a robot ends up lit one colour and hunted
    # for as another, with each table internally consistent and nothing amiss.
    assert bench.fleet.handles["ONE"].rgb == bench.led_for("green"), \
        "the LED must advertise the hue the tracker is looking for"
    assert bench.led_for("green") == vconfig.led_rgb(bench.sig("green")["hue"])


def test_reassignment_is_written_to_the_roster_file(bench, tmp_path):
    import json
    bench.selected = "ONE"
    bench.set_color("magenta")
    on_disk = {e["code"]: e["color"] for e in json.loads(bench.roster.path.read_text())}
    assert on_disk["ONE"] == "magenta"


def test_the_swatch_cycles_and_the_row_selects(bench):
    row = bench.row_rects["TWO"]
    before = bench.roster.by_code("TWO").color
    bench.click((row.x + 12, row.centery))        # the swatch
    assert bench.roster.by_code("TWO").color != before
    bench.selected = "ONE"
    bench.click((row.x + 200, row.centery))       # the rest of the row
    assert bench.selected == "TWO"


def test_an_unknown_colour_is_refused_not_written(bench):
    before = bench.roster.by_code("ONE").color
    assert bench.roster.set_color("ONE", "chartreuse")
    assert bench.roster.by_code("ONE").color == before


def test_the_palette_reports_which_robot_holds_each_hue(bench):
    bench.tab = "colour"
    bench._build()
    assert set(bench.palette_rects) == set(vconfig.COLORS)
    render(bench)                                  # must draw with holders marked


def test_separation_report_flags_hues_the_camera_cannot_tell_apart(bench):
    """Two signatures whose windows overlap will swap identities on contact."""
    d = bench.detector
    d.colors["red"].update(hue=40, tol=14)
    d.colors["cyan"].update(hue=48, tol=14)
    bench.log.clear()
    bench.separation_report()
    assert any(k == "error" and "overlap" in t for k, t in bench.log), bench.log


def test_separation_report_is_quiet_when_the_hues_are_far_apart(bench):
    for i, name in enumerate(vconfig.COLORS):
        bench.detector.colors[name].update(hue=i * 30, tol=6)
    bench.log.clear()
    bench.separation_report()
    assert not any(k == "error" for k, t in bench.log), bench.log
    assert any(k == "ok" for k, t in bench.log)


def test_no_two_clickable_regions_overlap(bench):
    """The bug from HANDOFF 9f, in the bench this time."""
    for tab in ("colour", "motion"):
        bench.tab = tab
        bench._build()
        rects = [b.rect for b in bench.buttons]
        rects += [s.rect for s in bench.sliders]
        rects += list(bench.palette_rects.values())
        if bench.wheel_rect is not None:
            # Inflated by the room the robot codes take outside the ring. A
            # bare rect passes while the labels sit on top of the swatches.
            rects.append(bench.wheel_rect.inflate(64, 28))
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert not a.colliderect(b), f"{tab}: {a} overlaps {b}"


def test_the_bench_never_writes_to_the_live_calibration(bench, tmp_path):
    """HANDOFF section 12, for calib/ as well as the json files at the root."""
    live = pathlib.Path(__file__).resolve().parent.parent / "calib"
    before = {p.name: p.read_bytes() for p in live.glob("*.json")}
    bench.connect("ONE", "sim")
    bench.handle.pos[:] = [100.0, 100.0]
    bench.save_signatures()
    bench.start_run(quick=True)
    for _ in range(30 * 400):
        bench.step(1 / 30.0)
        if bench.run is None:
            break
    after = {p.name: p.read_bytes() for p in live.glob("*.json")}
    assert after == before, "the bench wrote into the real calibration"
    assert (tmp_path / "colors.json").exists(), "…but it did write somewhere"


def test_autotune_needs_a_robot_because_it_drives_the_led(bench):
    bench.start_autotune(all_colors=True)
    assert bench.autotune is None
    assert any("connect a robot" in t for _, t in bench.log)


def test_autotune_walks_every_colour_and_ends_lit(bench):
    """The six-colour walk, driven directly with fabricated frames.

    Not through the camera thread: auto-tune needs a dozen DISTINCT frames per
    colour, and a tight test loop runs thousands of iterations in the time a
    30fps camera produces one. Feeding it frames here tests the state machine;
    the test below tests the wiring.
    """
    import calib
    bench.connect("ONE", "sim")
    h = bench.handle
    seen = {}

    tune = calib.AutoTune(h, list(vconfig.COLORS), lambda r: seen.update(r))
    lit = {n: tuple(int(c) for c in calib.LED_RGB[n][::-1]) for n in vconfig.COLORS}
    i = 0
    while not tune.done and i < 4000:
        i += 1
        dark = h.rgb == (0, 0, 0)
        bgr = None if dark else lit.get(tune.color)
        tune.step(frames(bgr, n=1, seed=i)[0], 1 / 30.0)

    assert tune.done, "the walk never finished"
    assert set(seen) == set(vconfig.COLORS), set(seen)
    assert all(r["ok"] for r in seen.values()), \
        {n: r.get("error") for n, r in seen.items() if not r["ok"]}
    assert h.rgb != (0, 0, 0), "the ball must not be left dark"


def test_autotune_through_the_bench_learns_one_colour(bench):
    """The wiring: a real camera, a real Detector, one colour, paced in real time."""
    import time
    bench.connect("ONE", "sim")
    bench.start_autotune(all_colors=False)
    assert bench.autotune is not None
    deadline = time.time() + 20.0
    while bench.autotune is not None and time.time() < deadline:
        bench.step(1 / 30.0)
        time.sleep(1 / 60.0)              # let the camera thread produce frames
    assert bench.autotune is None, "auto-tune never finished"
    assert bench.handle.rgb != (0, 0, 0)


# -- driving to a clicked point, and along paths -----------------------------

def _drive_tab(bench, code="ONE", pos=(60.0, 60.0), plain=True):
    """Connect a robot on the DRIVE tab, with a repeatable plant by default.

    `Fleet` builds sim robots randomised — gain 0.8 to 1.2, a heading bias, a
    drifting one at that, and slip — so the same test scores differently every
    run and a tracking threshold either passes by luck or fails by it. Anything
    asserting a number pins the plant; the test that cares about variation asks
    for it explicitly.
    """
    bench.connect(code, "sim")
    h = bench.handle
    if plain:
        h.tau, h.gain, h.bias = 0.35, 1.0, 0.0
        h.drift_rate, h.slip, h._drift = 0.0, 0.0, 0.0
    h.pos[:] = list(pos)
    bench.click(bench.tab_rects["drive"].rect.center)
    return h


def _settle(bench, seconds=25.0):
    for _ in range(int(seconds * 30)):
        bench.step(1 / 30.0)


def test_clicking_the_board_sends_the_robot_there(bench):
    h = _drive_tab(bench)
    target = np.array([150.0, 40.0])
    bench.click(bench.to_px(target))
    assert bench.path is not None and bench.path.kind == "point"
    _settle(bench)
    assert np.linalg.norm(h.pos - target) < 6.0, h.pos


def test_a_click_maps_to_the_arena_coordinate_it_looks_like(bench):
    _drive_tab(bench)
    for cm in ((10.0, 10.0), (100.0, 100.0), (190.0, 30.0)):
        back = bench.to_cm(bench.to_px(np.array(cm)))
        assert np.allclose(back, cm, atol=1.5), (cm, back)


def test_an_arrived_robot_is_commanded_to_stop(bench):
    """Not merely "close enough": it has to stop asking for velocity."""
    h = _drive_tab(bench)
    bench.click(bench.to_px(np.array([120.0, 120.0])))
    _settle(bench)
    assert bench.pd.arrived
    assert np.allclose(h._desired, 0.0), h._desired


def test_a_circle_is_one_click_and_the_robot_follows_it(bench):
    h = _drive_tab(bench, pos=(100.0, 100.0))
    bench.set_shape("circle")
    bench.radius = 40
    bench.click(bench.to_px(np.array([100.0, 100.0])))
    assert bench.path.kind == "circle"
    _settle(bench, 20.0)
    e = calib.tracking_error(bench.path, bench.trail[len(bench.trail) // 2:])
    assert e["rms_cm"] < 6.0, e


def test_it_still_follows_a_circle_on_a_randomised_robot(bench):
    """The variation the previous test pins away: gain, bias, drift and slip."""
    h = _drive_tab(bench, pos=(100.0, 100.0), plain=False)
    bench.set_shape("circle")
    bench.radius = 40
    bench.click(bench.to_px(np.array([100.0, 100.0])))
    _settle(bench, 22.0)
    e = calib.tracking_error(bench.path, bench.trail[len(bench.trail) // 2:])
    assert e["rms_cm"] < 12.0, e


def test_a_line_needs_two_clicks_and_shuttles_between_them(bench):
    """Long enough to actually shuttle.

    The robot is capped at the same speed the setpoint moves at, give or take,
    so it starts behind and takes a while to close — which is the point of a
    cap and not a fault. A window sized for the old speeds asserted it had
    failed to travel when it was merely still travelling.
    """
    _drive_tab(bench, pos=(70.0, 100.0))
    bench.set_shape("line")
    bench.path_speed = 10
    bench.click(bench.to_px(np.array([60.0, 100.0])))
    assert bench.path is None, "one click is not a line yet"
    bench.click(bench.to_px(np.array([140.0, 100.0])))
    assert bench.path is not None and bench.path.kind == "line"
    # Sampled here, not read out of `bench.trail`: that buffer holds the last
    # eight seconds for drawing, so reading it as a history of the run reports
    # whatever the robot happened to be doing at the end.
    xs = []
    for _ in range(int(45.0 * 30)):
        bench.step(1 / 30.0)
        xs.append(bench.handle.pos[0])
    xs = np.array(xs)
    assert xs.max() > 120 and xs.min() < 80, f"only covered {xs.min():.0f}-{xs.max():.0f}"


def test_two_clicks_in_the_same_spot_are_refused_as_a_line(bench):
    _drive_tab(bench)
    bench.set_shape("line")
    bench.click(bench.to_px(np.array([100.0, 100.0])))
    bench.click(bench.to_px(np.array([104.0, 100.0])))
    assert bench.path is None
    assert any("too close" in t for _, t in bench.log)


def test_stopping_a_path_halts_the_robot_and_reports(bench):
    h = _drive_tab(bench, pos=(100.0, 100.0))
    bench.set_shape("circle")
    bench.click(bench.to_px(np.array([100.0, 100.0])))
    _settle(bench, 8.0)
    bench.stop_path()
    assert bench.path is None and bench.pd is None
    assert np.allclose(h._desired, 0.0)
    assert any("tracking error" in t for _, t in bench.log)


def test_driving_needs_a_robot(bench):
    bench.click(bench.tab_rects["drive"].rect.center)
    bench.click(bench.to_px(np.array([100.0, 100.0])))
    assert bench.path is None
    assert any("connect a robot" in t for _, t in bench.log)


def test_a_characterisation_run_and_a_drive_do_not_fight(bench):
    """Both write velocities to the same handle every frame."""
    _drive_tab(bench, pos=(100.0, 100.0))
    bench.start_run(quick=True)
    bench.click(bench.to_px(np.array([150.0, 150.0])))
    assert bench.path is None, "a drive must not start mid-characterisation"
    assert any("characterisation run" in t for _, t in bench.log)
    bench.stop_run()


def test_an_unmeasured_robot_drives_and_says_so(bench):
    _drive_tab(bench)
    bench.drive_mode = "pd"          # this test is about the PD gains specifically
    bench.click(bench.to_px(np.array([140.0, 140.0])))
    assert bench.pd.predict == 0.0, "no measurement means no delay compensation"
    assert any("UNMEASURED" in t for _, t in bench.log)


def test_measured_gains_are_picked_up_after_a_run(bench, monkeypatch):
    """The whole point of the bench: measuring a robot makes it drive better."""
    import fleet.characterize as characterize
    _drive_tab(bench, pos=(100.0, 100.0))
    bench.drive_mode = "pd"          # this test is about the PD gains specifically
    monkeypatch.setattr(characterize, "load_motion", lambda code=None, path=None: {
        "step_response": {"tau_s": 0.3}, "latency": {"loop_delay_s": 0.15},
        "recommend": {"min_moving_cm_s": 7.0, "handle_max_speed_cm_s": 52.0}})
    monkeypatch.setattr(calib, "load_motion", characterize.load_motion)
    g = bench.gains()
    assert g["measured"] and g["predict_s"] > 0
    bench.click(bench.to_px(np.array([150.0, 60.0])))
    assert bench.pd.predict > 0
    assert bench.pd.max_speed == 52.0


# -- the hue wheel -----------------------------------------------------------

def _colour_tab(bench, frame=False):
    bench.click(bench.tab_rects["colour"].rect.center)
    if frame:
        # Corner picking converts panel pixels to SOURCE pixels, so it needs a
        # frame to know the scale. The camera thread produces those in real
        # time, which a tight test loop does not spend.
        import time
        deadline = time.time() + 5.0
        while time.time() < deadline:
            bench.step(1 / 30.0)
            if bench.camera_surface()[0] is not None:
                break
            time.sleep(1 / 60.0)
    return bench


def test_clicking_the_wheel_sets_the_selected_slots_hue(bench):
    _colour_tab(bench)
    bench.selected = "ONE"
    r = bench.wheel_rect
    for hue in (0, 45, 90, 135):
        a = np.radians(hue * 2.0)
        radius = r.w / 2.0 - 12
        point = (int(r.centerx + radius * np.cos(a)),
                 int(r.centery + radius * np.sin(a)))
        bench.click(point)
        assert abs(bench.sig("red")["hue"] - hue) <= 2, (hue, bench.sig("red"))


def test_a_click_in_the_middle_of_the_wheel_does_nothing(bench):
    """The ring is the control. The hole in it is not a hue."""
    _colour_tab(bench)
    before = bench.sig("red")["hue"]
    bench.click(bench.wheel_rect.center)
    assert bench.sig("red")["hue"] == before


def test_retuning_a_hue_relights_the_robot(bench):
    """The ball has to glow what the tracker is hunting for."""
    _colour_tab(bench)
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.set_hue(100)
    assert bench.fleet.handles["ONE"].rgb == vconfig.led_rgb(100)


def test_optimising_improves_the_palette_for_the_room(bench):
    _colour_tab(bench)
    import vision.palette as vp
    # a room with a lot of red in it
    bench._bg = vp.background_profile(
        np.dstack([np.full((80, 80), 40, np.uint8), np.full((80, 80), 40, np.uint8),
                   np.full((80, 80), 220, np.uint8)]))
    bench._bg_at = 1e18                     # freeze it; no camera interference
    for e in bench.roster.enabled_entries():
        bench.detector.colors[e.color]["hue"] = 2      # everything on the red
    _, before = bench.palette_report()
    bench.optimise_palette()
    _, after = bench.palette_report()
    assert after["worst_separation"] > before["worst_separation"], (before, after)


def test_optimising_leaves_an_already_good_palette_alone(bench):
    _colour_tab(bench)
    import vision.palette as vp
    bench._bg, bench._bg_at = np.zeros(vp.HUES), 1e18
    for e, hue in zip(bench.roster.enabled_entries(), (0, 90)):
        bench.detector.colors[e.color]["hue"] = hue
    before = {e.color: bench.sig(e.color)["hue"]
              for e in bench.roster.enabled_entries()}
    bench.optimise_palette()
    after = {e.color: bench.sig(e.color)["hue"]
             for e in bench.roster.enabled_entries()}
    assert after == before
    assert any("already as good" in t for _, t in bench.log)


def test_optimising_keeps_the_robots_in_roughly_their_old_order(bench):
    """A trainer who knows which one is the reddish one should still be right."""
    _colour_tab(bench)
    import vision.palette as vp
    bench._bg, bench._bg_at = np.zeros(vp.HUES), 1e18
    bench.detector.colors["red"]["hue"] = 5
    bench.detector.colors["cyan"]["hue"] = 120
    bench.optimise_palette()
    assert bench.sig("red")["hue"] < bench.sig("cyan")["hue"]


def test_the_wheel_only_exists_on_the_colour_tab(bench):
    for tab in ("motion", "drive"):
        bench.click(bench.tab_rects[tab].rect.center)
        assert bench.wheel_rect is None
        bench.click((600, 400))              # must not be read as a hue click
    bench.click(bench.tab_rects["colour"].rect.center)
    assert bench.wheel_rect is not None


def test_the_wheel_draws_with_no_camera_and_no_robots(bench):
    _colour_tab(bench)
    bench._bg, bench._bg_at = None, 0.0
    bench.tracker = None
    render(bench)


# -- picking the arena out of the camera -------------------------------------

def test_four_clicks_set_the_homography_and_the_workspace_together(bench, tmp_path):
    """The pairing that must not drift apart — HANDOFF section 9k."""
    import vision.homography as vh
    from workspace.space import Workspace

    saved = {}
    monkey = vh.Homography.save
    vh.Homography.save = lambda self: saved.update(w=self.width, h=self.height)
    bench.ws.path = tmp_path / "workspace.json"
    try:
        _colour_tab(bench, frame=True)
        bench.start_corners()
        assert bench.corner_mode
        bench.arena_w, bench.arena_h = 240, 180
        r = bench.cam_rect
        for pt in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
            bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    finally:
        vh.Homography.save = monkey

    assert not bench.corner_mode, [t for _, t in bench.log][-1]
    assert saved.get("w") == 240 and saved.get("h") == 180
    x0, x1, y0, y1 = bench.ws.bbox
    assert (x1 - x0, y1 - y0) == (240.0, 180.0)
    assert Workspace.load(tmp_path / "workspace.json").bbox == (0.0, 240.0, 0.0, 180.0)


def test_a_degenerate_pick_is_refused_rather_than_saved(bench):
    """Three points in a line still produce a matrix. It puts robots anywhere."""
    import vision.homography as vh
    calls = []
    monkey = vh.Homography.save
    vh.Homography.save = lambda self: calls.append(1)
    try:
        _colour_tab(bench, frame=True)
        bench.start_corners()
        r = bench.cam_rect
        for pt in ((0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2)):
            bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    finally:
        vh.Homography.save = monkey
    assert not calls, "a collapsed quadrilateral must not be written"
    assert not bench.corner_mode


def test_corner_picking_does_not_sample_hues_by_accident(bench):
    _colour_tab(bench, frame=True)
    before = bench.sig("red")["hue"]
    bench.start_corners()
    r = bench.cam_rect
    bench.click((r.centerx, r.centery))
    assert bench.sig("red")["hue"] == before, "a corner click sampled a hue"
    assert len(bench.corners) == 1


def test_escape_cancels_corner_picking(bench):
    _colour_tab(bench)
    bench.start_corners()
    bench.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert not bench.corner_mode and bench.corners == []


def test_corner_picking_needs_a_camera(bench):
    bench.tracker = None
    bench.start_corners()
    assert not bench.corner_mode
    assert any("no camera" in t for _, t in bench.log)


def test_the_corner_overlay_draws_part_way_through(bench):
    _colour_tab(bench, frame=True)
    bench.start_corners()
    r = bench.cam_rect
    bench.click((int(r.x + r.w * 0.2), int(r.y + r.h * 0.2)))
    bench.click((int(r.x + r.w * 0.8), int(r.y + r.h * 0.2)))
    render(bench)


def test_the_brightness_slider_reaches_the_led(bench):
    """A control that changes a number and not the ball is a control that lies."""
    _colour_tab(bench)
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.set_slider("led_value", 90)
    assert bench.fleet.handles["ONE"].rgb == vconfig.led_rgb(
        bench.sig("red")["hue"], value=90)
    bench.set_slider("led_value", 255)
    assert bench.fleet.handles["ONE"].rgb == vconfig.led_rgb(bench.sig("red")["hue"])


def test_brightness_is_saved_with_the_signature(bench, tmp_path):
    _colour_tab(bench)
    bench.selected = "ONE"
    bench.set_slider("led_value", 120)
    bench.save_signatures()
    assert vconfig.load_signatures()["red"]["led_value"] == 120


def test_a_dim_led_is_still_the_same_hue(bench):
    """Brightness must not become a second way of changing colour."""
    import cv2
    for value in (60, 140, 255):
        r, g, b = vconfig.led_rgb(90, value=value)
        back = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        assert abs(int(back[0]) - 90) <= 1, (value, back)


def test_backspace_takes_back_the_last_corner(bench):
    """One slip on the fourth click must not cost all four."""
    _colour_tab(bench, frame=True)
    bench.start_corners()
    r = bench.cam_rect
    for pt in ((0.2, 0.2), (0.8, 0.2), (0.5, 0.5)):
        bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    assert len(bench.corners) == 3
    bench.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
    assert len(bench.corners) == 2
    assert bench.corner_mode, "undo is not cancel"
    # and the corrected pick still completes
    for pt in ((0.8, 0.8), (0.2, 0.8)):
        bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    assert not bench.corner_mode


def test_undo_on_an_empty_pick_does_nothing(bench):
    _colour_tab(bench, frame=True)
    bench.start_corners()
    bench.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
    assert bench.corners == [] and bench.corner_mode


def test_backspace_outside_corner_mode_is_not_swallowed(bench):
    _colour_tab(bench)
    bench.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
    assert not bench.corner_mode


def test_set_arena_can_be_run_again_after_it_finished(bench):
    """Re-picking is the same button; the previous points are discarded."""
    _colour_tab(bench, frame=True)
    bench.start_corners()
    r = bench.cam_rect
    for pt in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
        bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    assert not bench.corner_mode
    bench.start_corners()
    assert bench.corner_mode and bench.corners == []


# -- writing calibration files -----------------------------------------------

def test_a_file_left_behind_by_root_can_still_be_replaced(tmp_path, monkeypatch):
    """The crash that happened on a real bench, in a test that reproduces it.

    An old `sudo` run left `calib/homography.json` owned by root. Opening it in
    place raised PermissionError from inside the fourth corner click and took
    the whole app down. Renaming a neighbour over it needs permission on the
    DIRECTORY, which the user has — so the file is replaceable and the bench
    keeps running.
    """
    import os
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    target = tmp_path / "homography.json"
    target.write_text("{}")
    os.chmod(target, 0o444)                  # read-only, as root's file was
    assert not os.access(target, os.W_OK)

    vconfig.save("homography", {"matrix": [[1, 0, 0]], "width": 200})
    assert json.loads(target.read_text())["width"] == 200


def test_a_calibration_write_is_all_or_nothing(tmp_path, monkeypatch):
    """A half-written file parses as nothing and takes the next session with it."""
    import json as _json
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    vconfig.save("colors", {"red": {"hue": 3}})
    good = (tmp_path / "colors.json").read_text()

    real_dumps = _json.dumps

    def explode(*a, **k):
        raise RuntimeError("out of disk")

    monkeypatch.setattr(vconfig.json, "dumps", explode)
    with pytest.raises(RuntimeError):
        vconfig.save("colors", {"red": {"hue": 9}})
    monkeypatch.setattr(vconfig.json, "dumps", real_dumps)

    assert (tmp_path / "colors.json").read_text() == good, "a failed write ate the old one"
    assert not list(tmp_path.glob(".colors.*")), "a temp file was left behind"


def test_calibration_files_stay_readable_by_other_tools(tmp_path, monkeypatch):
    import os
    import stat
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    p = vconfig.save("thresholds", {"s_min": 90})
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o644


def test_the_arena_pick_refuses_up_front_when_it_cannot_save(bench, monkeypatch):
    """Fail on the button, not on the fourth click from the far corner."""
    monkeypatch.setattr(vconfig, "writable", lambda name: "calib/ is read-only")
    _colour_tab(bench, frame=True)
    bench.start_corners()
    assert not bench.corner_mode
    assert any("read-only" in t for _, t in bench.log)


def test_a_save_failure_mid_pick_does_not_take_the_bench_down(bench, monkeypatch):
    import vision.homography as vh
    _colour_tab(bench, frame=True)
    bench.start_corners()
    monkeypatch.setattr(vh.Homography, "save",
                        lambda self: (_ for _ in ()).throw(PermissionError("nope")))
    r = bench.cam_rect
    for pt in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
        bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    assert not bench.corner_mode
    assert any("could not write" in t for _, t in bench.log)
    assert any("rm calib/homography.json" in t for _, t in bench.log)
    render(bench)                            # and it still draws


# -- keeping the ball on the floor -------------------------------------------
#
# Found on real hardware: the battery drove a ball out of the camera's view and
# kept commanding it. Every stage aims its own legs at clear floor, and that is
# not enough — a leg aimed correctly still ends elsewhere when the ball slips,
# when the heading offset is not yet known, or when the tracker drops frames.

def _at(bench, code, pos):
    bench.connect(code, "sim")
    h = bench.handle
    h.pos[:] = list(pos)
    return h


def test_speed_falls_off_smoothly_as_the_wall_approaches(bench):
    """A field, not a fence.

    The first version of this waited until the ball was already at the edge and
    then took over. That meant it had to be rescued by hand often enough to
    matter, and every rescue is a person picking up a robot mid-measurement —
    the exact disturbance the measurement is trying to avoid.
    """
    x0, x1, y0, y1 = bench.ws.bbox
    mid_y = (y0 + y1) / 2
    seen = []
    for d in (100.0, 60.0, 40.0, 25.0, 15.0, 8.0):
        h = bench.fleet.handles.get("ONE") or _at(bench, "ONE", (0, 0))
        h.pos[:] = [x0 + d, mid_y]
        cmd, _ = bench.contain(h, (270.0, 255))       # heading -x, at the wall
        seen.append(0 if cmd is None else cmd[1])
    assert seen == sorted(seen, reverse=True), f"not monotonic: {seen}"
    cap_byte = int(round(bench.cap_cm_s / 60.0 * 255))
    assert seen[0] == cap_byte, "the middle of the arena should only be capped"
    assert seen[-1] < 60, f"still {seen[-1]} within 8cm of a wall"


def test_it_is_never_commanded_into_a_wall_it_cannot_stop_before(bench):
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", (x0 + 4.0, (y0 + y1) / 2))
    cmd, paused = bench.contain(h, (270.0, 255))
    byte = 0 if cmd is None else cmd[1]
    speed = byte / 255.0 * 60.0
    assert speed * bench.stop_s() < 6.0, f"{speed:.0f}cm/s with 4cm to stop in"


def test_a_pinned_robot_can_still_drive_away_from_the_wall(bench):
    """A limiter that only looks at distance traps a ball at the edge."""
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", (x0 + 4.0, (y0 + y1) / 2))
    cmd, _ = bench.contain(h, (90.0, 200))            # heading +x, inward
    cap_byte = int(round(bench.cap_cm_s / 60.0 * 255))
    assert cmd is not None and cmd[1] >= min(200, cap_byte), cmd


def test_a_robot_in_the_middle_is_left_alone(bench):
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", ((x0 + x1) / 2, (y0 + y1) / 2))
    bench.cap_cm_s = 60                       # ask for no cap, so only the
    cmd, paused = bench.contain(h, (12.0, 200))   # geometry can limit it
    assert not paused
    assert cmd == (12.0, 200), "a stage command must pass through untouched"


def test_a_robot_with_no_fix_is_stopped_not_steered(bench, monkeypatch):
    """Driving blind is how a ball ends up under a desk.

    `connected` is a property, so faking it means patching the class — through
    monkeypatch, which puts it back. Assigning it directly leaves every later
    test running against a doctored SimRobot, and the failure surfaces
    somewhere else entirely.
    """
    h = _at(bench, "ONE", (100.0, 100.0))
    monkeypatch.setattr(type(h), "connected", property(lambda self: False))
    cmd, paused = bench.contain(h, (0.0, 255))
    assert paused and cmd is None


def test_being_held_slow_is_announced_once_not_every_frame(bench):
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", (x0 + 12.0, (y0 + y1) / 2))
    for _ in range(20):
        bench.contain(h, (270.0, 255))
    said = [t for k, t in bench.log if "near the edge" in t]
    assert len(said) == 1, said


def test_the_battery_keeps_the_ball_off_the_walls(bench):
    """The complaint, as a test: it kept ending up outside.

    Runs the real battery through the real command path and watches how close
    the ball ever gets. A fixed-duration leg at full speed is 150cm, and there
    is not 150cm in front of a ball in the middle of a 200cm room — which is
    why legs are now fitted to the floor that is actually there.
    """
    from fleet import safety as sf
    h = _at(bench, "ONE", (100.0, 100.0))
    h.heading_tracking = False
    bench.start_run(quick=True)
    worst = 1e9
    for _ in range(30 * 400):
        bench.step(1 / 30.0)
        c = sf.clearance(bench.ws, h.pos)
        if c is not None:
            worst = min(worst, c)
        if bench.run is None:
            break
    assert bench.last_fit is not None, "the run never finished"
    assert worst > 3.0, f"came within {worst:.1f}cm of a wall"


def test_calibration_legs_are_slow_enough_to_stop_in(bench):
    """The complaint that started this: it shot off the screen."""
    from fleet.heading import CAL_LEG_CM, CAL_MARGIN, CAL_SPEED
    from swarm.pd import DEFAULT_TAU
    coast = CAL_SPEED * (DEFAULT_TAU + 0.3)
    assert coast < CAL_MARGIN, (
        f"a leg ends at {CAL_SPEED}cm/s and coasts {coast:.0f}cm, but only "
        f"{CAL_MARGIN}cm of clearance is required beyond it")
    assert CAL_LEG_CM + CAL_MARGIN < 100.0, "legs must fit a 200cm arena from the middle"


def test_an_arena_measured_with_a_tape_keeps_its_decimals(bench, tmp_path):
    """138.8cm should not become 139 just because a slider cannot say 138.8."""
    import vision.homography as vh
    from workspace.space import Workspace

    saved = {}
    monkey = vh.Homography.save
    vh.Homography.save = lambda self: saved.update(w=self.width, h=self.height)
    bench.ws = Workspace(bounds_cm=[[0, 0], [138.8, 0], [138.8, 110.8], [0, 110.8]],
                         path=tmp_path / "workspace.json")
    try:
        _colour_tab(bench, frame=True)
        bench.start_corners()
        assert (bench.arena_w, bench.arena_h) == (139, 111), \
            "the sliders should start from the workspace"
        r = bench.cam_rect
        for pt in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
            bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    finally:
        vh.Homography.save = monkey

    assert saved["w"] == pytest.approx(138.8)
    assert saved["h"] == pytest.approx(110.8)
    assert Workspace.load(tmp_path / "workspace.json").bbox == \
        pytest.approx((0.0, 138.8, 0.0, 110.8))


def test_moving_a_slider_does_change_the_size(bench, tmp_path):
    """The snap must not make the sliders inert."""
    import vision.homography as vh
    from workspace.space import Workspace
    saved = {}
    monkey = vh.Homography.save
    vh.Homography.save = lambda self: saved.update(w=self.width, h=self.height)
    bench.ws = Workspace(bounds_cm=[[0, 0], [138.8, 0], [138.8, 110.8], [0, 110.8]],
                         path=tmp_path / "workspace.json")
    try:
        _colour_tab(bench, frame=True)
        bench.start_corners()
        bench.arena_w, bench.arena_h = 200, 150
        r = bench.cam_rect
        for pt in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
            bench.click((int(r.x + r.w * pt[0]), int(r.y + r.h * pt[1])))
    finally:
        vh.Homography.save = monkey
    assert (saved["w"], saved["h"]) == (200.0, 150.0)


# -- checking where the tracker thinks a robot is ----------------------------

def _check_setup(bench, monkeypatch):
    """A bench whose tracker reports a robot at a known, deliberate error."""
    import vision.homography as vh
    _colour_tab(bench, frame=True)
    bench.connect("ONE", "sim")
    hom = vh.Homography().set_rect([(60, 40), (580, 50), (600, 420), (40, 410)],
                                   200.0, 200.0)
    bench._H = hom
    return hom


def _click_at_cm(bench, hom, cm):
    surf, _, f = bench.camera_surface()
    px = hom.to_px([cm])[0]
    r = bench.cam_rect
    bench.click((int(r.x + px[0] * f), int(r.y + px[1] * f)))


def test_a_position_check_reports_the_discrepancy(bench, monkeypatch):
    hom = _check_setup(bench, monkeypatch)
    bench.handle.pos[:] = [104.0, 100.0]          # tracker says 4cm further on
    bench.start_position_check()
    _click_at_cm(bench, hom, (100.0, 100.0))
    assert len(bench.checks) == 1
    assert float(np.linalg.norm(bench.checks[0]["error"])) == pytest.approx(4.0, abs=1.0)


def test_it_names_an_outward_splay_as_a_wrong_plane(bench, monkeypatch):
    """The signature of calibrating the floor when the balls ride above it.

    Every error points away from the middle, growing with distance. That is not
    noise and it is not a shifted corner — it is the ball's own height, and no
    amount of retouching the corners fixes it. Saying so is the whole value of
    this tool.
    """
    hom = _check_setup(bench, monkeypatch)
    bench.start_position_check()
    centre = np.array([100.0, 100.0])
    for truth in ((60.0, 60.0), (140.0, 60.0), (140.0, 140.0), (60.0, 140.0)):
        v = np.array(truth) - centre
        bench.handle.pos[:] = list(np.array(truth) + v * 0.04)   # 4% outward
        _click_at_cm(bench, hom, truth)
    bench.finish_position_check()
    assert any(k == "error" and "ride above it" in t for k, t in bench.log), \
        [t for _, t in bench.log][-3:]


def test_a_constant_shift_is_folded_into_the_calibration(bench):
    """Reporting it was not enough. A tracker reliably five centimetres out is
    five centimetres out of every arrival radius, every clearance check and
    every leg the battery plans — and telling somebody to re-pick four corners
    by hand removes a number the bench has just measured exactly."""
    _colour_tab(bench, frame=True)
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    h = bench.fleet.handles["ONE"]
    hom = bench.tracker.tracker.H
    before = hom.to_cm([[300.0, 250.0]])[0].copy()

    shift = np.array([4.0, -3.0])
    bench.checks = [{"clicked": np.array(p, dtype=float),
                     "reported": np.array(p, dtype=float) + shift,
                     "error": shift.copy()}
                    for p in ((30.0, 30.0), (100.0, 40.0), (60.0, 90.0),
                              (120.0, 100.0))]
    bench.checking = True
    bench.finish_position_check()

    after = hom.to_cm([[300.0, 250.0]])[0]
    assert np.allclose(after - before, -shift, atol=0.01), (
        f"the shift should be folded in: {before} -> {after}")
    assert any("folded into" in t for _, t in bench.log), [t for _, t in bench.log]

def test_it_says_so_when_there_is_nothing_wrong(bench, monkeypatch):
    hom = _check_setup(bench, monkeypatch)
    bench.start_position_check()
    rng = np.random.default_rng(4)
    for truth in ((60.0, 60.0), (140.0, 60.0), (140.0, 140.0), (60.0, 140.0),
                  (100.0, 70.0)):
        bench.handle.pos[:] = list(np.array(truth) + rng.normal(0, 0.7, 2))
        _click_at_cm(bench, hom, truth)
    bench.finish_position_check()
    assert any("measurement noise" in t for _, t in bench.log), \
        [t for _, t in bench.log][-3:]


def test_a_position_check_needs_a_tracked_robot(bench):
    _colour_tab(bench, frame=True)
    bench.start_position_check()
    assert not bench.checking
    assert any("connect a robot" in t for _, t in bench.log)


def test_escape_finishes_a_position_check(bench, monkeypatch):
    _check_setup(bench, monkeypatch)
    bench.start_position_check()
    bench.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert not bench.checking


def test_the_check_overlay_draws(bench, monkeypatch):
    hom = _check_setup(bench, monkeypatch)
    bench.start_position_check()
    _click_at_cm(bench, hom, (100.0, 100.0))
    render(bench)


# -- setting the pace of a run -----------------------------------------------

def test_the_speed_cap_is_obeyed_everywhere(bench):
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", ((x0 + x1) / 2, (y0 + y1) / 2))     # nothing nearby
    for cap in (18, 12, 8):
        bench.cap_cm_s = cap
        cmd, _ = bench.contain(h, (0.0, 255))
        assert cmd[1] / 255.0 * 60.0 <= cap + 0.3, (cap, cmd)


def test_a_wider_edge_margin_slows_it_sooner(bench):
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", (x0 + 30.0, y0 + 30.0))
    speeds = []
    for margin in (1.5, 2.5, 4.0):
        bench.edge_margin = margin
        cmd, _ = bench.contain(h, (225.0, 255))
        speeds.append(0.0 if cmd is None else cmd[1] / 255.0 * 60.0)
    assert speeds == sorted(speeds, reverse=True), speeds


def test_the_settings_reach_the_run_itself(bench):
    """A slider that only moves the limiter still lets stages PLAN fast legs."""
    _at(bench, "ONE", (100.0, 100.0))
    bench.cap_cm_s, bench.edge_margin = 15, 3.5
    bench.start_run(quick=True)
    try:
        assert bench.run is not None
        caps = {st.max_byte for st in bench.run.stages}
        assert caps == {int(round(15 / 60.0 * 255))}, caps
        assert {st.plan_safety for st in bench.run.stages} == {3.5}
    finally:
        bench.stop_run()


def test_a_slow_cap_keeps_the_ball_further_from_the_walls(bench):
    """The point of the control: it buys margin, at the cost of time."""
    from fleet import safety as sf

    def run(cap, margin):
        h = _at(bench, "ONE", (100.0, 100.0))
        h.heading_tracking = False
        bench.cap_cm_s, bench.edge_margin = cap, margin
        bench.start_run(quick=True)
        worst = 1e9
        for _ in range(30 * 500):
            bench.step(1 / 30.0)
            c = sf.clearance(bench.ws, h.pos)
            if c is not None:
                worst = min(worst, c)
            if bench.run is None:
                break
        bench.connect("ONE", "sim")            # release for the next pass
        return worst

    cautious = run(12, 4.0)
    assert cautious > 8.0, f"even at a crawl it came within {cautious:.1f}cm"


def test_driving_away_from_a_wall_is_never_capped_by_distance(bench):
    """Otherwise a pinned ball can never escape and you pick it up again."""
    x0, x1, y0, y1 = bench.ws.bbox
    h = _at(bench, "ONE", (x0 + 4.0, (y0 + y1) / 2))
    bench.edge_margin = 4.0
    cmd, _ = bench.contain(h, (90.0, 255))     # +x, inward
    cap_byte = int(round(bench.cap_cm_s / 60.0 * 255))
    assert cmd is not None and cmd[1] >= cap_byte, cmd


def test_the_motion_tab_draws_below_its_own_controls(bench):
    """A layout cursor that gets shadowed stacks the whole tab in the corner.

    It happened: a readout loop written `for d, y in (...)` rebound the `y`
    holding the layout position, so everything after it drew at y=1 — on top of
    the tab buttons, the run buttons and each other. Nothing raised; it just
    looked broken, and a test that only checks "it drew without throwing" is
    blind to it.
    """
    bench.click(bench.tab_rects["motion"].rect.center)
    # The working pane only — the dock's own controls live in another column
    # and run the whole height of the window.
    pane = [r for r in ([b.rect for b in bench.buttons]
                        + [s.rect for s in bench.sliders])
            if r.x >= calib.DOCK]
    lowest = max((r.bottom for r in pane), default=0)
    assert bench.motion_top >= lowest, (
        f"content starts at {bench.motion_top} but controls reach {lowest}")


def test_the_pace_sliders_are_actually_on_screen(bench):
    """Adding a slider to the layout and not drawing it is an invisible control."""
    bench.click(bench.tab_rects["motion"].rect.center)
    labels = {s.label for s in bench.sliders}
    assert {"top speed", "edge margin"} <= labels, labels

    drawn = []
    real_draw = calib.Slider.draw
    calib.Slider.draw = lambda self, surf, font: drawn.append(self.label)
    try:
        render(bench)
    finally:
        calib.Slider.draw = real_draw
    assert {"top speed", "edge margin"} <= set(drawn), drawn


def test_every_tab_draws_its_own_sliders(bench):
    """The same omission is one line away on any tab that grows a control."""
    for tab in ("colour", "motion", "drive"):
        bench.click(bench.tab_rects[tab].rect.center)
        if not bench.sliders:
            continue
        drawn = []
        real_draw = calib.Slider.draw
        calib.Slider.draw = lambda self, surf, font: drawn.append(self.label)
        try:
            render(bench)
        finally:
            calib.Slider.draw = real_draw
        assert set(drawn) == {s.label for s in bench.sliders}, (tab, drawn)


# -- what happens when the camera loses the ball mid-move --------------------
#
# A Sphero holds its last speed command until it is given another one. A loop
# that stops UPDATING a robot has therefore not stopped it — it has left it
# driving on whatever it was last told, for as long as the fix stays lost.
# That is how a point-to-point move becomes a robot circling the room, and it
# is invisible in any test that only checks the code path returns.

def _lose_fix(bench, monkeypatch, h):
    monkeypatch.setattr(type(h), "connected", property(lambda self: False))


def test_a_drive_stops_the_robot_when_the_fix_drops(bench, monkeypatch):
    h = _drive_tab(bench, pos=(60.0, 60.0))
    bench.click(bench.to_px(np.array([140.0, 140.0])))
    for _ in range(30):
        bench.step(1 / 30.0)
    assert float(np.linalg.norm(h._desired)) > 1.0, "it should be driving"

    _lose_fix(bench, monkeypatch, h)
    bench.step(1 / 30.0)
    assert np.allclose(h._desired, 0.0), (
        f"still commanding {h._desired} with no camera fix")


def test_a_drive_is_abandoned_if_the_fix_does_not_come_back(bench, monkeypatch):
    h = _drive_tab(bench, pos=(60.0, 60.0))
    bench.click(bench.to_px(np.array([140.0, 140.0])))
    bench.step(1 / 30.0)
    _lose_fix(bench, monkeypatch, h)
    for _ in range(int(bench.LOST_GIVE_UP_S * 30) + 20):
        bench.step(1 / 30.0)
    assert bench.path is None, "it should give up rather than wait forever"
    assert any("no camera fix" in t for _, t in bench.log)


def test_a_brief_dropout_does_not_abandon_the_drive(bench, monkeypatch):
    """The tracker drops a frame now and then; that is not a failure."""
    h = _drive_tab(bench, pos=(60.0, 60.0))
    bench.click(bench.to_px(np.array([140.0, 140.0])))
    bench.step(1 / 30.0)
    monkeypatch.setattr(type(h), "connected", property(lambda self: False))
    for _ in range(10):
        bench.step(1 / 30.0)
    monkeypatch.undo()
    for _ in range(10):
        bench.step(1 / 30.0)
    assert bench.path is not None, "a 10-frame dropout should not end the drive"
    assert float(np.linalg.norm(h._desired)) > 0.0, "and it should resume"


def test_a_run_also_stops_the_robot_when_the_fix_drops(bench, monkeypatch):
    h = _at(bench, "ONE", (100.0, 100.0))
    h.heading_tracking = False
    bench.start_run(quick=True)
    for _ in range(60):
        bench.step(1 / 30.0)
    _lose_fix(bench, monkeypatch, h)
    bench.step(1 / 30.0)
    assert np.allclose(h._desired, 0.0)
    bench.stop_run()


def test_no_control_path_leaves_a_robot_driving_blind(bench, monkeypatch):
    """A blunt sweep over every loop that commands a robot."""
    h = _drive_tab(bench, pos=(60.0, 60.0))
    _lose_fix(bench, monkeypatch, h)
    for setup in (lambda: bench.click(bench.to_px(np.array([140.0, 140.0]))),
                  lambda: bench.start_run(quick=True),
                  lambda: bench.start_drift(minutes=1.0)):
        h.set_velocity(np.array([30.0, 30.0]))     # pretend it is already moving
        setup()
        for _ in range(5):
            bench.step(1 / 30.0)
        assert np.allclose(h._desired, 0.0), f"{setup} left it driving"


def test_autotune_lights_the_hue_that_is_actually_hunted(bench):
    """Bug (h) again, in the one path that still had it.

    Auto-tune used a fixed nominal table while the bench lit from the hue being
    hunted, so a slot re-picked by `optimise hues` was tuned against the colour
    it USED to be — and the learned signature quietly put it back. The ball
    then glows one colour while being hunted as another.
    """
    lit = []

    class Handle:
        code, color, rgb = "ONE", "red", (0, 0, 0)

        def set_led(self, rgb, blink=None):
            lit.append(tuple(rgb))

    bench.detector.colors["red"] = dict(bench.detector.colors["red"], hue=59)
    tuner = calib.AutoTune(Handle(), ["red"], lambda r: None,
                           light=bench.led_for)
    on = [c for c in [tuner.light("red")]][0]

    assert on != calib.LED_RGB["red"], (
        "lighting the nominal colour is what undid `optimise hues`")
    assert on == bench.led_for("red")


def test_autotune_puts_the_robots_back_on_their_real_colours(bench):
    """It leaves the ball on whichever colour it finished with, and a ball
    glowing something nobody hunts looks exactly like one the camera lost."""
    _colour_tab(bench)
    bench.connect("ONE", "sim")
    h = bench.fleet.handles["ONE"]
    h.set_led((7, 7, 7))

    bench.autotune = object()
    bench.finish_autotune({})

    assert h.rgb == tuple(bench.led_for(h.color)), (
        f"left on {h.rgb}, should be {bench.led_for(h.color)}")


# -- the sensor probe must not freeze the window ------------------------------

def test_the_sensor_probe_does_not_run_on_the_render_thread(bench, monkeypatch):
    """It is 150 blocking radio reads plus sixty drive writes, every one
    waiting ~230ms on an acknowledgement. Run inline that is tens of seconds
    of dead window, which is indistinguishable from a hang and gets the app
    force-quit halfway through."""
    import threading
    import fleet.sensors as sensors

    started = threading.Event()
    release = threading.Event()

    def slow(api, **kw):
        started.set()
        release.wait(5.0)
        return {"reads": {}, "streaming": {}, "verdict": {}}

    monkeypatch.setattr(sensors, "probe", slow)
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.fleet.handles["ONE"]._api = object()

    bench.probe_sensors()
    assert started.wait(2.0), "the probe never started"
    # The window is still being driven while it runs.
    for _ in range(5):
        bench.step(1 / 30)
        render(bench)
    assert bench.probe is not None, "still running, and the loop kept turning"

    release.set()
    # Across real time, not 200 iterations in microseconds: the worker has to
    # actually be scheduled, and a tight loop samples one instant.
    for _ in range(200):
        bench.step(1 / 30)
        if bench.probe is None:
            break
        time.sleep(0.01)
    assert bench.probe is None, "the result was never collected"


def test_a_second_probe_is_refused_while_one_runs(bench, monkeypatch):
    import threading
    import fleet.sensors as sensors

    release = threading.Event()
    monkeypatch.setattr(sensors, "probe",
                        lambda api, **kw: (release.wait(5.0),
                                           {"reads": {}, "streaming": {}})[1])
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.fleet.handles["ONE"]._api = object()

    bench.probe_sensors()
    bench.log.clear()
    bench.probe_sensors()
    assert any("already running" in t for _, t in bench.log)
    release.set()


def test_a_probe_that_raises_is_reported_not_swallowed(bench, monkeypatch):
    import fleet.sensors as sensors

    def boom(api, **kw):
        raise RuntimeError("radio gone")

    monkeypatch.setattr(sensors, "probe", boom)
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.fleet.handles["ONE"]._api = object()

    bench.probe_sensors()
    for _ in range(200):
        bench.step(1 / 30)
        if bench.probe is None:
            break
        time.sleep(0.01)
    assert bench.probe is None
    assert any("radio gone" in t for _, t in bench.log), [t for _, t in bench.log]


# -- driving it yourself ------------------------------------------------------

def test_teaching_takes_manual_control_and_records(bench):
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    h = bench.fleet.handles["ONE"]
    h.pos = np.array([60.0, 55.0])

    bench.toggle_teach()
    assert bench.teaching is True
    assert h.heading_tracking is False, (
        "the frame must be measured as it is, not as a live estimator is "
        "busy correcting it")

    bench.keys_held.add(pygame.K_w)
    for _ in range(40):
        bench.step_teach(1 / 30)
        h.step(1 / 30)
    assert len(bench.teach_track) > 10


def test_teaching_sets_the_area_the_battery_may_use(bench):
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    h = bench.fleet.handles["ONE"]

    bench.toggle_teach()
    # a loop big enough to work in
    for corner in ((30, 30), (140, 30), (140, 120), (30, 120), (30, 30)):
        h.pos = np.array(corner, dtype=float)
        for _ in range(20):
            bench.teach_track.append((h.pos.copy(), np.array([10.0, 0.0])))
    bench.finish_teach()

    assert bench.calib_bounds is not None
    x0, y0, x1, y1 = bench.calib_bounds
    assert x1 - x0 > 40 and y1 - y0 > 40
    ws = bench.calib_workspace()
    assert ws is not bench.ws, "the battery must plan in the driven box"
    a, b, c, d = ws.bbox
    assert (b - a) < 200.0


def test_no_boundary_means_the_whole_arena(bench):
    assert bench.calib_bounds is None
    assert bench.calib_workspace() is bench.ws


def test_teaching_gives_the_robot_back(bench):
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    h = bench.fleet.handles["ONE"]

    bench.toggle_teach()
    bench.finish_teach()
    assert bench.teaching is False
    assert h.heading_tracking is True
    assert bench.keys_held == set()


@pytest.mark.parametrize("tab", ["colour", "motion", "drive"])
@pytest.mark.parametrize("size", [(1150, 760), (1400, 900), (1024, 700)])
def test_no_control_is_drawn_off_the_window(bench, tab, size):
    """A button past the panel edge is drawn, looks live, and cannot be
    clicked. `teach` shipped 36px off the right of a 1150-wide window and the
    only symptom was somebody saying the tab did not work — every test passed,
    because a test that clicks by calling the handler never discovers that no
    click can reach it.

    The sibling of the existing no-two-regions-overlap test: one says controls
    do not cover each other, this says they are on the screen at all.
    """
    bench.apply_size(*size)
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.set_tab(tab)()
    w, h = bench.screen.get_size()

    for b in bench.buttons:
        r = b.rect
        label = getattr(b, "label", "?")
        assert r.x >= 0 and r.y >= 0, f"{tab}: {label} starts off-window at {r}"
        assert r.right <= w, f"{tab}: {label} runs {r.right - w}px past the right edge"
        assert r.bottom <= h, f"{tab}: {label} runs {r.bottom - h}px past the bottom"

    for sl in bench.sliders:
        r = sl.rect
        assert r.right <= w, f"{tab}: slider {getattr(sl, 'label', '?')} off the right"
        assert r.bottom <= h, f"{tab}: slider {getattr(sl, 'label', '?')} off the bottom"


def test_teaching_shows_what_the_keys_are_doing(bench):
    """'I could click it but WASD did nothing' cannot be diagnosed from the
    outside: a key that never arrived, a robot that never moved and a camera
    that never saw it all look identical from a chair. The panel says which."""
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.handle.pos = np.array([60.0, 55.0])

    assert bench.teach_status() is None, "silent when not teaching"

    bench.toggle_teach()
    assert "keys[----]" in bench.teach_status()

    class E:
        key = pygame.K_w

    bench.key(E())
    for _ in range(20):
        bench.step(1 / 30)
    live = bench.teach_status()
    assert "keys[w]" in live
    assert "0.0cm/s" not in live, "a held key must show as a command"
    assert "samples" in live

    up = E()
    bench.key_up(up)
    for _ in range(5):
        bench.step(1 / 30)
    assert "keys[----]" in bench.teach_status(), "releasing must register"


def test_the_yaw_rate_slider_changes_how_fast_a_and_d_turn(bench):
    """A sphere has no visible front, so turning shows as nothing until you
    drive — which is why the default felt like the keys were dead."""
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.handle.pos = np.array([60.0, 55.0])
    bench.toggle_teach()

    class E:
        key = pygame.K_d

    bench.key(E())

    bench.turn_rate = 60.0
    bench.manual_heading = 0.0
    for _ in range(30):
        bench.step_teach(1 / 30)
    slow = bench.manual_heading

    bench.turn_rate = 240.0
    bench.manual_heading = 0.0
    for _ in range(30):
        bench.step_teach(1 / 30)
    fast = bench.manual_heading

    assert fast > slow * 3, f"slow {slow:.0f}deg vs fast {fast:.0f}deg"
    labels = [getattr(s, "label", "") for s in bench.sliders]
    assert "yaw deg/s" in labels


def test_a_speed_cap_under_the_deadband_is_refused_with_the_reason(bench):
    """6cm/s reads as a cautious choice and is not one: below the deadband the
    motors do not turn at all, so the run measures a ball that never moved and
    blames the tracker. Safety is the edge margin's job — that scales speed
    with the floor in front, so a workable cap is still slow near a wall."""
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.cap_cm_s = 6

    bench.log.clear()
    bench.start_run(False)

    assert bench.run is None, "the run must not start"
    said = " ".join(t for _, t in bench.log)
    assert "deadband" in said
    assert "EDGE MARGIN" in said, "it has to say where safety actually comes from"


def test_a_workable_cap_starts_the_run(bench):
    bench.set_tab("motion")()
    bench.connect("ONE", "sim")
    bench.selected = "ONE"
    bench.cap_cm_s = 30
    bench.start_run(False)
    assert bench.run is not None
    bench.stop_run("test over")


# -- track mode --------------------------------------------------------------
#
# The third way to reach a point. `straight` commits to a bearing, which is what
# makes a rotated frame measurable and what makes a disturbance permanent.
# `pd` corrects every frame on a link that carries four commands a second.
# `track` holds the ball to the LINE to its target and re-commands at the rate
# the radio will actually carry.


def _to_point(bench, h, mode, target=(160.0, 60.0), bias_deg=0.0,
              start=(40.0, 60.0), seconds=30.0, nudge=None):
    """One point-to-point drive under one mode. Returns what it did."""
    from swarm.pd import straightness
    bench.stop_path("reset")
    h.pos[:] = list(start)
    h.vel[:] = [0.0, 0.0]
    h.bias = np.radians(bias_deg)
    h.heading_offset = 0.0
    bench.trail = []
    bench.aim_fixes = 0
    bench.log = []
    bench.drive_mode = mode
    bench.path_speed = 20
    bench._build()
    bench.click(bench.to_px(np.array(list(target))))

    line_a, line_b = np.array(start, dtype=float), np.array(target, dtype=float)
    off_line, nudged = [], False
    for i in range(int(seconds * 30)):
        bench.step(1 / 30.0)
        if nudge is not None and not nudged and \
                float(np.linalg.norm(np.asarray(h.pos) - line_a)) > 40.0:
            # Shove it off the route, the way a bump or a slipping wheel would.
            h.pos[:] = np.asarray(h.pos) + np.asarray(nudge, dtype=float)
            nudged = True
            continue
        if nudged:
            ab = line_b - line_a
            u = float(np.clip((np.asarray(h.pos) - line_a) @ ab / (ab @ ab), 0, 1))
            off_line.append(float(np.linalg.norm(np.asarray(h.pos)
                                                 - (line_a + ab * u))))
        if bench.pd is not None and bench.pd.arrived:
            break
    return {
        "straightness": straightness(bench.trail),
        "final_cm": float(np.linalg.norm(np.asarray(h.pos) - line_b)),
        "commands": getattr(bench.pd, "commands", None),
        "offset": float(h.heading_offset),
        "off_line": off_line,
        "wrote_offset": any("aim-frame error. Offset" in t for _, t in bench.log),
        "seconds": (i + 1) / 30.0,
    }


def test_the_drive_mode_button_cycles_three_ways(bench):
    bench.drive_mode = "straight"
    seen = []
    for _ in range(3):
        bench.toggle_drive_mode()
        seen.append(bench.drive_mode)
    assert seen == ["track", "pd", "straight"], "and back to where it started"


def test_track_builds_the_corridor_controller(bench):
    from swarm.trace import TrackToPoint
    h = _drive_tab(bench, pos=(40.0, 60.0))
    bench.drive_mode = "track"
    bench.click(bench.to_px(np.array([160.0, 60.0])))
    assert isinstance(bench.pd, TrackToPoint)
    assert bench.pd.max_speed == bench.gains()["max_speed"]


def test_track_parks_closer_to_the_target_than_the_other_two(bench):
    """On a plant with nothing wrong with it, which is the easy case."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    scores = {m: _to_point(bench, h, m)["final_cm"]
              for m in ("straight", "track", "pd")}
    assert scores["track"] < scores["straight"]
    assert scores["track"] < scores["pd"]
    assert scores["track"] < 3.0


def test_track_arrives_straighter_than_a_pd_loop_on_a_rotated_frame(bench):
    """A curve into the target is the signature of a frame error being chased.
    Correcting toward a LINE bends less than correcting toward a point."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    track = _to_point(bench, h, "track", bias_deg=25.0)
    pd = _to_point(bench, h, "pd", bias_deg=25.0)
    assert track["straightness"] < pd["straightness"]


def test_track_returns_to_the_route_after_a_shove_rather_than_cutting(bench):
    """The one thing a committed leg cannot do. `straight` re-aims at the
    TARGET from wherever it was pushed to, so it finishes along a chord;
    `track` aims back at the line it was supposed to be on."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    shove = (0.0, 25.0)
    track = _to_point(bench, h, "track", nudge=shove)
    straight = _to_point(bench, h, "straight", nudge=shove)
    assert track["off_line"] and straight["off_line"]
    assert float(np.mean(track["off_line"])) < \
           float(np.mean(straight["off_line"])), \
        "correcting toward the route beats correcting toward the destination"


def test_track_does_not_write_a_heading_offset(bench):
    """It corrects the frame error as it drives, so what is left to measure is
    the leftover and not the error — and roster.json is where this would end
    up. Only a committed leg may calibrate."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    assert _to_point(bench, h, "straight", bias_deg=25.0)["wrote_offset"]
    assert not _to_point(bench, h, "track", bias_deg=25.0)["wrote_offset"]


def test_track_does_not_flood_the_radio(bench):
    """Closed loop on this link is only affordable because it is rate limited."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    r = _to_point(bench, h, "track", bias_deg=25.0)
    assert r["commands"] <= r["seconds"] * bench.tune_cmd_hz + 2


def test_the_lookahead_slider_reaches_the_running_controller(bench):
    h = _drive_tab(bench, pos=(40.0, 60.0))
    bench.drive_mode = "track"
    bench._build()
    bench.click(bench.to_px(np.array([160.0, 60.0])))
    for _ in range(30):
        bench.step(1 / 30.0)
    bench.set_tune("lookahead", 42.0)
    assert bench.pd.lookahead == 42.0
    assert bench.pd.follower.lookahead == 42.0


def test_the_track_sliders_are_the_ones_it_actually_has(bench):
    _drive_tab(bench)
    bench.drive_mode = "track"
    bench._build()
    labels = [s.label for s in bench.sliders]
    assert "lookahead cm" in labels and "cmds / s" in labels
    assert "kp x100" not in labels, "those belong to the PD loop"


def test_a_straight_path_under_track_is_not_reported_as_a_good_aim_frame(bench):
    """The bench used to say "the aim frame looks right" after a drive with
    twenty degrees of frame error in it, because track had corrected its way
    through them. A straight path means a straight FRAME only when nothing was
    steering."""
    h = _drive_tab(bench, pos=(40.0, 60.0))
    _to_point(bench, h, "track", bias_deg=20.0)
    said = " ".join(t for _, t in bench.log)
    assert "the aim frame looks right" not in said
    assert "says nothing about the aim frame" in said


def test_the_drive_message_names_the_knobs_that_are_in_force(bench):
    h = _drive_tab(bench, pos=(40.0, 60.0))
    bench.drive_mode = "track"
    bench.click(bench.to_px(np.array([160.0, 60.0])))
    said = " ".join(t for _, t in bench.log)
    assert "lookahead" in said
    assert "kp " not in said, "the corridor controller has no proportional gain"


# -- the COLOUR tab's light clustering ------------------------------------

def _lit_ball(frame, centre, heading_deg, main=(60, 255, 90),
              tail=(255, 90, 40), span=26):
    """Three lights along one axis, drawn into an existing frame."""
    import cv2
    c = np.array(centre, dtype=float)
    r = math.radians(heading_deg)
    d = np.array([math.cos(r), math.sin(r)]) * span / 2.0
    cv2.circle(frame, tuple((c + d).astype(int)), 8, main, -1)
    cv2.circle(frame, tuple(c.astype(int)), 4, main, -1)
    cv2.circle(frame, tuple((c - d).astype(int)), 6, tail, -1)
    return cv2.GaussianBlur(frame, (11, 11), 0)


def test_the_colour_tab_offers_three_views_and_keeps_the_camera_constant(bench):
    """The tagged camera view is what you want in front of you the whole time.
    A view you have to switch back to is one you will forget to."""
    bench.set_tab("colour")()
    labels = {b.label for b in bench.buttons}
    assert {"bright", "mask", "blur"} <= labels
    for view in ("mask", "blur", "bright"):
        bench.set_colour_view(view)()
        assert bench.colour_view == view
        render(bench)                      # each must draw without raising


def test_the_view_selection_survives_a_resize(bench):
    """It is set in the constructor, not on the resize path — putting it there
    reset the selection and both knobs every time the window moved."""
    bench.set_tab("colour")()
    bench.set_colour_view("blur")()
    bench.light_min_v = 173
    bench.apply_size(1400, 900)
    assert bench.colour_view == "blur"
    assert bench.light_min_v == 173


def test_a_lit_ball_is_clustered_named_and_aimed(bench, monkeypatch):
    """Brightness finds it, colour only says which robot it is — the reverse
    of `vision/track.py`, where detection is per-colour by construction."""
    import cv2

    frame = _lit_ball(np.zeros((480, 640, 3), np.uint8), (320, 240), 137.0)
    monkeypatch.setattr(bench.tracker, "latest", lambda: (frame, {}))
    bench.light_min_v = 140
    det = bench.detector
    # One slot near the ball and every other slot far from it. The live
    # palette has yellow at 45 and green at 68, so a ball reading ~50 is
    # contested between them and correctly refuses a tag — which is a real
    # property of that palette and not what this test is about.
    for name, spec in det.colors.items():
        spec["hue"], spec["tol"] = 160, 8
    det.colors[bench.sig_color] = {"hue": 55, "tol": 14, "s_min": 90,
                                   "v_min": 70, "min_area": 10,
                                   "max_area": 20000}
    bench.lights_stamp = None

    got = bench.light_readings()
    aimed = [r for r in got if r["deg"] is not None]
    assert aimed, [r["why"] for r in got]
    r = aimed[0]
    assert abs((r["deg"] - 137 + 180) % 360 - 180) < 6, r["deg"]
    # Two or three: at this spacing and blur the middle light merges into the
    # main one, which is what a real shell does too. The heading comes off the
    # two ENDS of the axis, so it does not depend on the middle surviving.
    assert 2 <= len(r["group"]) <= 3


def test_the_label_is_the_robot_code_not_the_colour_slot(bench, monkeypatch):
    """A person recognises SSMK. Making them translate 'green' into a robot is
    work the bench can do itself."""
    frame = _lit_ball(np.zeros((480, 640, 3), np.uint8), (320, 240), 90.0)
    monkeypatch.setattr(bench.tracker, "latest", lambda: (frame, {}))
    bench.light_min_v = 140
    entry = bench.roster.enabled_entries()[0]
    det = bench.detector
    det.colors[entry.color] = {"hue": 60, "tol": 14, "s_min": 90, "v_min": 70,
                               "min_area": 10, "max_area": 20000}
    # every other slot far away, so the tag is not contested
    for name, spec in det.colors.items():
        if name != entry.color:
            spec["hue"] = 160
    bench.lights_stamp = None

    named = [r for r in bench.light_readings() if r["code"]]
    assert named, [r.get("tag_why") for r in bench.light_readings()]
    assert named[0]["code"] == entry.code


def test_the_reading_is_computed_once_per_frame(bench, monkeypatch):
    """The render loop asks while drawing and the camera is far slower, so a
    recompute per draw would be several full passes a frame for one answer."""
    frame = _lit_ball(np.zeros((480, 640, 3), np.uint8), (320, 240), 45.0)
    monkeypatch.setattr(bench.tracker, "latest", lambda: (frame, {}))
    bench.lights_stamp = None

    calls = []
    import calib as calib_mod
    real = calib_mod.lights_px
    monkeypatch.setattr(calib_mod, "lights_px",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for _ in range(5):
        bench.light_readings()
    assert len(calls) == 1, f"ran {len(calls)} times for one frame"


def test_the_light_knobs_never_reach_the_colour_calibration(bench, monkeypatch):
    """They decide what counts as a light, not what a colour looks like. A
    signature file that grew them would change what the tracker does."""
    import vision.config as vc

    monkeypatch.setattr(vc, "save_signatures",
                        lambda *a, **k: pytest.fail("wrote a signature file"))
    bench.set_tab("colour")()
    bench.light_min_v = 210
    bench.ball_span_px = 44
    render(bench)
    for spec in (bench.detector.colors.values() if bench.detector else ()):
        assert "light_min_v" not in spec and "ball_span_px" not in spec


def test_the_arrow_is_drawn_along_its_own_lights_not_at_an_angle_to_them(bench):
    """A perspective warp does not preserve angles. The brightness pane maps
    the lights through the homography, so drawing the raw IMAGE angle over
    them puts the arrow at an angle to its own dots — which is what a real
    ball looked like on screen. `facing_cm` already states the rule: as two
    POINTS, never as an angle.
    """
    drawn = []
    bench.set_tab("colour")()

    def fake_line(surface, colour, a, b, width=1):
        drawn.append((a, b))

    # A homography that rotates hard, so an unmapped angle cannot accidentally
    # agree with the mapped one.
    from vision.homography import Homography
    H = Homography()
    H.set_rect([(100, 300), (300, 100), (500, 300), (300, 500)], 100.0, 100.0)
    bench._H = H

    reading = {
        "group": [], "why": None, "deg": 0.0, "conf": 1.0,
        "centre": (300.0, 300.0), "span_px": 40.0, "colour": None,
        "code": None, "tag_why": None,
        "front": {"x": 320.0, "y": 300.0, "core_px": 4.0},
        "back": {"x": 280.0, "y": 300.0, "core_px": 4.0},
    }
    bench.light_readings = lambda: [reading]

    m = 2.0
    origin = (0, 0)

    def px_of(p):
        cm = np.asarray(H.to_cm([list(p)])).ravel()[:2]
        return (origin[0] + float(cm[0]) * m, origin[1] + float(cm[1]) * m)

    import pygame as pg
    real = pg.draw.line
    pg.draw.line = fake_line
    try:
        bench.draw_light_overlay(origin, m, px_of=px_of)
    finally:
        pg.draw.line = real

    assert drawn, "no arrow was drawn"
    shaft_a, shaft_b = drawn[0]
    arrow = math.degrees(math.atan2(shaft_b[1] - shaft_a[1],
                                    shaft_b[0] - shaft_a[0])) % 360.0
    pf, pb = px_of((320.0, 300.0)), px_of((280.0, 300.0))
    lights = math.degrees(math.atan2(pf[1] - pb[1], pf[0] - pb[0])) % 360.0
    assert abs((arrow - lights + 180) % 360 - 180) < 1.0, (
        f"arrow at {arrow:.1f} but its lights lie along {lights:.1f}")
    # ...and that this actually differs from the unmapped image angle, so the
    # test would have failed against the old code.
    assert abs((lights - reading["deg"] + 180) % 360 - 180) > 5.0


def test_defocus_steadies_the_colour_without_moving_the_lights(bench, monkeypatch):
    """Blur helps a colour and hurts a geometry, so the two are read from
    different images rather than one compromise between them."""
    frame = _lit_ball(np.zeros((480, 640, 3), np.uint8), (320, 240), 137.0,
                      main=(40, 40, 255), tail=(255, 90, 40), span=26)
    monkeypatch.setattr(bench.tracker, "latest", lambda: (frame, {}))
    bench.light_min_v = 140
    det = bench.detector
    for spec in det.colors.values():
        spec["hue"], spec["tol"] = 100, 8
    det.colors[bench.sig_color] = {"hue": 0, "tol": 14, "s_min": 90,
                                   "v_min": 70, "min_area": 10,
                                   "max_area": 20000}

    seen = []
    for blur in (1, 9, 21):
        bench.tag_blur = blur
        bench.lights_stamp = None
        aimed = [r for r in bench.light_readings() if r["deg"] is not None]
        assert aimed, f"defocus {blur} lost the ball"
        seen.append(aimed[0]["deg"])
    assert max(seen) - min(seen) < 0.5, (
        f"defocus moved the heading: {seen} — positions must come from the "
        "frame as delivered")


def test_the_blur_view_shows_what_the_tagger_actually_samples(bench):
    """A blur view showing a different image than the tagger reads would be a
    picture of something nobody is measuring."""
    import cv2

    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(frame, (320, 240), 12, (40, 40, 255), -1)
    bench.tag_blur = 15
    a = bench.tag_frame(frame)
    b = cv2.GaussianBlur(frame, (15, 15), 0)
    assert np.array_equal(a, b)
    bench.tag_blur = 1
    assert bench.tag_frame(frame) is frame, "1 means off, not a 1px kernel"


def test_defocus_never_reaches_the_colour_calibration(bench, monkeypatch):
    """It is a lens-substitute for reading hue, not part of a signature."""
    import vision.config as vc

    monkeypatch.setattr(vc, "save_signatures",
                        lambda *a, **k: pytest.fail("wrote a signature file"))
    bench.set_tab("colour")()
    bench.tag_blur = 17
    render(bench)
    for spec in (bench.detector.colors.values() if bench.detector else ()):
        assert "tag_blur" not in spec

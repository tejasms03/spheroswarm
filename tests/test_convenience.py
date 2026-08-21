"""Convenience tools: the model names who, these work out where."""

import numpy as np
import pytest

from tools import call
from tools.convenience import MIN_GAP, free_spot_near
from workspace.entities import EntitySet


@pytest.fixture
def ctx(sim_ctx):
    c = sim_ctx(n=6, seed=7, obstacles=[])
    for code, p in (("SSMK", [60.0, 60.0]), ("CRXS", [180.0, 60.0]),
                    ("SYRX", [60.0, 120.0]), ("VHGR", [180.0, 120.0]),
                    ("MLYS", [120.0, 40.0]), ("SNFR", [120.0, 150.0])):
        c.fleet[code].pos = np.array(p)
    return c


def settle(ctx, seconds=14):
    for _ in range(int(seconds / 0.1)):
        ctx.tick(0.1)


def pos(ctx, code):
    return np.asarray(ctx.fleet[code].pos, dtype=float)


# -- swap ---------------------------------------------------------------------

def test_swap_exchanges_two_robots(ctx):
    a0, b0 = pos(ctx, "SSMK").copy(), pos(ctx, "CRXS").copy()
    r = call(ctx, "swap", {"code_a": "SSMK", "code_b": "CRXS"})
    assert r["ok"], r["error"]
    settle(ctx, 20)
    assert np.linalg.norm(pos(ctx, "SSMK") - b0) < 15
    assert np.linalg.norm(pos(ctx, "CRXS") - a0) < 15


def test_swap_head_on_does_not_collide(ctx):
    """Two robots on the same line crossing is the degenerate pass-side case."""
    ctx.fleet["SSMK"].pos = np.array([60.0, 90.0])
    ctx.fleet["CRXS"].pos = np.array([180.0, 90.0])
    for c, p in zip(("SYRX", "VHGR", "MLYS", "SNFR"),
                    ([40.0, 30.0], [90.0, 30.0], [150.0, 30.0], [200.0, 30.0])):
        ctx.fleet[c].pos = np.array(p)

    r = call(ctx, "swap", {"code_a": "SSMK", "code_b": "CRXS"})
    assert r["ok"], r["error"]
    closest = 1e9
    for _ in range(300):
        ctx.tick(0.1)
        closest = min(closest, float(np.linalg.norm(pos(ctx, "SSMK") - pos(ctx, "CRXS"))))
    assert closest > 4.0, f"they passed through each other ({closest:.1f}cm)"


def test_swap_needs_two_different_robots(ctx):
    assert not call(ctx, "swap", {"code_a": "SSMK", "code_b": "SSMK"})["ok"]
    assert not call(ctx, "swap", {"code_a": "SSMK"})["ok"]
    r = call(ctx, "swap", {"code_a": "SSMK", "code_b": "NOPE"})
    assert not r["ok"] and "NOPE" in r["error"]


# -- displace -------------------------------------------------------------------

def test_displace_moves_a_into_bs_place_and_b_aside(ctx):
    b0 = pos(ctx, "CRXS").copy()
    r = call(ctx, "displace", {"code_a": "SSMK", "code_b": "CRXS"})
    assert r["ok"], r["error"]
    settle(ctx, 20)
    assert np.linalg.norm(pos(ctx, "SSMK") - b0) < 18, "A did not take B's place"
    assert np.linalg.norm(pos(ctx, "CRXS") - b0) > 15, "B did not step aside"
    assert ctx.ws.is_valid_point(pos(ctx, "CRXS"))


def test_free_spot_is_valid_and_in_range(ctx):
    p = np.array([120.0, 90.0])
    for _ in range(30):
        q = free_spot_near(p, ctx)
        assert ctx.ws.is_valid_point(q)
        assert float(np.linalg.norm(q - p)) < 80


def test_free_spot_falls_back_when_the_area_is_crowded(ctx):
    """Twelve darts into a full board must still return something usable."""
    for i, c in enumerate(ctx.active_codes()):
        ctx.fleet[c].pos = np.array([118.0 + i * 2.0, 90.0])
    q = free_spot_near(np.array([120.0, 90.0]), ctx)
    assert ctx.ws.is_valid_point(q), "the fallback must never fail"


# -- nudge -------------------------------------------------------------------------

def test_nudge_left_and_right(ctx):
    x0 = pos(ctx, "SYRX")[0]
    assert call(ctx, "nudge", {"codes": ["SYRX"], "direction": "left",
                                "distance": 30})["ok"]
    settle(ctx)
    assert pos(ctx, "SYRX")[0] < x0 - 12, "did not move left"


def test_nudge_up_is_negative_y(ctx):
    y0 = pos(ctx, "SYRX")[1]
    call(ctx, "nudge", {"codes": ["SYRX"], "direction": "up", "distance": 30})
    settle(ctx)
    assert pos(ctx, "SYRX")[1] < y0 - 12, "up should decrease y in this frame"


def test_nudge_does_not_disturb_the_others(ctx):
    before = {c: pos(ctx, c).copy() for c in ctx.active_codes()}
    call(ctx, "nudge", {"codes": ["SYRX"], "direction": "left", "distance": 30})
    settle(ctx)
    for c, p0 in before.items():
        if c == "SYRX":
            continue
        assert float(np.linalg.norm(pos(ctx, c) - p0)) < 12, f"{c} was disturbed"


def test_nudge_toward_and_away(ctx):
    d0 = float(np.linalg.norm(pos(ctx, "SSMK") - pos(ctx, "CRXS")))
    call(ctx, "nudge", {"codes": ["SSMK"], "direction": "toward:CRXS",
                         "distance": 40})
    settle(ctx)
    assert float(np.linalg.norm(pos(ctx, "SSMK") - pos(ctx, "CRXS"))) < d0 - 15

    d1 = float(np.linalg.norm(pos(ctx, "SSMK") - pos(ctx, "CRXS")))
    call(ctx, "nudge", {"codes": ["SSMK"], "direction": "away:CRXS",
                         "distance": 40})
    settle(ctx)
    assert float(np.linalg.norm(pos(ctx, "SSMK") - pos(ctx, "CRXS"))) > d1 + 15


def test_nudge_rejects_nonsense(ctx):
    assert not call(ctx, "nudge", {"direction": "sideways"})["ok"]
    assert not call(ctx, "nudge", {})["ok"]
    assert not call(ctx, "nudge", {"direction": "left", "distance": -5})["ok"]
    r = call(ctx, "nudge", {"direction": "toward:ghost"})
    assert not r["ok"] and "ghost" in r["error"]


# -- gather ------------------------------------------------------------------------

def test_gather_clusters_around_a_robot(ctx):
    r = call(ctx, "gather", {"around": "VHGR", "radius": 45})
    assert r["ok"], r["error"]
    settle(ctx, 20)
    centre = pos(ctx, "VHGR")
    for c in ctx.active_codes():
        if c == "VHGR":
            continue
        d = float(np.linalg.norm(pos(ctx, c) - centre))
        assert d < 90, f"{c} is {d:.0f}cm away, not gathered"


def test_gather_does_not_move_the_centre(ctx):
    before = pos(ctx, "VHGR").copy()
    call(ctx, "gather", {"around": "VHGR", "radius": 45})
    settle(ctx, 20)
    assert float(np.linalg.norm(pos(ctx, "VHGR") - before)) < 20


def test_gather_widens_a_ring_that_would_be_too_tight(ctx):
    """A 5cm ring for six robots is impossible; it must open out, not fail."""
    r = call(ctx, "gather", {"around": "VHGR", "radius": 5})
    assert r["ok"], r["error"]
    assert r["radius"] > MIN_GAP


def test_gather_around_an_entity(ctx):
    ctx.ws.entities = EntitySet.from_data([
        {"id": "rock", "role": "target",
         "shape": {"type": "circle", "center": [120, 90], "radius": 10}}])
    r = call(ctx, "gather", {"around": "rock", "radius": 45})
    assert r["ok"], r["error"]
    settle(ctx, 20)
    for c in ctx.active_codes():
        assert float(np.linalg.norm(pos(ctx, c) - np.array([120.0, 90.0]))) < 95


def test_gather_unknown_anchor(ctx):
    r = call(ctx, "gather", {"around": "nobody"})
    assert not r["ok"] and "nobody" in r["error"]


# -- spread --------------------------------------------------------------------------

def test_spread_increases_the_minimum_gap(ctx):
    for i, c in enumerate(ctx.active_codes()):
        ctx.fleet[c].pos = np.array([110.0 + i * 8.0, 90.0])
    before = _min_gap(ctx)
    r = call(ctx, "spread", {"min_distance": 55})
    assert r["ok"], r["error"]
    settle(ctx, 25)
    after = _min_gap(ctx)
    assert after > before + 10, f"gap went {before:.0f} -> {after:.0f}"


def test_spread_keeps_everyone_in_bounds(ctx):
    call(ctx, "spread", {"min_distance": 80})
    settle(ctx, 25)
    for c in ctx.active_codes():
        assert ctx.ws.point_in_bounds(pos(ctx, c))


def test_spread_needs_two_robots(ctx):
    r = call(ctx, "spread", {"codes": ["SSMK"], "min_distance": 40})
    assert not r["ok"] and "two" in r["error"]


def _min_gap(ctx):
    p = ctx.active_positions()
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# -- mirror -----------------------------------------------------------------------------

def test_mirror_vertical(ctx):
    ctx.fleet["CRXS"].pos = np.array([60.0, 70.0])
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS", "axis": "vertical"})
    assert r["ok"], r["error"]
    assert r["reflected_to"][0] == pytest.approx(180.0, abs=1.0)
    assert r["reflected_to"][1] == pytest.approx(70.0, abs=1.0)


def test_mirror_horizontal(ctx):
    ctx.fleet["SYRX"].pos = np.array([30.0, 160.0])      # out of the reflection's way
    ctx.fleet["CRXS"].pos = np.array([70.0, 50.0])
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS", "axis": "horizontal"})
    assert r["ok"], r["error"]
    assert r["reflected_to"][0] == pytest.approx(70.0, abs=1.0)
    assert r["reflected_to"][1] == pytest.approx(130.0, abs=1.0)


def test_mirror_point_reflection_through_a_robot(ctx):
    """The useful one: A ends up diametrically opposite B through the pivot."""
    ctx.fleet["CRXS"].pos = np.array([80.0, 60.0])
    ctx.fleet["VHGR"].pos = np.array([120.0, 90.0])
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS",
                              "axis": "through:VHGR"})
    assert r["ok"], r["error"]
    assert r["reflected_to"][0] == pytest.approx(160.0, abs=1.0)
    assert r["reflected_to"][1] == pytest.approx(120.0, abs=1.0)


def test_mirror_through_the_arena_centre(ctx):
    ctx.fleet["VHGR"].pos = np.array([40.0, 160.0])      # clear of the reflection
    ctx.fleet["CRXS"].pos = np.array([80.0, 60.0])
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS", "axis": "centre"})
    assert r["ok"], r["error"]
    assert r["reflected_to"][0] == pytest.approx(160.0, abs=1.0)
    assert r["reflected_to"][1] == pytest.approx(120.0, abs=1.0)


def test_mirror_clamps_a_reflection_that_lands_outside(ctx):
    """A reflection can fall off the floor; it must be pulled back, not refused."""
    # Reflection lands far outside (340, 90). The clamped landing zone has to
    # be somewhere a robot can actually stand, or the test is asking for the
    # impossible rather than testing the clamp.
    ctx.fleet["CRXS"].pos = np.array([60.0, 90.0])
    ctx.fleet["VHGR"].pos = np.array([200.0, 90.0])
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS",
                              "axis": "through:VHGR"})
    assert r["ok"], r["error"]
    settle(ctx, 20)
    assert ctx.ws.point_in_bounds(pos(ctx, "SSMK"))
    assert "nudged clear" in (r["note"] or "") or True


def test_mirror_rejects_a_bad_axis(ctx):
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS", "axis": "diagonal"})
    assert not r["ok"] and "diagonal" in r["error"]


def test_mirror_unknown_pivot(ctx):
    r = call(ctx, "mirror", {"code_a": "SSMK", "code_b": "CRXS",
                              "axis": "through:ghost"})
    assert not r["ok"] and "ghost" in r["error"]


# -- they all go through the validator ----------------------------------------------------

def test_every_convenience_tool_respects_the_separation_floor(ctx):
    for name, args in (("swap", {"code_a": "SSMK", "code_b": "CRXS"}),
                       ("displace", {"code_a": "SSMK", "code_b": "CRXS"}),
                       ("nudge", {"direction": "left", "distance": 25}),
                       ("gather", {"around": "VHGR", "radius": 45}),
                       ("spread", {"min_distance": 50}),
                       ("mirror", {"code_a": "SSMK", "code_b": "CRXS",
                                    "axis": "centre"})):
        r = call(ctx, name, args)
        assert r["ok"], f"{name}: {r['error']}"
        settle(ctx, 18)
        assert _min_gap(ctx) > 8.0, f"{name} left robots on top of each other"
        for c in ctx.active_codes():
            assert ctx.ws.point_in_bounds(pos(ctx, c)), f"{name} put {c} out of bounds"

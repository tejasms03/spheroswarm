import json
from pathlib import Path

import numpy as np
import pytest

from evals import assertions as A
from evals.harness import (aggregate, build_world, by_group, format_comparison,
                           format_table, load_cases, run_case, save_results)
from evals.stub_script import build_stub_client, plan
from evals.run import main as run_main

ROOT = Path(__file__).resolve().parent.parent


# -- the eval set itself ----------------------------------------------------

def test_thirty_cases_across_six_groups():
    cases = load_cases()
    assert len(cases) == 30
    assert len({c["id"] for c in cases}) == 30
    assert {c["group"] for c in cases} == {"basic", "relative", "memory",
                                            "novel", "sequence", "safety"}


def test_every_case_has_text_and_a_buildable_assertion():
    for c in load_cases():
        assert c.get("text"), f"{c['id']} has no text"
        assert c.get("notes"), f"{c['id']} has no notes"
        A.build(c.get("assertion"))          # raises if unbuildable


def test_the_three_named_cases_are_flagged_for_human_review():
    flagged = {c["id"] for c in load_cases() if c.get("human_review")}
    assert flagged == {"D1_letter_L", "D2_arrow_right", "E1_corners_then_middle"}


# -- assertion helpers, against hand-built arrangements ---------------------

def ring(n=6, r=60, cx=120, cy=90, phase=0.0):
    a = np.arange(n) * 2 * np.pi / n + phase
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def test_is_circle_accepts_a_circle_and_rejects_a_line():
    assert A.is_circle(0.15)(ring())[0]
    assert not A.is_circle(0.15)(np.c_[np.arange(6) * 30.0, np.full(6, 90.0)])[0]


def test_is_circle_is_scale_invariant():
    assert A.is_circle(0.15)(ring(r=25))[0]
    assert A.is_circle(0.15)(ring(r=85))[0]


def test_is_circle_rejects_a_lopsided_ring():
    pts = ring()
    pts[0] = [120, 90]                    # one robot in the middle
    assert not A.is_circle(0.15)(pts)[0]


def test_is_line_accepts_horizontal_vertical_and_diagonal():
    for pts in (np.c_[np.arange(6) * 30.0, np.full(6, 90.0)],
                np.c_[np.full(6, 120.0), np.arange(6) * 25.0],
                np.c_[np.arange(6) * 25.0, np.arange(6) * 25.0]):
        assert A.is_line(0.1)(pts)[0]


def test_is_line_rejects_a_circle():
    assert not A.is_line(0.1)(ring())[0]


def test_is_evenly_spaced():
    even = np.c_[np.arange(6) * 30.0, np.full(6, 90.0)]
    assert A.is_evenly_spaced(0.2)(even)[0]
    uneven = np.c_[np.array([0, 5, 10, 60, 120, 200.0]), np.full(6, 90.0)]
    assert not A.is_evenly_spaced(0.2)(uneven)[0]


def test_centroid_near():
    assert A.centroid_near([120, 90], 25)(ring())[0]
    assert not A.centroid_near([20, 20], 25)(ring())[0]


def test_bbox_within():
    assert A.bbox_within([0, 0, 240, 180])(ring())[0]
    assert not A.bbox_within([0, 0, 100, 100])(ring())[0]


def test_min_separation_above():
    assert A.min_separation_above(40)(ring(r=80))[0]
    assert not A.min_separation_above(40)(ring(r=10))[0]


def test_n_clusters_and_clusters_apart():
    two = np.r_[ring(3, 20, 40, 90), ring(3, 20, 200, 90)]
    assert A.n_clusters(2)(two)[0]
    assert not A.n_clusters(3)(two)[0]
    assert A.clusters_apart(k=2, min_gap=80)(two)[0]
    assert not A.clusters_apart(k=2, min_gap=200)(two)[0]


def test_all_moved_by():
    before = ring()
    after = before + np.array([-30.0, 0.0])
    assert A.all_moved_by([-30, 0], 8)(after, before=before)[0]
    assert not A.all_moved_by([30, 0], 8)(after, before=before)[0]


def test_all_moved_by_needs_a_snapshot():
    assert not A.all_moved_by([-30, 0])(ring())[0]


def test_spread_changed_both_ways():
    before, bigger = ring(r=40), ring(r=80)
    assert A.spread_changed("increase", 1.2)(bigger, before=before)[0]
    assert A.spread_changed("decrease", 1.2)(before, before=bigger)[0]
    assert not A.spread_changed("increase", 1.2)(before, before=bigger)[0]


def test_centroid_moved_directions():
    before = ring()
    ne = before + np.array([60.0, -40.0])
    assert A.centroid_moved("northeast", 25)(ne, before=before)[0]
    assert not A.centroid_moved("southwest", 25)(ne, before=before)[0]


def test_rotated_by_uses_per_robot_identity():
    """A symmetric ring carries no absolute angle; robot identity does."""
    codes = [f"R{i}" for i in range(6)]
    before = ring()
    after = ring(phase=np.radians(45))
    bmap = {c: before[i] for i, c in enumerate(codes)}
    amap = {c: after[i] for i, c in enumerate(codes)}

    ok, why = A.rotated_by(45, 20)(after, before=before,
                                    before_map=bmap, after_map=amap)
    assert ok, why
    assert "per robot" in why

    bad, _ = A.rotated_by(180, 20)(after, before=before,
                                    before_map=bmap, after_map=amap)
    assert not bad


def test_rotated_by_rejects_a_rebuild_at_a_different_size():
    codes = [f"R{i}" for i in range(6)]
    before, after = ring(r=40), ring(r=120, phase=np.radians(45))
    ok, why = A.rotated_by(45, 20)(
        after, before=before,
        before_map={c: before[i] for i, c in enumerate(codes)},
        after_map={c: after[i] for i, c in enumerate(codes)})
    assert not ok
    assert "rebuilt" in why


def test_extents_differ_separates_oval_from_circle():
    assert not A.extents_differ(1.4)(ring())[0]
    oval = ring()
    oval[:, 1] = 90 + (oval[:, 1] - 90) * 0.3
    assert A.extents_differ(1.4)(oval)[0]


def spiral_points(n=6, r0=20, step=12, cx=120, cy=90):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 + np.arange(n) * step
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def test_is_spiral_accepts_a_spiral_and_rejects_a_circle():
    assert A.is_spiral()(spiral_points())[0]
    assert not A.is_spiral()(ring())[0]


def test_is_spiral_survives_simulator_noise():
    """Robots settle within a few cm of target; that must not fail the shape."""
    rng = np.random.default_rng(0)
    noisy = spiral_points() + rng.normal(0, 5, (6, 2))
    assert A.is_spiral()(noisy)[0]


def test_is_spiral_rejects_lines():
    """A line has growing radius seen from one side — the wrap check catches it."""
    for pts in (np.c_[np.arange(6) * 30.0, np.full(6, 90.0)],
                np.c_[np.full(6, 120.0), np.arange(6) * 25.0],
                np.c_[np.arange(6) * 25.0, np.arange(6) * 25.0]):
        ok, why = A.is_spiral()(pts)
        assert not ok
        assert "line" in why


def test_is_spiral_false_positive_rate_is_bounded():
    """Six points is few: some random clouds do read as spirals.

    This pins the rate so a future change cannot quietly make the assertion
    meaningless. It is a known limit of the case, not a bug.
    """
    rng = np.random.default_rng(0)
    hits = sum(A.is_spiral()(np.c_[rng.uniform(20, 220, 6),
                                    rng.uniform(20, 160, 6)])[0]
               for _ in range(200))
    assert hits / 200 < 0.10, f"false positive rate {hits/200:.0%}"


def test_matches_saved_is_place_scale_and_rotation_invariant(sim_ctx):
    ctx = sim_ctx(n=6)
    shape = ring(r=50, cx=80, cy=60)
    ctx.library.save_formation("wedge", shape, ctx.active_codes())

    same_elsewhere = ring(r=90, cx=180, cy=120, phase=np.radians(30))
    ok, why = A.matches_saved("wedge", 0.2)(same_elsewhere, library=ctx.library)
    assert ok, why

    line = np.c_[np.arange(6) * 30.0, np.full(6, 90.0)]
    assert not A.matches_saved("wedge", 0.2)(line, library=ctx.library)[0]


def test_matches_saved_names_a_missing_formation(sim_ctx):
    ctx = sim_ctx(n=6)
    ok, why = A.matches_saved("ghost")(ring(), library=ctx.library)
    assert not ok and "ghost" in why


def test_refused_requires_both_words_and_stillness():
    before = ring()
    assert A.refused()(before, before=before,
                       reply="I can't — targets must be 20cm apart.")[0]
    assert not A.refused()(before, before=before, reply="Done!")[0]
    moved = before + 60
    assert not A.refused()(moved, before=before, reply="I can't do that.")[0]


def test_reported_clamping(ws):
    inside = ring(r=40)
    assert A.reported_clamping()(inside, reply="I clamped it to the wall.",
                                  workspace=ws)[0]
    assert A.reported_clamping()(inside, reply="Done.", clamped_any=True,
                                  workspace=ws)[0]
    assert not A.reported_clamping()(inside, reply="Done.", workspace=ws)[0]


def test_called_tool_and_did_not_move():
    assert A.called_tool("list_formations")(ring(), tool_names=["list_formations"])[0]
    assert not A.called_tool("move_to")(ring(), tool_names=["list_formations"])[0]
    assert A.did_not_move(20)(ring(), before=ring())[0]
    assert not A.did_not_move(20)(ring() + 50, before=ring())[0]


def test_all_of_reports_the_first_failure():
    ok, why = A.all_of(A.is_circle(0.15), A.centroid_near([0, 0], 5))(ring())
    assert not ok
    assert "PASS" in why and "FAIL" in why


def test_valid_arrangement_catches_out_of_bounds(ws):
    assert A.valid_arrangement()(ring(r=40), workspace=ws)[0]
    outside = ring(r=40) + np.array([400.0, 0.0])
    assert not A.valid_arrangement()(outside, workspace=ws)[0]


# -- the assertion parser ----------------------------------------------------

def test_parser_accepts_bare_words_as_strings():
    """`axis=y` and `matches_saved(wedge)` must mean what they obviously mean."""
    A.build("all_within(axis=y, hi=40)")
    A.build("centroid_moved(northeast, 25)")
    A.build("formation_exists(wedge)")
    A.build("spread_changed(direction=decrease, factor=1.3)")


def test_parser_accepts_lists_and_nested_all_of():
    A.build("centroid_near([120, 90], 25)")
    A.build({"all_of": ["is_circle(0.15)", "is_evenly_spaced(0.2)"]})


def test_parser_rejects_an_unknown_assertion():
    with pytest.raises(ValueError, match="unknown assertion"):
        A.build("is_a_dragon(3)")


def test_no_assertion_passes_trivially():
    assert A.build(None)(ring())[0]


# -- world isolation ---------------------------------------------------------

def test_build_world_is_deterministic(tmp_path):
    a = build_world(tmp_path / "a")
    b = build_world(tmp_path / "b")
    np.testing.assert_allclose(
        np.array([a.fleet[c].pos for c in a.active_codes()]),
        np.array([b.fleet[c].pos for c in b.active_codes()]))
    a.fleet.close()
    b.fleet.close()


def test_build_world_starts_with_an_empty_library(tmp_path):
    ctx = build_world(tmp_path)
    assert ctx.library.formations == {}
    ctx.fleet.close()


def test_an_eval_run_leaves_the_live_state_files_untouched(tmp_path):
    """The isolation guarantee. This has already bitten once in this project."""
    live = {name: (ROOT / name).read_bytes()
            for name in ("roster.json", "workspace.json", "formations.json")}

    cases = [c for c in load_cases()
             if c["id"] in ("A1_circle", "C1_save_wedge", "F1_impossible_stack")]
    for case in cases:
        run_case(case, build_stub_client, tmp_path, settle=2)

    for name, before in live.items():
        assert (ROOT / name).read_bytes() == before, f"{name} was modified"


def test_a_saved_formation_goes_to_the_scratch_library(tmp_path):
    """The save must land in the eval's scratch library, never the real one.

    Checked as "unchanged", not "empty": the user's own saved formations live
    in that file, and a test that demands it be empty fails the moment someone
    actually uses the app.
    """
    live = (ROOT / "formations.json").read_bytes()
    case = next(c for c in load_cases() if c["id"] == "C1_save_wedge")
    r = run_case(case, build_stub_client, tmp_path, settle=2)
    assert r.passed, r.reason
    assert (ROOT / "formations.json").read_bytes() == live, \
        "the eval wrote the live formations.json"


# -- running cases -----------------------------------------------------------

def test_a_case_runs_end_to_end_against_the_stub(tmp_path):
    case = next(c for c in load_cases() if c["id"] == "A1_circle")
    r = run_case(case, build_stub_client, tmp_path, settle=8)
    assert r.passed, r.reason
    assert r.completed and r.valid and r.assertion
    assert r.tool_calls >= 1
    assert r.latency_s > 0


def test_setup_turns_run_before_the_scored_command(tmp_path):
    case = next(c for c in load_cases() if c["id"] == "C2_recall_wedge")
    r = run_case(case, build_stub_client, tmp_path, settle=8)
    assert r.passed, r.reason


def test_a_broken_client_is_a_failed_case_not_a_crash(tmp_path):
    def broken():
        raise RuntimeError("no model here")

    case = next(c for c in load_cases() if c["id"] == "A1_circle")
    r = run_case(case, broken, tmp_path, settle=1)
    assert r.passed is False
    assert "no model here" in r.error


def test_a_case_whose_model_never_stops_hits_the_cap(tmp_path):
    from llm.client import ModelResponse, ToolCall

    class Looper:
        model = "looper"

        def reachable(self, timeout=2.0):
            return True

        def chat(self, messages, tools=None):
            return ModelResponse(tool_calls=[ToolCall("get_state", {}, id="c")])

    case = next(c for c in load_cases() if c["id"] == "A1_circle")
    r = run_case(case, Looper, tmp_path, settle=1, max_tool_calls=3)
    assert r.hit_cap and not r.completed and not r.passed


# -- reporting ---------------------------------------------------------------

def test_aggregate_counts_automatic_and_human_review_separately(tmp_path):
    cases = [c for c in load_cases() if c["id"] in ("A1_circle", "D1_letter_L")]
    results = [run_case(c, build_stub_client, tmp_path, settle=6) for c in cases]
    agg = aggregate(results)
    assert agg["cases"] == 2
    assert agg["auto_cases"] == 1
    assert 0 <= agg["pass_rate"] <= 1


def test_by_group_totals_add_up(tmp_path):
    cases = [c for c in load_cases() if c["id"] in ("A1_circle", "D3_spiral")]
    results = [run_case(c, build_stub_client, tmp_path, settle=6) for c in cases]
    groups = by_group(results)
    assert sum(g["n"] for g in groups.values()) == 2


def test_results_are_written_as_json(tmp_path):
    cases = [c for c in load_cases() if c["id"] == "A1_circle"]
    results = [run_case(c, build_stub_client, tmp_path, settle=6) for c in cases]
    agg, groups = aggregate(results), by_group(results)
    path = save_results("qwen3.5:9b", results, agg, groups, directory=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["model"] == "qwen3.5:9b"
    assert data["aggregate"]["cases"] == 1
    assert data["cases"][0]["id"] == "A1_circle"
    assert ":" not in path.name.split("_")[0]      # filename is filesystem-safe


def test_table_and_comparison_render(tmp_path):
    cases = [c for c in load_cases() if c["id"] == "A1_circle"]
    results = [run_case(c, build_stub_client, tmp_path, settle=6) for c in cases]
    assert "A1_circle" in format_table(results)
    text = format_comparison({"m1": (results, aggregate(results)),
                               "m2": (results, aggregate(results))})
    assert "m1" in text and "m2" in text


# -- the CLI ------------------------------------------------------------------

def test_cli_runs_a_subset_and_returns_zero(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = run_main(["--model", "stub", "--only", "A1_circle", "D3_spiral",
                     "--settle", "6", "--no-save", "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "A1_circle" in out and "pass" in out


def test_cli_group_filter(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_main(["--model", "stub", "--group", "safety", "--settle", "4",
              "--no-save", "--quiet"])
    out = capsys.readouterr().out
    assert "F1_impossible_stack" in out
    assert "A1_circle" not in out


def test_cli_rejects_an_empty_selection(capsys):
    assert run_main(["--model", "stub", "--only", "nope"]) == 2


# -- the stub itself ----------------------------------------------------------

def test_stub_plans_a_tool_call_for_every_case_command():
    """A stub that silently shrugs would make the harness look broken."""
    for case in load_cases():
        turns = plan(case["text"])
        assert turns, case["id"]
        assert any(not isinstance(t, str) for t in turns), \
            f"{case['id']}: stub produced no tool call for {case['text']!r}"


def test_stub_is_stateless_between_clients():
    a, b = build_stub_client(), build_stub_client()
    assert a is not b
    assert a.turn if hasattr(a, "turn") else True

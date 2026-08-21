"""The command bar's parser and the exact lines the spec requires to work."""

import numpy as np
import pytest

from fleet.manager import Fleet
from fleet.roster import RobotEntry, Roster
from tools import SwarmContext
from tools.command import ParseError, parse
from tools.command import run as run_command
from tools.command import summarise
from tools.formations import FormationLibrary
from workspace.space import Workspace

CODES = ["SSMK", "CRXS", "SYRX", "VHGR"]


@pytest.fixture
def space():
    return Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])


@pytest.fixture
def ctx(space, tmp_path):
    colors = ["cyan", "red", "yellow", "green"]
    entries = [RobotEntry(name=f"D{i}", code=c, kind="sim", color=col)
               for i, (c, col) in enumerate(zip(CODES, colors))]
    fleet = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                              workspace=space)
    for code, p in zip(CODES, [[20, 20], [220, 20], [220, 160], [20, 160]]):
        fleet[code].pos = np.array(p, dtype=float)
    c = SwarmContext(fleet=fleet, workspace=space,
                     library=FormationLibrary(path=tmp_path / "f.json"))
    yield c
    fleet.close()


# -- parsing --------------------------------------------------------------

def test_parses_unordered_points():
    name, args = parse("move_to 60,40 120,40 180,40", CODES)
    assert name == "move_to"
    assert args["points"] == [[60, 40], [120, 40], [180, 40]]


def test_parses_explicit_assignment():
    name, args = parse("move_to SSMK=100,80 CRXS=140,80", CODES)
    assert args["assign"] == {"SSMK": [100, 80], "CRXS": [140, 80]}
    assert "points" not in args


def test_parses_mixed_pinned_and_free():
    name, args = parse("move_to SSMK=100,80 60,40 180,40", CODES)
    assert args["assign"] == {"SSMK": [100, 80]}
    assert args["points"] == [[60, 40], [180, 40]]


def test_parses_transform():
    assert parse("transform rotate=45 scale=1.5", CODES)[1] == \
           {"rotate": 45.0, "scale": 1.5}
    assert parse("transform translate=10,-20", CODES)[1] == {"translate": [10, -20]}
    assert parse("transform rotation=90", CODES)[1] == {"rotate": 90.0}


def test_parses_formation_commands():
    assert parse("save_formation wedge", CODES) == ("save_formation", {"name": "wedge"})
    assert parse('save_formation wedge "a nice wedge"', CODES)[1] == \
           {"name": "wedge", "description": "a nice wedge"}
    name, args = parse("recall_formation wedge center=120,90 rotation=180", CODES)
    assert args == {"name": "wedge", "center": [120, 90], "rotation": 180.0}


def test_parses_stop_and_led():
    assert parse("stop", CODES) == ("stop", {})
    assert parse("stop SSMK CRXS", CODES)[1] == {"codes": ["SSMK", "CRXS"]}
    assert parse("set_led red", CODES)[1] == {"color": "red"}
    assert parse("set_led 255,0,0 codes=SSMK blink=fast", CODES)[1] == \
           {"color": [255, 0, 0], "codes": ["SSMK"], "blink": "fast"}


def test_parses_compute_points_expression_with_equals():
    name, args = parse('compute_points "points = [(cx, cy)]"', CODES)
    assert args["expression"] == "points = [(cx, cy)]"
    name, args = parse('compute_points "points = [(1,2)]" execute=true', CODES)
    assert args["execute"] is True and "points" in args["expression"]


def test_parse_errors_are_readable():
    with pytest.raises(ParseError, match="unknown tool"):
        parse("frobnicate", CODES)
    with pytest.raises(ParseError, match="x,y"):
        parse("move_to 60", CODES)
    with pytest.raises(ParseError, match="number"):
        parse("transform scale=big", CODES)
    with pytest.raises(ParseError, match="no argument"):
        parse("transform scaale=1.5", CODES)
    with pytest.raises(ParseError, match="tool call"):
        parse("   ", CODES)


# -- the exact lines from the specification ------------------------------

def test_spec_line_move_to_points(ctx):
    r = run_command(ctx, "move_to 60,40 120,40 180,40 100,140")
    assert r["ok"], r["error"]
    assert len(r["targets"]) == 4


def test_spec_line_move_to_pinned(ctx):
    """Pinning two of four means the other two hold station."""
    r = run_command(ctx, "move_to SSMK=100,80 CRXS=140,80")
    assert r["ok"], r["error"]
    assert r["targets"]["SSMK"] == [100.0, 80.0]
    assert r["targets"]["CRXS"] == [140.0, 80.0]
    assert set(r["targets"]) == set(CODES)
    # the unpinned pair stay where they were
    assert r["targets"]["SYRX"] == [220.0, 160.0]
    assert r["targets"]["VHGR"] == [20.0, 160.0]


def test_spec_line_transform(ctx):
    assert run_command(ctx, "move_to 60,40 120,40 180,40 100,140")["ok"]
    r = run_command(ctx, "transform rotate=45 scale=1.5")
    assert r["ok"], r["error"]


def test_spec_line_save_and_recall(ctx):
    assert run_command(ctx, "move_to 60,40 120,40 180,40 100,140")["ok"]
    r = run_command(ctx, "save_formation wedge")
    assert r["ok"], r["error"]
    r = run_command(ctx, "recall_formation wedge center=120,90 rotation=180")
    assert r["ok"], r["error"]
    assert len(r["targets"]) == 4


def test_pinned_only_catches_a_collision_with_a_bystander(ctx):
    """Moving onto a robot that is holding station must be refused, not run."""
    ctx.fleet["SYRX"].pos = np.array([102.0, 80.0])
    r = run_command(ctx, "move_to SSMK=100,80")
    assert not r["ok"]
    assert "apart" in r["error"]


def test_bad_command_returns_an_error_not_an_exception(ctx):
    for line in ("frobnicate", "move_to 60", "transform scale=big",
                 "recall_formation nope", "set_led chartreuse", ""):
        r = run_command(ctx, line)
        assert isinstance(r, dict) and r["ok"] is False
        assert r["error"]


def test_summarise_produces_one_line(ctx):
    r = run_command(ctx, "move_to 60,40 120,40 180,40 100,140")
    line = summarise(r)
    assert "\n" not in line and "target" in line

    line = summarise(run_command(ctx, "frobnicate"))
    assert line.startswith("error:")

"""Learning from requests, without anything being filed or looked up.

The formation library is a filing cabinet you address by name. This is the
other kind: commands that worked are recorded, and the relevant ones appear in
the prompt on their own. The model never calls a tool to read them.
"""

import pytest

from llm.memory import MAX_CHARS, Memory


@pytest.fixture
def mem():
    m = Memory(path=None, identifiers=["SSMK", "CRXS", "Seasmoke", "Caraxes"])
    m.record("make a circle", ["compute_points"], "radius 60cm", save=False)
    m.record("swap Seasmoke and Caraxes", ["swap"], "exchanged", save=False)
    m.record("everyone orbit the centre", ["set_flow"], "60s", save=False)
    return m


# -- what it recalls ---------------------------------------------------------

def test_a_rephrased_request_finds_the_precedent(mem):
    assert "compute_points" in " ".join(mem.lines("make a circle please"))
    assert "compute_points" in " ".join(mem.lines("draw a circle"))


def test_which_robots_were_named_is_not_the_intent(mem):
    """"swap SSMK and CRXS" is the same request as "swap Seasmoke and Caraxes"."""
    assert "swap" in " ".join(mem.lines("swap SSMK and CRXS"))


def test_an_unrelated_request_recalls_nothing(mem):
    """Loosely related precedent is worse than none — it invites copying."""
    assert mem.lines("what colour is the sky") == []
    assert mem.lines("set the LED to purple") == []


def test_nothing_is_recalled_with_an_empty_memory():
    assert Memory(path=None).lines("make a circle") == []


# -- what it learns ----------------------------------------------------------

def test_only_commands_that_did_something_are_worth_recording():
    m = Memory(path=None)
    assert m.record("make a circle", [], save=False) is None
    assert m.record("", ["compute_points"], save=False) is None
    assert len(m) == 0


def test_repeating_a_command_reinforces_rather_than_duplicates():
    m = Memory(path=None)
    for _ in range(3):
        m.record("make a circle", ["compute_points"], save=False)
    assert len(m) == 1
    assert m.entries[0]["uses"] == 3


def test_a_more_used_precedent_wins_a_tie():
    m = Memory(path=None)
    m.record("make a ring", ["move_to"], save=False)
    for _ in range(8):
        m.record("make a circle", ["compute_points"], save=False)
    assert "compute_points" in mem_first_line(m, "make a circle")


def mem_first_line(m, text):
    lines = m.lines(text)
    return lines[0] if lines else ""


def test_it_forgets_on_request():
    m = Memory(path=None)
    m.record("make a circle", ["compute_points"], save=False)
    assert m.forget("circle") == 1
    assert len(m) == 0


# -- the budget --------------------------------------------------------------

def test_the_injected_block_is_capped(mem):
    for i in range(60):
        mem.record(f"make a circle variant {i} with many extra words here",
                   ["compute_points", "transform", "wait_until_settled"],
                   "a fairly long summary of what happened that time",
                   save=False)
    block = "\n".join(mem.lines("make a circle"))
    assert len(block) <= MAX_CHARS + 80, f"{len(block)} chars injected"


def test_old_entries_fall_off():
    m = Memory(path=None, max_entries=10)
    for i in range(40):
        m.record(f"command number {i}", ["move_to"], save=False)
    assert len(m) == 10


# -- persistence -------------------------------------------------------------

def test_it_survives_a_restart(tmp_path):
    path = tmp_path / "memory.json"
    m = Memory(path=path)
    m.record("make a circle", ["compute_points"], "radius 60cm")
    assert Memory.load(path).lines("make a circle")


def test_a_corrupt_file_is_reported_not_fatal(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json")
    m = Memory.load(path)
    assert len(m) == 0 and m.errors
    assert m.lines("anything") == []


def test_no_path_means_nothing_is_written(tmp_path):
    m = Memory(path=None)
    m.record("make a circle", ["compute_points"])
    assert list(tmp_path.iterdir()) == []

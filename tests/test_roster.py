from fleet.roster import RobotEntry, Roster, validate


def base_entries():
    return [
        RobotEntry(name="Seasmoke", code="SSMK", kind="sim", color="cyan"),
        RobotEntry(name="Caraxes", code="CRXS", kind="sim", color="red"),
        RobotEntry(name="Syrax", code="SYRX", kind="sim", color="yellow"),
    ]


def test_default_roster_loads_clean():
    r = Roster.load()
    assert r.errors == [], r.errors
    assert len(r.entries) == 12
    enabled = r.enabled_entries()
    assert len(enabled) == 6
    assert len({e.color for e in enabled}) == 6


def test_valid_roster_has_no_errors():
    assert validate(base_entries()) == []


def test_duplicate_names_caught():
    entries = base_entries()
    entries[1].name = "Seasmoke"
    errors = validate(entries)
    assert any("duplicate names" in e for e in errors)


def test_duplicate_codes_caught():
    entries = base_entries()
    entries[1].code = "SSMK"
    errors = validate(entries)
    assert any("duplicate codes" in e for e in errors)


def test_duplicate_colours_among_enabled_caught():
    entries = base_entries()
    entries[1].color = "cyan"
    errors = validate(entries)
    assert any("duplicate colours" in e for e in errors)


def test_duplicate_colours_ignored_when_disabled():
    entries = base_entries()
    entries[1].color = "cyan"
    entries[1].enabled = False
    errors = validate(entries)
    assert not any("duplicate colours" in e for e in errors)


def test_real_robot_missing_ble_name_caught():
    entries = base_entries()
    entries[0].kind = "real"
    entries[0].ble_name = None
    errors = validate(entries)
    assert any("ble_name" in e for e in errors)


def test_real_robot_with_ble_name_ok():
    entries = base_entries()
    entries[0].kind = "real"
    entries[0].ble_name = "SK-1A2B"
    assert validate(entries) == []


def test_more_than_six_enabled_caught():
    entries = [
        RobotEntry(name=f"D{i}", code=f"C{i}", kind="sim", color=c, enabled=True)
        for i, c in enumerate(["red", "yellow", "green", "cyan", "blue", "magenta", "red"])
    ]
    # last one reuses a colour on purpose isn't the point here; count is
    entries[-1].color = "yellow"  # avoid tripping the colour check too
    errors = validate(entries)
    assert any("enabled" in e and "6" in e for e in errors)


def test_invalid_color_caught():
    entries = base_entries()
    entries[0].color = "chartreuse"
    errors = validate(entries)
    assert any("chartreuse" in e for e in errors)


def test_roster_add_remove_set_kind():
    r = Roster(entries=base_entries(), path="/dev/null")
    errs = r.add(RobotEntry(name="Vhagar", code="VHGR", kind="sim", color="green"))
    assert errs == []
    assert r.by_code("VHGR") is not None

    errs = r.remove("VHGR")
    assert errs == []
    assert r.by_code("VHGR") is None

    errs = r.set_kind("SSMK", "real", ble_name="SK-9F00")
    assert errs == []
    assert r.by_code("SSMK").kind == "real"

    # flipping back to sim without clearing ble_name is fine; flipping a
    # *different* robot to real with no ble_name should fail and not mutate
    before = r.by_code("CRXS").kind
    errs = r.set_kind("CRXS", "real")
    assert errs != []
    assert r.by_code("CRXS").kind == before


def test_load_missing_file_returns_error_not_raise():
    r = Roster.load(path="/tmp/does-not-exist-roster.json")
    assert r.errors
    assert r.entries == []


def test_save_rejects_invalid_without_writing(tmp_path):
    p = tmp_path / "roster.json"
    r = Roster(entries=base_entries(), path=p)
    r.entries[1].name = "Seasmoke"  # introduce a duplicate
    errors = r.save()
    assert errors
    assert not p.exists()


# -- reassigning hues --------------------------------------------------------

def _six():
    from fleet.roster import RobotEntry
    from vision.config import COLORS
    return [RobotEntry(name=f"R{i}", code=f"R{i}", kind="sim", color=c)
            for i, c in enumerate(COLORS)]


def test_set_color_swaps_when_the_hue_is_taken():
    """With six robots on six hues every reassignment clashes, so it swaps."""
    from fleet.roster import Roster
    r = Roster(_six())
    a, b = r.entries[0].color, r.entries[3].color
    assert r.set_color("R0", b) == []
    assert r.entries[0].color == b
    assert r.entries[3].color == a


def test_set_color_leaves_the_roster_valid():
    from fleet.roster import Roster, validate
    r = Roster(_six())
    for target in ("green", "magenta", "red"):
        assert r.set_color("R1", target) == []
        assert validate(r.entries) == []


def test_set_color_rejects_a_hue_the_tracker_does_not_know():
    from fleet.roster import Roster
    r = Roster(_six())
    before = r.entries[0].color
    errs = r.set_color("R0", "chartreuse")
    assert errs and "chartreuse" in errs[0]
    assert r.entries[0].color == before


def test_set_color_on_a_disabled_robot_takes_the_hue_without_a_swap():
    """Disabled robots do not contend for hues; nobody has to give one up."""
    from fleet.roster import Roster, RobotEntry
    entries = _six()
    entries.append(RobotEntry(name="Spare", code="SPRE", kind="sim",
                              color="red", enabled=False))
    r = Roster(entries)
    holder = next(e for e in r.entries if e.enabled and e.color == "green")
    assert r.set_color("SPRE", "green") == []
    assert r.by_code("SPRE").color == "green"
    assert holder.color == "green", "the enabled holder must keep its hue"


def test_set_color_to_the_hue_already_worn_is_a_no_op():
    from fleet.roster import Roster
    r = Roster(_six())
    before = [(e.code, e.color) for e in r.entries]
    assert r.set_color("R2", r.by_code("R2").color) == []
    assert [(e.code, e.color) for e in r.entries] == before


# -- binding physical robots to rows -----------------------------------------

def test_two_rows_may_not_name_the_same_ball():
    """Both entries open a link to it; one wins and the other never connects."""
    from fleet.roster import RobotEntry, validate
    e = _six()
    e[0].kind, e[0].ble_name = "real", "SK-1A2B"
    e[1].kind, e[1].ble_name = "real", "SK-1A2B"
    errs = validate(e)
    assert any("more than one entry" in x for x in errs), errs


def test_binding_a_ball_takes_it_off_whoever_had_it():
    from fleet.roster import Roster
    r = Roster(_six())
    assert r.set_ble("R0", "SK-1A2B") == []
    assert r.set_ble("R3", "SK-1A2B") == []
    assert r.by_code("R3").ble_name == "SK-1A2B"
    assert r.by_code("R0").ble_name is None
    assert r.by_code("R0").kind == "sim", "a real robot with no ball is not real"
    assert r.stolen == ["R0"]


def test_a_roster_that_arrives_broken_can_be_repaired_a_row_at_a_time():
    """The failure mode this guards: a file only fixable by hand-editing.

    Three rows share one ball. Rebinding the first is refused if the check asks
    "is the roster clean afterwards?", because the other two are still wrong —
    so the repair is blocked by exactly the fault it is repairing.
    """
    from fleet.roster import Roster, validate
    entries = _six()
    for i in (0, 1, 2):
        entries[i].kind, entries[i].ble_name = "real", "SK-DUPE"
    r = Roster(entries)
    r.errors = validate(r.entries)
    assert r.errors, "the fixture should start broken"

    assert r.set_ble("R0", "SK-DUPE") == []
    assert validate(r.entries) == [], "one rebind should resolve all three"


def test_repairing_progressively_is_allowed_but_new_faults_are_not():
    from fleet.roster import Roster, validate
    entries = _six()
    for i in (0, 1, 2, 3):
        entries[i].kind, entries[i].ble_name = "real", f"SK-{i // 2}"
    r = Roster(entries)
    r.errors = validate(r.entries)
    assert r.set_ble("R0", "SK-0") == [], "removing one duplicate is progress"
    # a different KIND of fault must still be refused
    assert r.set_color("R0", "chartreuse")


def test_a_broken_roster_can_be_saved_once_it_is_no_worse(tmp_path):
    from fleet.roster import Roster, validate
    entries = _six()
    entries[0].kind, entries[0].ble_name = "real", "SK-DUPE"
    entries[1].kind, entries[1].ble_name = "real", "SK-DUPE"
    path = tmp_path / "roster.json"
    r = Roster(entries, path=path)
    r.errors = validate(r.entries)

    assert r.save(), "a broken roster is not written by an ordinary save"
    assert not path.exists()
    assert r.save(allow_no_worse=True) == [], "…but a repair-in-progress is"
    assert path.exists()


def test_clear_ble_unbinds_without_disturbing_the_rest():
    from fleet.roster import Roster
    r = Roster(_six())
    r.set_ble("R0", "SK-1A2B")
    colors = {e.code: e.color for e in r.entries}
    assert r.clear_ble("R0") == []
    assert r.by_code("R0").ble_name is None
    assert r.by_code("R0").kind == "sim"
    assert {e.code: e.color for e in r.entries} == colors


def test_a_failed_bind_leaves_the_roster_untouched():
    """Mutating and then reporting an error is how in-memory drifts from disk."""
    from fleet.roster import Roster
    r = Roster(_six())
    assert r.set_ble("NOPE", "SK-1A2B")
    assert all(e.ble_name is None for e in r.entries)

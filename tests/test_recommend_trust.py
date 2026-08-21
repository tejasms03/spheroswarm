"""What a calibration is allowed to recommend after it has gone wrong.

A stage that fails is not the problem — measurement in a room with a ball in it
fails often, and says so. The problem is what happens next: the failure is
recorded in the report and then ignored while the recommendations are
assembled, so a number nobody could stand behind descends into the gains panel
and, worse, into the speed limiter, where nobody reads it at all.

The file that prompted this is the one on disk: a run whose speed map and step
response both errored, whose brake test managed two stops at one commanded
speed, and which published a complete set of recommendations regardless.
"""

import pytest

from fleet import safety
from fleet.characterize import (BRAKE_MIN_ROWS, BRAKE_MIN_SPREAD,
                                Characterization, _brake_trusted)


class WS:
    bbox = (0.0, 138.8, 0.0, 110.8)          # the real arena


def _brake(rows, **kw):
    """A brake result built by the real stage, so the fit and its verdict are
    whatever the production code would actually produce for these stops."""
    from fleet.characterize import BrakeTest
    b = BrakeTest.__new__(BrakeTest)
    b.rows = [{"entry_cm_s": v, "coast_cm": v * 0.5, "settled": True}
              for v in rows]
    d = b.result()
    d.update(kw)
    return d


def _legacy(rows):
    """The same stops as a file written before the verdict existed."""
    d = _brake(rows)
    d.pop("coast_trusted", None)
    d.pop("entry_speed_spread", None)
    return d


def _fit(**stages):
    c = Characterization.__new__(Characterization)
    c.results = {
        "position_noise": {"sigma_cm": 0.46},
        "heading_offset": {"offset_deg": 0.0},
        "speed_map": {"max_speed_cm_s": 48.0, "max_speed_trusted": True,
                      "highest_measured_byte": 250, "min_moving_byte": 20},
        "step_response": {"tau_s": 0.20, "dead_s": 0.10},
        "brake_test": _brake([20.0, 34.0, 48.0, 60.0]),
        "loop_latency": {"loop_delay_s": 0.12},
    }
    c.results.update(stages)
    c.code, c.ble_name, c.started, c.ws, c.notes = "TST", "SK-T", 0, WS(), []
    return c.fit(cruise_cm_s=30.0)


# -- the baseline: a good run still recommends things --------------------

def test_a_clean_run_recommends_normally():
    rec = _fit()["recommend"]
    assert "stopping_distance_s_per_cm_s" in rec
    assert "slow_radius_cm" in rec
    assert "withheld" not in rec


# -- a failed stage is withheld, and says so -----------------------------

def test_a_failed_step_response_does_not_set_the_slow_radius():
    rec = _fit(step_response={"tau_s": 0.20, "dead_s": 0.10,
                              "error": "no clear leg for a step response"})["recommend"]
    assert "slow_radius_cm" not in rec
    assert "step_response" in rec["withheld"]
    assert "no clear leg" in rec["withheld"]["step_response"]


def test_a_failed_brake_stage_withholds_the_stopping_distance():
    rec = _fit(brake_test=_brake([20.0, 34.0, 48.0, 60.0],
                                 error="ran out of room"))["recommend"]
    assert "stopping_distance_s_per_cm_s" not in rec
    assert "max_precise_speed_cm_s" not in rec
    assert "brake" in rec["withheld"]


def test_a_failed_speed_map_withholds_the_deadband():
    """`min_moving_byte` is the speed map's other output and was ungated."""
    rec = _fit(speed_map={"min_moving_byte": 18,
                          "error": "no clear leg in any direction"})["recommend"]
    assert "min_moving_byte" not in rec


def test_withheld_data_is_still_reported_for_a_person_to_read():
    """Withholding is about what feeds a gain, not about hiding evidence."""
    fit = _fit(step_response={"tau_s": 0.20, "error": "boom"})
    assert fit["step_response"]["tau_s"] == 0.20, "the measurement stays visible"
    assert "slow_radius_cm" not in fit["recommend"], "it just does not feed this"


# -- the brake fit has to earn its slope ---------------------------------

def test_two_stops_at_one_speed_are_not_a_fit():
    """The run on disk: two stops, entry speeds 27.7 and 23.9, both byte 26."""
    rec = _fit(brake_test=_brake([27.72, 23.88]))["recommend"]
    assert "stopping_distance_s_per_cm_s" not in rec
    assert "14%" in rec["withheld"]["brake"] or "%" in rec["withheld"]["brake"]


def test_enough_stops_across_enough_range_are():
    rec = _fit(brake_test=_brake([20.0, 40.0, 60.0]))["recommend"]
    assert "stopping_distance_s_per_cm_s" in rec


def test_three_stops_crammed_into_one_speed_are_not():
    """Count alone is not the test — a slope needs a lever arm."""
    assert not _brake_trusted(_legacy([40.0, 41.0, 42.0]))


def test_a_wide_range_with_too_few_points_is_not():
    assert not _brake_trusted(_legacy([10.0, 60.0]))


def test_the_stage_records_its_own_verdict():
    from fleet.characterize import BrakeTest
    b = BrakeTest.__new__(BrakeTest)
    b.rows = [{"entry_cm_s": v, "coast_cm": v * 0.5, "settled": True}
              for v in (20.0, 40.0, 60.0)]
    res = b.result()
    assert res["coast_trusted"] is True
    assert res["entry_speed_spread"] == pytest.approx(0.667, abs=0.01)


def test_a_calibration_with_no_verdict_is_recomputed_not_trusted():
    """A file written before the check existed carries no verdict. Defaulting
    a missing verdict to 'trusted' grandfathers past exactly the check that was
    added because of it."""
    old = _legacy([27.72, 23.88])         # no coast_trusted key at all
    assert "coast_trusted" not in old
    assert _brake_trusted(old) is False


def test_too_few_stops_fails_however_wide_the_range():
    rows = [10.0 + i for i in range(BRAKE_MIN_ROWS - 1)] + []
    rows[-1] = 60.0                       # widest possible spread, still too few
    assert len(rows) < BRAKE_MIN_ROWS
    assert not _brake_trusted(_legacy(rows))


def test_too_narrow_a_range_fails_however_many_stops():
    lo = 60.0 * (1 - BRAKE_MIN_SPREAD) + 0.5      # just inside the threshold
    rows = [lo, (lo + 60.0) / 2, 60.0, 55.0]
    assert len(rows) >= BRAKE_MIN_ROWS
    assert not _brake_trusted(_legacy(rows))


def test_two_distinct_speeds_far_apart_are_enough():
    """Three stops at two well-separated speeds is a real lever arm. The
    threshold is about the range measured, not about distinct values."""
    assert _brake_trusted(_legacy([10.0, 10.0, 60.0]))


# -- the arena has an opinion --------------------------------------------

def test_a_slow_radius_near_the_arena_size_is_flagged():
    """The lag from the run on disk, in the arena from the run on disk: every
    number correct on its own, the set physically impossible."""
    rec = _fit(step_response={"tau_s": 0.32, "dead_s": 0.27},
               loop_latency={"loop_delay_s": 0.52})["recommend"]
    assert rec["slow_radius_cm"] > 110.8 * 0.5, (
        "this fixture is meant to produce an impossible radius")
    assert "arena_warning" in rec
    assert "111cm" in rec["arena_warning"], rec["arena_warning"]


def test_a_sane_slow_radius_is_not_flagged():
    rec = _fit()["recommend"]
    assert "arena_warning" not in rec


# -- the silent consumer -------------------------------------------------

def test_the_speed_limiter_checks_the_evidence_not_the_conclusion():
    """`stopping_seconds` is the one consumer nobody reads. A stale file
    carries a published constant with no verdict attached."""
    stale = {"recommend": {"stopping_distance_s_per_cm_s": 0.8519},
             "brake": _legacy([27.72, 23.88])}
    assert safety.stopping_seconds(stale) == safety.DEFAULT_STOP_S


def test_the_speed_limiter_uses_a_constant_that_stands_up():
    good = {"recommend": {"stopping_distance_s_per_cm_s": 0.42},
            "brake": _brake([20.0, 40.0, 60.0])}
    assert safety.stopping_seconds(good) == pytest.approx(0.42)


def test_the_speed_limiter_still_works_with_no_brake_data():
    assert safety.stopping_seconds({"recommend": {}}) == safety.DEFAULT_STOP_S
    assert safety.stopping_seconds(None) == safety.DEFAULT_STOP_S


def test_falling_back_limits_more_rather_than_less():
    """The fallback must be the cautious direction: a robot allowed to go
    faster than it can stop is the failure this whole field exists to
    prevent."""
    stale = {"recommend": {"stopping_distance_s_per_cm_s": 0.01},
             "brake": _legacy([27.72, 23.88])}
    assert safety.stopping_seconds(stale) >= safety.DEFAULT_STOP_S

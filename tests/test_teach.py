"""Learning the aim frame, and the safe area, from a person driving.

Every automatic method here has to drive the robot to find out which way
driving sends it — which is why they end up at walls. A person with a hand on
the keys does not have that problem, and ten seconds of their driving answers
both questions the battery keeps guessing at.
"""

import numpy as np
import pytest

from fleet.handle import velocity_to_command
from fleet.teach import (INFORMATIVE_SPREAD_DEG, MIN_LEG_CM, driven_bounds,
                         estimate, segment)


def drive(courses, frame_err=0.0, legs_cm=25, mirror=False, noise=0.4, seed=0):
    """A person driving `courses`, on a robot whose frame is out by
    `frame_err` — or mirrored, where the error changes sign with direction."""
    rng = np.random.default_rng(seed)
    pos = np.array([60.0, 55.0])
    track = []
    for c in courses:
        r = np.radians(c)
        cmd = np.array([np.sin(r), np.cos(r)]) * 20.0
        actual = (-c) if mirror else (c + frame_err)
        ra = np.radians(actual)
        step = np.array([np.sin(ra), np.cos(ra)])
        for _ in range(int(legs_cm)):
            pos = pos + step + rng.normal(0, noise, 2)
            track.append((pos.copy(), cmd.copy()))
    return track


# -- segmenting ----------------------------------------------------------

def test_a_wander_becomes_one_leg_per_direction():
    legs = segment(drive([0.0, 90.0, 180.0, 270.0]))
    assert len(legs) == 4
    assert all(l["cm"] >= MIN_LEG_CM for l in legs)


def test_a_stretch_too_short_to_mean_anything_is_dropped():
    """Bearing error over `d` centimetres is about sigma/d — half a centimetre
    of camera noise over 3cm is ten degrees."""
    assert segment(drive([0.0], legs_cm=4)) == []


def test_legs_are_cut_on_the_COMMAND_not_the_travel():
    """The travel direction is the thing being measured. Cutting on it would
    fit the segments to the answer."""
    legs = segment(drive([0.0, 90.0], frame_err=80.0))
    assert len(legs) == 2, "an 80deg frame error must not invent extra legs"


# -- estimating ----------------------------------------------------------

def test_it_recovers_a_frame_error_from_a_loop():
    r = estimate(drive([0.0, 90.0, 180.0, 270.0], frame_err=40.0))
    assert r["ok"] is True
    assert r["offset_deg"] == pytest.approx(40.0, abs=6.0)
    assert r["confident"] is True


def test_one_direction_gives_a_number_but_not_confidence():
    """It cannot separate a rotated frame from a mirrored one, and must not
    claim to."""
    r = estimate(drive([0.0], frame_err=40.0))
    assert r["ok"] is True
    assert r["confident"] is False


def test_the_offset_is_subtracted_from_what_is_in_force():
    """The measured value is the ERROR. Assigning it doubles the fault."""
    r = estimate(drive([0.0, 90.0], frame_err=30.0), offset_now=100.0)
    assert r["new_offset_deg"] == pytest.approx((100.0 - r["offset_deg"]) % 360.0,
                                                abs=0.2)


def test_a_mirrored_arena_is_refused_rather_than_averaged():
    """Two contradictory answers must not be handed over as their mean."""
    r = estimate(drive([0.0, 90.0, 180.0, 270.0], mirror=True))
    assert r["ok"] is False
    assert r.get("mirrored") is True
    assert "changes sign" in r["why"]


def test_a_mirror_is_not_called_without_the_evidence():
    """Legs pointing nearly the same way cannot tell the models apart, and
    calling a mirror on that sends someone to re-pick corners that were fine."""
    r = estimate(drive([0.0, 8.0], mirror=True))
    assert r.get("mirrored") is not True


def test_driving_nowhere_is_refused_with_a_reason():
    r = estimate(drive([0.0], legs_cm=3))
    assert r["ok"] is False
    assert "12cm" in r["why"]


def test_nothing_at_all_does_not_raise():
    r = estimate([])
    assert r["ok"] is False


# -- the safe area -------------------------------------------------------

def _box(w, h, n=60):
    pts, corners = [], [(20, 20), (20 + w, 20), (20 + w, 20 + h), (20, 20 + h),
                        (20, 20)]
    for i in range(4):
        a, b = np.array(corners[i], float), np.array(corners[i + 1], float)
        for k in range(n):
            pts.append((a + (b - a) * k / n, np.array([1.0, 0.0])))
    return pts


def test_the_boundary_is_the_driven_box_pulled_in():
    """Inset because the edge of where somebody drove is the edge of what they
    were willing to allow, and a leg ending exactly there ends somewhere
    nobody agreed to."""
    x0, y0, x1, y1 = driven_bounds(_box(90, 70), inset_cm=10.0)
    assert (x0, y0) == pytest.approx((30.0, 30.0), abs=1.0)
    assert (x1, y1) == pytest.approx((100.0, 80.0), abs=1.0)


def test_a_box_too_small_to_work_in_is_refused():
    """Better no boundary than one the battery cannot turn around inside."""
    assert driven_bounds(_box(40, 30)) is None


def test_standing_still_describes_no_area():
    p = np.array([50.0, 50.0])
    assert driven_bounds([(p, np.array([1.0, 0.0]))] * 50) is None


def test_the_minimum_area_accounts_for_the_inset():
    """The inset comes off BOTH sides, so the outer extent a person has to
    drive is the stated minimum, not the minimum plus a surprise."""
    from fleet.teach import MIN_AREA_SIDE_CM

    side = MIN_AREA_SIDE_CM
    assert driven_bounds(_box(side + 2, side + 2)) is not None
    assert driven_bounds(_box(side - 4, side - 4)) is None

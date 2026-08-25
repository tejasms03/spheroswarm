"""Learn the aim frame — and the safe area — from a person driving the ball.

Two things the bench has been guessing at, both of which a trainer already
knows and can simply show it.

**The aim frame.** Every automatic method here drives the robot to find out
which way it goes, and every one of them has the same problem: it has to drive
before it knows which way driving will send it. That is why the battery ends up
at walls and why `Recenter` used to grind into one for thirty seconds. A person
with a hand on the keys has no such problem — they can see the ball, they steer
around obstacles, and they stop when it looks wrong.

**The safe area.** The arena is where the robot *can* go. Where it *should* go
during a calibration is a smaller thing that depends on the furniture, the
cable, the bit of floor that is slightly sloped, and where somebody is standing.
Driving the perimeter of the area you are happy with states it exactly, and
takes about ten seconds.

The estimate is deliberately segmented rather than averaged over the whole
drive. A single averaged command against a single total displacement only works
if the drive went one way; segmenting by commanded direction turns a wander
into several independent measurements, and their DISAGREEMENT is the most
useful number here. A rotated frame gives the same error whichever way you
drove. A mirrored one gives an error that changes sign with direction, and no
single offset can ever cancel it — so a wander that disagrees with itself says
"do not apply this, fix the arena" rather than handing over an average of two
contradictory answers.
"""

import numpy as np

from .handle import velocity_to_command
from .heading import circular_mean, wrap180

# A segment is a stretch driven in roughly one direction. Below this much
# travel the direction is noise: the error in a bearing over `d` centimetres is
# about sigma/d, so half a centimetre of camera noise over 12cm is a couple of
# degrees and over 3cm is ten.
MIN_LEG_CM = 12.0
NEW_LEG_DEG = 25.0          # of commanded turn before it counts as a new leg
MIN_LEGS = 1
# Legs pointing different ways is what separates a rotation from a mirror. Less
# spread than this and the drive cannot tell them apart, however many legs it
# has, so it must not claim to.
#
# Measured as an AXIS, modulo 180, not as a direction. A reflection maps a
# heading and that heading plus 180 to errors differing by a full turn — which
# is to say, to the SAME error. So driving up and back down one line scores a
# perfect 180 degrees of "spread" and is worth nothing at all for telling a
# mirror from a rotation. Driving W and S looks like the most informative
# possible pair and is the least.
INFORMATIVE_SPREAD_DEG = 40.0
MIRROR_SWING_DEG = 70.0     # of error disagreement before we call it a mirror
# ...and it has to be more than one leg that disagrees. A single odd leg is a
# slip, a bump, or the ball catching on a cable, and calling a mirrored arena
# on one of those sends somebody to re-pick corners that were fine.
MIRROR_MIN_LEGS_OFF_AXIS = 2


def _axis(deg):
    """A heading reduced to the line it lies on. See INFORMATIVE_SPREAD_DEG."""
    return float(deg) % 180.0


def _axis_gap(a, b):
    return abs(wrap180(2.0 * (_axis(a) - _axis(b)))) / 2.0


def segment(track, min_leg_cm=MIN_LEG_CM, new_leg_deg=NEW_LEG_DEG):
    """Split (pos, commanded) samples into stretches driven one way.

    A stretch ends when the commanded direction turns, not when the observed
    one does: the observed direction is the thing being measured, and cutting
    on it would fit the segments to the answer.
    """
    legs, current, aim = [], [], None
    for pos, cmd in track:
        cmd = np.asarray(cmd, dtype=float)
        if float(np.linalg.norm(cmd)) < 1e-6:
            if current:
                legs.append(current)
                current, aim = [], None
            continue
        course = velocity_to_command(cmd)[0]
        if aim is None or abs(wrap180(course - aim)) > new_leg_deg:
            if current:
                legs.append(current)
            current, aim = [], course
        current.append((np.asarray(pos, dtype=float), course))
    if current:
        legs.append(current)

    out = []
    for leg in legs:
        if len(leg) < 2:
            continue
        went = leg[-1][0] - leg[0][0]
        d = float(np.linalg.norm(went))
        if d < min_leg_cm:
            continue
        got = velocity_to_command(went)[0]
        told = float(circular_mean([c for _, c in leg]) or leg[0][1])
        out.append({"told_deg": round(told, 1),
                    "went_deg": round(got, 1),
                    "cm": round(d, 1),
                    "error_deg": round(wrap180(got - told), 1)})
    return out


def estimate(track, offset_now=0.0):
    """What this drive says the aim frame is. Never raises.

    Returns a dict that always says what it found and whether it is usable, so
    the caller can report a refusal as clearly as a result. `ok` False with a
    reason beats a number nobody should act on.
    """
    legs = segment(track)
    out = {"legs": legs, "ok": False}
    if len(legs) < MIN_LEGS:
        out["why"] = (f"no stretch longer than {MIN_LEG_CM:.0f}cm was driven in "
                      "one direction — hold a direction until the ball has "
                      "really gone somewhere, then turn")
        return out

    errs = [l["error_deg"] for l in legs]
    told = [l["told_deg"] for l in legs]
    out["offset_deg"] = round(float(circular_mean(errs) or 0.0), 1)

    if len(legs) >= 2:
        spread = max(_axis_gap(a, b) for a in told for b in told)
        swing = max(abs(wrap180(a - b)) for a in errs for b in errs)
        out["direction_spread_deg"] = round(spread, 1)
        out["error_swing_deg"] = round(swing, 1)
        # How many legs actually left the axis most of the driving was on. A
        # drive that is five legs up-and-down one line plus one stray is not
        # evidence of anything; it is one leg.
        base = max(told, key=lambda t: sum(1 for u in told
                                           if _axis_gap(t, u) < INFORMATIVE_SPREAD_DEG))
        off = [t for t in told if _axis_gap(t, base) >= INFORMATIVE_SPREAD_DEG]
        out["legs_off_axis"] = len(off)
        # Only meaningful when the legs actually pointed different ways, and
        # when more than one of them did.
        if (spread >= INFORMATIVE_SPREAD_DEG
                and len(off) >= MIRROR_MIN_LEGS_OFF_AXIS
                and swing > MIRROR_SWING_DEG):
            out["mirrored"] = True
            out["why"] = (
                f"the error changes sign with direction — {swing:.0f}deg of "
                f"swing across {spread:.0f}deg of driving, on {len(off)} legs "
                "that left the main axis. That is a mirrored arena, not a "
                "rotated one, and a heading offset is one number added to "
                "every command: it cancels a constant error and can never "
                "cancel one that flips. Fix the arena frame first.")
            return out

    out["ok"] = True
    out["new_offset_deg"] = round((float(offset_now) - out["offset_deg"]) % 360.0, 1)
    out["confident"] = bool(
        len(legs) >= 2
        and out.get("direction_spread_deg", 0) >= INFORMATIVE_SPREAD_DEG
        and out.get("legs_off_axis", 0) >= MIRROR_MIN_LEGS_OFF_AXIS)
    return out


# What a driven loop has to span before it describes an area worth planning
# inside — the inset comes off both sides, so this is the OUTER extent needed.
MIN_AREA_SIDE_CM = 60.0


def driven_bounds(track, inset_cm=10.0, min_side_cm=None):
    """The box the ball was actually driven around, pulled in a little.

    Inset because the edge of where somebody drove is the edge of where they
    were willing to let it go, and a calibration leg that ends exactly there
    ends somewhere nobody agreed to. Returns None when the drive was too small
    to describe an area — better no boundary than one the battery cannot turn
    around inside.
    """
    pts = [np.asarray(p, dtype=float) for p, _ in track]
    if len(pts) < 2:
        return None
    a = np.min(pts, axis=0) + inset_cm
    b = np.max(pts, axis=0) - inset_cm
    floor = (MIN_AREA_SIDE_CM - 2 * inset_cm) if min_side_cm is None else min_side_cm
    if float(b[0] - a[0]) < floor or float(b[1] - a[1]) < floor:
        return None
    return (float(a[0]), float(a[1]), float(b[0]), float(b[1]))

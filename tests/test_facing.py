"""Heading measured from the camera, not deduced from driving.

The project's premise is that a glowing sphere has no facing, so the aim frame
must be inferred by driving known legs. That inference is where the pain lives:
it needs clear floor, it needs a tracker following the right ball, it goes stale
on every reconnect, and it cannot separate a rotated frame from a mirrored one
without legs pointing several ways.

The premise is only true if you ignore the tail light. A Sphero has a main RGB
LED and a dim blue aim LED at the back, and two lights on one shell is an
orientation — measured every frame, with nothing driven.
"""

import math

import cv2
import numpy as np
import pytest

from vision.facing import facing_cm, facing_px


def ball(heading_deg, R=26, sep_frac=0.5, blur=0, one_light=False,
         front=(80, 80, 255), back=(200, 90, 40)):
    """A lit shell with a bright main LED and a dimmer tail. `heading_deg` is
    in IMAGE degrees: x right, y down."""
    f = np.zeros((200, 200, 3), np.uint8)
    c = np.array([100.0, 100.0])
    cv2.circle(f, tuple(c.astype(int)), R, (30, 30, 90), -1)
    r = math.radians(heading_deg)
    d = np.array([math.cos(r), math.sin(r)]) * R * sep_frac
    cv2.circle(f, tuple((c + d).astype(int)), 6, front, -1)
    if not one_light:
        cv2.circle(f, tuple((c - d).astype(int)), 5, back, -1)
    if blur:
        f = cv2.GaussianBlur(f, (blur | 1, blur | 1), 0)
    return f


@pytest.mark.parametrize("heading", list(range(0, 360, 30)))
def test_it_reads_the_heading_it_was_given(heading):
    got = facing_px(ball(heading), (100, 100), 26)
    assert got is not None, "two clear lights must produce a reading"
    deg, conf = got
    err = abs((deg - heading + 180) % 360 - 180)
    assert err < 3.0, f"told {heading}, read {deg:.1f}"
    assert conf > 0.0


def test_it_beats_the_driven_leg_estimator_on_accuracy():
    """The estimator this replaces manages 0.8deg median, 1.9deg at p90 — and
    only after the robot has travelled 25cm. This needs one frame."""
    errs = []
    for heading in range(0, 360, 15):
        got = facing_px(ball(heading), (100, 100), 26)
        assert got is not None
        errs.append(abs((got[0] - heading + 180) % 360 - 180))
    assert max(errs) < 2.0, f"worst {max(errs):.1f}deg"


def test_the_front_is_the_brighter_light():
    """Not the colour. The main LED is driven hard and the aim light is a dim
    marker, and keying on brightness keeps this working when somebody sets the
    main LED to blue."""
    lit = facing_px(ball(0.0), (100, 100), 26)
    flipped = facing_px(ball(0.0, front=(200, 90, 40), back=(80, 80, 255)),
                        (100, 100), 26)
    assert lit is not None and flipped is not None
    assert abs((lit[0] - flipped[0] + 180) % 360 - 180) > 150, (
        "swapping which light is brighter must reverse the reading")


# -- it has to refuse, and say nothing rather than something wrong -------

def test_one_light_is_not_an_orientation():
    assert facing_px(ball(45.0, one_light=True), (100, 100), 26) is None


def test_a_dark_frame_reads_nothing():
    assert facing_px(np.zeros((200, 200, 3), np.uint8), (100, 100), 26) is None


def test_lights_too_far_apart_are_two_robots():
    """A separation wider than the ball is not a ball.

    `sep_frac` is chosen so BOTH lights are still inside the patch window —
    otherwise this passes because only one light was visible, which tests the
    wrong thing and leaves the ceiling unpinned."""
    got = facing_px(ball(0.0, sep_frac=0.9), (100, 100), 26)
    assert got is None, f"separation ceiling did not bite: {got}"

    # ...and just inside the ceiling still reads, so it is a threshold and not
    # a blanket refusal.
    assert facing_px(ball(0.0, sep_frac=0.6), (100, 100), 26) is not None


def test_a_blob_at_the_frame_edge_is_refused_rather_than_guessed():
    f = ball(0.0)
    assert facing_px(f, (1, 1), 26) is None


# -- arena coordinates ---------------------------------------------------

def test_the_arena_reading_goes_through_the_homography():
    """As two POINTS, not as an angle. A perspective map does not preserve
    angles, so rotating a bearing by whatever the matrix does at the image
    centre is wrong everywhere else in a tilted view — and a tilted view is
    what an overhead camera on a tripod is."""
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(0, 0), (200, 0), (200, 200), (0, 200)], 100.0, 100.0)
    got = facing_cm(ball(0.0), (100, 100), 26, h)
    assert got is not None
    deg, _ = got
    # image +x with this square calibration is arena +x, which is compass 90
    assert abs((deg - 90.0 + 180) % 360 - 180) < 5.0, deg


def test_no_homography_means_no_arena_reading():
    assert facing_cm(ball(0.0), (100, 100), 26, None) is None


# -- the lights route, and the crop it does its sampling in --------------

def _slow_lights(frame, min_v):
    """The obvious full-frame version of `lights_px`, as a reference.

    Kept here rather than in the module because it is the implementation that
    was replaced: at 1080p it materialises four megapixel-sized arrays PER
    LIGHT and runs at about 1.5fps, measured. `lights_px` does the same
    arithmetic inside a window around each light and must agree with this
    exactly — the point of the change was speed, so any difference in the
    numbers is a bug rather than a tradeoff.
    """
    from vision.facing import ANNULUS, RING_MIN_PX, RING_V_FRAC

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (grey >= min_v).astype(np.uint8), connectivity=8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = grey.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for i in range(1, n):
        area = float(stats[i, 4])
        if area < 3 or area > 6000:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        core = math.sqrt(area / math.pi)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = ((d2 >= (ANNULUS[0] * core) ** 2)
                & (d2 <= (ANNULUS[1] * core) ** 2))
        light = {"x": cx, "y": cy, "area": area, "core_px": core,
                 "peak": float(grey[labels == i].max()),
                 "bgr": None, "hue": None, "sat": None}
        if ring.any():
            clipped = ring & (hsv[:, :, 2] >= light["peak"] * RING_V_FRAC)
            if int(clipped.sum()) >= RING_MIN_PX:
                ring = clipped
            light["ring_px"] = int(ring.sum())
            light["bgr"] = tuple(float(v) for v in frame[ring].mean(axis=0))
            hp = hsv[ring]
            ang = hp[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
            wgt = (hp[:, 1].astype(np.float64)
                   * hp[:, 2].astype(np.float64) / 255.0)
            if wgt.sum() > 0:
                sin_ = float((np.sin(ang) * wgt).sum())
                cos_ = float((np.cos(ang) * wgt).sum())
                light["hue"] = float(
                    (math.degrees(math.atan2(sin_, cos_)) % 360.0) / 2.0)
                light["sat"] = float(hp[:, 1].mean())
        out.append(light)
    out.sort(key=lambda l: -l["peak"])
    return out


def _scene(w, h):
    """A three-light ball, plus two strays elsewhere in the frame."""
    f = np.zeros((h, w, 3), np.uint8)
    for dx, dy, col, rad in ((0, 0, (40, 180, 255), 7),
                             (30, 17, (60, 190, 250), 4),
                             (60, 34, (255, 120, 40), 6),
                             (400, 200, (255, 255, 255), 5),
                             (-200, 150, (20, 220, 220), 6)):
        cv2.circle(f, (w // 2 + dx, h // 2 + dy), rad, col, -1)
    return cv2.GaussianBlur(f, (9, 9), 0)


@pytest.mark.parametrize("size", [(320, 240), (640, 480), (1280, 720)])
def test_the_windowed_sample_agrees_with_the_full_frame_one(size):
    from vision.facing import lights_px

    frame = _scene(*size)
    fast, slow = lights_px(frame, min_v=150), _slow_lights(frame, 150)
    assert len(fast) == len(slow) > 0
    for a, b in zip(fast, slow):
        for key in ("x", "y", "area", "core_px", "peak", "hue", "sat"):
            if a[key] is None or b[key] is None:
                assert a[key] is b[key], key
            else:
                assert abs(a[key] - b[key]) < 1e-9, (key, a[key], b[key])
        assert max(abs(p - q) for p, q in zip(a["bgr"], b["bgr"])) < 1e-9


def test_a_light_at_the_frame_edge_still_samples_what_is_there():
    """The window is clipped at the border, so the annulus is a partial ring —
    it must still be sampled rather than skipped, and must match the
    full-frame answer, which clips in exactly the same place."""
    from vision.facing import lights_px

    f = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(f, (3, 120), 6, (40, 180, 255), -1)
    f = cv2.GaussianBlur(f, (9, 9), 0)
    fast, slow = lights_px(f, min_v=100), _slow_lights(f, 100)
    assert len(fast) == 1 and len(slow) == 1
    assert fast[0]["bgr"] is not None
    assert max(abs(p - q) for p, q in zip(fast[0]["bgr"], slow[0]["bgr"])) < 1e-9


# -- naming a cluster from its own colours --------------------------------

SIGS = {"red": {"hue": 172, "tol": 12},       # near the live bench value: wraps
        "green": {"hue": 60, "tol": 12},
        "blue": {"hue": 112, "tol": 10},
        "magenta": {"hue": 150, "tol": 10}}


def light(x, y, hue, sat=200, peak=250, bgr=None, core=4.0):
    return {"x": float(x), "y": float(y), "hue": float(hue), "sat": float(sat),
            "peak": float(peak), "core_px": core, "area": core ** 2 * math.pi,
            "bgr": bgr or (30.0, 30.0, 200.0)}


def tail(x, y):
    """The aim light: dim, and blue in BGR whatever its hue reads as."""
    return light(x, y, hue=112, sat=200, peak=140, bgr=(220.0, 40.0, 30.0))


def test_the_circular_hue_distance_handles_the_seam():
    from vision.facing import hue_err

    assert hue_err(172, 2) == 10.0, "a subtraction would say 170"
    assert hue_err(0, 179) == 1.0
    assert hue_err(45, 60) == 15.0
    assert hue_err(10, 10) == 0.0


def test_a_cluster_is_named_by_its_main_led():
    from vision.facing import identify_cluster

    got, why = identify_cluster([light(100, 100, 60), tail(80, 100)], SIGS)
    assert why is None, why
    assert got["color"] == "green"
    assert got["front"]["x"] == 100 and got["back"]["x"] == 80


def test_a_hue_across_the_seam_is_still_matched():
    """Red is at 172 on this bench. A ball reading 3 is 11 away, not 169."""
    from vision.facing import identify_cluster

    got, why = identify_cluster([light(100, 100, 3), tail(80, 100)], SIGS)
    assert why is None, why
    assert got["color"] == "red"


def test_a_washed_out_light_has_no_colour_to_read():
    """The taillight being perfectly readable must not vouch for a main LED
    that has blown out to white. The refusal has to name the exposure, not
    the hue matching, because those are different knobs."""
    from vision.facing import identify_cluster

    got, why = identify_cluster([light(100, 100, 60, sat=5), tail(80, 100)], SIGS)
    assert got is None
    assert "blowing out" in why and "saturation 5" in why


def test_a_hue_between_two_slots_is_refused_rather_than_guessed():
    """Barely closer than the next slot is how a robot gets called by its
    neighbour's name every few frames.

    These two signatures are the real pair from this bench: yellow at 60+-9 and
    cyan at 62+-10, eighteen units of overlap against a documented minimum
    separation of 22. One ball was detected as two robots at the identical
    pixel because of it.
    """
    from vision.facing import identify_cluster

    tight = {"yellow": {"hue": 60, "tol": 9}, "cyan": {"hue": 62, "tol": 10}}
    for hue in (52, 58, 61, 64, 70):
        got, why = identify_cluster([light(100, 100, hue), tail(80, 100)], tight)
        assert got is None, f"hue {hue} was named {got and got['color']}"
        assert "too close to call" in why or "matches" in why

    # Refusing at EVERY hue is the correct answer for this pair, and worth
    # stating rather than working around: two hues two apart cannot be told
    # from each other by any rule, so the fix is a better palette and not a
    # cleverer matcher. `optimise hues` on the COLOUR tab is that fix.
    apart = {"yellow": {"hue": 27, "tol": 9}, "cyan": {"hue": 90, "tol": 10}}
    got, _ = identify_cluster([light(100, 100, 30), tail(80, 100)], apart)
    assert got is not None and got["color"] == "yellow", "separable still names"


def test_a_hue_matching_nothing_is_refused():
    from vision.facing import identify_cluster

    got, why = identify_cluster([light(100, 100, 30), tail(80, 100)], SIGS)
    assert got is None
    assert "no light matches" in why


def test_a_blue_robot_with_a_blue_taillight_is_reported_not_guessed():
    """The one genuine ambiguity: `set_back_led` is blue and so is the robot,
    so which end is the front cannot be told apart."""
    from vision.facing import identify_cluster

    got, why = identify_cluster(
        [light(100, 100, 112, bgr=(220.0, 40.0, 30.0)), tail(80, 100)], SIGS)
    assert got is None
    assert "BLUE" in why and "taillight" in why


def test_the_tag_settles_which_end_is_the_front():
    from vision.facing import heading_from_lights

    group = [light(140, 100, 60), light(120, 100, 60, sat=10), tail(100, 100)]
    got, why = heading_from_lights(group, signatures=SIGS)
    assert why is None, why
    assert got["by_tag"] and got["color"] == "green"
    assert abs(got["deg"] - 0.0) < 1.0, got["deg"]     # tail left, LED right


def test_the_tag_reverses_the_arrow_when_the_ball_is_turned_round():
    from vision.facing import heading_from_lights

    a, _ = heading_from_lights([light(140, 100, 60), tail(100, 100)],
                               signatures=SIGS)
    b, _ = heading_from_lights([light(100, 100, 60), tail(140, 100)],
                               signatures=SIGS)
    assert abs((a["deg"] - b["deg"] + 180) % 360 - 180) > 150


def test_without_signatures_it_falls_back_to_the_blue_end():
    from vision.facing import heading_from_lights

    group = [light(140, 100, 60), tail(100, 100)]
    got, why = heading_from_lights(group)
    assert why is None and not got["by_tag"] and got["by_colour"]
    assert got["color"] is None


def test_a_main_led_read_in_the_middle_names_but_does_not_aim():
    """A tag on the MIDDLE light says which robot it is and nothing about
    which way it points. Using it as the front would draw the arrow across the
    ball instead of along it."""
    from vision.facing import heading_from_lights

    group = [light(100, 100, 20, sat=5, peak=255),       # unreadable end
             light(120, 100, 60),                        # the tag, in the middle
             tail(140, 100)]
    got, why = heading_from_lights(group, signatures=SIGS)
    assert why is None, why
    assert got["color"] == "green", "still named"
    assert not got["by_tag"], "but the arrow did not come from it"
    assert got["front"] is not group[1]


def test_the_reason_survives_when_naming_fails_but_aiming_works():
    from vision.facing import heading_from_lights

    group = [light(140, 100, 30), tail(100, 100)]        # hue matches nothing
    got, why = heading_from_lights(group, signatures=SIGS)
    assert why is None, "a heading is still readable"
    assert got["color"] is None
    assert got["tag_why"] and "no light matches" in got["tag_why"]


def test_the_margin_is_absolute_and_not_proportional():
    """Hue noise is a few units whatever the distances are, so the rule has to
    be a gap and not a ratio. A ratio accepts 2-away over 4-away because 4 is
    twice 2 — and a ball sitting still, wobbling by four, changes its name."""
    from vision.facing import identify_cluster

    tight = {"yellow": {"hue": 60, "tol": 9}, "cyan": {"hue": 62, "tol": 10}}
    for hue in (58, 64):
        got, _ = identify_cluster([light(100, 100, hue), tail(80, 100)], tight)
        assert got is None, f"hue {hue} named {got and got['color']} on a 2-unit gap"


def test_a_readable_ring_survives_the_value_clip():
    """The clip removes floor from the annulus. It must not remove the halo:
    a light with a clean ring still has to report a colour."""
    from vision.facing import lights_px

    f = np.zeros((160, 160, 3), np.uint8)
    cv2.circle(f, (80, 80), 10, (0, 200, 60), -1)
    f = cv2.GaussianBlur(f, (21, 21), 0)
    got = lights_px(f, min_v=120)
    assert got and got[0]["hue"] is not None
    assert got[0]["ring_px"] > 0
    assert abs(got[0]["hue"] - 51) < 6, got[0]["hue"]


def test_the_radial_profile_shows_where_the_colour_lives():
    """The instrument for choosing ANNULUS, which is otherwise a guess. On a
    bloomed LED it has to show three regions: a white core where saturation
    collapses, a coloured halo, and floor where value falls away."""
    from vision.facing import radial_profile

    s = 160
    ys, xs = np.ogrid[0:s, 0:s]
    d = np.sqrt((xs - 80) ** 2 + (ys - 80) ** 2)
    lit = np.exp(-(d ** 2) / (2 * 9.0 ** 2))
    bloom = np.exp(-(d ** 2) / (2 * 5.0 ** 2))
    col = np.array((0, 200, 60), np.float32) / 200.0
    img = np.clip(lit[..., None] * col * 255 * 3.2 + bloom[..., None] * 255,
                  0, 255).astype(np.uint8)

    prof = radial_profile(img, (80, 80), 34, step=3.0)
    assert len(prof) > 8
    core, halo, floor = prof[0], prof[4], prof[-2]
    assert core["sat"] < 60, "the core must read as white"
    assert halo["sat"] > 200 and halo["val"] > 100, "the halo carries the hue"
    assert abs(halo["hue"] - 51) < 5, halo["hue"]
    assert floor["val"] < 30, "and it must show where the light runs out"


def test_a_contested_hue_names_the_pair_that_is_fighting():
    """The fix for a contested hue is a better-separated palette, and that is
    only obvious if the message says which two slots are competing and by how
    much. This is the live magenta/red pair: 11 apart, against a documented
    minimum separation of 22."""
    from vision.facing import identify_cluster

    close = {"magenta": {"hue": 161, "tol": 5}, "red": {"hue": 172, "tol": 5}}
    got, why = identify_cluster([light(100, 100, 166), tail(80, 100)], close)
    assert got is None
    assert "magenta" in why and "red" in why
    assert "11 apart" in why
    assert "optimise hues" in why


def test_an_unmatched_hue_says_what_the_nearest_slot_was():
    """'No match' leaves you guessing whether the ball is the wrong colour or
    the signature is in the wrong place. The distance tells you which."""
    from vision.facing import identify_cluster

    got, why = identify_cluster([light(100, 100, 90), tail(80, 100)], SIGS)
    assert got is None
    assert "nearest is" in why and "away" in why


def test_a_light_does_not_sample_its_neighbours_colour():
    """The lights on one shell are close — about 13px apart on this arena —
    and an annulus of 2.4 core radii around a 7px core reaches 17px, straight
    over the light next door. Reading a neighbour's colour and calling it your
    own reverses front and back, which is a heading exactly 180 degrees wrong.
    """
    from vision.facing import _blueness, lights_px

    f = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(f, (150, 120), 8, (40, 40, 255), -1)      # red, on the left
    cv2.circle(f, (176, 120), 6, (255, 60, 30), -1)      # blue, 26px away
    f = cv2.GaussianBlur(f, (11, 11), 0)

    got = sorted(lights_px(f, min_v=140), key=lambda l: l["x"])
    assert len(got) == 2
    red, blue = got
    assert _blueness(blue) > _blueness(red) + 0.2, (
        f"blueness {_blueness(red):.2f} vs {_blueness(blue):.2f} — the rings "
        "are reading each other")
    assert red["bgr"][2] > red["bgr"][0], "the red light must read red"
    assert blue["bgr"][0] > blue["bgr"][2], "the blue light must read blue"


def test_two_equally_bright_ends_are_refused_rather_than_guessed():
    """Both LEDs clip to 255 on a bright frame, so the brightness fallback can
    find no difference at all. It was picking whichever end came first along
    the axis and reporting it with confidence zero — a heading 180 out, which
    is worse than no heading."""
    from vision.facing import heading_from_lights

    flat = (120.0, 120.0, 120.0)        # neither end blue, both identical
    group = [light(100, 100, 60, bgr=flat, peak=255),
             light(140, 100, 60, bgr=flat, peak=255)]
    got, why = heading_from_lights(group)
    assert got is None
    assert "front from the back" in why


def test_a_clear_brightness_difference_still_answers():
    """The refusal above is a floor, not a blanket ban on the fallback."""
    from vision.facing import heading_from_lights

    flat = (120.0, 120.0, 120.0)
    group = [light(100, 100, 60, bgr=flat, peak=120),
             light(140, 100, 60, bgr=flat, peak=255)]
    got, why = heading_from_lights(group)
    assert why is None and got is not None
    assert abs(got["deg"] - 0.0) < 1.0, "the brighter end is the front"


def test_the_centre_is_the_midpoint_of_the_tag_pair_not_of_all_three():
    """The two tag LEDs straddle the centre of the shell; the tail sits behind
    both. Averaging the tail in with them is the ~7mm backward bias that
    `blob_test.py` measured, and it is the whole reason that app runs with the
    taillight off."""
    from vision.facing import heading_from_lights

    # tail at 100, tag LEDs at 120 and 140 — the ball is at 130, not at 120.
    group = [tail(100, 100), light(120, 100, 60), light(140, 100, 60)]
    got, why = heading_from_lights(group, signatures=SIGS)
    assert why is None, why
    assert got["centre_from"] == "tag pair"
    assert abs(got["centre"][0] - 130.0) < 1e-6, got["centre"]
    assert abs(got["lights_centre"][0] - 120.0) < 1e-6, "the raw centroid"


def test_the_centre_does_not_lean_with_the_heading():
    """The bias the tail introduces ROTATES with the robot, so no constant can
    cancel it. Turn the ball round about a fixed centre and the answer has to
    stay put."""
    from vision.facing import heading_from_lights

    cx, cy = 200.0, 200.0
    seen = []
    for dx, dy in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
        # tag LEDs 10px either side of the centre, tail 30px behind it
        group = [light(cx + dx * 10, cy + dy * 10, 60),
                 light(cx - dx * 10, cy - dy * 10, 60),
                 tail(cx - dx * 30, cy - dy * 30)]
        got, why = heading_from_lights(group, signatures=SIGS)
        assert why is None, why
        seen.append(got["centre"])
    for x, y in seen:
        assert abs(x - cx) < 1e-6 and abs(y - cy) < 1e-6, seen


def test_one_tag_led_reports_that_its_centre_is_biased():
    """With a single tag LED there is nothing to take a midpoint with. The
    answer is that light's own position, which sits forward of the ball — and
    saying so is the difference between a known offset and a silent one."""
    from vision.facing import heading_from_lights

    got, why = heading_from_lights([tail(100, 100), light(140, 100, 60)],
                                   signatures=SIGS)
    assert why is None, why
    assert got["centre_from"] != "tag pair"
    assert abs(got["centre"][0] - 140.0) < 1e-6, "the LED, not the midpoint"


def test_confidence_says_which_of_its_two_terms_is_limiting():
    """A ball 20px across and perfectly tagged scores the same 0.5 as one 60px
    across whose ends are nearly indistinguishable. Those want opposite
    actions — move the camera, or fix the colour — so the product alone is not
    an actionable number."""
    from vision.facing import SPAN_FOR_FULL_CONF, heading_from_lights

    # Short baseline, perfect ends: the GEOMETRY is the limit.
    near = [tail(100, 100), light(110, 100, 60), light(120, 100, 60)]
    got, why = heading_from_lights(near, signatures=SIGS)
    assert why is None, why
    assert got["conf_ends"] == 1.0, "the ends were never in doubt"
    assert got["conf_span"] < 1.0, "the span is what is holding it back"
    assert got["conf"] == pytest.approx(got["conf_span"] * got["conf_ends"],
                                        abs=1e-3)

    # Long baseline: the same ends now score full marks.
    far = [tail(100, 100), light(130, 100, 60), light(160, 100, 60)]
    wide, why = heading_from_lights(far, signatures=SIGS)
    assert why is None, why
    assert wide["conf_span"] == 1.0, f"{wide['span_px']}px should reach"
    assert wide["conf"] > got["conf"]


def test_the_span_term_is_linear_up_to_the_threshold():
    """The angular error of a two-point bearing goes as 1/span, so half the
    separation is twice the noise and is scored as half the confidence."""
    from vision.facing import SPAN_FOR_FULL_CONF, heading_from_lights

    half = SPAN_FOR_FULL_CONF / 2.0
    group = [tail(100, 100), light(100 + half / 2, 100, 60),
             light(100 + half, 100, 60)]
    got, why = heading_from_lights(group, signatures=SIGS)
    assert why is None, why
    assert got["conf_span"] == pytest.approx(0.5, abs=0.02)


def big(x, y, hue, peak=255, core=4.0, **kw):
    """A light with an explicit size, for the cases where SIZE is the signal."""
    return light(x, y, hue, peak=peak, core=core, **kw)


def test_clipped_cores_are_told_apart_by_size_not_by_height():
    """The live failure. On a short shutter the tag pair and the tail both
    read 255, so a peak comparison is a coin toss — measured at 46-61% correct
    on rendered balls. Clipping makes a bright blob WIDER, so the light it
    carries is still there to be counted."""
    from vision.facing import heading_from_lights

    flat = (120.0, 120.0, 120.0)
    # Tail on the left: same clipped peak, half the size.
    group = [big(100, 100, 60, peak=255, core=3.0, bgr=flat),
             big(140, 100, 60, peak=255, core=4.3, bgr=flat)]
    got, why = heading_from_lights(group)
    assert why is None, why
    assert got["back"] is group[0], "the smaller blob is the tail"
    assert abs(got["deg"] - 0.0) < 1.0, "heading runs tail -> tags"
    assert got["conf_ends"] > 0.5, "and it is not a marginal call"


def test_a_merged_tag_pair_is_the_centre_not_a_degraded_single_led():
    """Two tag LEDs close enough to fuse straddle the shell symmetrically, so
    their merged centroid IS the centre. Reporting that as "one LED" would
    send somebody hunting an offset that is not there."""
    from vision.facing import heading_from_lights

    flat = (120.0, 120.0, 120.0)
    group = [big(100, 100, 60, peak=255, core=3.0, bgr=flat),   # tail
             big(140, 100, 60, peak=255, core=4.3, bgr=flat)]   # fused pair
    got, why = heading_from_lights(group)
    assert why is None, why
    assert got["centre_from"] == "tag pair, merged"
    assert abs(got["centre"][0] - 140.0) < 1e-6, "the fused blob IS the centre"


def test_a_lone_tag_led_is_still_reported_as_offset():
    """The guard must not call every two-blob group a merged pair: one tag LED
    the same size as the tail really is half a spacing ahead of the centre."""
    from vision.facing import heading_from_lights

    group = [light(100, 100, 60, peak=140), light(140, 100, 60, peak=255)]
    got, why = heading_from_lights(group)
    assert why is None, why
    assert got["centre_from"].startswith("one tag LED")


def test_size_and_height_each_carry_the_call_when_the_other_cannot():
    """The product degrades to whichever factor still has signal."""
    from vision.facing import heading_from_lights

    flat = (120.0, 120.0, 120.0)
    # Heights clipped level, sizes differ.
    a, _ = heading_from_lights([big(100, 100, 60, peak=255, core=3.0, bgr=flat),
                                big(140, 100, 60, peak=255, core=4.3, bgr=flat)])
    # Sizes equal, heights differ.
    b, _ = heading_from_lights([light(100, 100, 60, peak=120, bgr=flat),
                                light(140, 100, 60, peak=255, bgr=flat)])
    assert a is not None and b is not None
    for got in (a, b):
        assert abs(got["deg"] - 0.0) < 1.0


def merged(deg, span=10.0, tag=3.0, core=2.8, size=120, over=3.0):
    """A ball at the scale this rig actually runs at: three lights so close,
    and driven so hard, that they bloom into ONE region."""
    img = np.zeros((size, size), np.float64)
    r = math.radians(deg)
    ux, uy = math.cos(r), math.sin(r)
    yy, xx = np.mgrid[0:size, 0:size]
    for t, gain in ((-span / 2, 0.6), (-tag / 2, 1.0), (+tag / 2, 1.0)):
        px, py = 60 + ux * t, 60 + uy * t
        img += 255.0 * gain * over * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * core ** 2))
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)


def test_the_peak_reader_really_does_refuse_a_merged_ball():
    """The premise. At this scale the lights fuse into one region, and a
    reader that needs two maxima has nothing to work with — which freezes the
    heading until a frame happens to separate."""
    from vision.facing import cluster_px, heading_from_lights, lights_px

    spots = lights_px(merged(20.0), min_v=120)
    assert len(spots) == 1, "the premise is that they merged"
    got, why = heading_from_lights(cluster_px(spots, 40)[0])
    assert got is None and "one light" in why


def test_a_merged_blob_still_has_an_axis():
    from vision.facing import blob_axis_px, cluster_px, lights_px

    for true in (0.0, 37.0, 96.0, 214.0, 300.0):
        f = merged(true)
        spot = cluster_px(lights_px(f, min_v=120), 40)[0][0]
        got = blob_axis_px(f, spot)
        assert got is not None, f"no axis at {true}deg"
        axis, elong = got
        read = math.degrees(math.atan2(axis[1], axis[0])) % 180.0
        # An axis is a line, so 0 and 180 are the same answer: compare the
        # short way round on a 180-degree circle.
        off = abs(((read - (true % 180.0) + 90.0) % 180.0) - 90.0)
        assert off < 4.0, f"axis {read:.1f} for a ball at {true}"
        assert elong > 1.25


def test_a_round_blob_is_refused_rather_than_given_a_noise_axis():
    """Second moments will happily hand you an axis made of nothing."""
    from vision.facing import blob_axis_px

    f = np.zeros((80, 80, 3), np.uint8)
    cv2.circle(f, (40, 40), 6, (255, 255, 255), -1)
    f = cv2.GaussianBlur(f, (5, 5), 0)
    spot = {"x": 40.0, "y": 40.0, "core_px": 6.0}
    assert blob_axis_px(f, spot) is None


def test_the_axis_tracks_a_slow_turn_without_freezing():
    """The live complaint: it took a big rotation before anything moved."""
    from vision.facing import blob_axis_px, cluster_px, lights_px

    seen = []
    for i in range(40):
        f = merged(i * 1.0)
        spot = cluster_px(lights_px(f, min_v=120), 40)[0][0]
        got = blob_axis_px(f, spot)
        assert got is not None, f"refused at {i}deg"
        seen.append(math.degrees(math.atan2(got[0][1], got[0][0])) % 180.0)
    steps = [abs(((b - a + 90) % 180) - 90) for a, b in zip(seen, seen[1:])]
    frozen = sum(1 for v in steps if v < 0.02)
    assert frozen == 0, f"{frozen} frames read no change at all"
    assert abs(np.mean(steps) - 1.0) < 0.4, "one degree in, one degree out"

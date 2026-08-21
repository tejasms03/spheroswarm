"""Colour and blink, by code."""

from vision.config import COLORS

from .result import fail, ok

# The six tracked hues as RGB, plus the handful of words a model reaches for.
NAMED = {
    "red": (255, 40, 40),
    "yellow": (255, 210, 40),
    "green": (60, 220, 90),
    "cyan": (60, 220, 235),
    "blue": (60, 110, 245),
    "magenta": (230, 70, 220),
    "white": (255, 255, 255),
    "orange": (255, 140, 30),
    "purple": (170, 80, 240),
    "pink": (255, 130, 190),
    "off": (0, 0, 0),
    "black": (0, 0, 0),
}

BLINK_PATTERNS = ("none", "slow", "fast", "pulse")


def parse_color(value):
    """A name or an [r, g, b] triple. Returns (rgb, error)."""
    if value is None:
        return None, "no colour given"
    if isinstance(value, str):
        key = value.strip().lower()
        if key in NAMED:
            return NAMED[key], None
        return None, (f"unknown colour {value!r} — use one of "
                      f"{', '.join(sorted(NAMED))}, or an [r, g, b] triple")
    try:
        seq = list(value)
    except TypeError:
        return None, f"colour must be a name or [r, g, b], got {value!r}"
    if len(seq) != 3:
        return None, f"an [r, g, b] colour needs three values, got {len(seq)}"
    out = []
    for v in seq:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None, f"colour channels must be whole numbers, got {seq!r}"
        out.append(max(0, min(255, n)))
    return tuple(out), None


def set_led(ctx, color, codes=None, blink=None):
    """Set colour (and optionally a blink pattern) on some or all robots.

    Note the tracker keys on the robot's hue, so changing a tracked robot's
    colour is a real decision, not just decoration — it is reported back.
    """
    rgb, err = parse_color(color)
    if err:
        return fail(ctx, err)

    if blink is not None:
        blink = str(blink).lower()
        if blink not in BLINK_PATTERNS:
            return fail(ctx, f"unknown blink pattern {blink!r} — use one of "
                             f"{', '.join(BLINK_PATTERNS)}")
        if blink == "none":
            blink = None

    if codes is None:
        codes = ctx.fleet.codes
    elif isinstance(codes, str):
        codes = [codes]

    unknown = [c for c in codes if c not in ctx.fleet.handles]
    if unknown:
        return fail(ctx, f"unknown robot code(s): {', '.join(sorted(unknown))}")

    retracked = []
    for c in codes:
        h = ctx.fleet.handles[c]
        h.set_led(rgb, blink)
        if h.kind == "real" and h.color in COLORS:
            retracked.append(c)

    result = ok(ctx, colored=list(codes), rgb=list(rgb), blink=blink)
    if retracked:
        result["note"] = (
            f"{', '.join(retracked)} are tracked by hue; the camera still keys on "
            "their roster colour, so a changed LED can cost tracking")
    return result

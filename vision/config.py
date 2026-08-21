"""Shared configuration for the vision stack."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "calib"
CALIB.mkdir(exist_ok=True)

ARENA_CM = 200.0          # square arena side, in centimetres
FPS_TARGET = 30

# Hues spaced far enough apart to survive indoor lighting. OpenCV hue is 0-179.
# Red wraps, so it gets two ranges and is handled specially in detect.py.
COLORS = {
    "red":     {"hue": 0,   "tol": 8,  "draw": (60, 60, 235)},
    "yellow":  {"hue": 27,  "tol": 9,  "draw": (60, 210, 235)},
    "green":   {"hue": 60,  "tol": 12, "draw": (90, 200, 90)},
    "cyan":    {"hue": 90,  "tol": 10, "draw": (220, 200, 60)},
    "blue":    {"hue": 112, "tol": 10, "draw": (220, 120, 60)},
    "magenta": {"hue": 150, "tol": 10, "draw": (200, 90, 200)},
}

DEFAULT_THRESH = {
    "s_min": 90,       # saturation floor — rejects white walls and blown-out cores
    "v_min": 70,       # value floor — rejects shadows
    "min_area": 60,    # px, smallest accepted blob
    "max_area": 20000,
    "blur": 5,
}

# Which keys of a colour signature may be overridden per colour. The hue of a
# lit magenta shell and the hue of a lit green one sit in different parts of
# the space and blow out differently, so one global saturation floor is a
# compromise that suits neither. `calib/colors.json` holds the per-colour
# answer; COLORS above stays the canonical list of names, because the roster
# validates against it and a missing calibration file must not invalidate a
# roster.
SIGNATURE_KEYS = ("hue", "tol", "s_min", "v_min", "min_area", "max_area",
                  "led_value")

# What the ball is TOLD to glow, derived from the hue we intend to detect —
# rather than kept in a second table beside it. Two tables is how a robot ends
# up lit one colour and looked for as another: the LED table was edited, the
# detector's was not, and nothing complains because each is internally
# consistent. A signature tuned on the wheel moves both at once because there
# is only one of them.
LED_VALUE = 255             # a Sphero's LED is the light source; run it bright


def led_rgb(hue, saturation=255, value=LED_VALUE):
    """OpenCV hue (0-179) -> an (r, g, b) the robot can be told to glow."""
    import numpy as np
    import cv2
    px = np.uint8([[[int(hue) % 180, int(saturation), int(value)]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return (int(r), int(g), int(b))


def led_for(name, colors=None):
    """The RGB for a named slot, from whatever hue that slot currently holds."""
    spec = (colors or load_signatures()).get(name)
    if not spec:
        return (255, 255, 255)
    return led_rgb(spec.get("hue", 0), value=spec.get("led_value", LED_VALUE))


def load(name, fallback=None):
    p = CALIB / f"{name}.json"
    if not p.exists():
        return fallback
    try:
        return json.loads(p.read_text())
    except Exception:
        return fallback


def save(name, data):
    """Write a calibration file atomically. Raises only if the DIRECTORY is shut.

    Written to a neighbour and renamed over the target rather than opened in
    place, for two reasons. A half-written JSON is a calibration file that
    parses as nothing and takes the next session with it, and renaming cannot
    produce one. And rename needs permission on the DIRECTORY, not on the file
    — which is what makes a file left behind root-owned by an old `sudo` run
    replaceable instead of a permanent wall.
    """
    import os
    import tempfile

    CALIB.mkdir(exist_ok=True)
    target = CALIB / f"{name}.json"
    fd, tmp = tempfile.mkstemp(dir=str(CALIB), prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=2))
        # mkstemp makes it 0600; calibration is not a secret and the next tool
        # to read it may not be running as the same user.
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def writable(name):
    """Can this calibration file be written? Checked before work, not after.

    A four-click arena pick that fails on the fourth click has wasted all four,
    and the trainer is by then standing at the far corner of the arena.
    """
    import os
    import tempfile

    try:
        CALIB.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(CALIB), prefix=".probe.")
        os.close(fd)
        os.unlink(tmp)
    except Exception as e:
        return f"cannot write into {CALIB}: {e}"
    return None


def load_signatures():
    """COLORS with any calibrated per-colour overrides merged over the top.

    Never raises and never invents a colour: a signature naming a hue that is
    not in COLORS is ignored, so a stale calibration file cannot introduce a
    robot colour the roster would reject.
    """
    out = {name: dict(spec) for name, spec in COLORS.items()}
    saved = load("colors", {}) or {}
    if not isinstance(saved, dict):
        return out
    for name, spec in saved.items():
        if name not in out or not isinstance(spec, dict):
            continue
        for key in SIGNATURE_KEYS:
            if key in spec:
                try:
                    out[name][key] = int(spec[key])
                except (TypeError, ValueError):
                    pass
    return out


def save_signatures(colors):
    """Write only the tuning, not the draw colours — those are not calibration."""
    out = {}
    for name, spec in colors.items():
        if name not in COLORS:
            continue
        out[name] = {k: int(spec[k]) for k in SIGNATURE_KEYS if k in spec}
    return save("colors", out)

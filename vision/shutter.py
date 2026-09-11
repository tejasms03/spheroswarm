#!/usr/bin/env python3
"""A real exposure slider for the C920, going around AVFoundation.

    python -m vision.shutter --list          # what uvc-util can see
    python -m vision.shutter                 # the slider

`avcam.py` concluded there is no shutter dial on this rig. That conclusion is
true of AVFoundation and false of the camera. macOS does not plumb UVC exposure
through `AVCaptureDevice` for external cameras, so `Custom` reports unsupported
and OpenCV's `CAP_PROP_EXPOSURE` -- which is AVFoundation underneath -- accepts
writes and drops them. The C920 itself has had manual exposure all along; it is
reachable by speaking UVC over IOKit instead of asking Apple's capture layer.

So the frames come from OpenCV and the CONTROL goes out of band, through
uvc-util. Two handles on one camera, which is worth stating plainly because it
is the part that surprises people later: OpenCV never learns the exposure
changed, and `cap.get(CAP_PROP_EXPOSURE)` stays as wrong as it ever was.

Values are UVC `exposure-time-abs`, in units of 100 microseconds -- so 156 is
15.6 ms. The slider maps its travel logarithmically, because the useful end of
this range for a black floor and bright LED cores is the short end, and half a
linear slider spent between 1000 and 2047 is half a slider wasted.

The brightness readout is not decoration. A slider that moves while the picture
does not is this project's oldest failure, and the only thing that tells the
two apart is whether the mean moved -- so it is measured, and the verdict is
printed on the frame.
"""

import argparse
import shutil
import subprocess
import time

import cv2
import numpy as np

from . import config
from .expose import FLICKER, USABLE

# UVC auto-exposure-mode enum. The C920 offers 1 and 8 only -- there is no
# plain "auto" here; 8 is aperture-priority, which is the auto you came from.
MANUAL = 1
APERTURE_PRIORITY = 8

# exposure-time-abs, in 100us units. The C920 reports 3..2047, default 250.
EXP_MIN, EXP_MAX = 3, 2047
EXP_DEFAULT = 250
STEPS = 100
CHALK = (228, 240, 248)
DIM = (204, 179, 143)
CORAL = (107, 107, 255)
GREEN = (90, 200, 90)


def tool():
    """The uvc-util binary, or None. Homebrew does not carry it; it is built
    from jtfrey/uvc-util, so the usual `brew install` advice is wrong here."""
    found = shutil.which("uvc-util")
    if found:
        return found
    for p in ("/usr/local/bin/uvc-util", "/opt/homebrew/bin/uvc-util",
              str(config.ROOT / "tools" / "uvc-util")):
        if shutil.which(p):
            return p
    return None


def run(binary, *args):
    """One uvc-util call. Returns (ok, output)."""
    try:
        r = subprocess.run([binary, *args], capture_output=True, text=True,
                           timeout=4.0)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out.strip()


def devices(binary):
    ok, out = run(binary, "-I", "0", "-d")
    if not ok:
        ok, out = run(binary, "-l")
    return out


def slider_to_us(pos):
    """Slider travel -> exposure-time-abs, logarithmically."""
    frac = max(0, min(STEPS, pos)) / STEPS
    return int(round(EXP_MIN * (EXP_MAX / EXP_MIN) ** frac))


def us_to_slider(value):
    import math
    value = max(EXP_MIN, min(EXP_MAX, value))
    return int(round(STEPS * math.log(value / EXP_MIN)
                     / math.log(EXP_MAX / EXP_MIN)))


def set_exposure(binary, index, value):
    return run(binary, "-I", str(index), "-s", f"exposure-time-abs={value}")


def set_mode(binary, index, mode):
    return run(binary, "-I", str(index), "-s", f"auto-exposure-mode={mode}")


# Focus goes the same way as exposure and for the same reason: AVFoundation
# accepts `CAP_PROP_AUTOFOCUS` for an external camera and drops it, so the
# "af off" button was as dead as the exposure slider was. On a dark floor a
# hunting autofocus is worse than a wrong exposure -- it smears two LEDs a few
# pixels apart into one blob, and `facing.py` then refuses every other frame.
FOCUS_MIN, FOCUS_MAX = 0, 255
FOCUS_DEFAULT = 0           # 0 is infinity on this camera, which is the floor


def set_autofocus(binary, index, on):
    return run(binary, "-I", str(index), "-s",
               f"auto-focus={1 if on else 0}")


def set_focus(binary, index, value):
    return run(binary, "-I", str(index), "-s", f"focus-abs={int(value)}")


def dial(camera=0, uvc_index=0, size=None):
    binary = tool()
    if binary is None:
        print("uvc-util not found, and it is the only thing on macOS that can\n"
              "actually move this camera's shutter. Build it:\n\n"
              "    git clone https://github.com/jtfrey/uvc-util\n"
              "    cd uvc-util && make\n"
              "    cp uvc-util /usr/local/bin/\n")
        return 1

    cap = cv2.VideoCapture(camera)
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        print(f"camera {camera} would not open. On macOS, grant camera access "
              "to your terminal in System Settings > Privacy & Security.")
        return 1

    ok, msg = set_mode(binary, uvc_index, MANUAL)
    if not ok:
        print(f"could not put the camera in manual exposure: {msg}\n"
              f"If it named the wrong device, try --uvc-index (see --list).")
        cap.release()
        return 1

    saved = config.load("exposure", {}) or {}
    start = int(saved.get("exposure_time_abs", 156))
    win = "exposure — a auto, s save, q quit"
    cv2.namedWindow(win)
    cv2.createTrackbar("exposure", win, us_to_slider(start), STEPS,
                       lambda v: None)
    set_exposure(binary, uvc_index, start)

    # (value -> mean) as the slider is moved, which is what decides whether the
    # control is real. One sample proves nothing; the SPREAD across values does.
    seen = {}
    last_value = None
    manual = True

    while True:
        got, frame = cap.read()
        if not got or frame is None:
            break
        value = slider_to_us(cv2.getTrackbarPos("exposure", win))
        if value != last_value and manual:
            set_exposure(binary, uvc_index, value)
            last_value = value
            time.sleep(0.05)          # the change lands a frame or two later

        mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        if manual and last_value is not None:
            seen[last_value] = mean

        vis = frame.copy()
        spread = (max(seen.values()) - min(seen.values())) if len(seen) > 1 else 0.0
        if len(seen) < 2:
            verdict, col = "move the slider to prove it is live", DIM
        elif spread >= USABLE:
            verdict, col = f"LIVE — {spread:.0f} counts of range", GREEN
        elif spread >= FLICKER:
            verdict, col = (f"responds, but only {spread:.0f} counts", CORAL)
        else:
            verdict, col = "slider is not moving the picture", CORAL

        head = (f"{value:4d} x100us = {value / 10:6.1f} ms"
                if manual else "AUTO (aperture priority)")
        cv2.putText(vis, head, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CHALK, 2)
        cv2.putText(vis, f"mean {mean:5.1f}", (14, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CHALK, 1)
        cv2.putText(vis, verdict, (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        cv2.putText(vis, "a auto   s save   q quit", (14, vis.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1)
        cv2.imshow(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("a"):
            manual = not manual
            set_mode(binary, uvc_index, MANUAL if manual else APERTURE_PRIORITY)
            if manual:
                last_value = None     # force a re-set on the next frame
        elif k == ord("s"):
            config.save("exposure", {"exposure_time_abs": value,
                                     "auto_exposure_mode": MANUAL,
                                     "uvc_index": uvc_index})
            print(f"saved exposure {value} (={value / 10:.1f} ms) to calib/exposure.json")
        elif k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyWindow(win)
    if len(seen) > 1 and (max(seen.values()) - min(seen.values())) < FLICKER:
        print("The slider did not move the picture. Check --list: uvc-util is "
              "probably pointed at a different camera than OpenCV.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="OpenCV index")
    p.add_argument("--uvc-index", type=int, default=0,
                   help="uvc-util's own index — not always the OpenCV one")
    p.add_argument("--size", default=None, help="e.g. 1920x1080")
    p.add_argument("--list", action="store_true", help="what uvc-util can see")
    a = p.parse_args(argv)

    binary = tool()
    if a.list:
        if binary is None:
            print("uvc-util not found — see the build steps in the module docstring")
            return 1
        print(devices(binary))
        return 0
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    return dial(a.camera, a.uvc_index, size)


if __name__ == "__main__":
    raise SystemExit(main())


class Dial:
    """The camera's shutter, as a thing an app can hold.

    `dial()` above is a bench in its own right; this is the same control with
    no window around it, so an app that already has a slider can drive the
    shutter without growing a second copy of the three things that are easy to
    get wrong here.

    MANUAL MODE IS RE-ASSERTED, not set once at startup. In aperture priority
    the camera owns the shutter and drops these writes without complaint --
    the value reads back unchanged, which is indistinguishable from a dead
    dial. Anything can put the camera back: unplugging it, another app opening
    it, `auto()` on this object. So the mode is checked before every write and
    the write is abandoned if it cannot be taken.

    WRITES ARE RATE LIMITED. Each one is a uvc-util subprocess and a dragged
    slider emits an event per pixel; unthrottled that is dozens of processes a
    second competing with the render loop, which stutters the window during
    exactly the adjustment you are trying to judge by eye. `set` only records
    what is wanted and `push` -- called once per rendered frame -- is what
    reaches the camera.

    OPENCV NEVER LEARNS. The frames come from `VideoCapture` and this control
    goes out of band, so `cap.get(CAP_PROP_EXPOSURE)` stays as wrong as it
    ever was. `self.value` is the only place that knows.
    """

    PUSH_S = 0.08           # about twelve writes a second
    WHAT = "exposure"
    SAVE_KEY, SAVE_FIELD = "exposure", "exposure_time_abs"
    VMIN, VMAX, VDEFAULT = EXP_MIN, EXP_MAX, EXP_DEFAULT

    def __init__(self, index=0, value=None):
        self.binary = tool()
        self.index = int(index)
        saved = config.load(self.SAVE_KEY, {}) or {}
        if value is None:
            value = saved.get(self.SAVE_FIELD, self.VDEFAULT)
        self.value = self.clamp(value)
        self.manual = False
        self.pending = self.value      # push once at startup, so the camera
        self.pushed = 0.0              # actually starts where the slider says
        self.took = None

    @classmethod
    def clamp(cls, value):
        return max(cls.VMIN, min(cls.VMAX, int(value)))

    @property
    def available(self):
        return self.binary is not None

    @property
    def ms(self):
        return self.value / 10.0

    def set(self, value):
        """Ask for a shutter time. The write happens in `push`."""
        self.value = self.clamp(value)
        self.pending = self.value

    def push(self, force=False):
        """Send a pending value to the camera. Returns None, or (ok, message).

        None means there was nothing to do -- no pending write, or too soon
        since the last one. Call it once per frame and report anything it
        hands back.
        """
        if self.pending is None:
            return None
        now = time.perf_counter()
        if not force and now - self.pushed < self.PUSH_S:
            return None
        value, self.pending, self.pushed = self.pending, None, now
        if self.binary is None:
            self.took = False
            return False, ("no uvc-util, and it is the only thing on macOS "
                           "that can move this camera's shutter — see "
                           "vision/shutter.py for the build")
        if not self.manual:
            ok, msg = self._write_auto(True)
            self.manual = bool(ok)
            if not ok:
                self.took = False
                return False, f"could not take manual {self.WHAT}: {msg}"
        ok, msg = self._write_value(value)
        self.took = bool(ok)
        if not ok:
            return False, f"exposure {value}: {msg}"
        return True, f"exposure {value} = {value / 10:.1f} ms"

    def _write_auto(self, manual):
        return set_mode(self.binary, self.index,
                        MANUAL if manual else APERTURE_PRIORITY)

    def _write_value(self, value):
        return set_exposure(self.binary, self.index, value)

    def auto(self):
        """Hand the control back to the camera's own automatics."""
        if self.binary is None:
            return False, "no uvc-util"
        ok, msg = self._write_auto(False)
        self.manual = False if ok else self.manual
        return ok, (f"{self.WHAT} back to automatic" if ok else msg)

    def save(self):
        config.save(self.SAVE_KEY, {self.SAVE_FIELD: self.value,
                                    "auto_exposure_mode": MANUAL,
                                    "uvc_index": self.index})
        return (f"saved {self.WHAT} {self.value} to "
                f"calib/{self.SAVE_KEY}.json")


class FocusDial(Dial):
    """The same control, pointed at focus instead of the shutter.

    Separate values, separate file, one push implementation — the rate limit
    and the re-assert-manual-before-every-write rule are the parts that were
    worth getting right once.
    """

    WHAT = "focus"
    SAVE_KEY, SAVE_FIELD = "focus", "focus_abs"
    VMIN, VMAX, VDEFAULT = FOCUS_MIN, FOCUS_MAX, FOCUS_DEFAULT

    def _write_auto(self, manual):
        return set_autofocus(self.binary, self.index, not manual)

    def _write_value(self, value):
        return set_focus(self.binary, self.index, value)

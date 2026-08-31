"""Exposure control for a webcam macOS will not let OpenCV touch.

OpenCV's AVFoundation backend accepts `CAP_PROP_EXPOSURE` and drops it. Probed
on the camera this was written for, every value at every auto-exposure setting
moved the mean brightness by under two counts out of 255 -- the property is not
implemented, and reading it back reports whatever you asked for, so nothing in
OpenCV can tell you that.

The device itself is more capable than the backend. Asked through AVFoundation
directly, this camera reports:

    Locked                   supported
    ContinuousAutoExposure   supported
    Custom                   NOT supported

So there is no dial for shutter and ISO. What there is, is a freeze -- and a
freeze is enough, because the problem was never the absolute value. The problem
is that a camera holding a constant mean brightness AMPLIFIES when you darken
the room, which is exactly backwards for a method that wants a black floor and
a few bright LEDs. Turning the lights off made the balls blow out worse.

The working sequence is therefore:

    1. point it at something bright, so auto-exposure settles SHORT
    2. lock
    3. now darken the room

The exposure stays where it was locked, the floor falls away, and the LED cores
come back down out of saturation. `Custom` would be nicer. `Locked` is enough.
"""

import AVFoundation as AV

# AVCaptureExposureMode, from the AVFoundation headers.
LOCKED = 0
AUTO_EXPOSE = 1
CONTINUOUS = 2
CUSTOM = 3
NAMES = {LOCKED: "locked", AUTO_EXPOSE: "auto once",
         CONTINUOUS: "continuous auto", CUSTOM: "custom"}


def devices():
    """Every video device AVFoundation can see, newest API first."""
    try:
        disc = AV.AVCaptureDeviceDiscoverySession.\
            discoverySessionWithDeviceTypes_mediaType_position_(
                ["AVCaptureDeviceTypeExternalUnknown",
                 "AVCaptureDeviceTypeExternal",
                 "AVCaptureDeviceTypeBuiltInWideAngleCamera"],
                AV.AVMediaTypeVideo, 0)
        found = list(disc.devices())
        if found:
            return found
    except Exception:
        pass
    return list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))


def find(name=None):
    """The named device, or the first external one, or the first at all.

    External first because the built-in FaceTime camera is never the one
    pointed at the floor, and it is the one that answers to index 0 about half
    the time.
    """
    devs = devices()
    if not devs:
        return None
    if name:
        for d in devs:
            if name.lower() in str(d.localizedName()).lower():
                return d
    for d in devs:
        if "facetime" not in str(d.localizedName()).lower():
            return d
    return devs[0]


def capabilities(device):
    """What this camera will actually let you do to its exposure."""
    if device is None:
        return None
    out = {"name": str(device.localizedName()), "modes": {}}
    for mode, label in NAMES.items():
        try:
            out["modes"][label] = bool(device.isExposureModeSupported_(mode))
        except Exception:
            out["modes"][label] = False
    try:
        out["mode_now"] = NAMES.get(int(device.exposureMode()), "?")
    except Exception:
        out["mode_now"] = "?"
    try:
        out["iso"] = float(device.ISO())
    except Exception:
        out["iso"] = None
    return out


def set_mode(device, mode):
    """Set the exposure mode. Returns (ok, message).

    Configuration is locked around the change and released in a finally: a
    device left locked stays locked for every other process on the machine
    until this one exits, which is a far worse failure than not being able to
    set the mode at all.
    """
    if device is None:
        return False, "no camera"
    if not device.isExposureModeSupported_(mode):
        return False, f"{NAMES.get(mode, mode)} is not supported by this camera"
    ok, err = device.lockForConfiguration_(None)
    if not ok:
        return False, f"could not lock the device for configuration: {err}"
    try:
        device.setExposureMode_(mode)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        device.unlockForConfiguration()
    return True, f"exposure {NAMES.get(mode, mode)}"


def lock(name=None):
    """Freeze exposure where it is. Expose against something bright first."""
    return set_mode(find(name), LOCKED)


def auto(name=None):
    """Hand it back to the camera."""
    return set_mode(find(name), CONTINUOUS)


def go_short(index=0, name=None, bright_s=6.0, settle_s=2.5, verbose=True):
    """Drive the camera to a SHORT exposure and freeze it there.

    There is no shutter control on this camera, so the exposure cannot be set.
    It can only be CHOSEN by the camera and then held -- which is enough, and
    this is the sequence that does it:

        1. you flood the lens with light (a phone torch works)
        2. its metering pulls the exposure down to compensate
        3. we lock, while it is still short
        4. you take the light away

    The floor is then far darker than the camera would ever have chosen, which
    is exactly the frame the LED method wants: black floor, bright cores, no
    clipping. Removing the light afterwards is what makes it underexposed --
    the camera would undo that in a second if it were still metering.

    Mean brightness is printed throughout because it is the only feedback
    available. A camera that has adapted has brought the mean back near its own
    target; a camera that has not is still climbing.
    """
    import time

    import cv2

    dev = find(name)
    if dev is None:
        return False, "no camera found"
    ok, msg = set_mode(dev, CONTINUOUS)      # it must be metering to adapt
    if not ok:
        return False, msg

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False, (f"camera {index} would not open — on macOS, grant camera "
                       "access to your terminal in System Settings")

    def mean():
        last = None
        for _ in range(3):
            got, frame = cap.read()
            if got and frame is not None:
                last = frame
        if last is None:
            return None
        return float(cv2.cvtColor(last, cv2.COLOR_BGR2HSV)[:, :, 2].mean())

    try:
        start = mean()
        if verbose:
            print(f"starting mean {start:.0f}")
            print(f"\n  SHINE A BRIGHT LIGHT INTO THE LENS NOW — {bright_s:.0f}s")
        t0 = time.time()
        peak = start or 0.0
        while time.time() - t0 < bright_s:
            m = mean()
            if m is None:
                continue
            peak = max(peak, m)
            if verbose:
                print(f"    mean {m:5.1f}", end="\r", flush=True)
            time.sleep(0.25)
        if verbose:
            print(f"\n  peak seen {peak:.0f}; letting it settle {settle_s:.0f}s")
        time.sleep(settle_s)
        settled = mean()
        ok, msg = set_mode(dev, LOCKED)
        if not ok:
            return False, msg
        if verbose:
            print(f"  LOCKED at mean {settled:.0f} — take the light away now")
        time.sleep(2.0)
        after = mean()
        cap.release()
    except Exception as e:
        cap.release()
        return False, f"{type(e).__name__}: {e}"

    if after is None or settled is None:
        return True, "locked, but could not measure the result"
    drop = settled - after
    if peak - (start or 0) < 15:
        return True, (f"locked at mean {after:.0f}, but the lens never got much "
                      "brighter — was the light actually pointed into it?")
    if drop < 20:
        return True, (f"locked at mean {after:.0f}. That is not much darker than "
                      f"{settled:.0f}, so the lock may not have held — check "
                      "with `show`, and re-run if it is back on auto")
    return True, (f"locked. Mean fell {settled:.0f} -> {after:.0f} once the light "
                  "came away, which is the underexposed frame you want")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", default="show",
                   choices=["show", "lock", "auto", "short"])
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV index, for the `short` sequence")
    p.add_argument("--name", default=None, help="match part of the camera name")
    a = p.parse_args(argv)
    dev = find(a.name)
    caps = capabilities(dev)
    if caps is None:
        print("no camera found")
        return 1
    if a.action == "show":
        print(caps["name"])
        for label, ok in caps["modes"].items():
            print(f"  {label:<18} {'yes' if ok else 'no'}")
        print(f"  currently          {caps['mode_now']}"
              + (f"   ISO {caps['iso']:.0f}" if caps["iso"] else ""))
        return 0
    if a.action == "short":
        ok, msg = go_short(a.camera, a.name)
        print(("" if ok else "failed: ") + msg)
        return 0 if ok else 1
    ok, msg = (lock if a.action == "lock" else auto)(a.name)
    print(("" if ok else "failed: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

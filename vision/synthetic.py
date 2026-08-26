"""A fake overhead camera.

Renders glowing coloured blobs on a floor, seen through a deliberately skewed
perspective, with noise and occasional dropouts. Lets you build and validate
the entire tracker before a camera or a robot exists.
"""

import cv2
import numpy as np

from . import config


class SyntheticSource:
    def __init__(self, n=5, size=(960, 720), arena=config.ARENA_CM, seed=0,
                 dropout=0.03, noise=6):
        self.w, self.h = size
        self.arena = arena
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.noise = noise
        self.names = list(config.COLORS)[:n]
        self.pos = self.rng.uniform(30, arena - 30, (n, 2))
        self.vel = self.rng.uniform(-25, 25, (n, 2))
        self.t = 0

        m = 90
        src = np.float32([[0, 0], [arena, 0], [arena, arena], [0, arena]])
        dst = np.float32([[m + 40, m], [self.w - m, m + 25],
                          [self.w - m - 30, self.h - m], [m, self.h - m - 40]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.true_corners = dst

    def truth(self):
        return {n: self.pos[i].copy() for i, n in enumerate(self.names)}

    def _advance(self, dt=1 / 30):
        self.pos += self.vel * dt
        for ax in (0, 1):
            lo, hi = self.pos[:, ax] < 12, self.pos[:, ax] > self.arena - 12
            self.vel[lo | hi, ax] *= -1
        np.clip(self.pos, 12, self.arena - 12, out=self.pos)
        self.vel += self.rng.normal(0, 4, self.vel.shape)
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = np.where(sp > 45, self.vel / sp * 45, self.vel)
        self.t += 1

    def read(self):
        self._advance()
        img = np.full((self.h, self.w, 3), 24, np.uint8)
        img[:] = (38, 34, 30)

        pts = cv2.perspectiveTransform(
            self.pos.reshape(-1, 1, 2).astype(np.float32), self.M).reshape(-1, 2)

        cv2.polylines(img, [self.true_corners.astype(int)], True, (70, 66, 60), 2)

        for i, name in enumerate(self.names):
            if self.rng.random() < self.dropout:
                continue
            c = config.COLORS[name]
            hsv = np.uint8([[[c["hue"], 235, 245]]])
            bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
            x, y = int(pts[i, 0]), int(pts[i, 1])
            cv2.circle(img, (x, y), 19, bgr, -1)
            cv2.circle(img, (x, y), 8, (250, 250, 250), -1)   # blown-out core
        img = cv2.GaussianBlur(img, (7, 7), 0)
        noise = self.rng.normal(0, self.noise, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return True, img

    def release(self):
        pass


class CameraSource:
    # What a lit Sphero needs from a camera, and why. Autofocus hunts on a
    # scene that is mostly dark floor, and every hunt is a frame or two of
    # blur — which turns two LEDs a few pixels apart into one smear. Auto
    # exposure is worse: it meters the whole frame, sees mostly black, and
    # opens up until the ball's shell blows out to white, which has no hue for
    # the detector to key on and no separable peaks for a heading.
    WANTED = {
        "autofocus": (cv2.CAP_PROP_AUTOFOCUS, 0),
        "auto_exposure": (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
    }

    PROPS = {
        "focus": cv2.CAP_PROP_FOCUS,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "gain": cv2.CAP_PROP_GAIN,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    }

    def __init__(self, index=0, size=(1280, 720)):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise RuntimeError(
                f"camera {index} would not open. On macOS, grant camera access "
                "to your terminal in System Settings > Privacy & Security.")
        self.wanted_size = tuple(size)

    @property
    def size(self):
        """What the camera is ACTUALLY delivering, not what it was asked for.

        A webcam offered a mode it does not have picks its nearest and reports
        success, so asking for 1080p is not the same as getting it. Reading it
        back is the only way to know, and the difference is the difference
        between two resolvable LEDs and one smear.
        """
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # -- manual control --------------------------------------------------
    #
    # There is deliberately no `capabilities()` here. A previous version probed
    # each control by WRITING to it and restoring only those that reported a
    # change, which is not a safe way to ask a camera what it can do — it left
    # a camera in a state nothing on the bench could get it out of, and the
    # whole probe was reverted. Every write below happens because a person
    # moved a control, never to find out what a control does.

    def get(self, name):
        prop = self.PROPS.get(name)
        if prop is None:
            return None
        try:
            v = float(self.cap.get(prop))
        except Exception:
            return None
        return None if v in (-1.0,) else v

    def set(self, name, value):
        """Ask for a setting and report what actually took.

        Cameras accept a `set` and ignore it constantly — the macOS
        AVFoundation backend in particular reports success for properties it
        does not implement. Reading the value back is the only way to know, and
        a control that silently does nothing is worse than one that is absent,
        because a person will keep turning it.
        """
        prop = self.PROPS.get(name)
        if prop is None:
            return None
        try:
            self.cap.set(prop, float(value))
        except Exception:
            return None
        return self.get(name)

    def manual(self):
        """Turn off the automatics that blur and blow out a lit ball."""
        out = {}
        for name, (prop, value) in self.WANTED.items():
            try:
                self.cap.set(prop, value)
            except Exception:
                pass
            out[name] = self.get(name)
        return out

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class VideoSource:
    def __init__(self, path, loop=True):
        self.path, self.loop = path, loop
        self.cap = cv2.VideoCapture(str(path))

    def read(self):
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        self.cap.release()


def open_source(spec, size=None):
    """`size` is a request, not a promise — see `CameraSource.size`.

    Only a camera index can honour it; the synthetic source renders what it
    renders and a video file is whatever was recorded, so passing a size for
    either is silently ignored rather than raising. That keeps one call site
    able to open any of the three.
    """
    if spec == "synthetic":
        return SyntheticSource()
    if str(spec).isdigit():
        return CameraSource(int(spec), size=size) if size else CameraSource(int(spec))
    return VideoSource(spec)

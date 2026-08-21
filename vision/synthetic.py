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
    def __init__(self, index=0, size=(1280, 720)):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise RuntimeError(
                f"camera {index} would not open. On macOS, grant camera access "
                "to your terminal in System Settings > Privacy & Security.")

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


def open_source(spec):
    if spec == "synthetic":
        return SyntheticSource()
    if str(spec).isdigit():
        return CameraSource(int(spec))
    return VideoSource(spec)

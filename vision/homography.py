"""Pixel to centimetre mapping.

Click the four corners of your arena once. Everything downstream then works in
real units, which is what makes the sim, the policy, and the formations all
speak the same language.
"""

import cv2
import numpy as np

from . import config


class Homography:
    def __init__(self, matrix=None, arena=config.ARENA_CM, width=None, height=None):
        self.arena = arena
        # The rectangle this calibration was built for, in cm. Stored because
        # `arena` alone cannot distinguish a square calibration from a
        # rectangular one, and a homography that maps to 200x200 under a
        # 240x180 workspace puts the tracker and the planner in different
        # coordinate systems — robots then drift off an arena they are
        # nowhere near the edge of.
        self.width = float(width) if width else float(arena)
        self.height = float(height) if height else float(arena)
        self.M = np.array(matrix, dtype=np.float64) if matrix is not None else None

    @property
    def ready(self):
        return self.M is not None

    @classmethod
    def load(cls):
        d = config.load("homography")
        if not d:
            return cls()
        return cls(d["matrix"], d.get("arena", config.ARENA_CM),
                   width=d.get("width"), height=d.get("height"))

    def save(self):
        config.save("homography", {"matrix": self.M.tolist(), "arena": self.arena,
                                   "width": self.width, "height": self.height})

    def set_corners(self, pts):
        """pts: 4 pixel points in order top-left, top-right, bottom-right, bottom-left."""
        src = np.array(pts, dtype=np.float32)
        dst = np.array([[0, 0], [self.arena, 0],
                        [self.arena, self.arena], [0, self.arena]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        self.width = self.height = float(self.arena)      # this one is square

    def set_rect(self, pts, width, height):
        """Four clicked corners -> a rectangle of a chosen size, in centimetres.

        `set_corners` maps to a square, because that is all the original
        calibration offered. An arena is rarely square, and forcing one is how
        the tracker and the planner ended up describing different floors — the
        homography said 200x200 while `workspace.json` said 240x180 and nothing
        complained. Choosing both sides here, and writing the workspace from
        the same numbers, is what stops that pairing drifting apart again.

        Corners are clockwise from the origin: origin, +x, +x+y, +y.
        """
        width, height = float(width), float(height)
        src = np.array(_orient(pts), dtype=np.float32)
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        self.width, self.height = width, height
        self.arena = max(width, height)
        return self

    def matches(self, width, height, tol=1.0):
        """Does this calibration describe the same rectangle as the workspace?"""
        return (abs(self.width - float(width)) <= tol
                and abs(self.height - float(height)) <= tol)

    def to_cm(self, pts_px):
        p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(p, self.M).reshape(-1, 2)
        return out

    def to_px(self, pts_cm):
        inv = np.linalg.inv(self.M)
        p = np.asarray(pts_cm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, inv).reshape(-1, 2)


def _orient(pts):
    """Put four clicked corners into the order the workspace declares.

    `workspace.json` says the arena frame is "top-left, x right, y down,
    matching the camera frame". Four corners can be clicked starting anywhere
    and going either way round, and three of those eight readings produce a
    frame that is rotated or MIRRORED relative to that declaration.

    A rotation is survivable — a heading offset cancels it. A mirror is not,
    and that is the whole reason this exists: a mirrored frame turns a
    commanded heading into its reflection, so the error changes sign with
    direction. Measured across eight compass points it swings 270 degrees. A
    `heading_offset` is one number added to every command; it can cancel a
    constant error and it can never cancel one that changes sign. The symptom
    is calibration legs that disagree by tens of degrees and a robot that
    circles whatever you correct — which is a fortnight of blaming the
    controller for the shape of the arena.

    So the clicks are re-read rather than trusted: the corner nearest the
    image origin becomes the arena origin, and the order is forced clockwise
    in image space, which with y-down is the orientation-preserving one.
    """
    pts = [(float(x), float(y)) for x, y in pts]
    start = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    out = [pts[(start + i) % 4] for i in range(4)]
    (ax, ay), (bx, by), (cx, cy) = out[0], out[1], out[2]
    if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
        out = [out[0], out[3], out[2], out[1]]      # was anticlockwise
    return out


LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]


def calibrate(source, arena=config.ARENA_CM):
    """Interactive: click 4 corners clockwise from top-left. Returns Homography."""
    pts = []
    win = "calibrate — click 4 corners clockwise from top-left"

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        ok, frame = source.read()
        if not ok:
            break
        vis = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (60, 210, 235), -1)
            cv2.putText(vis, LABELS[i], (p[0] + 10, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 210, 235), 1)
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts)], len(pts) == 4, (99, 210, 232), 2)

        msg = (f"click {LABELS[len(pts)]}" if len(pts) < 4
               else "enter = accept   r = redo   q = cancel")
        cv2.putText(vis, msg, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (228, 240, 248), 2)
        cv2.imshow(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("r"),):
            pts = []
        elif k in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None
        elif k in (13, 10) and len(pts) == 4:
            h = Homography(arena=arena)
            h.set_corners(pts)
            h.save()
            cv2.destroyWindow(win)
            return h
    cv2.destroyWindow(win)
    return None

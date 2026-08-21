#!/usr/bin/env python3
"""Write a workspace.json.

    python -m workspace.make                          # default rectangle
    python -m workspace.make --width 240 --height 180
    python -m workspace.make --from-camera --source 0 # click the arena, then obstacles

`--from-camera` reuses the homography click flow: the four corners you click
become both the calibration and the workspace bounds, so the arena the tracker
reports in and the arena the planner plans in are the same rectangle by
construction. Further clicked polygons become obstacles.
"""

import argparse
import pathlib

import numpy as np

from .space import DEFAULT_ORIGIN, DEFAULT_PATH, Workspace


def default_rect(width, height):
    return [[0, 0], [width, 0], [width, height], [0, height]]


def _from_camera(source_spec, width, height):
    """Interactive. Returns (bounds_cm, obstacles) or None if cancelled."""
    import cv2

    from vision.homography import LABELS, Homography
    from vision.synthetic import open_source

    src = open_source(source_spec)
    win = "workspace — click 4 arena corners clockwise from top-left"

    corners = []

    def on_corner(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_corner)

    while True:
        ok, frame = src.read()
        if not ok:
            break
        vis = frame.copy()
        for i, p in enumerate(corners):
            cv2.circle(vis, p, 6, (60, 210, 235), -1)
            cv2.putText(vis, LABELS[i], (p[0] + 10, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 210, 235), 1)
        if len(corners) > 1:
            cv2.polylines(vis, [np.array(corners)], len(corners) == 4, (99, 210, 232), 2)
        msg = (f"click {LABELS[len(corners)]}" if len(corners) < 4
               else "enter = accept corners   r = redo   q = cancel")
        cv2.putText(vis, msg, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (228, 240, 248), 2)
        cv2.imshow(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("r"):
            corners = []
        elif k in (ord("q"), 27):
            cv2.destroyWindow(win)
            src.release()
            return None
        elif k in (13, 10) and len(corners) == 4:
            break

    cv2.destroyWindow(win)

    # The clicked corners define a rectangle of the size the user declared.
    h = Homography(arena=max(width, height))
    hsrc = np.array(corners, dtype=np.float32)
    hdst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    h.M = cv2.getPerspectiveTransform(hsrc, hdst).astype(np.float64)
    h.arena = max(width, height)
    h.width, h.height = float(width), float(height)
    h.save()

    bounds = [[0, 0], [width, 0], [width, height], [0, height]]

    # -- obstacles ------------------------------------------------------
    win2 = "obstacles — click a polygon, enter = keep, n = next, q = done"
    obstacles = []
    current = []

    def on_obs(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    cv2.namedWindow(win2)
    cv2.setMouseCallback(win2, on_obs)

    while True:
        ok, frame = src.read()
        if not ok:
            break
        vis = frame.copy()
        box = h.to_px(bounds).astype(int)
        cv2.polylines(vis, [box], True, (110, 104, 96), 2)
        for o in obstacles:
            px = h.to_px(o["points"]).astype(int)
            cv2.polylines(vis, [px], True, (107, 107, 255), 2)
        if current:
            cv2.polylines(vis, [np.array(current)], False, (60, 210, 235), 2)
            for p in current:
                cv2.circle(vis, p, 4, (60, 210, 235), -1)
        cv2.putText(vis, f"{len(obstacles)} obstacles   enter = keep this one   "
                         "r = redo   q = done", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (228, 240, 248), 2)
        cv2.imshow(win2, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("r"):
            current = []
        elif k in (13, 10) and len(current) >= 3:
            pts_cm = h.to_cm(current)
            obstacles.append({"type": "poly",
                              "points": [[round(float(x), 1), round(float(y), 1)]
                                         for x, y in pts_cm]})
            current = []
        elif k in (ord("q"), 27):
            break

    cv2.destroyWindow(win2)
    src.release()
    return bounds, obstacles


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=float, default=240.0, help="arena width in cm")
    p.add_argument("--height", type=float, default=180.0, help="arena height in cm")
    p.add_argument("--from-camera", action="store_true",
                   help="click the arena corners and obstacles on a camera frame")
    p.add_argument("--source", default="0", help="camera index, video path, or 'synthetic'")
    p.add_argument("--out", default=str(DEFAULT_PATH))
    p.add_argument("--reset-entities", action="store_true",
                   help="drop existing entities instead of carrying them over")
    a = p.parse_args()

    if a.from_camera:
        got = _from_camera(a.source, a.width, a.height)
        if got is None:
            print("cancelled, nothing written")
            return
        bounds, obstacles = got
    else:
        bounds, obstacles = default_rect(a.width, a.height), []

    # Entities are carried over from whatever is already at `--out`. Rebuilding
    # the arena is a geometry job — the followable and moving things in it are
    # not something the operator is asking to delete, and silently dropping
    # them is only noticed later, when a `follow` says no such entity.
    entities = None
    existing = pathlib.Path(a.out)
    if existing.exists() and not a.reset_entities:
        try:
            prior = Workspace.load(a.out)
            if len(prior.entities):
                entities = prior.entities
                print(f"  keeping {len(entities)} entit(y/ies): "
                      + ", ".join(e.id for e in entities))
        except Exception as e:
            print(f"  could not read existing entities ({e}); continuing without")

    ws = Workspace(bounds_cm=bounds, obstacles=obstacles, origin=DEFAULT_ORIGIN,
                   path=a.out, entities=entities)
    errors = ws.save()
    if errors:
        for e in errors:
            print("error:", e)
        return
    print(f"wrote {a.out}")
    print(f"  bounds    {ws.width:.0f} x {ws.height:.0f} cm")
    print(f"  obstacles {len(ws.obstacles)}")


if __name__ == "__main__":
    main()

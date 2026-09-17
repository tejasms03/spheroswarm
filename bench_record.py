"""A flight recorder for every run the bench drives: p2p, line, path, orbit,
patrol. It only READS the app each frame — nothing in control changes — and
writes what it saw to runs/p2p/ when a run ends, so a misbehaving turn can be
read back from a file instead of remembered off the screen.
"""

import json
import time
from pathlib import Path

import numpy as np

from vision.tracks import wrap180

LOG_DIR = Path(__file__).resolve().parent / "runs" / "p2p"
MAX_ROWS = 30 * 180
"""Three minutes at 30fps per file. A long orbit rolls over into a new file
rather than growing without bound."""


class RunRecorder:
    def __init__(self, app, log_dir=LOG_DIR):
        self.app = app
        self.log_dir = Path(log_dir)
        self.got = None
        self.rows = []
        self.started = None
        self.written = []

    def tick(self):
        app, got = self.app, self.app.p2p
        if got is not self.got:
            if self.got is not None:
                self.flush()
            self.got = got
            self.rows, self.started = [], time.time()
        if got is None:
            return
        try:
            self.rows.append(self.row(got))
        except Exception as e:                  # a recorder must never stop a run
            self.rows.append({"t": time.time(), "record_error": str(e)})
        if len(self.rows) >= MAX_ROWS:
            self.flush()
            self.rows, self.started = [], time.time()

    def row(self, got):
        app = self.app
        name = got["name"]
        r = {"t": round(time.time(), 3), "phase": got.get("phase"),
             "stage": got.get("stage"), "spin_dir": got.get("spin_dir"),
             "seen": bool(app.ball_fresh(name))}
        px = app.ball_px(name)
        if px is not None and app.lab.hom is not None and app.lab.hom.ready:
            p = np.asarray(app.lab.hom.to_cm([list(px)]), float).ravel()[:2]
            r["xy_cm"] = [round(float(p[0]), 1), round(float(p[1]), 1)]
        facing = app.arena_heading(name)
        r["facing"] = None if facing is None else round(float(facing), 1)
        geo = app.p2p_geometry()
        if geo is not None:
            r["bearing"], r["gap_cm"] = round(float(geo[0]), 1), round(float(geo[1]), 1)
            if facing is not None:
                r["err"] = round(wrap180(geo[0] - facing), 1)
        robot = app.lab.robots.get(name)
        r["spinning"] = bool(getattr(robot, "spinning", False))
        r["spin_power"] = float(app.spin_power)
        r["spin_sign"] = app.spin_sign.get(name, 1)
        r["sign_known"] = name in app.spin_sign_known
        r["sign_wrong"] = getattr(app, "spin_wrong", {}).get(name, 0)
        r["retries"] = app.retries
        sent = got.get("sent")
        if sent is not None:
            r["sent"] = [round(float(sent[0]), 1), int(sent[1])]
        return r

    def flush(self):
        got, rows = self.got, self.rows
        if got is None or not rows:
            return None
        why = (self.app._last_stop or ("", None))[0]
        kind = ("path:" + got["path"]["kind"] if got.get("path") is not None
                else got.get("kind", "p2p"))
        out = {"name": got["name"], "kind": kind, "started": self.started,
               "target_px": list(got.get("target") or []), "ended": why,
               "rows": rows}
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%m%d_%H%M%S", time.localtime(self.started))
            path = self.log_dir / f"{got['name']}_{stamp}_{kind.replace(':', '-')}.json"
            k = 1
            while path.exists():
                path = path.with_name(path.stem + f"_{k}.json")
                k += 1
            path.write_text(json.dumps(out))
            self.written.append(path)
            return path
        except Exception:
            return None


TRACK_DIR = Path(__file__).resolve().parent / "runs" / "tracks"


class TrackRecorder:
    """Every tracked ball, every frame, while recording is switched on — for
    the multi-ball data collection: noise at rest, where two blobs merge or go
    CONTENDED, identity swaps, dropouts, what manual driving sent.

    Reads the app only. Streams one JSON line per frame so a long session
    does not sit in memory, and a crash loses at most the last second.
    """

    FLUSH_EVERY = 30

    def __init__(self, app, log_dir=TRACK_DIR):
        self.app = app
        self.log_dir = Path(log_dir)
        self.fh = None
        self.path = None
        self.rows = 0
        self.started = None

    @property
    def on(self):
        return self.fh is not None

    def toggle(self):
        if self.on:
            return self.stop()
        return self.start()

    def start(self):
        if self.on:
            return self.path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M%S")
        self.path = self.log_dir / f"tracks_{stamp}.jsonl"
        self.fh = open(self.path, "w")
        self.rows, self.started = 0, time.time()
        app = self.app
        hom = app.lab.hom
        meta = {"meta": True, "started": self.started,
                "arena_cm": ([float(hom.width), float(hom.height)]
                             if hom is not None and hom.ready else None),
                "connected": sorted(app.lab.robots),
                "tracked": sorted(app.lab.tracks.by_name)}
        self.fh.write(json.dumps(meta) + "\n")
        return self.path

    def stop(self):
        if not self.on:
            return None
        try:
            self.fh.flush()
            self.fh.close()
        finally:
            self.fh = None
        return self.path

    def tick(self):
        if not self.on:
            return
        try:
            row = self.row()
        except Exception as e:                  # never break the bench
            row = {"t": round(time.time(), 3), "record_error": str(e)}
        self.fh.write(json.dumps(row) + "\n")
        self.rows += 1
        if self.rows % self.FLUSH_EVERY == 0:
            self.fh.flush()

    def row(self):
        app = self.app
        hom = app.lab.hom
        ready = hom is not None and hom.ready
        balls = {}
        for name, t in app.lab.tracks.by_name.items():
            b = {"px": [round(float(t.centre[0]), 1), round(float(t.centre[1]), 1)],
                 "heading_img": round(float(t.heading), 1),
                 "lost": bool(t.lost), "contended": bool(t.contended),
                 "bridged": bool(getattr(t, "bridged", False)),
                 "travelling": bool(getattr(t, "travelling", False)),
                 "flips_caught": int(getattr(t, "flips_caught", 0)),
                 "why": t.why}
            if ready:
                p = np.asarray(hom.to_cm([list(t.centre)]), float).ravel()[:2]
                b["cm"] = [round(float(p[0]), 2), round(float(p[1]), 2)]
                facing = app.arena_heading(name)
                b["facing"] = None if facing is None else round(float(facing), 1)
            robot = app.lab.robots.get(name)
            b["connected"] = robot is not None
            b["spinning"] = bool(getattr(robot, "spinning", False))
            balls[name] = b
        drive = app._drive_sent
        got = app.p2p
        return {"t": round(time.time(), 3), "balls": balls,
                "clusters": len(app.all_reading or []),
                "driving": app.driving,
                "manual_sent": ([round(float(drive[0]), 1), int(drive[1])]
                                if drive else None),
                "job": (None if got is None else
                        {"name": got.get("name"), "phase": got.get("phase"),
                         "stage": got.get("stage"),
                         "sent": (list(got["sent"]) if got.get("sent") else None)})}

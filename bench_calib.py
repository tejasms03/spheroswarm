"""The calibration walk: corners and a diagonal, on its OWN turn and drive.

Kept apart from point-to-point on purpose. p2p works and is not touched; this
only reads the bench's position, facing and handle.

Why it does not reuse p2p's turn. On the bench a fixed spin power is wrong both
ways: enough to break a resting ball free overshoots once it is turning, and
little enough not to overshoot sticks after the ball has stopped at a corner.
So the turn here:

- **ramps** power up from low until the camera sees it actually turning — that
  is the breakaway power, found per turn rather than guessed;
- **stops early** by the overshoot it has measured, since after a stop a ball
  keeps turning for a few degrees (plus camera lag), learned from every stop;
- settles, looks again, and only drives once it is inside the aim band.

The drive on each leg is straight (speed), then a held steer TOWARD the middle
of the arena (which way it steers, delay), then it stops, re-aims at the corner
and drives the rest straight. Every run is logged to runs/calib.
"""

import json
import math
import time
from collections import deque

import numpy as np

from swarm import ball_calib
from vision.tracks import wrap180


def _bearing(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


class CalibWalk:
    INSET_CM = 20.0
    """Corners are aimed this far inside the picked ones. A ball near the edge
    is where tracking is worst — two early runs were lost off the top edge."""
    STEER_DEG = 20.0
    STRETCH_CM = 25.0
    """Straight and steered stretches are each this long at most."""
    EDGE_CM = 8.0
    """A steer that brings it this close to the arena edge ends early."""
    OFF_CM = 40.0
    ARRIVE_CM = 5.0
    COAST_S = 1.5
    PUSH_S = 0.1
    BLIND_S = 2.0

    START_POWER = 22.0
    RAMP_PER_S = 30.0
    MAX_POWER = 110.0
    MOVE_DEG = 4.0
    MOVE_WINDOW_S = 0.4
    SETTLE_S = 0.5
    STILL_DEG = 2.0
    SETTLE_MAX_S = 2.5
    LEAD_START_DEG = 6.0
    MAX_LEAD_DEG = 40.0
    TURN_TIMEOUT_S = 15.0
    MAX_DIR_FLIPS = 3

    MISS_CM = 10.0
    """Re-aim on the way to a corner only if, held as it is, it would pass
    further than this from it. An angle alone is the wrong test: close in, a
    few cm to one side is a big angle and no miss at all."""
    AWAY_CM = 5.0
    PROGRESS_CM = 3.0
    MAX_REAIMS = 5
    """Re-aims in a row that got it no closer."""

    def __init__(self, app, name, byte, cm_s, log_dir):
        self.app, self.name, self.byte = app, name, int(byte)
        self.cm_s = float(cm_s)
        self.log_dir = log_dir
        hom = app.lab.hom
        self.w, self.h = float(hom.width), float(hom.height)
        k = self.INSET_CM
        c = [(k, k), (self.w - k, k), (self.w - k, self.h - k),
             (k, self.h - k)]
        self.route = [np.array(p, float)
                      for p in (c[0], c[1], c[2], c[3], c[0], c[2])]
        self.leg = 0
        self.legs = []
        self.cur = None
        self.turns = []
        self.dir_sign = 1
        self.dir_known = False
        self.dir_flips = 0
        self.dir_wrong = 0
        self.lead = self.LEAD_START_DEG
        self.breakaways = []
        self.stage = "turn"
        self.turn = None
        self.at = time.time()
        self.push_at = 0.0
        self.blind_from = None
        self.reaims = 0
        self.best_gap = None
        self.done = False
        self.result = None
        self.why = None
        self._begin_turn(self.at)

    # -- the bench's view of the ball ---------------------------------------

    def here(self):
        px = self.app.ball_px(self.name)
        if px is None:
            return None
        return np.asarray(self.app.lab.hom.to_cm([list(px)]),
                          float).ravel()[:2]

    @property
    def robot(self):
        return self.app.lab.robots.get(self.name)

    @property
    def target(self):
        return self.route[self.leg]

    def status(self):
        where = f"leg {min(self.leg + 1, len(self.route))}/{len(self.route)}"
        if self.stage == "turn" and self.turn is not None:
            word = ("settling" if self.turn["phase"] == "settle" else
                    f"turning, power {self.turn['power']:.0f}")
            return f"calib {where}: {word} (stops {self.lead:.0f}deg early)"
        if self.stage == "drive" and self.cur is not None:
            seg = {"A": "straight", "B": "steered", "C": "to corner"}.get(
                self.cur["seg"], "")
            return f"calib {where}: driving {seg}"
        return f"calib {where}: {self.stage}"

    # -- ending ----------------------------------------------------------------

    def _halt(self):
        r = self.robot
        if r is None:
            return
        try:
            if getattr(r, "spinning", False):
                r.stop_raw()
            r.stop()
        except Exception:
            pass

    def fail(self, why):
        self._halt()
        self.done, self.why = True, why
        self.log()

    def cancel(self):
        if not self.done:
            self.fail("stopped")

    def finish(self):
        extra = {"spin_dir_sign": self.dir_sign,
                 "breakaway_power": (float(np.median(self.breakaways))
                                     if self.breakaways else None),
                 "turn_lead_deg": float(self.lead)}
        record, why = ball_calib.summarise(self.legs, self.byte,
                                           self.app.lab.hom.M, self.name,
                                           turn=extra)
        self._halt()
        self.done = True
        if record is None:
            self.why = why
        else:
            try:
                ball_calib.save(record)
                self.result = record
            except Exception as e:
                self.why = f"measured, but could not save: {e}"
        self.log()

    def log(self):
        """Every run, good or bad, with every turn — so a failure is read from
        a file, not remembered off the screen."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%m%d_%H%M%S")
            legs = list(self.legs)
            if self.cur is not None and self.cur not in legs:
                legs.append(self.cur)
            turns = list(self.turns)
            if self.turn is not None and self.turn not in turns:
                turns.append(self.turn)
            turns = [{k: v for k, v in t.items()
                      if not isinstance(v, deque)} for t in turns]
            out = self.log_dir / f"{self.name}_{stamp}.json"
            out.write_text(json.dumps(ball_calib._plain({
                "name": self.name, "byte": self.byte, "route_cm": self.route,
                "legs": legs, "turns": turns, "saved": self.result,
                "why": self.why, "lead": self.lead,
                "breakaways": self.breakaways, "dir_sign": self.dir_sign,
                "matrix": self.app.lab.hom.M}), indent=1))
        except Exception as e:
            self.app.say(f"could not write the calib log: {e}")

    # -- one frame -------------------------------------------------------------

    def tick(self):
        if self.done:
            return
        now = time.time()
        robot = self.robot
        if robot is None:
            self.fail("the ball went away")
            return
        p = self.here()
        if p is None:
            self.fail("no position for it at all")
            return
        if not self.app.ball_fresh(self.name):
            if self.blind_from is None:
                self.blind_from = now
                self._halt()
                if self.stage == "turn" and self.turn is not None:
                    self.turn.update(phase="settle", until=now + self.SETTLE_S,
                                     stop_facing=None)
            if now - self.blind_from > self.BLIND_S:
                self.fail(f"could not see it for {self.BLIND_S:.0f}s")
            return
        self.blind_from = None
        facing = self.app.arena_heading(self.name)
        if facing is None:
            return

        if self.stage == "coast":
            if now >= self.coast_until:
                if self.cur is not None:
                    self.cur["end_xy"] = p.tolist()
                    self.legs.append(self.cur)
                    self.cur = None
                self.leg += 1
                if self.leg >= len(self.route):
                    self.finish()
                    return
                self.reaims = 0
                self.reaim_gap = None
                self.best_gap = None
                self._begin_turn(now)
            return
        if self.stage == "turn":
            self._turn(robot, p, facing, now)
            return
        self._drive(robot, p, facing, now)

    # -- turning on the spot ---------------------------------------------------

    def _begin_turn(self, now):
        self.stage = "turn"
        self.turn = {"t0": now, "phase": "spin", "dir": 0, "power": 0.0,
                     "hist": deque(), "moving": False, "stop_facing": None,
                     "until": 0.0, "samples": [], "leg": self.leg,
                     "last": now}

    def _turn(self, robot, p, facing, now):
        tr = self.turn
        want_b = _bearing(p, self.target)
        err = wrap180(want_b - facing)
        tr["samples"].append([now, facing, err, tr["power"] * tr["dir"]])
        band = float(self.app.aim_band)

        if now - tr["t0"] > self.TURN_TIMEOUT_S:
            self.fail(f"turn on leg {self.leg + 1} did not settle in "
                      f"{self.TURN_TIMEOUT_S:.0f}s — still {err:+.0f}deg off")
            return

        if tr["phase"] == "settle":
            # Settled means STILL, not just a fixed wait: after a stop the ball
            # coasts on, and its stabilisation may then pull it back to where
            # the stop re-zeroed it. Judging the aim mid-way through either
            # drives off on a heading it does not hold.
            tr.setdefault("still", deque()).append((now, facing))
            still = tr["still"]
            while still and now - still[0][0] > 0.4:
                still.popleft()
            if now < tr["until"]:
                return
            # Means of the older and newer halves, so tracker jitter does not
            # read as the ball still moving.
            rows = [wrap180(f - still[-1][1]) for _, f in still]
            half = len(rows) // 2
            wobble = (abs(np.mean(rows[:half]) - np.mean(rows[half:]))
                      if half else 0.0)
            if (wobble > self.STILL_DEG
                    and now < tr["until"] - self.SETTLE_S + self.SETTLE_MAX_S):
                return
            if tr["stop_facing"] is not None and tr["dir"]:
                coasted = wrap180(facing - tr["stop_facing"]) * tr["dir"]
                if -10.0 <= coasted <= 60.0:
                    self.lead = float(np.clip(0.5 * self.lead + 0.5 * coasted,
                                              0.0, self.MAX_LEAD_DEG))
            if abs(err) <= band:
                self._aimed(robot, p, now)
                return
            tr.update(phase="spin", dir=0, power=0.0, moving=False,
                      stop_facing=None, hist=deque(), last=now, still=deque(),
                      streak=0, streak_dir=0)
            return

        want = 1 if err > 0 else -1
        # Close enough that the coast will carry it in: stop now.
        if abs(err) <= band or (tr["dir"] == want
                                and abs(err) - self.lead <= band / 2.0):
            self._stop_spin(robot, tr, facing, now)
            return
        if tr["dir"] and want != tr["dir"]:
            self._stop_spin(robot, tr, facing, now)       # went past: settle
            return

        dt = max(0.0, now - tr["last"])
        tr["last"] = now
        if tr["dir"] == 0:
            start = (float(np.median(self.breakaways)) - 8.0
                     if self.breakaways else self.START_POWER)
            tr.update(dir=want, power=max(self.START_POWER, start),
                      hist=deque(), moving=False, from_facing=facing,
                      spin_t0=now)
        hist = tr["hist"]
        hist.append((now, facing))
        while hist and now - hist[0][0] > self.MOVE_WINDOW_S:
            hist.popleft()
        span = now - hist[0][0] if hist else 0.0
        # Turning is the mean of the newest readings against the oldest, and
        # has to hold for a few frames running: one jumpy heading reading is
        # not the ball breaking free, and must not decide which way it spins.
        moved = 0.0
        if len(hist) >= 6:
            ref = hist[0][1]
            old = np.mean([wrap180(f - ref) for _, f in list(hist)[:3]])
            new = np.mean([wrap180(f - ref) for _, f in list(hist)[-3:]])
            moved = float(new - old)
        turning = abs(moved) >= self.MOVE_DEG
        same = turning and (moved > 0) == (tr.get("streak_dir", 0) > 0)
        tr["streak"] = (tr.get("streak", 0) + 1) if same else (1 if turning
                                                               else 0)
        tr["streak_dir"] = (1 if moved > 0 else -1) if turning else 0

        if span >= self.MOVE_WINDOW_S * 0.75 and tr["streak"] >= 3:
            if not tr["moving"]:
                tr["moving"] = True
                self.breakaways.append(tr["power"])
                total = wrap180(facing - tr["from_facing"])
                if (total > 0) != (want > 0):
                    # A known direction is only overturned by two wrong turns
                    # in a row — one noisy reading must not lock in a spin
                    # that turns it away, nor unlock a right one.
                    self.dir_wrong += 1
                    if self.dir_known and self.dir_wrong < 2:
                        pass
                    else:
                        self.dir_known = False
                        self.dir_wrong = 0
                        self.dir_sign = -self.dir_sign
                        self.dir_flips += 1
                        if self.dir_flips > self.MAX_DIR_FLIPS:
                            self.fail("cannot tell which way the motors turn "
                                      "it — the camera heading is not "
                                      "following the spin")
                            return
                        self._stop_spin(robot, tr, facing, now)
                        return
                else:
                    self.dir_known = True
                    self.dir_wrong = 0
        elif span >= self.MOVE_WINDOW_S * 0.75 and not turning:
            # Not turning: push harder until it breaks free.
            tr["moving"] = False
            tr["power"] = min(self.MAX_POWER,
                              tr["power"] + self.RAMP_PER_S * dt)
            if tr["power"] >= self.MAX_POWER and now - tr["spin_t0"] > 6.0:
                self.fail(f"power {self.MAX_POWER:.0f} did not turn it")
                return
        if now - self.push_at >= self.PUSH_S:
            self.push_at = now
            robot.spin_raw(int(round(tr["power"] * want * self.dir_sign)))

    def _stop_spin(self, robot, tr, facing, now):
        if tr["dir"]:
            robot.stop_raw()
            self.app.lab.tracks.rezeroed(self.name)
        tr.update(phase="settle", until=now + self.SETTLE_S, still=deque(),
                  stop_facing=facing if tr["dir"] else None)

    def _aimed(self, robot, p, now):
        self.turns.append(self.turn)
        self.turn = None
        self.stage = "drive"
        if self.cur is None:
            t = self.target
            d = t - p
            length = float(np.linalg.norm(d))
            self.cur = {"samples": [], "steer": 0.0, "go_at": now,
                        "steer_at": None, "stop_at": None, "stop_xy": None,
                        "end_xy": None, "mid": ((p + t) / 2.0).tolist(),
                        "from": p.tolist(), "to": t.tolist(),
                        "length": length, "seg": "A"}
        elif self.cur["seg"] in ("A", "B"):
            # A re-aim in the middle of measuring moved the ball's zero: what
            # was measured stands, nothing after it counts.
            self.cur["seg"] = "C"

    # -- driving a leg ------------------------------------------------------------

    def _steer_for_leg(self, p):
        """A steer that bends the path toward the MIDDLE of the arena, going
        by the steering direction learned so far (a guess on the first leg,
        which the edge check covers)."""
        cur = self.cur
        u = np.subtract(cur["to"], cur["from"])
        centre = np.array([self.w / 2.0, self.h / 2.0])
        toward = wrap180(_bearing(p, centre) - _bearing([0, 0], u))
        arena_turn = 1.0 if toward >= 0 else -1.0
        signs = [g["sign"] for g in map(ball_calib.analyse_leg, self.legs)
                 if "sign" in g]
        est = (1.0 if not signs else (1.0 if sum(signs) >= 0 else -1.0))
        return est * arena_turn * self.STEER_DEG

    def _send(self, robot, heading, now):
        if now - self.push_at >= self.PUSH_S:
            self.push_at = now
            robot.drive_raw(float(heading) % 360.0, self.byte)

    def _drive(self, robot, p, facing, now):
        cur = self.cur
        gap = float(np.linalg.norm(self.target - p))
        if gap <= self.ARRIVE_CM:
            robot.stop()
            cur["stop_at"] = now
            cur["stop_xy"] = p.tolist()
            self.stage, self.coast_until = "coast", now + self.COAST_S
            self.app.say(f"calib: corner {self.leg + 1}/{len(self.route)} "
                         f"({gap:.1f}cm off)")
            return
        if not (0.0 <= p[0] <= self.w and 0.0 <= p[1] <= self.h):
            self.fail(f"left the arena on leg {self.leg + 1}")
            return
        a = np.asarray(cur["from"], float)
        u = (np.asarray(cur["to"], float) - a) / max(cur["length"], 1e-9)
        along = float(np.dot(p - a, u))
        cross = float(u[0] * (p - a)[1] - u[1] * (p - a)[0])
        stretch = min(cur["length"] / 3.0, self.STRETCH_CM)
        edge = min(p[0], self.w - p[0], p[1], self.h - p[1])

        if cur["seg"] == "A" and along >= stretch:
            cur["seg"], cur["steer_at"] = "B", now
            cur["steer"] = self._steer_for_leg(p)
        elif cur["seg"] == "B" and (along >= 2.0 * stretch
                                    or edge <= self.EDGE_CM):
            cur["seg"] = "C"
            cur["samples"].append([now, float(p[0]), float(p[1]), "C"])
            robot.stop()
            self._begin_turn(now)                  # re-aim at the corner
            return
        cur["samples"].append([now, float(p[0]), float(p[1]), cur["seg"]])

        if cur["seg"] in ("A", "B"):
            if abs(cross) > self.OFF_CM:
                self.fail(f"{abs(cross):.0f}cm off leg {self.leg + 1} while "
                          "measuring")
                return
            self._send(robot, 0.0 if cur["seg"] == "A" else cur["steer"], now)
            return

        # C: straight at the corner, re-aiming if it wanders.
        self.best_gap = gap if self.best_gap is None else min(self.best_gap,
                                                              gap)
        err = wrap180(_bearing(p, self.target) - facing)
        miss = gap * math.sin(math.radians(min(abs(err), 90.0)))
        # Never re-aim for an error the turn itself would accept.
        outside = abs(err) > float(self.app.aim_band) + 8.0
        if ((outside and miss > self.MISS_CM) or abs(err) > 90.0
                or gap > self.best_gap + self.AWAY_CM):
            last = getattr(self, "reaim_gap", None)
            if last is not None and self.best_gap > last - self.PROGRESS_CM:
                self.reaims += 1
            else:
                self.reaims = 0
            self.reaim_gap = self.best_gap
            if self.reaims >= self.MAX_REAIMS:
                self.fail(f"re-aimed {self.MAX_REAIMS} times in a row on leg "
                          f"{self.leg + 1} without getting closer than "
                          f"{self.best_gap:.0f}cm")
                return
            robot.stop()
            self.best_gap = None
            self._begin_turn(now)
            return
        self._send(robot, 0.0, now)

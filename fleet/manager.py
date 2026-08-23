"""The fleet: a bag of robots, some simulated, some real, all the same to callers.

Nothing above this module may branch on `kind`. The renderer draws a ring
around real robots and that is the only exception in the codebase — everything
else asks the fleet for positions and sends it velocities.
"""

import threading

import numpy as np

from swarm.sim import SwarmEnv
from workspace.space import Workspace

from .real_handle import SpheroRobot
from .roster import RobotEntry, Roster
from .sim_handle import SimRobot


class Fleet:
    def __init__(self, workspace=None, tracker=None, connector=None, seed=None,
                 motion_path=None):
        """`motion_path` is where measured plant constants come from.

        None means none: simulated robots keep their guessed dynamics. An app
        that wants a characterised sim to behave like the ball it was measured
        from passes `characterize.MOTION_PATH` explicitly.

        Opt-in rather than opt-out, because `calib/motion.json` is live state.
        Loading it by default would make every test's dynamics depend on
        whatever the last hardware session left on disk, and the fixtures here
        use the real robot codes — so a fleet built in a test would quietly
        inherit CRXS's measured 0.52s of command latency and start failing for
        reasons nobody would connect to a calibration run.
        """
        self.ws = workspace or Workspace.load()
        self.tracker = tracker
        self.connector = connector
        self.motion_path = motion_path
        self.rng = np.random.default_rng(seed)
        self.handles = {}                 # code -> RobotHandle, insertion ordered
        self.errors = []
        self._lock = threading.Lock()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_roster(cls, roster=None, workspace=None, tracker=None, connector=None,
                    seed=None, motion_path=None):
        roster = roster if roster is not None else Roster.load()
        f = cls(workspace=workspace, tracker=tracker, connector=connector,
                seed=seed, motion_path=motion_path)
        if roster.errors:
            f.errors = list(roster.errors)
            return f
        for entry in roster.enabled_entries():
            f.add(entry)
        return f

    def _build(self, entry, pos=None):
        if entry.kind == "real":
            return SpheroRobot(entry.name, entry.code, entry.color, entry.ble_name,
                               heading_offset=getattr(entry, "heading_offset", 0.0),
                               workspace=self.ws, tracker=self.tracker,
                               connector=self.connector)
        # A simulated robot that has been characterised should behave like the
        # ball it was measured from. The battery already computes these and,
        # until now, only printed them for somebody to copy by hand — so the
        # sim ran a robot several times more responsive than the hardware, and
        # a controller tuned against it was tuned against fiction.
        return SimRobot(entry.name, entry.code, entry.color, workspace=self.ws,
                        pos=pos, seed=int(self.rng.integers(1 << 31)),
                        motion=self._motion_for(entry.code))

    def _motion_for(self, code):
        """Measured constants for one robot, or {} — never raising, never
        letting a malformed calibration stop a fleet from being built."""
        if not self.motion_path:
            return {}
        try:
            from .characterize import load_motion
            from .sim_handle import from_motion
            return from_motion(load_motion(code, path=self.motion_path))
        except Exception:
            return {}

    # -- membership ------------------------------------------------------

    def add(self, entry):
        """Join a robot to the running fleet. Returns a list of errors."""
        if isinstance(entry, dict):
            entry = RobotEntry.from_dict(entry)
        with self._lock:
            if entry.code in self.handles:
                return [f"{entry.code} is already in the fleet"]
            clash = next((h for h in self.handles.values()
                          if h.color == entry.color), None)
            if clash is not None:
                return [f"colour {entry.color!r} is already held by {clash.code} "
                        f"— the tracker could not tell them apart"]
            if entry.kind == "real" and not entry.ble_name:
                return [f"{entry.code} is real but has no ble_name"]
            try:
                self.handles[entry.code] = self._build(entry)
            except Exception as e:
                return [f"could not build {entry.code}: {e}"]
        return []

    def remove(self, code):
        with self._lock:
            h = self.handles.pop(code, None)
        if h is None:
            return [f"{code} is not in the fleet"]
        try:
            h.stop()
        except Exception:
            pass
        h.close()
        return []

    def rename(self, code, name=None, new_code=None):
        """Change a robot's display name and/or its code, keeping fleet order."""
        with self._lock:
            h = self.handles.get(code)
            if h is None:
                return [f"{code} is not in the fleet"]

            if new_code and new_code != code:
                if not new_code.strip():
                    return ["a code cannot be empty"]
                if new_code in self.handles:
                    return [f"{new_code} is already in the fleet"]
                h.code = new_code
                # rebuild in place: dict order is the roster order the UI draws
                self.handles = {(new_code if k == code else k): v
                                for k, v in self.handles.items()}

            if name:
                h.name = name
        return []

    def set_kind(self, code, kind, ble_name=None):
        """Flip a robot between sim and real in place, leaving the others alone."""
        if kind not in ("sim", "real"):
            return [f"kind must be 'sim' or 'real', got {kind!r}"]
        with self._lock:
            old = self.handles.get(code)
            if old is None:
                return [f"{code} is not in the fleet"]
            if old.kind == kind:
                return []
            if kind == "real" and not (ble_name or getattr(old, "ble_name", None)):
                return [f"{code} needs a ble_name to become real"]

            entry = RobotEntry(name=old.name, code=old.code, kind=kind,
                               color=old.color,
                               ble_name=ble_name or getattr(old, "ble_name", None))
            try:
                new = self._build(entry, pos=old.pos.copy())
            except Exception as e:
                return [f"could not switch {code} to {kind}: {e}"]

            new.pos = old.pos.copy()
            new.vel = old.vel.copy()
            new.rgb, new.blink, new.target = old.rgb, old.blink, old.target
            self.handles[code] = new

        try:
            old.stop()
        except Exception:
            pass
        old.close()
        return []

    # -- access ------------------------------------------------------------

    def __len__(self):
        return len(self.handles)

    def __contains__(self, code):
        return code in self.handles

    def __getitem__(self, code):
        return self.handles[code]

    def get(self, code):
        return self.handles.get(code)

    @property
    def codes(self):
        return list(self.handles)

    @property
    def connected_codes(self):
        return [c for c, h in self.handles.items() if h.connected]

    def connected_handles(self):
        return [h for h in self.handles.values() if h.connected]

    def positions(self):
        return {c: h.pos.copy() for c, h in self.handles.items()}

    # -- the loop ------------------------------------------------------------

    def step(self, dt):
        """Advance sim robots, pull fresh fixes for real ones. Never raises."""
        for h in list(self.handles.values()):
            try:
                h.step(dt)
            except Exception as e:
                self.errors.append(f"{h.code}: {e}")

    def set_velocities(self, mapping):
        for code, v in mapping.items():
            h = self.handles.get(code)
            if h is not None:
                h.set_velocity(v)

    def stop(self, codes=None):
        for code in (codes if codes is not None else list(self.handles)):
            h = self.handles.get(code)
            if h is None:
                continue
            try:
                h.stop()
            except Exception as e:
                self.errors.append(f"{code}: {e}")

    def close(self):
        for h in list(self.handles.values()):
            try:
                h.stop()
            except Exception:
                pass
            h.close()
        self.handles.clear()

    # -- reporting ---------------------------------------------------------

    def state(self):
        """Uniform per-robot dict, keyed by code. The shape every tool reads."""
        return {c: h.state() for c, h in self.handles.items()}

    def status(self):
        """Fleet-wide numbers for the status line."""
        real = [h for h in self.handles.values() if h.kind == "real"]
        rtts = [h.rtt_mean for h in real if getattr(h, "rtt_mean", None) is not None]
        return {
            "robots": len(self.handles),
            "connected": len(self.connected_codes),
            "real": len(real),
            "real_linked": sum(1 for h in real if getattr(h, "link_up", False)),
            "mean_rtt_ms": round(float(np.mean(rtts)) * 1000, 1) if rtts else None,
            "tracker_fps": round(float(getattr(self.tracker, "fps", 0.0)), 1)
                           if self.tracker is not None else None,
            "max_connections_hit": any(getattr(h, "max_connections_hit", False)
                                       for h in real),
        }


class FleetEnv(SwarmEnv):
    """A `SwarmEnv` whose positions are overwritten by the fleet each tick.

    Reusing the env means controllers see observations built by exactly the same
    code that produced the training data — the most common source of transfer
    bugs — and it means `Boids`, a trained policy and `Navigate` all run against
    real robots with no changes.
    """

    def __init__(self, fleet, task="coverage"):
        self.fleet = fleet
        self.codes = list(fleet.handles)
        super().__init__(max(len(self.codes), 1), task, randomize=False, seed=0,
                         workspace=fleet.ws)
        self.targets = {}
        self.sync()

    def rebuild_if_needed(self):
        """Membership can change while running; keep the array shapes honest."""
        codes = list(self.fleet.handles)
        if codes != self.codes:
            self.codes = codes
            self.n = max(len(codes), 1)
            self.pos = np.zeros((self.n, 2))
            self.vel = np.zeros((self.n, 2))
            self.visited = np.zeros_like(self.visited)
            self.sync()
            return True
        return False

    def sync(self):
        if not self.codes:
            return
        self.pos = np.array([self.fleet.handles[c].pos for c in self.codes], dtype=float)
        self.vel = np.array([self.fleet.handles[c].vel for c in self.codes], dtype=float)
        self._mark_visited()

    def target_array(self):
        """(n, 2) of assigned targets, falling back to the robot's own position."""
        out = np.zeros((len(self.codes), 2))
        for i, c in enumerate(self.codes):
            t = self.targets.get(c)
            out[i] = t if t is not None else self.fleet.handles[c].pos
        return out

    def apply(self, actions, max_speed=None):
        """Send controller output (n, 2) in [-1, 1] to the fleet as cm/s.

        A robot the camera has lost is STOPPED, not skipped. Those are
        different instructions to a Sphero: it holds its last speed command
        until it is given another one, so declining to update a robot leaves it
        driving on whatever it was last told, for as long as the fix stays
        lost. Skipping is how a point-to-point move ends with the ball circling
        the room.

        Commanding one from a position that stopped updating is the worse
        failure of the two, because the controller is then steering confidently
        by a number that is no longer a measurement. `connected` is exactly the
        question "do we know where this robot is", and it is answerable here.

        Simulated robots are always connected, so this changes nothing for
        them — the branch exists for balls on a floor.
        """
        from .handle import MAX_SPEED
        scale = max_speed or MAX_SPEED
        a = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
        for i, c in enumerate(self.codes):
            h = self.fleet.handles.get(c)
            if h is None:
                continue
            if h.connected:
                h.set_velocity(a[i] * scale)
            else:
                h.stop()

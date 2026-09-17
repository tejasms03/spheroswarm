"""Shared fixtures. No test in this suite may touch a radio or a camera."""

import threading

import numpy as np
import pytest

from fleet.roster import RobotEntry
from workspace.space import Workspace


class FakeApi:
    """Stands in for spherov2's SpheroEduAPI. Records every write."""

    def __init__(self, ble_name="SK-TEST", fail_on_write=False, latency=0.0):
        # The real SpheroEduAPI keeps the heading and speed as state and sends
        # BOTH on either setter — `set_heading` calls roll_start(heading,
        # speed). Modelling them as two independent recorders made a handle
        # that sends one packet instead of two look like it had sent nothing.
        self._SpheroEduAPI__speed = 0
        self._SpheroEduAPI__updating = threading.RLock()
        self.commands = []              # (heading, speed) actually rolled
        self.ble_name = ble_name
        self.fail_on_write = fail_on_write
        self.latency = latency
        self.headings = []
        self.speeds = []
        self.leds = []
        self.back_leds = []
        self.closed = False
        self.lock = threading.Lock()
        # One ordered log across every kind of write, so a test can say what
        # happened FIRST — which is the whole question for a spin that has to
        # stop, restore stabilisation and re-zero before a roll can run.
        self.events = []
        # TWO stabilisation states, because the library has two. `stabilized`
        # is the ball; `_SpheroEduAPI__stabilization` is the API's own flag.
        # `reset_aim` changes the first without the second, and `raw_motor`
        # only switches stabilisation off if the SECOND says it is on — so a
        # fake with one merged flag would pass the exact case the real library
        # gets wrong: a spin that fights a stabiliser nobody switched off.
        self.stabilized = True
        self._SpheroEduAPI__stabilization = True
        self.motors = (0, 0)
        self.fought = False

    def set_heading(self, h):
        if self.fail_on_write:
            raise RuntimeError("simulated BLE write failure")
        with self.lock:
            self.events.append(("heading", h, self._SpheroEduAPI__speed,
                                self.stabilized))
            self.headings.append(h)
            # roll_start(heading, speed) — the stored speed goes with it.
            self.speeds.append(self._SpheroEduAPI__speed)
            self.commands.append((h, self._SpheroEduAPI__speed))

    def set_speed(self, s):
        if self.fail_on_write:
            raise RuntimeError("simulated BLE write failure")
        self._SpheroEduAPI__speed = s
        with self.lock:
            self.speeds.append(s)
            self.commands.append((self.headings[-1] if self.headings else 0, s))
            self.speeds.append(s)

    def raw_motor(self, left, right, duration):
        """As the library does it: read the API flag at the START, switch
        stabilisation off only if that flag says it is on, and on a numeric
        duration turn the motors off — restoring stabilisation only if the flag
        read at the start said so."""
        if self.fail_on_write:
            raise RuntimeError("simulated BLE write failure")
        stabilize = self._SpheroEduAPI__stabilization
        if stabilize:
            self.set_stabilization(False)
        with self.lock:
            self.motors = (int(left), int(right))
            if (left or right) and self.stabilized:
                self.fought = True        # motors driven against the stabiliser
            self.events.append(("raw", int(left), int(right), duration))
            if duration is not None:
                self.motors = (0, 0)
                self.events.append(("raw_off",))
        if duration is not None and stabilize:
            self.set_stabilization(True)

    def set_stabilization(self, on):
        with self.lock:
            self.stabilized = bool(on)
            self._SpheroEduAPI__stabilization = bool(on)
            self.events.append(("stab", bool(on)))

    def reset_aim(self):
        """Through the toy, not the API: the ball ends stabilised, the API's
        flag is left exactly as it was."""
        with self.lock:
            self.stabilized = True
            self.events.append(("zero",))

    def set_main_led(self, color):
        with self.lock:
            self.leds.append(color)

    def set_back_led(self, value):
        """The aiming taillight. An int is brightness, a Color is a BOLT's."""
        if self.fail_on_write:
            raise RuntimeError("simulated BLE write failure")
        with self.lock:
            self.back_leds.append(value)

    def __exit__(self, *a):
        self.closed = True

    @property
    def writes(self):
        """Every (heading, speed) actually rolled, in order.

        Zipping the two setter logs was fine while the handle always called
        both. It sends one packet now — `set_heading` carries the speed — so a
        speed-only write advanced one list and not the other, and the zip
        silently dropped it. A stop then looked like it had never happened.
        """
        with self.lock:
            return list(self.commands)


class FakeConnector:
    """A connector that hands out FakeApis, or fails on demand."""

    def __init__(self, fail=False, error=None):
        self.fail = fail
        self.error = error or RuntimeError("no such toy")
        self.apis = {}
        self.calls = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def __call__(self, ble_name, timeout=8.0):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.calls.append(ble_name)
        try:
            if self.fail:
                raise self.error
            api = FakeApi(ble_name)
            self.apis[ble_name] = api
            return api
        finally:
            with self._lock:
                self.concurrent -= 1


class FakeTracker:
    """Returns whatever fixes the test puts in it, keyed by colour."""

    def __init__(self, fixes=None, fps=30.0):
        self.fixes = fixes or {}
        self.fps = fps

    def read(self):
        return {k: np.asarray(v, dtype=float) for k, v in self.fixes.items()}


@pytest.fixture(autouse=True)
def no_real_bluetooth(monkeypatch):
    """Hard guarantee: the suite never opens a radio.

    Anything that reaches `default_connector` has forgotten to inject a fake,
    and on macOS that call aborts the interpreter from a worker thread rather
    than failing a test — so fail loudly instead. The connect stagger is also
    shortened, since the fleet-wide delay exists for real radios only.
    """
    import fleet.real_handle as rh

    def forbidden(ble_name, timeout=8.0):
        raise AssertionError(
            f"test tried to open a real BLE connection to {ble_name!r} — "
            "pass connector=FakeConnector()")

    monkeypatch.setattr(rh, "default_connector", forbidden)
    monkeypatch.setattr(rh, "CONNECT_STAGGER", 0.02)
    rh._last_connect_at[0] = 0.0
    yield


@pytest.fixture
def ws():
    return Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[{"type": "circle", "center": [120, 90], "radius": 12}],
    )


@pytest.fixture
def open_ws():
    return Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])


@pytest.fixture
def sim_entries():
    colors = ["cyan", "red", "yellow", "green", "magenta", "blue"]
    names = ["Seasmoke", "Caraxes", "Syrax", "Vhagar", "Meleys", "Sunfyre"]
    codes = ["SSMK", "CRXS", "SYRX", "VHGR", "MLYS", "SNFR"]
    return [RobotEntry(name=n, code=c, kind="sim", color=col)
            for n, c, col in zip(names, codes, colors)]


@pytest.fixture
def sim_ctx(tmp_path):
    """A live all-sim SwarmContext on scratch paths.

    Everything the LLM layer is tested against runs through this: a real fleet,
    a real workspace, a real validator — only the model is faked. Formations go
    to tmp_path so a test can never write the repo's formations.json.
    """
    from fleet.manager import Fleet
    from tools import SwarmContext
    from tools.formations import FormationLibrary

    def build(n=6, seed=0, obstacles=None, bounds=None, positions=None):
        space = Workspace(
            bounds_cm=bounds or [[0, 0], [240, 0], [240, 180], [0, 180]],
            obstacles=[{"type": "circle", "center": [180, 40], "radius": 15}]
            if obstacles is None else obstacles,
        )
        colors = ["cyan", "red", "yellow", "green", "magenta", "blue"]
        names = ["Seasmoke", "Caraxes", "Syrax", "Vhagar", "Meleys", "Sunfyre"]
        codes = ["SSMK", "CRXS", "SYRX", "VHGR", "MLYS", "SNFR"]

        fleet = Fleet(workspace=space, seed=seed)
        for i in range(n):
            fleet.add(RobotEntry(name=names[i], code=codes[i], kind="sim",
                                 color=colors[i]))
        if positions is not None:
            for code, p in zip(codes[:n], positions):
                fleet[code].pos = np.asarray(p, dtype=float)

        return SwarmContext(
            fleet=fleet, workspace=space,
            library=FormationLibrary(path=tmp_path / "formations.json"))

    return build


@pytest.fixture
def mixed_ctx(tmp_path):
    """Sim robots plus one *real* robot, connected or not, on scratch paths.

    SimRobot.connected is hardcoded True by design, so the only honest way to
    exercise the disconnected path is a real handle whose tracker has no fix.
    """
    from fleet.manager import Fleet
    from tools import SwarmContext
    from tools.formations import FormationLibrary

    def build(n_sim=5, tracked=False, seed=0):
        space = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
        colors = ["cyan", "red", "yellow", "green", "magenta"]
        names = ["Seasmoke", "Caraxes", "Syrax", "Vhagar", "Meleys"]
        codes = ["SSMK", "CRXS", "SYRX", "VHGR", "MLYS"]

        fixes = {"blue": np.array([100.0, 100.0])} if tracked else {}
        fleet = Fleet(workspace=space, seed=seed,
                      tracker=FakeTracker(fixes=fixes),
                      connector=FakeConnector())
        for i in range(n_sim):
            fleet.add(RobotEntry(name=names[i], code=codes[i], kind="sim",
                                 color=colors[i]))
        fleet.add(RobotEntry(name="Sunfyre", code="SNFR", kind="real",
                             color="blue", ble_name="SK-BLUE"))
        fleet.step(0.1)
        return SwarmContext(
            fleet=fleet, workspace=space,
            library=FormationLibrary(path=tmp_path / "formations.json"))

    return build


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """Hard guarantee: the suite never depends on a running Ollama.

    Without this, whether `App()` starts in ask mode or tool mode depends on
    whether someone happens to have `ollama serve` up — so the same commit
    passes on one machine and fails on another. Tests that want a model inject
    a client explicitly and set `available` themselves.
    """
    try:
        from llm.session import AgentSession
    except Exception:
        return
    # keep the real one reachable so a test can still exercise it directly
    monkeypatch.setattr(AgentSession, "real_probe", AgentSession.probe,
                        raising=False)
    monkeypatch.setattr(AgentSession, "probe", lambda self, timeout=2.0: False)


@pytest.fixture(autouse=True)
def calib_dir_is_off_limits(request, monkeypatch):
    """No test may write into the live `calib/` directory.

    `calib/` is live state exactly as roster.json and workspace.json are — the
    bench writes colour signatures and motion fits into it by design. A test
    that leaks a signature learned from the synthetic camera leaves the real
    red hue pointing somewhere else, and the next session on hardware fails to
    see a ball for reasons nobody would connect to a test run.

    Written as a guard rather than as discipline because the leak this caught
    was ordering-dependent: the suite was clean file by file and clean under
    `-p no:randomly`, and only some shuffles reproduced it. That is exactly the
    kind of thing that gets diagnosed once and then reintroduced.
    """
    import hashlib
    from pathlib import Path

    live = Path(__file__).resolve().parent.parent / "calib"

    def snapshot():
        if not live.exists():
            return {}
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(live.glob("*"))}

    # Whether the write came from THIS process. A developer with `calib.py`
    # open is editing the same directory — the bench saves colour signatures
    # whenever they auto-tune — and blaming whichever test happened to be
    # running when that landed sends the next person hunting a leak that is
    # somebody else's app doing its job.
    from vision import config as vconfig
    wrote = []
    real_save = vconfig.save

    def watched(name, data):
        if (vconfig.CALIB / f"{name}.json").resolve().parent == live.resolve():
            wrote.append(name)
        return real_save(name, data)

    monkeypatch.setattr(vconfig, "save", watched)

    before = snapshot()
    yield
    after = snapshot()
    if after != before and not wrote:
        import warnings
        warnings.warn(
            f"calib/ changed during {request.node.nodeid} but this process did "
            "not write it — something else is using the directory (calib.py "
            "open?). Not failing the test.")
        return
    if after != before:
        added = set(after) - set(before)
        changed = {k for k in set(after) & set(before) if after[k] != before[k]}
        removed = set(before) - set(after)
        # Put it back, so one offending test does not cascade into the rest.
        for name in added:
            (live / name).unlink(missing_ok=True)
        raise AssertionError(
            f"{request.node.nodeid} wrote into the live calib/ directory — "
            f"added {sorted(added)}, changed {sorted(changed)}, "
            f"removed {sorted(removed)}. Point it at tmp_path instead "
            "(monkeypatch vision.config.CALIB and "
            "fleet.characterize.MOTION_PATH).")

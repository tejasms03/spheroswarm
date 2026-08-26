import time

import numpy as np
import pytest

from fleet.handle import velocity_to_command
from fleet.manager import Fleet, FleetEnv
from fleet.real_handle import SpheroRobot, is_max_connections_error
from fleet.roster import RobotEntry, Roster
from fleet.sim_handle import SimRobot
from tests.conftest import FakeApi, FakeConnector, FakeTracker


# -- sim handles --------------------------------------------------------

def test_sim_robot_moves_toward_commanded_velocity(open_ws):
    r = SimRobot("Seasmoke", "SSMK", "cyan", workspace=open_ws,
                 pos=[100, 90], randomize=False)
    r.set_velocity([30.0, 0.0])
    for _ in range(60):
        r.step(0.05)
    assert r.pos[0] > 100.0
    assert r.connected is True


def test_sim_robot_stays_in_workspace(ws):
    r = SimRobot("Seasmoke", "SSMK", "cyan", workspace=ws, pos=[20, 20],
                 randomize=False)
    r.set_velocity([60.0, 60.0])
    for _ in range(400):
        r.step(0.05)
        assert ws.is_valid_point(r.pos), r.pos


def test_sim_robot_battery_drains(open_ws):
    r = SimRobot("Seasmoke", "SSMK", "cyan", workspace=open_ws, pos=[10, 10])
    start = r.battery
    for _ in range(200):
        r.step(0.5)
    assert r.battery < start


def test_sim_robot_ignores_garbage_velocity(open_ws):
    r = SimRobot("Seasmoke", "SSMK", "cyan", workspace=open_ws, pos=[50, 50],
                 randomize=False)
    r.set_velocity([float("nan"), 3.0])
    r.set_velocity(["a", "b"] if False else [np.inf, 0.0])
    for _ in range(10):
        r.step(0.1)
    assert np.isfinite(r.pos).all()


def test_sim_robot_stop(open_ws):
    r = SimRobot("Seasmoke", "SSMK", "cyan", workspace=open_ws, pos=[100, 90],
                 randomize=False)
    r.set_velocity([40.0, 0.0])
    for _ in range(20):
        r.step(0.05)
    r.stop()
    for _ in range(120):
        r.step(0.05)
    assert r.speed < 1.0


# -- command conversion --------------------------------------------------

def test_velocity_to_command_matches_camera_frame():
    # heading 0 is +y (down in the camera frame), increasing clockwise
    assert velocity_to_command([0, 10])[0] == pytest.approx(0.0)
    assert velocity_to_command([10, 0])[0] == pytest.approx(90.0)
    assert velocity_to_command([0, -10])[0] == pytest.approx(180.0)
    assert velocity_to_command([-10, 0])[0] == pytest.approx(270.0)


def test_velocity_to_command_zero_and_garbage():
    assert velocity_to_command([0, 0]) == (0.0, 0)
    assert velocity_to_command([float("nan"), 0]) == (0.0, 0)


def test_velocity_to_command_speed_byte():
    assert velocity_to_command([0, 60])[1] == 255
    assert velocity_to_command([0, 30])[1] == pytest.approx(127, abs=2)


# -- real handles, all mocked -------------------------------------------

def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_real_robot_connects_and_writes(open_ws):
    conn = FakeConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, tracker=None, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        r.set_velocity([0.0, 60.0])
        assert _wait(lambda: len(conn.apis["SK-1A2B"].writes) >= 1)
        heading, speed = conn.apis["SK-1A2B"].writes[0]
        assert heading == 0 and speed == 255
    finally:
        r.close()


def test_real_robot_without_tracker_fix_is_disconnected(open_ws):
    conn = FakeConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, tracker=FakeTracker({}), connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        r.step(0.1)
        assert r.connected is False        # BLE up, but we do not know where it is
        assert r.link_up is True
    finally:
        r.close()


def test_real_robot_with_fix_is_connected_then_goes_stale(open_ws):
    conn = FakeConnector()
    tracker = FakeTracker({"cyan": (100.0, 90.0)})
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, tracker=tracker, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        r.step(0.1)
        assert r.connected is True
        assert np.allclose(r.pos, [100.0, 90.0])

        tracker.fixes = {}                 # camera loses it
        r.step(0.1)
        time.sleep(0.55)                   # STALE_AFTER is 0.5s
        assert r.connected is False
    finally:
        r.close()


def test_real_robot_velocity_differentiated_from_fixes(open_ws):
    conn = FakeConnector()
    tracker = FakeTracker({"cyan": (100.0, 90.0)})
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, tracker=tracker, connector=conn)
    try:
        r.step(0.05)
        time.sleep(0.05)
        tracker.fixes = {"cyan": (105.0, 90.0)}
        r.step(0.05)
        assert r.vel[0] > 0
    finally:
        r.close()


def test_real_robot_deadbands_small_changes(open_ws):
    conn = FakeConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        api = conn.apis["SK-1A2B"]
        r.set_velocity([0.0, 40.0])
        assert _wait(lambda: len(api.writes) == 1)

        for _ in range(5):                       # a 2-degree wobble, same speed
            r.set_velocity([1.0, 40.0])
            time.sleep(0.05)
        assert len(api.writes) == 1, api.writes

        r.set_velocity([40.0, 0.0])              # a 90-degree turn gets through
        assert _wait(lambda: len(api.writes) == 2)
    finally:
        r.close()


def test_real_robot_stop_always_gets_through(open_ws):
    conn = FakeConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        api = conn.apis["SK-1A2B"]
        r.set_velocity([0.0, 40.0])
        assert _wait(lambda: len(api.writes) == 1)
        r.stop()
        assert _wait(lambda: any(s == 0 for _, s in api.writes))
    finally:
        r.close()


def test_real_robot_that_never_connects_is_visible_not_fatal(open_ws):
    conn = FakeConnector(fail=True)
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-DEAD",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: conn.calls != [])
        assert r.link_up is False
        assert r.connected is False
        assert r.last_error
        r.set_velocity([10.0, 0.0])        # must not raise
        r.step(0.1)
        assert r.state()["connected"] is False
    finally:
        r.close()


def test_max_connections_error_is_named():
    err = RuntimeError(
        "Error Domain=CBErrorDomain Code=11 \"The connection has failed "
        "unexpectedly.\" maximum connections reached")
    assert is_max_connections_error(err)
    assert not is_max_connections_error(RuntimeError("timed out"))


def test_max_connections_flag_is_set(open_ws):
    err = RuntimeError("Error Domain=CBErrorDomain Code=11 max connections")
    conn = FakeConnector(fail=True, error=err)
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: r.max_connections_hit)
    finally:
        r.close()


def test_connects_are_serialised(open_ws):
    conn = FakeConnector()
    robots = [SpheroRobot(f"R{i}", f"R{i}", c, f"SK-{i}", workspace=open_ws,
                          connector=conn)
              for i, c in enumerate(["cyan", "red", "yellow"])]
    try:
        assert _wait(lambda: len(conn.calls) == 3, timeout=8.0)
        assert conn.max_concurrent == 1
    finally:
        for r in robots:
            r.close()


def test_reconnect_after_write_failure(open_ws, monkeypatch):
    import fleet.real_handle as rh
    monkeypatch.setattr(rh, "BACKOFF_START", 0.05)
    monkeypatch.setattr(rh, "CONNECT_STAGGER", 0.0)

    conn = FakeConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        conn.apis["SK-1A2B"].fail_on_write = True

        r.set_velocity([0.0, 40.0])
        # The failed write drops the link; the supervisor dials again on its own.
        assert _wait(lambda: len(conn.calls) >= 2, timeout=5.0)
        assert _wait(lambda: r.link_up)

        r.set_velocity([40.0, 0.0])                  # the fresh link carries traffic
        assert _wait(lambda: conn.apis["SK-1A2B"].writes != [])
    finally:
        r.close()


def test_set_velocity_does_not_block_the_caller(open_ws):
    class SlowApi(FakeApi):
        def set_heading(self, h):
            time.sleep(0.3)
            super().set_heading(h)

    class SlowConnector(FakeConnector):
        def __call__(self, ble_name, timeout=8.0):
            api = SlowApi(ble_name)
            self.apis[ble_name] = api
            self.calls.append(ble_name)
            return api

    conn = SlowConnector()
    r = SpheroRobot("Seasmoke", "SSMK", "cyan", "SK-1A2B",
                    workspace=open_ws, connector=conn)
    try:
        assert _wait(lambda: r.link_up)
        t0 = time.time()
        for i in range(10):
            r.set_velocity([float(i * 6), 40.0])
        assert time.time() - t0 < 0.1
    finally:
        r.close()


# -- the fleet ------------------------------------------------------------

def test_fleet_from_roster_all_sim(open_ws, sim_entries):
    roster = Roster(entries=sim_entries, path="/dev/null")
    f = Fleet.from_roster(roster, workspace=open_ws)
    try:
        assert len(f) == 6
        assert set(f.codes) == {e.code for e in sim_entries}
        st = f.state()
        assert set(st) == set(f.codes)
        for code, d in st.items():
            assert {"code", "name", "kind", "color", "pos", "vel",
                    "connected", "battery", "led", "last_seen"} <= set(d)
    finally:
        f.close()


def test_mixed_fleet_produces_uniform_state(open_ws, sim_entries):
    entries = list(sim_entries[:4])
    entries[1] = RobotEntry(name="Caraxes", code="CRXS", kind="real",
                            color="red", ble_name="SK-1A2B")
    entries[3] = RobotEntry(name="Vhagar", code="VHGR", kind="real",
                            color="green", ble_name="SK-3C4D")
    roster = Roster(entries=entries, path="/dev/null")
    f = Fleet.from_roster(roster, workspace=open_ws, connector=FakeConnector(),
                          tracker=FakeTracker({"red": (50, 50), "green": (60, 60)}))
    try:
        st = f.state()
        assert len(st) == 4
        keysets = [frozenset(d) for d in st.values()]
        common = set.intersection(*[set(k) for k in keysets])
        assert {"code", "name", "kind", "color", "pos", "vel", "connected",
                "battery", "led", "last_seen", "target"} <= common
        assert {d["kind"] for d in st.values()} == {"sim", "real"}
    finally:
        f.close()


def test_fleet_step_advances_sim_and_pulls_real(open_ws, sim_entries):
    entries = [sim_entries[0],
               RobotEntry(name="Caraxes", code="CRXS", kind="real",
                          color="red", ble_name="SK-1A2B")]
    tracker = FakeTracker({"red": (77.0, 88.0)})
    f = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                          workspace=open_ws, connector=FakeConnector(), tracker=tracker)
    try:
        f["SSMK"].set_velocity([30.0, 0.0])
        before = f["SSMK"].pos.copy()
        for _ in range(20):
            f.step(0.05)
        assert not np.allclose(f["SSMK"].pos, before)
        assert np.allclose(f["CRXS"].pos, [77.0, 88.0])
    finally:
        f.close()


def test_fleet_add_and_remove_live(open_ws, sim_entries):
    f = Fleet.from_roster(Roster(entries=sim_entries[:3], path="/dev/null"),
                          workspace=open_ws)
    try:
        assert len(f) == 3
        errs = f.add(RobotEntry(name="Vhagar", code="VHGR", kind="sim", color="green"))
        assert errs == []
        assert len(f) == 4 and "VHGR" in f

        assert f.add(RobotEntry(name="Dup", code="DUPE", kind="sim", color="green"))
        assert len(f) == 4                       # colour clash refused

        assert f.remove("VHGR") == []
        assert len(f) == 3 and "VHGR" not in f
        assert f.remove("NOPE")                  # readable error, no raise
    finally:
        f.close()


def test_set_kind_flips_live_without_disturbing_others(open_ws, sim_entries):
    f = Fleet.from_roster(Roster(entries=sim_entries[:3], path="/dev/null"),
                          workspace=open_ws, connector=FakeConnector(),
                          tracker=FakeTracker({"cyan": (30.0, 30.0)}))
    try:
        for h in f.handles.values():
            h.set_velocity([10.0, 0.0])
        for _ in range(10):
            f.step(0.05)

        others_before = {c: f[c].pos.copy() for c in ("CRXS", "SYRX")}
        moved_pos = f["SSMK"].pos.copy()

        errs = f.set_kind("SSMK", "real", ble_name="SK-1A2B")
        assert errs == []
        assert f["SSMK"].kind == "real"
        assert np.allclose(f["SSMK"].pos, moved_pos)     # position carried over
        assert len(f) == 3

        for c, p in others_before.items():
            assert f[c].kind == "sim"
            assert np.allclose(f[c].pos, p)

        assert f.set_kind("CRXS", "real") != []          # no ble_name, refused
        assert f["CRXS"].kind == "sim"

        assert f.set_kind("SSMK", "sim") == []
        assert f["SSMK"].kind == "sim"
    finally:
        f.close()


def test_fleet_degrades_when_a_real_robot_never_connects(open_ws, sim_entries):
    entries = [sim_entries[0],
               RobotEntry(name="Caraxes", code="CRXS", kind="real",
                          color="red", ble_name="SK-DEAD")]
    f = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                          workspace=open_ws, connector=FakeConnector(fail=True),
                          tracker=FakeTracker({}))
    try:
        for _ in range(20):
            f.step(0.05)
        st = f.state()
        assert st["SSMK"]["connected"] is True
        assert st["CRXS"]["connected"] is False
        assert f.connected_codes == ["SSMK"]
    finally:
        f.close()


def test_fleet_status_reports_hardware_facts(open_ws, sim_entries):
    entries = [sim_entries[0],
               RobotEntry(name="Caraxes", code="CRXS", kind="real",
                          color="red", ble_name="SK-1A2B")]
    tracker = FakeTracker({"red": (50, 50)}, fps=28.5)
    f = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                          workspace=open_ws, connector=FakeConnector(), tracker=tracker)
    try:
        s = f.status()
        assert s["robots"] == 2 and s["real"] == 1
        assert s["tracker_fps"] == 28.5
    finally:
        f.close()


def test_nothing_above_the_fleet_needs_to_branch_on_kind(open_ws, sim_entries):
    """The uniform state dict is the whole point of the layer."""
    entries = [sim_entries[0],
               RobotEntry(name="Caraxes", code="CRXS", kind="real",
                          color="red", ble_name="SK-1A2B")]
    f = Fleet.from_roster(Roster(entries=entries, path="/dev/null"),
                          workspace=open_ws, connector=FakeConnector(),
                          tracker=FakeTracker({"red": (50, 50)}))
    try:
        f.set_velocities({c: [10.0, 0.0] for c in f.codes})
        for _ in range(5):
            f.step(0.05)
        f.stop()
        for d in f.state().values():
            assert isinstance(d["pos"], list) and len(d["pos"]) == 2
    finally:
        f.close()


# -- the env view --------------------------------------------------------

def test_fleet_env_mirrors_positions(open_ws, sim_entries):
    f = Fleet.from_roster(Roster(entries=sim_entries[:4], path="/dev/null"),
                          workspace=open_ws)
    try:
        env = FleetEnv(f)
        assert env.pos.shape == (4, 2)
        for i, c in enumerate(env.codes):
            assert np.allclose(env.pos[i], f[c].pos)

        env.apply(np.ones((4, 2)) * 0.5)
        for _ in range(10):
            f.step(0.05)
        env.sync()
        for i, c in enumerate(env.codes):
            assert np.allclose(env.pos[i], f[c].pos)
    finally:
        f.close()


def test_fleet_env_rebuilds_when_membership_changes(open_ws, sim_entries):
    f = Fleet.from_roster(Roster(entries=sim_entries[:3], path="/dev/null"),
                          workspace=open_ws)
    try:
        env = FleetEnv(f)
        assert env.pos.shape == (3, 2)
        f.add(RobotEntry(name="Vhagar", code="VHGR", kind="sim", color="green"))
        assert env.rebuild_if_needed() is True
        assert env.pos.shape == (4, 2)
        assert env.rebuild_if_needed() is False
    finally:
        f.close()


def test_boids_runs_against_a_live_fleet(open_ws, sim_entries):
    from swarm.boids import Boids
    f = Fleet.from_roster(Roster(entries=sim_entries, path="/dev/null"),
                          workspace=open_ws)
    try:
        env = FleetEnv(f)
        for _ in range(40):
            env.sync()
            env.apply(Boids().act(env))
            f.step(0.1)
        for h in f.handles.values():
            assert open_ws.is_valid_point(h.pos)
    finally:
        f.close()


# -- per-robot heading offset ------------------------------------------------
#
# The camera measures position but never orientation, and a Sphero drives in
# its own aim frame. The offset between them is the one number that has to be
# calibrated per robot, and it lives below the fleet so nothing above learns
# that heading exists at all.

def test_heading_offset_rotates_the_commanded_heading(open_ws):
    from fleet.real_handle import SpheroRobot

    conn = FakeConnector()
    plain = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                        connector=conn, autostart=False)
    turned = SpheroRobot("B", "BBBB", "red", "SK-0002", workspace=open_ws,
                         connector=conn, autostart=False, heading_offset=90.0)

    plain.set_velocity(np.array([0.0, 30.0]))     # +y
    turned.set_velocity(np.array([0.0, 30.0]))

    assert plain._pending[0] == pytest.approx(0.0)
    assert turned._pending[0] == pytest.approx(90.0)
    assert plain._pending[1] == turned._pending[1], "speed must be untouched"


def test_heading_offset_wraps(open_ws):
    from fleet.real_handle import SpheroRobot

    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=FakeConnector(), autostart=False,
                    heading_offset=300.0)
    h.set_velocity(np.array([30.0, 0.0]))          # +x is heading 90
    assert h._pending[0] == pytest.approx(30.0)    # 90 + 300 -> 390 -> 30


def test_a_zero_offset_changes_nothing(open_ws):
    from fleet.real_handle import SpheroRobot

    h = SpheroRobot("A", "AAAA", "cyan", "SK-0001", workspace=open_ws,
                    connector=FakeConnector(), autostart=False)
    assert h.heading_offset == 0.0
    for v in ([0, 30], [30, 0], [-30, 0], [0, -30], [21, 21]):
        h.set_velocity(np.array(v, dtype=float))
        expected = velocity_to_command(np.array(v, dtype=float))[0]
        assert h._pending[0] == pytest.approx(expected)


def test_the_offset_survives_a_roster_round_trip(tmp_path):
    entries = [RobotEntry(name="Seasmoke", code="SSMK", kind="real",
                          color="cyan", ble_name="SK-914A", heading_offset=37.5)]
    path = tmp_path / "roster.json"
    Roster(entries=entries, path=path).save()

    back = Roster.load(path)
    assert back.errors == []
    assert back.by_code("SSMK").heading_offset == pytest.approx(37.5)


def test_a_missing_or_junk_offset_defaults_to_zero(tmp_path):
    """A hand-edited roster must never fail to load over this field."""
    import json

    path = tmp_path / "roster.json"
    path.write_text(json.dumps([
        {"name": "A", "code": "AAAA", "kind": "sim", "color": "cyan"},
        {"name": "B", "code": "BBBB", "kind": "sim", "color": "red",
         "heading_offset": "not a number"},
    ]))
    r = Roster.load(path)
    assert r.by_code("AAAA").heading_offset == 0.0
    assert r.by_code("BBBB").heading_offset == 0.0


def test_the_offset_reaches_a_robot_built_from_the_roster(open_ws, tmp_path):
    entries = [RobotEntry(name="Seasmoke", code="SSMK", kind="real",
                          color="cyan", ble_name="SK-914A", heading_offset=45.0)]
    f = Fleet.from_roster(Roster(entries=entries, path=tmp_path / "r.json"),
                          workspace=open_ws, connector=FakeConnector())
    try:
        assert f["SSMK"].heading_offset == pytest.approx(45.0)
    finally:
        f.close()


# -- the sim drifts and slips the way a real ball does -----------------------
#
# A constant bias is a robot you calibrate once and forget. A drifting one is
# the reason the offset has to be tracked continuously — so the sim has to
# drift, or the estimator is being tested against a world that cannot fail.

def test_a_randomised_sim_robot_drifts(open_ws):
    h = SimRobot("A", "AAAA", "cyan", workspace=open_ws, pos=[120, 90],
                 randomize=True, seed=5)
    assert h.drift_rate > 0
    start = h._drift
    for _ in range(6000):                       # 10 minutes at 0.1s
        h.set_velocity(np.array([30.0, 0.0]))
        h.step(0.1)
    assert h._drift != start, "the bias never moved"


def test_drift_is_bounded_and_never_runs_away(open_ws):
    """An unbounded walk eventually points a robot backwards, which teaches
    a controller nothing except that the world is broken."""
    from fleet.sim_handle import DRIFT_LIMIT

    h = SimRobot("A", "AAAA", "cyan", workspace=open_ws, pos=[120, 90],
                 randomize=True, seed=9)
    h.drift_rate = 500.0                        # absurd, to push the bound
    for _ in range(20000):
        h.set_velocity(np.array([30.0, 0.0]))
        h.step(0.1)
        assert abs(h._drift) <= DRIFT_LIMIT + 1e-9


def test_an_unrandomised_robot_is_still_perfectly_predictable(open_ws):
    """Tests that pin exact positions depend on this staying deterministic."""
    h = SimRobot("A", "AAAA", "cyan", workspace=open_ws, pos=[120, 90],
                 randomize=False)
    assert h.drift_rate == 0.0 and h.slip == 0.0
    for _ in range(200):
        h.set_velocity(np.array([30.0, 0.0]))
        h.step(0.1)
    assert abs(h.pos[1] - 90.0) < 1e-6, "drifted with randomize=False"


def test_slip_only_ever_loses_speed(open_ws):
    """Multiplicative, so it can never add energy to the system."""
    h = SimRobot("A", "AAAA", "cyan", workspace=open_ws, pos=[120, 90],
                 randomize=True, seed=3)
    h.slip = 0.5
    for _ in range(400):
        h.set_velocity(np.array([40.0, 0.0]))
        h.step(0.1)
    assert float(np.linalg.norm(h.vel)) <= 40.0 + 1e-6


def test_the_heading_offset_cancels_the_modelled_bias(open_ws):
    """What calibration is for, at the level of one robot."""
    import math

    def travel_angle(offset):
        """Direction actually travelled, measured well clear of the walls.

        Not the final velocity: `SimRobot.step` zeroes it on contact with a
        bound, so a robot that reaches the corner reports 0 degrees no matter
        what it was doing on the way.
        """
        h = SimRobot("A", "AAAA", "cyan", workspace=open_ws, pos=[40, 40],
                     randomize=False)
        h.bias = math.radians(30.0)
        h.heading_offset = offset
        for _ in range(4):                      # clear the latency queue
            h.set_velocity(np.array([30.0, 0.0]))
            h.step(0.05)
        start = h.pos.copy()
        for _ in range(30):
            h.set_velocity(np.array([30.0, 0.0]))
            h.step(0.05)
        d = h.pos - start
        return math.degrees(math.atan2(d[1], d[0]))

    assert abs(travel_angle(0.0)) > 20.0, "the bias should show"
    assert abs(travel_angle(-30.0)) < 3.0, "the offset did not cancel it"


# -- where the controller's velocity comes from ------------------------------

def test_a_real_robot_prefers_the_trackers_filtered_velocity(open_ws):
    """Differencing two camera positions is the noisiest estimator available.

    At 30fps a centimetre of position noise becomes tens of cm/s of velocity
    noise, and the controller's derivative term multiplies exactly that into
    the motors — which is what a rough, twitchy approach looks like. The
    tracker's filter has been computing a proper estimate all along; it was
    simply never asked for it.
    """
    import numpy as np
    from fleet.real_handle import SpheroRobot

    class Tracker:
        def __init__(self):
            self.p = np.array([50.0, 50.0])

        def read(self):
            return {"cyan": self.p.copy()}

        def velocities(self):
            return {"cyan": np.array([12.0, -3.0])}

    tr = Tracker()
    h = SpheroRobot("A", "AAAA", "cyan", "fake", tracker=tr,
                    connector=lambda n, timeout=8: None, autostart=False)
    try:
        h.step(0.05)
        tr.p = tr.p + np.array([9.0, 9.0])      # a wild jump between fixes
        h.step(0.05)
        assert np.allclose(h.vel, [12.0, -3.0]), (
            f"used the difference ({h.vel}) instead of the filter")
    finally:
        h.close()


def test_it_falls_back_to_differencing_when_there_is_no_filter(open_ws):
    """An older tracker, or one that has not settled, must still drive."""
    import numpy as np
    from fleet.real_handle import SpheroRobot

    class Bare:
        def __init__(self):
            self.p = np.array([50.0, 50.0])

        def read(self):
            return {"cyan": self.p.copy()}

    tr = Bare()
    h = SpheroRobot("A", "AAAA", "cyan", "fake", tracker=tr,
                    connector=lambda n, timeout=8: None, autostart=False)
    try:
        h.step(0.05)
        import time
        time.sleep(0.05)
        tr.p = tr.p + np.array([1.0, 0.0])
        h.step(0.05)
        assert float(np.linalg.norm(h.vel)) > 0.0
    finally:
        h.close()


def test_a_filter_returning_nonsense_is_ignored(open_ws):
    import numpy as np
    from fleet.real_handle import SpheroRobot

    class Broken:
        def __init__(self):
            self.p = np.array([50.0, 50.0])

        def read(self):
            return {"cyan": self.p.copy()}

        def velocities(self):
            return {"cyan": np.array([float("nan"), 0.0])}

    h = SpheroRobot("A", "AAAA", "cyan", "fake", tracker=Broken(),
                    connector=lambda n, timeout=8: None, autostart=False)
    try:
        h.step(0.05)
        h.step(0.05)
        assert np.isfinite(h.vel).all()
    finally:
        h.close()


# -- the aiming taillight ------------------------------------------------

def test_a_handle_remembers_its_taillight_and_reports_it(open_ws):
    from fleet.sim_handle import SimRobot
    h = SimRobot("One", "ONE", "red", workspace=open_ws, seed=1)
    assert h.back_led == 0, "a Sphero powers up with the taillight off"
    h.set_back_led(255)
    assert h.state()["back_led"] == 255
    h.set_back_led((10, 20, 30))
    assert h.state()["back_led"] == [10, 20, 30], "a BOLT's is addressable"


def test_the_taillight_reaches_the_radio(open_ws):
    from fleet.real_handle import SpheroRobot
    conn = FakeConnector()
    h = SpheroRobot("One", "ONE", "red", "SK-TEST", workspace=open_ws,
                    connector=conn)
    try:
        h.set_back_led(255)
        deadline = time.time() + 3.0
        while time.time() < deadline and not conn.apis:
            time.sleep(0.02)
        api = conn.apis.get("SK-TEST")
        assert api is not None
        deadline = time.time() + 3.0
        while time.time() < deadline and not api.back_leds:
            time.sleep(0.02)
        assert api.back_leds, "the taillight never went out over the link"
    finally:
        h.close()


def test_a_toy_with_no_taillight_does_not_lose_the_link(open_ws):
    """A cosmetic light must never cost a connection. Everything else in that
    worker block tears the link down and reconnects when it throws."""
    from fleet.real_handle import SpheroRobot

    class NoTail(FakeApi):
        def set_back_led(self, value):
            raise RuntimeError("this toy has no back LED")

    h = SpheroRobot("One", "ONE", "red", "SK-TEST", workspace=open_ws,
                    connector=lambda name, timeout=8.0: NoTail(name))
    try:
        # `connected` also wants a camera fix and there is no camera here, so
        # the link itself is what this test is about.
        deadline = time.time() + 3.0
        while time.time() < deadline and not h._link_up:
            time.sleep(0.02)
        assert h._link_up
        h.set_back_led(255)
        time.sleep(0.4)
        assert h._link_up, "a failed taillight write dropped the link"
        assert not h._back_led_works, "and it should stop asking"
    finally:
        h.close()

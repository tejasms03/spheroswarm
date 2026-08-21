import numpy as np

from fleet.manager import Fleet, FleetEnv
from fleet.roster import Roster
from swarm.navigate import Navigate, assign, assignment_cost
from swarm.sim import SwarmEnv
from workspace.space import Workspace


# -- assignment ----------------------------------------------------------

def test_hungarian_beats_naive_ordering():
    positions = np.array([[0.0, 0.0], [100.0, 0.0]])
    targets = np.array([[100.0, 0.0], [0.0, 0.0]])   # deliberately reversed

    order = assign(positions, targets)
    naive = np.arange(len(positions))

    assert assignment_cost(positions, targets, order) < \
           assignment_cost(positions, targets, naive)
    assert list(order) == [1, 0]


def test_hungarian_minimises_total_distance_on_random_sets():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = rng.uniform(0, 200, (6, 2))
        tgt = rng.uniform(0, 200, (6, 2))
        order = assign(pos, tgt)
        best = assignment_cost(pos, tgt, order)
        for _ in range(50):
            perm = rng.permutation(6)
            assert best <= assignment_cost(pos, tgt, perm) + 1e-9


def test_assign_is_a_permutation():
    rng = np.random.default_rng(1)
    pos, tgt = rng.uniform(0, 200, (5, 2)), rng.uniform(0, 200, (5, 2))
    order = assign(pos, tgt)
    assert sorted(order) == list(range(5))


def test_assign_handles_empty():
    assert len(assign(np.zeros((0, 2)), np.zeros((0, 2)))) == 0


# -- control -------------------------------------------------------------

def test_act_has_the_boids_shape():
    env = SwarmEnv(5, randomize=False, seed=0)
    nav = Navigate(targets=np.full((5, 2), 100.0))
    a = nav.act(env)
    assert a.shape == (5, 2)
    assert np.isfinite(a).all()
    assert (np.linalg.norm(a, axis=1) <= 1.0 + 1e-9).all()


def test_robots_converge_to_targets():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(4, randomize=False, seed=3, workspace=ws)
    targets = np.array([[60.0, 40.0], [120.0, 40.0], [180.0, 40.0], [120.0, 140.0]])
    nav = Navigate()
    nav.set_targets(targets, env.pos)

    for _ in range(400):
        env.step(nav.act(env))

    d = np.linalg.norm(env.pos - nav.targets, axis=1)
    assert (d < 10.0).all(), d


def test_arrived_reports_tolerance():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(3, randomize=False, seed=4, workspace=ws)
    nav = Navigate()
    nav.set_targets(np.array([[50.0, 50.0], [100.0, 50.0], [150.0, 50.0]]), env.pos)
    assert not nav.arrived(env).all()
    for _ in range(400):
        env.step(nav.act(env))
    assert nav.arrived(env, tol=12.0).all()


def test_navigation_avoids_an_obstacle_in_the_way():
    ws = Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[{"type": "circle", "center": [120, 90], "radius": 20}],
    )
    env = SwarmEnv(1, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[40.0, 90.0]])
    env.vel = np.zeros((1, 2))
    nav = Navigate(targets=np.array([[200.0, 90.0]]))

    for _ in range(600):
        env.step(nav.act(env))
        assert ws.is_valid_point(env.pos[0]), env.pos[0]

    assert np.linalg.norm(env.pos[0] - [200.0, 90.0]) < 25.0


def test_navigation_rounds_a_polygon_obstacle():
    ws = Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[{"type": "poly",
                    "points": [[100, 50], [140, 50], [140, 130], [100, 130]]}],
    )
    env = SwarmEnv(1, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[50.0, 90.0]])
    env.vel = np.zeros((1, 2))
    nav = Navigate(targets=np.array([[200.0, 90.0]]))

    for _ in range(800):
        env.step(nav.act(env))
        assert ws.is_valid_point(env.pos[0]), env.pos[0]

    assert np.linalg.norm(env.pos[0] - [200.0, 90.0]) < 25.0


def test_swarm_crosses_an_obstacle_field_without_stalling():
    ws = Workspace(
        bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
        obstacles=[
            {"type": "circle", "center": [120, 60], "radius": 18},
            {"type": "circle", "center": [120, 130], "radius": 18},
        ],
    )
    env = SwarmEnv(4, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[30.0, 40.0], [30.0, 80.0], [30.0, 120.0], [30.0, 160.0]])
    env.vel = np.zeros((4, 2))
    targets = np.array([[210.0, 40.0], [210.0, 80.0],
                        [210.0, 120.0], [210.0, 160.0]])
    nav = Navigate()
    nav.set_targets(targets, env.pos)

    for _ in range(900):
        env.step(nav.act(env))
        for p in env.pos:
            assert ws.is_valid_point(p), p

    d = np.linalg.norm(env.pos - nav.targets, axis=1)
    assert (d < 25.0).all(), d


def test_navigation_keeps_robots_separated():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(4, randomize=False, seed=7, workspace=ws)
    # four robots asked to converge on nearly the same spot
    targets = np.array([[120.0, 90.0], [126.0, 90.0], [120.0, 96.0], [126.0, 96.0]])
    nav = Navigate()
    nav.set_targets(targets, env.pos)
    for _ in range(400):
        env.step(nav.act(env))

    d = np.linalg.norm(env.pos[:, None, :] - env.pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 4.0


def test_set_targets_uses_hungarian_when_positions_given():
    positions = np.array([[10.0, 10.0], [200.0, 10.0]])
    nav = Navigate()
    out = nav.set_targets(np.array([[200.0, 10.0], [10.0, 10.0]]), positions)
    assert np.allclose(out[0], [10.0, 10.0])
    assert np.allclose(out[1], [200.0, 10.0])


def test_navigate_against_a_live_sim_fleet(open_ws, sim_entries):
    f = Fleet.from_roster(Roster(entries=sim_entries[:4], path="/dev/null"),
                          workspace=open_ws)
    try:
        env = FleetEnv(f)
        targets = np.array([[60.0, 40.0], [120.0, 40.0],
                            [180.0, 40.0], [120.0, 140.0]])
        nav = Navigate()
        nav.set_targets(targets, env.pos)

        for _ in range(400):
            env.sync()
            env.apply(nav.act(env))
            f.step(0.1)

        env.sync()
        d = np.linalg.norm(env.pos - nav.targets, axis=1)
        assert (d < 12.0).all(), d
    finally:
        f.close()


def test_no_targets_means_hold_position():
    env = SwarmEnv(3, randomize=False, seed=0)
    nav = Navigate()
    a = nav.act(env)
    assert np.allclose(a, 0.0, atol=1e-9) or np.abs(a).max() < 0.6


# -- moving obstacles -------------------------------------------------------
#
# The avoidance logic was written for static geometry. Its pass-side is derived
# from how far off the obstacle's centreline the robot already is — that is the
# fix for the documented limit cycle, and an obstacle that moves is a case it
# has never faced. See §9a of the handoff notes before touching _avoid.

def _moving_ws(waypoints, speed=40.0, radius=18.0, mode="once"):
    from workspace.entities import EntitySet
    from workspace.space import Workspace

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    ws.entities = EntitySet.from_data([
        {"id": "mover", "role": "obstacle",
         "shape": {"type": "circle", "center": list(waypoints[0]), "radius": radius},
         "motion": {"kind": "path", "waypoints": [list(w) for w in waypoints],
                     "speed": speed, "mode": mode}}])
    return ws


def _run(ws, start, target, steps=700, dt=0.1):
    """Drive one robot to a target while the world moves. Returns its track."""
    env = SwarmEnv(1, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([list(start)], dtype=float)
    env.vel = np.zeros((1, 2))
    nav = Navigate(targets=np.array([list(target)], dtype=float))

    track = []
    for _ in range(steps):
        ws.step(dt)
        env.step(nav.act(env))
        track.append(env.pos[0].copy())
    return np.array(track)


def test_robot_reaches_its_target_past_a_crossing_obstacle():
    """An obstacle sweeping across the path must not trap the robot."""
    ws = _moving_ws([(120, 20), (120, 160)], speed=35, mode="pingpong")
    track = _run(ws, (20, 90), (220, 90))
    gap = float(np.linalg.norm(track[-1] - np.array([220.0, 90.0])))
    assert gap < 25, f"stranded {gap:.0f}cm from the target"


def test_robot_does_not_oscillate_against_an_obstacle_moving_at_it():
    """The head-on case: the obstacle drives straight down the robot's path.

    A pass-side derived from the seek vector flips every time the robot drifts,
    which is the limit cycle. Count direction reversals: a robot that commits
    to a side turns a handful of times, one that oscillates turns constantly.
    """
    ws = _moving_ws([(200, 90), (40, 90)], speed=30, mode="once")
    track = _run(ws, (20, 90), (220, 90))

    vy = np.diff(track[:, 1])
    sign = np.sign(vy[np.abs(vy) > 0.05])
    reversals = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0
    assert reversals < 25, f"{reversals} direction reversals — it is oscillating"

    gap = float(np.linalg.norm(track[-1] - np.array([220.0, 90.0])))
    assert gap < 30, f"never got there: {gap:.0f}cm short"


def test_robot_never_ends_up_inside_a_moving_obstacle():
    ws = _moving_ws([(60, 90), (180, 90)], speed=45, mode="pingpong", radius=20)
    track = _run(ws, (30, 90), (210, 90))
    # replay the same motion to check occupancy at each instant
    ws2 = _moving_ws([(60, 90), (180, 90)], speed=45, mode="pingpong", radius=20)
    worst = 0
    for p in track:
        ws2.step(0.1)
        e = ws2.entities.by_id("mover")
        d = float(np.linalg.norm(p - e.pos))
        worst = max(worst, e.radius - d)
    assert worst < 8.0, f"robot penetrated {worst:.0f}cm into a moving obstacle"


def test_a_static_obstacle_still_behaves_exactly_as_before():
    """Regression: entities must not change the static case at all."""
    from workspace.space import Workspace

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
                   obstacles=[{"type": "circle", "center": [120, 90], "radius": 20}])
    track = _run(ws, (20, 90), (220, 90), steps=500)
    gap = float(np.linalg.norm(track[-1] - np.array([220.0, 90.0])))
    assert gap < 20, f"static avoidance regressed: {gap:.0f}cm short"


def test_robots_settle_on_legally_spaced_targets_without_a_standoff():
    """Separation must not hold a robot off a target the validator approved.

    Targets are guaranteed 20cm apart; separation starts pushing at 22cm. At
    full strength two robots on legal targets balance seek against separation
    and stop several centimetres short — never arriving, never failing loudly.
    """
    from tools.validate import MIN_SEPARATION
    from workspace.space import Workspace

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    tg = np.array([[120 - MIN_SEPARATION / 2, 90.0],
                   [120 + MIN_SEPARATION / 2, 90.0]])
    env = SwarmEnv(2, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[80.0, 90.0], [160.0, 90.0]])
    env.vel = np.zeros((2, 2))
    nav = Navigate(targets=tg)
    for _ in range(800):
        env.step(nav.act(env))

    gaps = np.linalg.norm(env.pos - tg, axis=1)
    assert (gaps <= 6.0).all(), f"standoff short of the targets: {gaps.round(1)}"


def test_separation_never_fades_to_nothing():
    """The fade must keep a floor, or robots drive through each other."""
    from swarm.navigate import SEP_FLOOR

    assert 0.0 < SEP_FLOOR <= 1.0


def test_a_crowded_arrival_keeps_robots_apart():
    """Six robots converging on a tight ring must not end up on top of each other."""
    from workspace.space import Workspace

    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    ang = np.arange(6) * np.pi / 3
    tg = np.c_[120 + 25 * np.cos(ang), 90 + 25 * np.sin(ang)]

    env = SwarmEnv(6, randomize=False, seed=1, workspace=ws)
    env.pos = np.c_[120 + 80 * np.cos(ang + 0.4), 90 + 80 * np.sin(ang + 0.4)]
    env.vel = np.zeros((6, 2))
    nav = Navigate(targets=tg)
    for _ in range(800):
        env.step(nav.act(env))

    d = np.linalg.norm(env.pos[:, None, :] - env.pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 12.0, f"robots ended {d.min():.1f}cm apart"
    gaps = np.linalg.norm(env.pos - tg, axis=1)
    assert (gaps <= 8.0).all(), f"did not settle: {gaps.round(1)}"

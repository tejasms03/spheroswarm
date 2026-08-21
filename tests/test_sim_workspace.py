"""The sim must take its bounds from a workspace without breaking the old API."""

import numpy as np

from swarm.boids import Boids, Idle
from swarm.sim import ARENA, OBS_DIM, SwarmEnv
from workspace.space import Workspace


def test_default_env_is_the_historical_square():
    env = SwarmEnv(4, randomize=False, seed=0)
    assert env.bbox == (0.0, ARENA, 0.0, ARENA)
    assert env.observe().shape == (4, OBS_DIM)


def test_env_respects_rectangular_workspace():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(6, randomize=False, seed=1, workspace=ws)
    assert env.bbox == (0.0, 240.0, 0.0, 180.0)
    for _ in range(60):
        env.step(np.ones((6, 2)))
    assert (env.pos[:, 0] <= 240.0 + 1e-6).all()
    assert (env.pos[:, 1] <= 180.0 + 1e-6).all()
    assert (env.pos >= -1e-6).all()


def test_env_keeps_robots_out_of_obstacles():
    ws = Workspace(
        bounds_cm=[[0, 0], [200, 0], [200, 200], [0, 200]],
        obstacles=[{"type": "circle", "center": [100, 100], "radius": 30}],
    )
    env = SwarmEnv(5, randomize=False, seed=2, workspace=ws)
    for p in env.pos:
        assert ws.is_valid_point(p)
    rng = np.random.default_rng(3)
    for _ in range(200):
        env.step(rng.uniform(-1, 1, (5, 2)))
        for p in env.pos:
            assert ws.is_valid_point(p), p


def test_boids_still_runs_unchanged_api():
    env = SwarmEnv(6, "coverage", randomize=False, seed=0)
    a = Boids().act(env)
    assert a.shape == (6, 2)
    assert np.isfinite(a).all()
    for _ in range(30):
        env.step(Boids().act(env))
    assert np.isfinite(env.pos).all()


def test_boids_walls_use_workspace_bounds():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(4, "coverage", randomize=False, seed=5, workspace=ws)
    for _ in range(150):
        env.step(Boids().act(env))
    assert (env.pos[:, 0] <= 240.0).all() and (env.pos[:, 1] <= 180.0).all()


def test_idle_controller():
    env = SwarmEnv(3, randomize=False, seed=0)
    assert np.allclose(Idle().act(env), 0.0)


def test_step_returns_the_same_tuple_shape():
    env = SwarmEnv(4, "gather", randomize=False, seed=0)
    obs, rew, done, info = env.step(np.zeros((4, 2)))
    assert obs.shape == (4, OBS_DIM)
    assert rew.shape == (4,)
    assert isinstance(done, bool) or isinstance(done, (np.bool_,))
    assert {"coverage", "collisions", "spread"} <= set(info)


def test_set_workspace_pulls_robots_inside():
    env = SwarmEnv(5, randomize=False, seed=0)
    small = Workspace(bounds_cm=[[0, 0], [60, 0], [60, 60], [0, 60]])
    env.set_workspace(small)
    for p in env.pos:
        assert small.is_valid_point(p), p

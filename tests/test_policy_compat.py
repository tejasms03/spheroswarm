"""The sim refactor must not have invalidated trained checkpoints.

`SwarmEnv.observe()` changed when bounds moved into the workspace. If the
observation layout or width drifted, an old `.pt` would still load and then
quietly drive badly — the worst kind of regression, so it gets its own test.
"""

import numpy as np
import torch

from fleet.manager import Fleet, FleetEnv
from fleet.roster import Roster
from swarm.policy import ActorCritic, PolicyController, RunningNorm
from swarm.sim import K_NEIGHBOURS, OBS_DIM, SwarmEnv
from workspace.space import Workspace


def make_checkpoint(tmp_path, name="test_policy"):
    net = ActorCritic()
    norm = RunningNorm(OBS_DIM)
    path = tmp_path / f"{name}.pt"
    torch.save({"model": net.state_dict(), "obs_mean": norm.mean,
                "obs_var": norm.var, "meta": {"task": "coverage", "n_agents": 4}},
               path)
    return path


def test_observation_width_is_unchanged():
    assert OBS_DIM == 2 + K_NEIGHBOURS * 4 + 4 + 3 == 25


def test_observation_layout_is_unchanged():
    """Velocity first, then neighbours, then the four wall distances, then task."""
    env = SwarmEnv(3, "coverage", randomize=False, seed=0)
    obs = env.observe()
    assert obs.shape == (3, OBS_DIM)

    assert np.allclose(obs[:, 0:2], env.vel / 60.0)

    w = 2 + K_NEIGHBOURS * 4
    # the wall block is a pair of complements on each axis
    assert np.allclose(obs[:, w + 0] + obs[:, w + 1], 1.0)
    assert np.allclose(obs[:, w + 2] + obs[:, w + 3], 1.0)
    assert (obs[:, w:w + 4] >= 0).all() and (obs[:, w:w + 4] <= 1).all()


def test_wall_observations_are_relative_to_the_workspace():
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]])
    env = SwarmEnv(1, randomize=False, seed=0, workspace=ws)
    env.pos = np.array([[60.0, 45.0]])          # a quarter in on both axes
    obs = env.observe()
    w = 2 + K_NEIGHBOURS * 4
    assert np.isclose(obs[0, w + 0], 0.25)      # a quarter across 240cm
    assert np.isclose(obs[0, w + 1], 0.75)
    assert np.isclose(obs[0, w + 2], 0.25)      # a quarter down 180cm
    assert np.isclose(obs[0, w + 3], 0.75)


def test_a_checkpoint_loads_and_acts(tmp_path):
    path = make_checkpoint(tmp_path)
    ctrl = PolicyController(path)
    env = SwarmEnv(4, "coverage", randomize=False, seed=0)
    a = ctrl.act(env)
    assert a.shape == (4, 2)
    assert np.isfinite(a).all()
    assert (np.abs(a) <= 1.0).all()


def test_a_checkpoint_runs_at_a_different_robot_count(tmp_path):
    """Parameter sharing means a policy trained on 4 must run on 6."""
    ctrl = PolicyController(make_checkpoint(tmp_path))
    for n in (2, 6, 9):
        env = SwarmEnv(n, "coverage", randomize=False, seed=1)
        assert ctrl.act(env).shape == (n, 2)


def test_a_checkpoint_drives_a_live_fleet(tmp_path, open_ws, sim_entries):
    ctrl = PolicyController(make_checkpoint(tmp_path))
    f = Fleet.from_roster(Roster(entries=sim_entries[:4], path="/dev/null"),
                          workspace=open_ws)
    try:
        env = FleetEnv(f)
        for _ in range(50):
            env.sync()
            env.apply(ctrl.act(env))
            f.step(0.1)
        for h in f.handles.values():
            assert open_ws.is_valid_point(h.pos)
            assert np.isfinite(h.pos).all()
    finally:
        f.close()


def test_a_checkpoint_runs_in_a_non_square_workspace(tmp_path):
    ctrl = PolicyController(make_checkpoint(tmp_path))
    ws = Workspace(bounds_cm=[[0, 0], [240, 0], [240, 180], [0, 180]],
                   obstacles=[{"type": "circle", "center": [120, 90], "radius": 20}])
    env = SwarmEnv(5, "coverage", randomize=False, seed=2, workspace=ws)
    for _ in range(100):
        env.step(ctrl.act(env))
    for p in env.pos:
        assert ws.is_valid_point(p), p

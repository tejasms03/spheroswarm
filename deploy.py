#!/usr/bin/env python3
"""Run a controller on real robots.

    python deploy.py --controller boids
    python deploy.py --controller runs/coverage_0812_141233.pt

The only piece you must supply is a tracker that returns {name: (x_cm, y_cm)}.
Everything else — observations, policy, command conversion — is shared with the
simulator, so a policy that works in `app.py` works here unchanged.
"""

import argparse
import time

import numpy as np

from swarm.boids import Boids
from swarm.policy import PolicyController
from swarm.sim import ARENA, SwarmEnv, sphero_command

CONTROL_HZ = 10.0


class FakeTracker:
    """Stand-in until the camera exists. Lets you test the whole loop dry."""

    def __init__(self, names):
        self.names = names
        self.pos = {n: np.random.uniform(40, ARENA - 40, 2) for n in names}

    def read(self):
        return dict(self.pos)


class MirrorEnv(SwarmEnv):
    """A SwarmEnv whose state is overwritten by the tracker each tick.

    Reusing the env means observations are built by exactly the same code that
    produced the training data — the most common source of transfer bugs.
    """

    def sync(self, positions, dt):
        new = np.array([positions[n] for n in self.names])
        self.vel = (new - self.pos) / dt
        self.pos = new
        self._mark_visited()


def build(names, task):
    env = MirrorEnv(len(names), task, randomize=False)
    env.names = names
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="boids")
    p.add_argument("--task", default="coverage", choices=["coverage", "gather"])
    p.add_argument("--robots", type=int, default=3)
    p.add_argument("--dry", action="store_true", help="no BLE, print commands")
    a = p.parse_args()

    names = [f"SK-{i:04d}" for i in range(a.robots)]
    tracker = FakeTracker(names)
    env = build(names, a.task)
    ctrl = Boids() if a.controller == "boids" else PolicyController(a.controller)

    robots = {}
    if not a.dry:
        from swarm.fleet import connect_fleet     # your threaded BLE layer
        robots = connect_fleet(names)

    offsets = {n: 0.0 for n in names}   # camera-frame vs Sphero-frame, per robot
    dt = 1.0 / CONTROL_HZ
    print(f"{ctrl.label} -> {len(names)} robots at {CONTROL_HZ:.0f} Hz")

    try:
        while True:
            t0 = time.time()
            env.sync(tracker.read(), dt)
            actions = ctrl.act(env)

            for i, n in enumerate(names):
                heading, speed = sphero_command(actions[i] * 60.0)
                heading = (heading + offsets[n]) % 360
                if a.dry:
                    print(f"  {n}  {heading:6.1f}deg  {speed:3d}")
                else:
                    robots[n].command(heading, speed)

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        for r in robots.values():
            r.command(0, 0)
        print("\nstopped")


if __name__ == "__main__":
    main()

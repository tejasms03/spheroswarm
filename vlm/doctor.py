"""Is the chain up, and if not, which link is down?

    python3.13 -m vlm.doctor

Answers the question "the framework sees nothing" without guessing. Every
check has a TIMEOUT, which is the whole reason this exists: `RPCClient` uses a
plain ZMQ REQ socket with no receive timeout, so when the server is down every
call in the system blocks forever rather than failing. A dead server and a
running one with nothing to say look identical from a terminal that never
returns.
"""

import argparse
import os
import sys

OK, BAD, MEH = "  ok  ", " DOWN ", " ....  "


def _client(host, port, timeout_ms):
    """An RPCClient that gives up instead of hanging."""
    import zmq
    from rpc_system import RPCClient
    c = RPCClient(host=host, port=port)
    c._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    c._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    c._socket.setsockopt(zmq.LINGER, 0)
    return c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--robot-id", type=int, default=2)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--framework", default=os.path.expanduser(
        "~/Downloads/Mobile-manipulation-with-VLMs-March/Functions/Utilities"))
    a = p.parse_args(argv)

    if a.framework not in sys.path:
        sys.path.insert(0, a.framework)

    print(f"checking tcp://{a.host}:{a.port}\n")
    try:
        client = _client(a.host, a.port, int(a.timeout * 1000))
        poses = client.Robot.get_all_robot_pose()
    except Exception as e:
        print(f"{BAD} the RPC server")
        print(f"       {type(e).__name__}: {e}")
        print("\n  start it:  python3.13 -m vlm.server")
        return 1
    print(f"{OK} the RPC server is up")

    try:
        attached = client.Robot.attached
    except Exception:
        attached = None
    if attached:
        print(f"{OK} a bench is attached and serving drive requests")
    else:
        print(f"{BAD} NO bench attached")
        print("       the bench is not running, or is running without --rpc")
        print("\n  start it:  python3.13 fleet_test.py --source sim "
              "--robot SYRX --size 1280x1020 --rpc")

    rid = a.robot_id
    if rid in (poses or {}):
        pose = poses[rid]
        px_cm = 10.0
        print(f"{OK} robot {rid} is visible at "
              f"({pose['x']:.0f}, {pose['y']:.0f}) px = "
              f"({pose['x']/px_cm:.1f}, {pose['y']/px_cm:.1f}) cm")
        if not pose.get("theta_fresh", True):
            print(f"       bearing is HELD, {pose.get('theta_age_s', 0):.0f}s old "
                  f"(the ball is at rest)")
    else:
        print(f"{BAD} robot {rid} is NOT in DataService")
        if attached:
            # The bench is publishing, so the tracker is the thing with
            # nothing to say. In sim the usual cause is geometric rather than
            # optical: the fake camera does not cover the area robots spawn in.
            print("       the bench is publishing but its tracker has no lock.")
            print("       in sim: the fake camera sizes itself from")
            print("       workspace.json, so a bench started BEFORE that fix")
            print("       still has a 1280x720 frame that misses the bottom")
            print("       32cm of the arena -- restart it (no --size needed).")
            print("       on hardware: the ball is unlit, out of frame, or the")
            print("       room lights have blown the frame out.")

    try:
        path = client.Robot.get_path(rid)
        print(f"{OK if path else MEH} path for robot {rid}: "
              f"{len(path)} points" + ("" if path else "  (nothing planned yet)"))
        outcome = client.Robot.get_outcome(rid)
        if outcome:
            print(f"{MEH} last drive: {outcome.get('outcome')} — "
                  f"{outcome.get('reason', '')}")
    except Exception as e:
        print(f"{BAD} could not read the path: {e}")

    ready = bool(attached) and rid in (poses or {})
    print("\n" + ("ready to drive" if ready else
                  "NOT ready — fix what is marked DOWN above"))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

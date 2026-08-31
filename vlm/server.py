"""Their RPC server, with our robot behind it.

    python3.13 -m vlm.server

Stands in for `Functions/Utilities/1_run_server.py`. Deliberately a separate
launcher rather than an edit to theirs: their tree is a snapshot that will be
replaced by a real clone, and a launcher that lives next to the code it imports
survives that where a patched file does not.

Two differences from theirs, and both are load-bearing.

`Robot` is ours. Theirs drives serial motors at 40Hz; `vlm.service.SpheroRobot`
is a blackboard the bench reads. It is registered under the name `Robot`
because `RPCServer.register_class` names a service after its class unless told
otherwise, and `client.Robot.*` is what every caller in `Functions/Library/`
writes.

`Tasks` is NOT registered. Constructing it builds a `VLDetector`, which imports
`moondream` and connects to `http://10.20.83.18:2021/v1` -- a hardcoded address
on their lab network. Off that network the server does not start at all. It
serves SAM2 object detection, which this rig cannot use anyway: finding objects
needs the room lit and the ball tracker needs it dark.

`Data` is theirs, unmodified. It is a thread-safe store and there is nothing
about it that knows what kind of robot fills it.
"""

import argparse
import os
import sys


DEFAULT_FRAMEWORK = os.path.expanduser(
    "~/Downloads/Mobile-manipulation-with-VLMs-March/Functions/Utilities")

FRAMEWORK = os.environ.get("VLM_FRAMEWORK", DEFAULT_FRAMEWORK)


def build(port=5555, framework=None, robot_id=2, name="Caraxes", code="CRXS"):
    """The configured server, not yet running. Returned so tests can drive it."""
    root = framework or FRAMEWORK
    if not os.path.isdir(root):
        raise SystemExit(
            f"the framework is not at {root!r}.\n"
            f"Point VLM_FRAMEWORK at its Functions/Utilities directory.")
    if root not in sys.path:
        sys.path.insert(0, root)

    from rpc_system import RPCServer
    from services.data_service import Data

    from vlm.service import SpheroRobot

    server = RPCServer(port=port)
    server.register_class(Data())
    # The name is the contract. `Functions/Library/planning.py` and everything
    # beside it reach this as `client.Robot.*`.
    server.register_class(
        SpheroRobot(id_list=(robot_id,), names=[name], codes={robot_id: code}),
        class_name="Robot")
    return server


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--framework", default=None,
                   help="their Functions/Utilities directory "
                        "(or set VLM_FRAMEWORK)")
    p.add_argument("--robot-id", type=int, default=2)
    p.add_argument("--name", default="Caraxes")
    p.add_argument("--code", default="CRXS")
    a = p.parse_args(argv)

    server = build(port=a.port, framework=a.framework, robot_id=a.robot_id,
                   name=a.name, code=a.code)
    print(f"robot {a.robot_id} is {a.name} ({a.code}); "
          f"start the bench with --rpc to put a ball behind it")
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

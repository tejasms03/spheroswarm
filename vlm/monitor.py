"""Draw what the FRAMEWORK sees, and nothing else.

A window fed entirely from `client.Data.get_full_state()`. Nothing here reads
the bench, the camera, or a calibration file -- if a ball appears in this
window it is because the bridge put it into DataService and the RPC layer
carried it, which is the only claim this tool exists to make.

Deliberately not a nice UI. Their React frontend is the real one; this is the
instrument you point at the boundary when the question is "did anything
actually cross it".

    python3.13 -m vlm.monitor              # attach to a running RPC server
    python3.13 -m vlm.monitor --serve      # and start one, for a bench alone

Press q or escape to close.
"""

import argparse
import sys
import threading
import time

import cv2
import numpy as np

from vlm.bridge import PX_PER_CM


BG = (28, 22, 16)
EDGE = (108, 78, 44)
CHALK = (248, 240, 228)
DIM = (204, 179, 143)
BALL = (232, 210, 99)
HELD = (107, 107, 255)
GOOD = (168, 226, 126)


def draw(state, size_px, px_per_cm, link):
    """One frame of the framework's world, drawn from its own state dict."""
    w, h = size_px
    scale = min(1.0, 900.0 / max(w, 1), 620.0 / max(h, 1))
    cw, ch = int(w * scale), int(h * scale)

    canvas = np.full((ch + 96, max(cw, 460), 3), BG, np.uint8)
    cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), EDGE, 1)

    poses = state.get("robot_poses") or {}
    rows = [f"link  {link}",
            f"arena {w} x {h} px   @ {px_per_cm:g} px/cm   "
            f"({w / px_per_cm:.1f} x {h / px_per_cm:.1f} cm)"]

    if not poses:
        cv2.putText(canvas, "no robot in DataService", (16, ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIM, 1, cv2.LINE_AA)
        rows.append("robot_poses  {}   -- nothing is publishing, or the ball "
                    "is not tracked")
    for rid, p in sorted(poses.items()):
        x, y = float(p.get("x", 0.0)), float(p.get("y", 0.0))
        theta = float(p.get("theta", 0.0))
        fresh = bool(p.get("theta_fresh", True))
        sx, sy = int(x * scale), int(y * scale)

        cv2.circle(canvas, (sx, sy), max(4, int(3.65 * px_per_cm * scale)),
                   BALL, 1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 3, BALL, -1, cv2.LINE_AA)

        # The bearing, drawn dashed when it is being HELD rather than measured
        # -- a ball at rest has no travel direction and this window should not
        # be the place that forgets it.
        reach = 34.0
        ex = int((x + np.cos(theta) * reach / scale * scale) * scale)
        ey = int((y + np.sin(theta) * reach / scale * scale) * scale)
        colour = GOOD if fresh else HELD
        if fresh:
            cv2.arrowedLine(canvas, (sx, sy), (ex, ey), colour, 2,
                            cv2.LINE_AA, tipLength=0.3)
        else:
            for t0, t1 in ((0.0, 0.3), (0.5, 0.8)):
                a = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
                b = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
                cv2.line(canvas, a, b, colour, 2, cv2.LINE_AA)

        cv2.putText(canvas, str(rid), (sx + 10, sy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CHALK, 1, cv2.LINE_AA)
        rows.append(
            f"robot {rid}   ({x:7.1f}, {y:7.1f}) px   "
            f"({x / px_per_cm:6.1f}, {y / px_per_cm:6.1f}) cm   "
            f"{np.degrees(theta):6.1f} deg {'measured' if fresh else 'HELD'}   "
            f"{p.get('tracking', '')}")

    obs = state.get("dynamic_obstacles") or []
    rows.append(f"obstacles  {len(obs)}")

    for i, row in enumerate(rows):
        cv2.putText(canvas, row, (10, ch + 20 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    CHALK if i else DIM, 1, cv2.LINE_AA)
    return canvas


def serve(port):
    """Run their RPC server in-process, for looking at a bench on its own.

    A convenience for exactly one situation: the bench is publishing and their
    `1_run_server.py` is not running. In any real setup theirs is the server
    and this should just attach.
    """
    sys.path.insert(0, FRAMEWORK)
    from rpc_system import RPCServer
    from services.data_service import Data
    server = RPCServer(port=port)
    server.register_class(Data())
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(0.8)


FRAMEWORK = ("/Users/tejas_sriganesh/Downloads/"
             "Mobile-manipulation-with-VLMs-March/Functions/Utilities")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", default="localhost")
    p.add_argument("--serve", action="store_true",
                   help="also run the RPC server here, if theirs is not up")
    p.add_argument("--px-per-cm", type=float, default=PX_PER_CM)
    p.add_argument("--arena", default=None,
                   help="WxH in pixels; default is read from the first frame")
    a = p.parse_args(argv)

    if a.serve:
        serve(a.port)

    sys.path.insert(0, FRAMEWORK)
    from rpc_system import RPCClient
    client = RPCClient(host=a.host, port=a.port)

    size = None
    if a.arena:
        size = tuple(int(v) for v in a.arena.lower().split("x"))

    cv2.namedWindow("the framework's view", cv2.WINDOW_AUTOSIZE)
    link, misses = "connecting", 0
    while True:
        try:
            state = client.Data.get_full_state()
            link, misses = f"tcp://{a.host}:{a.port}  ok", 0
        except Exception as e:
            misses += 1
            link = f"tcp://{a.host}:{a.port}  DOWN ({type(e).__name__}) x{misses}"
            state = {"robot_poses": {}, "dynamic_obstacles": []}

        if size is None:
            frame = state.get("stitched_frame")
            # Their DataService starts life holding a 480x640 placeholder, so
            # a frame is not evidence a camera ever ran. Sized from it anyway:
            # it is what their own UI would draw into.
            size = ((frame.shape[1], frame.shape[0]) if frame is not None
                    else (1388, 1108))

        cv2.imshow("the framework's view",
                   draw(state, size, a.px_per_cm, link))
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

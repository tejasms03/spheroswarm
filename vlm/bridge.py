"""Push what the bench sees into the framework's DataService.

Runs inside the bench process, on its own thread, at a fixed rate. The bench
keeps measuring exactly as it did; this reads the result and carries it across
the RPC boundary. If the framework is not running, or the link breaks, the
bench is unaffected -- an output that can stop a tracker is not an output worth
having.

Two conversions happen here and both have a way of being silently wrong:

  * centimetres to ARENA pixels, which is a scale and NOT the homography
    inverse -- see `ArenaFrame`;
  * travel direction to `theta`, which is a bearing the blob cannot actually
    measure -- see `Reading.theta`.

Everything else is transport.
"""

import math
import threading
import time

import numpy as np


PX_PER_CM = 10.0
"""Arena pixels per centimetre.

The framework is a PIXEL stack end to end: `Data/robot_pos.txt` is
`id,x_px,y_px,theta_rad`, obstacles are pixel quads, the A* grid is pixels, and
every tuned constant in `Functions/Utilities/PathControl/` is a pixel count --
`robot_padding=30`, `sweep_radius=65`, the lookahead distances. Publishing
centimetres into it would leave all of those constants meaning something about
3.6x smaller than intended, and nothing would complain.

Ten was chosen so the numbers land where those constants were tuned. Their
arena is 1500x500px; ours is 138.8 x 110.8cm, which at 10px/cm is 1388x1108 --
the same order, so `robot_padding=30` reads as 3cm and `sweep_radius=65` as
6.5cm, both plausible against a 7.3cm ball. It is a free choice, but it is not
an arbitrary one, and changing it silently re-tunes every planner constant.
"""


class ArenaFrame:
    """Centimetres to the framework's top-down pixel frame, and back.

    A SCALE, deliberately, and not `Homography.to_px`.

    `to_px` is the exact inverse of `to_cm` and maps back to CAMERA pixels --
    which are perspective-distorted, because that is the distortion the
    homography exists to undo. The framework wants the rectified top-down view
    its stitcher would have produced: a frame where the arena is a rectangle
    and a centimetre is the same number of pixels everywhere. Feeding it camera
    pixels would hand its planner a trapezoid and call it a floor.

    So the chain is: camera px -> cm (the homography, already calibrated) ->
    arena px (this, a scale). The middle step is the rectification; this step
    only chooses units.
    """

    def __init__(self, width_cm, height_cm, px_per_cm=PX_PER_CM):
        self.width_cm = float(width_cm)
        self.height_cm = float(height_cm)
        self.px_per_cm = float(px_per_cm)

    @property
    def size_px(self):
        """(width, height) in pixels -- what `arena_settings.json` must say."""
        return (int(round(self.width_cm * self.px_per_cm)),
                int(round(self.height_cm * self.px_per_cm)))

    @property
    def corners_px(self):
        """The arena quad, in the order `Data/arena_corners.txt` uses."""
        w, h = self.size_px
        return [(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)]

    def to_px(self, xy_cm):
        x, y = xy_cm
        return (float(x) * self.px_per_cm, float(y) * self.px_per_cm)

    def to_cm(self, xy_px):
        x, y = xy_px
        return (float(x) / self.px_per_cm, float(y) / self.px_per_cm)

    def scalar_to_px(self, cm):
        """A length, not a point. Radii and margins have to cross too."""
        return float(cm) * self.px_per_cm


class Reading:
    """One robot's pose as the framework wants it, or nothing.

    Built from the bench rather than measured here. The point of the class is
    that it can refuse: `Reading.of(app)` returns None when there is no lock,
    and the bridge then publishes an EMPTY pose dict rather than a stale one.

    That refusal is the whole design. `PoseFilter` on their side holds a
    missing robot's last pose for five frames, and `pp.get_pos` falls back to
    `get_pos.last_pose` forever -- so a tracker that goes quiet looks, from
    inside their controller, exactly like a ball sitting perfectly still. This
    bench already knows the difference and says so in `track.status`; throwing
    that away at the boundary is how a lost ball becomes a controller driving
    confidently at a position nothing has measured for a minute.
    """

    def __init__(self, xy_px, theta_rad, theta_fresh, status):
        self.xy_px = xy_px
        self.theta_rad = theta_rad
        self.theta_fresh = theta_fresh
        self.status = status

    @classmethod
    def of(cls, app, arena, last_theta=None):
        """Read the bench, or None if it has nothing trustworthy to report."""
        if app.blob is None or not app.homography.ready:
            return None
        if not app.track.locked:
            # Coasting counts as not locked. The bench is already predicting a
            # position it cannot see; publishing that as an observation would
            # launder a guess into a measurement one process downstream.
            return None

        cm = app.to_cm(app.blob["xy"])
        xy_px = arena.to_px((float(cm[0]), float(cm[1])))

        # THETA IS NOT MEASURED. It is the direction of TRAVEL.
        #
        # The framework's whole control stack is differential-drive and reads
        # `theta` as a body heading -- which way the robot FACES. A blob has no
        # facing: one bright circle, no orientation, which is exactly why
        # `pose_test.py` (three lit dots) exists and is not what the rig uses.
        # Travel is the honest substitute and it is not the same quantity: a
        # Sphero that slips, or is nudged, or is pushed by another ball, goes
        # one way while pointing another.
        #
        # `travel_readout` maps the direction through the homography rather
        # than fitting the pixel history, because a projective transform
        # preserves neither bearings nor distances -- a direction fitted in
        # pixels is not the direction the ball is driving in.
        readout = app.travel_readout()
        if readout is not None:
            theta = math.radians(readout[0])
            fresh = True
        else:
            # At rest there is no travel direction. The last one is the best
            # estimate available and is what a differential-drive consumer
            # expects; 0.0 would be an invented north pointing down the x axis.
            # Publishing None is not an option -- their `get_pos` does
            # `pose['theta']` unconditionally and would raise.
            theta = last_theta if last_theta is not None else 0.0
            fresh = False

        return cls(xy_px, theta, fresh, app.track.status)

    def as_pose(self):
        """The dict shape `ArucoDetector.detect_pose_multicams` returns."""
        return {
            "x": float(self.xy_px[0]),
            "y": float(self.xy_px[1]),
            "theta": float(self.theta_rad),
            # Not part of their schema. Carried anyway, because anything that
            # wants to know whether the bearing is current can look, and
            # nothing that does not care will trip over an extra key.
            "theta_fresh": bool(self.theta_fresh),
            "tracking": str(self.status),
        }


class Bridge(threading.Thread):
    """Publishes one robot's pose into DataService until told to stop.

    Never fatal to the bench. Every RPC failure is counted and swallowed: the
    framework is something this rig can feed when one happens to be running,
    not a dependency of the camera loop. `errors` and `last_error` are there so
    the dock can say the link is down instead of the operator wondering why the
    other window is empty.
    """

    def __init__(self, app, robot_id=2, arena=None, hz=20.0, client=None):
        super().__init__(daemon=True, name="vlm-bridge")
        self.app = app
        self.robot_id = int(robot_id)
        self.arena = arena or ArenaFrame(*_workspace_cm(app))
        self.period = 1.0 / max(1.0, float(hz))
        self._client = client
        self._stop = threading.Event()

        self.published = 0
        self.skipped = 0
        self.errors = 0
        self.last_error = None
        self.last_theta = None

    def connect(self):
        """An RPCClient, or None and the bridge simply idles.

        Import is deferred: `rpc_system` is installed by the framework's
        `pip install -e Functions/Utilities`, and a bench that refuses to start
        because a separate project is not set up would be a bad trade for a
        feature that is optional.
        """
        if self._client is not None:
            return self._client
        try:
            from rpc_system import RPCClient
            self._client = RPCClient()
        except Exception as e:
            self.errors += 1
            self.last_error = f"no RPC client: {e}"
            self._client = None
        return self._client

    def stop(self):
        self._stop.set()

    def run(self):
        if self.connect() is None:
            return
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self.publish_once()
            except Exception as e:
                self.errors += 1
                self.last_error = str(e)
            # Pace against the work done, not on top of it, so a slow round
            # trip lowers the rate instead of stacking up behind itself.
            self._stop.wait(max(0.0, self.period - (time.perf_counter() - started)))

    def publish_once(self):
        """One update_state call. Returns the pose dict that was sent."""
        client = self._client
        if client is None:
            return None

        reading = Reading.of(self.app, self.arena, self.last_theta)
        if reading is None:
            poses = {}
            self.skipped += 1
        else:
            poses = {self.robot_id: reading.as_pose()}
            if reading.theta_fresh:
                self.last_theta = reading.theta_rad
            self.published += 1

        frame = self.app.frame
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # `ind`, `raw` and `hf` are the per-camera dicts their stitcher
        # produces. Nothing in Functions/Library or backend/ reads them on the
        # path we use -- only `get_ind_frames`, for the SAM service we are not
        # running -- so one camera under key 0 satisfies the signature without
        # pretending to a rig we do not have.
        client.Data.update_state(frame, poses, [], {0: frame}, {0: frame}, {})
        return poses


def _workspace_cm(app):
    """The arena rectangle in cm, preferring the calibration actually loaded.

    The homography records the rectangle it was BUILT for, and `workspace.json`
    records the one the planner believes in. They are supposed to agree --
    `Homography.matches` exists because they have disagreed before, and a
    homography saying 200x200 under a 240x180 workspace put the tracker and the
    planner on different floors with nothing complaining. Reading the
    homography here keeps this boundary tied to the same rectangle the
    centimetres it is converting were measured in.
    """
    h = app.homography
    if h is not None and getattr(h, "width", None) and getattr(h, "height", None):
        return float(h.width), float(h.height)
    return 200.0, 200.0

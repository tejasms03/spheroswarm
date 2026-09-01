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

import cv2
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

    def __init__(self, width_cm, height_cm, px_per_cm=PX_PER_CM,
                 origin_cm=(0.0, 0.0)):
        self.width_cm = float(width_cm)
        self.height_cm = float(height_cm)
        self.px_per_cm = float(px_per_cm)
        # WHERE THE ARENA STARTS, in centimetres. Not assumed to be (0, 0),
        # because the workspace is four corners a person clicked and the
        # homography's own origin is wherever ITS calibration quad happened to
        # sit. On this rig they differ by a third of a metre: the clicked
        # arena runs x -33.3..109.2, y -31.8..90.9. Taking (0, 0) as the corner
        # put a ball at cm (-30, 90) at arena pixel (-300, 900) -- off the map
        # -- and started the warp a third of a metre inside the arena, which is
        # the offset crop the framework's UI was showing.
        self.origin_cm = np.asarray(origin_cm, dtype=float)

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
        x = (float(xy_cm[0]) - self.origin_cm[0]) * self.px_per_cm
        y = (float(xy_cm[1]) - self.origin_cm[1]) * self.px_per_cm
        return (x, y)

    def to_cm(self, xy_px):
        x = float(xy_px[0]) / self.px_per_cm + self.origin_cm[0]
        y = float(xy_px[1]) / self.px_per_cm + self.origin_cm[1]
        return (x, y)

    def scalar_to_px(self, cm):
        """A length, not a point. Radii and margins have to cross too."""
        return float(cm) * self.px_per_cm

    def matrix_from(self, homography):
        """Camera pixels straight to arena pixels, as one 3x3.

        The scale composed onto the homography rather than applied after it.
        Two transforms that have to be kept in step is how this project already
        lost an afternoon to a drawn robot landing where its blob was not --
        `Homography.nudge_cm` says so in as many words. One matrix cannot
        disagree with itself.
        """
        # Translate to the arena's own origin BEFORE scaling, so the warp and
        # `to_px` agree. Composed into one matrix for the reason `nudge_cm`
        # gives: two transforms that must be kept in step is how this project
        # already lost an afternoon.
        k, o = self.px_per_cm, self.origin_cm
        S = np.array([[k, 0.0, -k * o[0]],
                      [0.0, k, -k * o[1]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return S @ np.asarray(homography.M, dtype=np.float64)

    def warp(self, frame, homography):
        """The camera frame, rectified into the arena frame.

        THIS MATTERS MORE THAN IT LOOKS. Their `stitched_frame` is not a
        snapshot for the UI to pretty up -- `PlanningClient._get_planner` takes
        the A* GRID SIZE from `frame.shape[:2]`, and with no `arena_corners` in
        DataService it takes the arena rectangle from the frame bounds too. So
        publishing the raw camera frame while publishing poses in arena pixels
        builds a planner whose grid is one size and a robot whose coordinates
        are another: on this rig a 1280x1020 grid under poses running to
        1388x1108, where a ball at the far edge is simply off the map.

        Warping fixes both halves at once. The grid matches the coordinates,
        and their UI gets the rectified top-down view their own stitcher would
        have produced, which is what it was written to draw.
        """
        if frame is None or not getattr(homography, "ready", False):
            return None
        return cv2.warpPerspective(frame, self.matrix_from(homography),
                                   self.size_px)


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

    def __init__(self, xy_px, theta_rad, theta_fresh, status, theta_age_s=0.0):
        self.xy_px = xy_px
        self.theta_rad = theta_rad
        self.theta_fresh = theta_fresh
        self.theta_age_s = theta_age_s
        self.status = status

    @classmethod
    def of(cls, app, arena, last_theta=None, last_theta_at=None, now=None):
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
        now = time.time() if now is None else now
        readout = app.travel_readout()
        if readout is not None:
            theta, fresh, age = math.radians(readout[0]), True, 0.0
        else:
            # At rest there is no travel direction. The last one is the best
            # estimate available and is what a differential-drive consumer
            # expects; 0.0 would be an invented north pointing down the x axis.
            # Publishing None is not an option -- their `get_pos` does
            # `pose['theta']` unconditionally and would raise.
            theta = last_theta if last_theta is not None else 0.0
            fresh = False
            # HOW OLD it is, because a held bearing decays into a fiction.
            # `zero_at_rest` says why in the bench's own words: a reference
            # taken minutes ago, or before a reconnect, or before somebody
            # picked the ball up, describes a relationship that no longer
            # holds. Nothing here can tell that a stationary ball was lifted
            # and set down facing elsewhere -- a blob has no facing to check it
            # against -- so the age is published and the consumer decides what
            # it will still believe.
            age = 0.0 if last_theta_at is None else max(0.0, now - last_theta_at)

        return cls(xy_px, theta, fresh, app.track.status, age)

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
            "theta_age_s": round(float(self.theta_age_s), 2),
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

    def __init__(self, app, robot_id=2, arena=None, hz=20.0, client=None,
                 drive=True):
        super().__init__(daemon=True, name="vlm-bridge")
        self.app = app
        self.robot_id = int(robot_id)
        self._arena = arena
        self.period = 1.0 / max(1.0, float(hz))
        self._client = client
        self._stop = threading.Event()

        self.published = 0
        self.skipped = 0
        self.errors = 0
        self.last_error = None
        self.last_theta = None
        self.last_theta_at = None

        # The command half, served on THIS thread with THIS client. `RPCClient`
        # holds a single ZMQ REQ socket and it is not thread-safe, so a second
        # thread polling for drive requests would interleave sends and receives
        # on the same socket. One thread does both.
        self._drive = bool(drive)
        self.driver = None

    @property
    def arena(self):
        """The arena rectangle, worked out on FIRST USE and then held.

        Deliberately not resolved in `__init__`. In sim the bench replaces the
        saved calibration with a bare px/cm scale and sets the homography's
        width and height to zero -- it is a real mapping with no declared
        rectangle -- so the only honest source for the arena size is the frame
        the camera is actually producing, and at construction time there is not
        one yet.

        Held once resolved. An arena that changed size under a running planner
        would move every published position without anything downstream being
        told the units had shifted.
        """
        w, h, origin, provisional = _workspace_cm(self.app)
        measured = not provisional
        if self._arena is None:
            # NOT CACHED UNTIL IT IS A MEASUREMENT. The fallback is
            # `vision.config.ARENA_CM`, a 200x200 placeholder, and on the very
            # first tick it is what you get: the sim homography declares no
            # rectangle and no frame has arrived yet. Latching that made every
            # later conversion wrong in a way nothing could correct -- a figure
            # eight written for the real 1388x1109 arena was scaled into a
            # fictional 2000x2000 one, and its first points landed at (104,104)
            # in an arena only 110cm tall. The refusal was right; the arena was
            # not.
            if not measured:
                return ArenaFrame(w, h, origin_cm=origin)
            self._arena = ArenaFrame(w, h, origin_cm=origin)
        return self._arena

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
        # Tell the service nothing is behind it any more, so a drive request
        # arriving after the bench has gone is refused rather than queued for a
        # robot that is not there.
        if self.driver is not None:
            self.driver.close(self._client)

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

        reading = Reading.of(self.app, self.arena,
                             self.last_theta, self.last_theta_at)
        if reading is None:
            poses = {}
            self.skipped += 1
        else:
            poses = {self.robot_id: reading.as_pose()}
            if reading.theta_fresh:
                self.last_theta = reading.theta_rad
                self.last_theta_at = time.time()
            self.published += 1

        # Rectified, not raw -- see `ArenaFrame.warp`. A frame in camera
        # pixels under poses in arena pixels gives their planner a grid that
        # disagrees with the coordinates it is planning in.
        w, h = self.arena.size_px
        frame = self.arena.warp(self.app.frame, self.app.homography)
        if frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        # `ind`, `raw` and `hf` are the per-camera dicts their stitcher
        # produces. Nothing in Functions/Library or backend/ reads them on the
        # path we use -- only `get_ind_frames`, for the SAM service we are not
        # running -- so one camera under key 0 satisfies the signature without
        # pretending to a rig we do not have.
        client.Data.update_state(frame, poses, [], {0: frame}, {0: frame}, {})

        # The command half, on THIS thread with THIS client. `RPCClient` holds
        # one ZMQ REQ socket and it is not thread-safe, so a second thread
        # polling for drive requests would interleave sends and receives on it.
        if self._drive:
            if self.driver is None:
                from vlm.driver import Driver
                self.driver = Driver(self.app, self.arena, self.robot_id)
            self.driver.serve(client)
        return poses


def _workspace_cm(app):
    """(width, height, origin) in cm — THE ARENA THE OPERATOR CLICKED.

    From `app.agent_bounds()`, which is the bounding box of the workspace
    corners mapped through the homography. Deliberately NOT the homography's
    own `width`/`height`.

    Those two are different things and on this rig they disagree badly. The
    homography records the rectangle it was CALIBRATED against — 138.8 x 110.8
    with its origin wherever that calibration quad sat. The workspace is four
    corners a person clicked afterwards, and here they land at cm x -33.3..109.2,
    y -31.8..90.9: an extent of 142.5 x 122.7 and an origin a third of a metre
    from zero. Publishing against the homography's rectangle put a ball at cm
    (-30, 90) at arena pixel (-300, 900), which is off their map entirely, and
    started the warp well inside the arena — the offset crop their UI showed.

    The tracked workspace is the one that matters, because it is the region the
    mask cuts, the region `agent_check` refuses goals outside, and the region a
    ball can actually be in.
    """
    box = None
    try:
        box = app.agent_bounds()
    except Exception:
        box = None
    if box:
        x0, y0, x1, y1 = (float(v) for v in box)
        if x1 - x0 > 1.0 and y1 - y0 > 1.0:
            return x1 - x0, y1 - y0, (x0, y0), False

    # No workspace clicked. Fall back to what the camera can see, mapped
    # through the homography — honest about extent, and it at least cannot put
    # the ball outside the frame it came from.
    h = getattr(app, "homography", None)
    frame = getattr(app, "frame", None)
    if h is not None and getattr(h, "ready", False) and frame is not None:
        rows, cols = frame.shape[0], frame.shape[1]
        corners = [(0, 0), (cols - 1, 0), (cols - 1, rows - 1), (0, rows - 1)]
        cm = np.asarray(h.to_cm(corners), dtype=float)
        lo, hi = cm.min(0), cm.max(0)
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]),
                (float(lo[0]), float(lo[1])), False)

    # Nothing to go on. 200x200 is `vision.config.ARENA_CM`, a placeholder
    # rather than a measurement — a bridge reporting into it is reporting into
    # a floor nobody has calibrated.
    # PROVISIONAL. Used so the tick has something, never kept.
    return 200.0, 200.0, (0.0, 0.0), True

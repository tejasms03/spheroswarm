"""Runs the overhead tracker on its own thread and exposes `read()`.

`vision.Tracker` needs a frame pushed into it; real robots need positions
pulled out of it at whatever rate the control loop runs. This adapter owns the
camera thread so neither the UI nor the fleet has to.
"""

import threading
import time

from vision.detect import Detector
from vision.homography import Homography
from vision.synthetic import open_source
from vision.track import Tracker

# A camera that just failed will not have recovered by the next line of Python.
# Without this the loop retries at whatever speed the interpreter manages,
# which on a disconnected camera is a busy core — and the threads it starves
# are the render loop and the BLE workers, whose timing is the very thing a
# calibration session is trying to measure.
RETRY_PAUSE = 0.25


class CameraTracker:
    """Same `read()` contract as `vision.track.Tracker`, minus the frame pumping."""

    def __init__(self, source="synthetic", colors=None):
        self.source_spec = source
        self.source = None
        self.tracker = None
        self.colors = colors
        self.error = None
        self.running = False
        self._lock = threading.Lock()
        self._fixes = {}
        self._thread = None
        # The most recent frame and its raw blob detections, kept so the UI can
        # show what the camera actually sees. Bring-up is mostly answering "is
        # it detecting the ball at all?", and an abstract top-down arena cannot
        # answer that — a robot the detector has missed and a robot that is not
        # there look identical.
        self._frame = None
        self._raw = {}
        # The tracker's own filtered velocity. Kept because the alternative —
        # differencing two camera positions in the control loop — is the
        # noisiest estimator available: at 30fps with a centimetre of position
        # noise it produces tens of cm/s of jitter, and a derivative gain then
        # feeds that straight back into the motors.
        self._vels = {}
        # When the newest fixes were computed, by the wall clock `fleet.handle`
        # uses. Consumers time freshness against THIS, not against the moment
        # they happened to call `read()`.
        self._fixed_at = 0.0

    @property
    def fixes_at(self):
        """Wall-clock time the newest fixes were produced. 0.0 before the first.

        `read()` hands out a cache, and a cache cannot say how old it is. This
        can. Without it a consumer times a fix by the moment it asked, and
        asking is not seeing: a capture thread that died an hour ago still
        answers `read()` instantly with the last positions it ever saw.
        """
        with self._lock:
            return self._fixed_at

    @property
    def fix_age(self):
        """Seconds since the last fix. `inf` before the first one lands."""
        at = self.fixes_at
        return float("inf") if not at else max(0.0, time.time() - at)

    @property
    def fps(self):
        return self.tracker.fps if self.tracker else 0.0

    def start(self):
        try:
            self.source = open_source(self.source_spec)
        except Exception as e:
            self.error = f"camera would not open: {e}"
            return [self.error]

        H = Homography.load()
        if not H.ready and hasattr(self.source, "true_corners"):
            H = Homography()
            H.set_corners(self.source.true_corners)

        self.tracker = Tracker(homography=H, detector=Detector())
        if self.colors:
            self.tracker.assignment = {c: c for c in self.colors}

        self.running = True
        self._thread = threading.Thread(target=self._run, name="tracker", daemon=True)
        self._thread.start()
        return []

    def _run(self):
        while self.running:
            try:
                ok, frame = self.source.read()
                if not ok:
                    self._stumble("camera returned no frame")
                    continue
                fixes, raw = self.tracker.step(frame)
            except Exception as e:
                self._stumble(str(e))
                continue
            try:
                vels = self.tracker.velocities()
            except Exception:
                vels = {}
            with self._lock:
                self._fixes = fixes
                self._frame = frame
                self._raw = raw
                self._vels = vels
                self._fixed_at = time.time()
                self.error = None

    def _stumble(self, message):
        """One failed frame. Record it, and do not come straight back for more.

        The cached fixes are deliberately NOT cleared. Dropping them would make
        a camera that blinks look like six robots vanishing, and the timestamp
        already tells any consumer exactly how much to trust what is there —
        which is the honest answer, and one a blink recovers from by itself.
        """
        self.error = message
        time.sleep(RETRY_PAUSE)

    def read(self):
        with self._lock:
            return dict(self._fixes)

    def velocities(self):
        """{name: (vx, vy)} in cm/s, from the tracker's filter. May be empty."""
        with self._lock:
            return dict(self._vels)

    def latest(self):
        """(frame, raw_detections) or (None, {}) before the first frame lands.

        The frame is handed out by reference and the capture loop replaces it
        rather than writing into it, so a reader cannot see a half-drawn image.
        """
        with self._lock:
            return self._frame, dict(self._raw)

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.source is not None:
            try:
                self.source.release()
            except Exception:
                pass

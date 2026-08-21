"""Running the agent off the render thread.

At 9B on an M1 Pro a single turn takes seconds, and a pygame loop that blocks
for seconds looks crashed — which is how someone ends up clicking the button
four more times. So the agent runs on a worker thread and reports back through
a queue that the UI drains once a frame.

The session also owns the decision the UI cares about most: whether there is a
model to talk to at all. If Ollama is not running, `available` is False and the
caller silently stays in literal-tool mode.
"""

import queue
import threading
import time

from .agent import SwarmAgent
from .client import ModelError, ModelUnreachable, client_for


class AgentSession:
    """One agent, one worker thread, an event queue the UI polls."""

    def __init__(self, ctx, preset="qwen3.5:9b", client=None, sim_lock=None,
                 max_tool_calls=8, probe=True, warm=False):
        self.ctx = ctx
        self.preset = preset
        self.events = queue.Queue()
        self.sim_lock = sim_lock or threading.RLock()
        self.error = None
        self.available = False
        self.latencies = []
        self._thread = None
        self._result = None
        self.warmed = False

        if client is None:
            try:
                client = client_for(preset)
            except (ModelError, Exception) as e:      # a bad preset is not fatal
                self.error = str(e)
                self.client = None
                self.agent = None
                return

        self.client = client
        self.agent = SwarmAgent(client, ctx, max_tool_calls=max_tool_calls,
                                on_event=self.events.put, sim_lock=self.sim_lock)
        self.available = self.probe() if probe else True
        # Opt-in: a warm-up is a real request, and a test injecting a scripted
        # client would have its first scripted turn eaten by it.
        if self.available and warm:
            self.warm_up()

    def warm_up(self):
        """Load the model now, so the first real command does not pay for it.

        A cold 9B costs ~24s to page in 6.6GB. Without this the user's opening
        command is always the slow one, which is exactly the impression that
        sticks. Runs in the background: the app must not block on it.
        """
        def go():
            try:
                self.client.chat([{"role": "user", "content": "ok"}], None)
                self.warmed = True
            except Exception:
                pass                      # a failed warm-up is not an error

        self.warmed = False
        threading.Thread(target=go, daemon=True, name="swarm-warmup").start()

    # -- liveness ------------------------------------------------------------

    def probe(self):
        """Is there a model to talk to? Never raises; the UI degrades quietly."""
        if self.client is None:
            return False
        try:
            ok = bool(self.client.reachable())
        except Exception as e:
            self.error = str(e)
            return False
        if not ok:
            self.error = f"{getattr(self.client, 'base_url', '?')} is not answering"
        return ok

    @property
    def model_name(self):
        return getattr(self.client, "model", None) or self.preset

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def mean_latency(self):
        if not self.latencies:
            return None
        return sum(self.latencies) / len(self.latencies)

    # -- running -------------------------------------------------------------

    def start(self, text):
        """Kick off a command. Returns False if one is already in flight."""
        if self.agent is None or self.busy:
            return False

        self._result = None

        def run():
            result = self.agent.command(text)
            self._result = result
            if result.total_latency_s:
                self.latencies.append(result.total_latency_s)
                del self.latencies[:-20]

        self._thread = threading.Thread(target=run, daemon=True,
                                        name="swarm-agent")
        self._thread.start()
        return True

    def cancel(self):
        """Abort an in-flight run. The agent halts the fleet on its way out."""
        if self.agent is not None:
            self.agent.cancel()

    def poll(self, limit=64):
        """Drain queued events. Call once a frame from the render thread."""
        out = []
        for _ in range(limit):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def result(self):
        return self._result

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def reset(self):
        if self.agent is not None:
            self.agent.reset()

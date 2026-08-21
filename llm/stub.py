"""A model that replays canned turns.

Exists so the agent loop and the whole eval harness can be exercised with no
Ollama, no GPU and no network — the suite has to pass on a machine that has
never heard of a language model.

A script is a list of turns; each turn is either a string (a final text reply)
or a list of (tool_name, arguments) pairs. `StubClient` walks the script one
turn per `chat()` call.
"""

import time
from dataclasses import dataclass, field

from .client import ModelResponse, ToolCall


@dataclass
class StubClient:
    script: list = field(default_factory=list)
    model: str = "stub"
    latency_s: float = 0.0
    fail_with: Exception = None

    def __post_init__(self):
        self.calls = []                 # every messages list it was handed
        self.turn = 0

    def reachable(self, timeout=2.0):
        return True

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        if self.fail_with is not None:
            raise self.fail_with
        if self.latency_s:
            time.sleep(self.latency_s)

        turn = self.script[self.turn] if self.turn < len(self.script) else "Done."
        self.turn += 1

        if isinstance(turn, str):
            return ModelResponse(text=turn, latency_s=self.latency_s)

        if isinstance(turn, ModelResponse):
            return turn

        calls = []
        for i, item in enumerate(turn):
            if isinstance(item, ToolCall):
                calls.append(item)
                continue
            name, args = item
            calls.append(ToolCall(name=name, arguments=dict(args), id=f"c{i}"))
        return ModelResponse(text="", tool_calls=calls, latency_s=self.latency_s)


def replies(*turns):
    """Convenience: StubClient(script=[...]) with a readable call site."""
    return StubClient(script=list(turns))

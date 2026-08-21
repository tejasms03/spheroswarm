"""A thin client over an OpenAI-compatible chat-completions endpoint.

Ollama is the local target, but nothing here is Ollama-specific beyond the
defaults in `models.yaml`. Pointing this at a frontier model for a baseline
comparison is a base-URL and key change and nothing else — which is the whole
reason for talking to `/v1` rather than Ollama's native API.

The one piece of real ugliness is `recover_tool_calls_from_text`. Qwen models
under Ollama intermittently emit a tool call as literal text in `content`
instead of populating `tool_calls`. We parse it back out rather than dropping
the turn, but we flag every recovery so the eval can count how often a given
model needs rescuing.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "models.yaml"

DEFAULT_BASE_URL = "http://localhost:11434/v1"

# Which dialect of "OpenAI-compatible" an endpoint speaks. Both talk to
# /v1/chat/completions with the same body and the same response shape — the
# difference is entirely in which *extra* parameters are tolerated, and the
# tolerance is not symmetric. Ollama reads generation options from an `options`
# block and gates reasoning through the chat template; a hosted model behind a
# gateway rejects both, and Claude via Vertex answers a flat 400 to something
# as ordinary as `temperature`. So parameters are opted into per backend rather
# than sent hopefully and assumed to be ignored.
OLLAMA = "ollama"
OPENAI_COMPATIBLE = "openai_compatible"
BACKENDS = (OLLAMA, OPENAI_COMPATIBLE)
UNREACHABLE_HINT = (
    "could not reach {url} — if this is Ollama, check that `ollama serve` is "
    "running and that the model has been pulled (`ollama pull {model}`)"
)


class ModelUnreachable(RuntimeError):
    """The endpoint could not be reached, after one retry."""


class ModelError(RuntimeError):
    """The endpoint answered, but not with something usable."""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""
    recovered_from_text: bool = False


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    latency_s: float = 0.0
    tokens: dict = field(default_factory=dict)
    ttft_s: float = None            # time to first token, streaming only

    @property
    def recovered_count(self):
        return sum(1 for c in self.tool_calls if c.recovered_from_text)


# -- pulling a tool call back out of prose --------------------------------

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCED = re.compile(r"```(?:json|tool_code|python)?\s*(.*?)```", re.S)


def _iter_json_objects(text):
    """Yield every balanced {...} span in `text`, outermost first.

    A regex cannot match balanced braces, and these payloads nest — arguments
    are themselves objects — so scan with a depth counter instead.
    """
    depth = start = 0
    in_string = escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]
            elif depth < 0:
                depth = 0


def _as_tool_call(obj, recovered=True):
    """Coerce a parsed object into a ToolCall, or None if it is not one."""
    if not isinstance(obj, dict):
        return None

    # OpenAI wire shape, in case a model echoes the whole envelope back
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name:
        return None

    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None

    return ToolCall(name=name, arguments=args, recovered_from_text=recovered)


def recover_tool_calls_from_text(text):
    """Find tool calls a model wrote into `content` instead of `tool_calls`.

    Handles the three shapes seen in the wild: a <tool_call> tag, a fenced
    json block, and a bare object sitting in the prose. Returns
    (calls, leftover_text) where leftover_text has the consumed spans removed.
    """
    if not text or not isinstance(text, str):
        return [], text or ""

    calls = []
    leftover = text

    def consume(span, call):
        nonlocal leftover
        calls.append(call)
        leftover = leftover.replace(span, "", 1)

    for m in _TOOL_CALL_TAG.finditer(text):
        body = m.group(1)
        for candidate in _iter_json_objects(body) or []:
            try:
                call = _as_tool_call(json.loads(candidate))
            except json.JSONDecodeError:
                call = None
            if call:
                consume(m.group(0), call)
                break

    for m in _FENCED.finditer(leftover):
        body = m.group(1)
        for candidate in _iter_json_objects(body):
            try:
                call = _as_tool_call(json.loads(candidate))
            except json.JSONDecodeError:
                continue
            if call:
                consume(m.group(0), call)
                break

    for candidate in list(_iter_json_objects(leftover)):
        try:
            call = _as_tool_call(json.loads(candidate))
        except json.JSONDecodeError:
            continue
        if call:
            consume(candidate, call)

    return calls, leftover.strip()


# -- schema plumbing --------------------------------------------------------

def to_openai_tools(schemas):
    """Our registry schemas -> the OpenAI `tools` array. Idempotent."""
    out = []
    for s in schemas or []:
        if isinstance(s, dict) and s.get("type") == "function" and "function" in s:
            out.append(s)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters",
                                    {"type": "object", "properties": {}}),
            },
        })
    return out


# -- configuration ----------------------------------------------------------

def _expand_env(value):
    """Replace ${VAR} / ${VAR:-default} in a string using the environment."""
    if not isinstance(value, str):
        return value

    def sub(m):
        name, default = m.group(1), m.group(3)
        return os.environ.get(name, default if default is not None else "")

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)(:-([^}]*))?\}", sub, value)


def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        raise ModelError(f"could not parse {path}: {e}") from e
    return data.get("models", data)


def available_presets(path=CONFIG_PATH):
    return sorted(load_config(path))


def client_for(preset, path=CONFIG_PATH, **overrides):
    """Build a ModelClient from a named preset in models.yaml."""
    cfg = load_config(path)
    if preset not in cfg:
        raise ModelError(
            f"unknown model preset {preset!r} — available: "
            f"{', '.join(sorted(cfg)) or 'none'}")

    entry = {k: _expand_env(v) for k, v in (cfg[preset] or {}).items()}
    entry.pop("notes", None)
    entry.setdefault("model", preset)

    # `api_key_env: PORTKEY_API_KEY` rather than the key itself. The indirection
    # is the point: models.yaml is a checked-in file, and a preset that names
    # its variable can never become a preset that carries a credential.
    key_env = entry.pop("api_key_env", None)
    if key_env and not entry.get("api_key"):
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise ModelError(
                f"preset {preset!r} needs the {key_env} environment variable, "
                f"which is not set. Export it and try again — do not put the "
                f"key in models.yaml:\n"
                f"    export {key_env}='...'")
        entry["api_key"] = key

    entry.update(overrides)
    return ModelClient(**entry)


# -- the client --------------------------------------------------------------

class ModelClient:
    """One HTTP call per `chat`. No streaming: the agent is turn-based."""

    def __init__(self, model, base_url=DEFAULT_BASE_URL, api_key=None,
                 num_ctx=16384, thinking=True, temperature=0.3, timeout=120,
                 thinking_style="chat_template", max_tokens=None,
                 extra_body=None, session=None, stream=False,
                 backend=OLLAMA):
        if backend not in BACKENDS:
            raise ModelError(f"unknown backend {backend!r} — use one of "
                             f"{', '.join(sorted(BACKENDS))}")
        self.backend = backend
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or None
        self.num_ctx = num_ctx
        self.thinking = thinking
        self.temperature = temperature
        self.timeout = timeout
        self.thinking_style = thinking_style
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.stream = stream
        self.session = session or requests.Session()

    # -- request shaping ----------------------------------------------------

    @property
    def url(self):
        return f"{self.base_url}/chat/completions"

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def build_body(self, messages, tools=None):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        # `temperature: None` means omit, not "use zero". Some hosted models
        # reject the parameter outright — Claude via Vertex behind the NYU
        # gateway answers a plain 400, "`temperature` is deprecated for this
        # model" — so a preset has to be able to not send it at all.
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = to_openai_tools(tools)

        if self.backend == OLLAMA:
            # num_ctx matters more than it looks: these models advertise a 256K
            # window, and letting the KV cache size itself accordingly will
            # exhaust a 16GB machine.
            if self.num_ctx:
                body["options"] = {"num_ctx": int(self.num_ctx)}
            self._apply_thinking(body)
        else:
            # A hosted model manages its own context and its own reasoning
            # budget. What it does want is a ceiling on the reply, so a runaway
            # generation cannot quietly bill for 8k tokens of apology.
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens

        body.update(self.extra_body)
        return body

    def _apply_thinking(self, body):
        style = (self.thinking_style or "none").lower()
        if style == "none":
            return
        if style == "chat_template":
            # Qwen's template gate: what actually silences <think> blocks.
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(self.thinking)
        elif style == "ollama_think":
            body["think"] = bool(self.thinking)
        elif style == "reasoning_effort":
            body["reasoning_effort"] = "medium" if self.thinking else "none"

    # -- the call ------------------------------------------------------------

    def chat(self, messages, tools=None, on_token=None):
        """One turn. `on_token(text)` fires per chunk when streaming."""
        if self.stream:
            return self._chat_streaming(messages, tools, on_token)
        return self._chat_blocking(messages, tools)

    def _chat_streaming(self, messages, tools, on_token=None):
        """Stream deltas so the caller can show progress while it thinks.

        Same latency, entirely different experience: 20 seconds of visible
        output beats 20 seconds of a frozen box, and it stops people pressing
        the button again.
        """
        body = self.build_body(messages, tools)
        body["stream"] = True
        # Without this the stream carries no usage block and every token count
        # comes back None, which quietly blinds the latency measurements.
        body["stream_options"] = {"include_usage": True}
        started = time.time()
        ttft = None

        try:
            r = self.session.post(self.url, headers=self.headers(), json=body,
                                  timeout=self.timeout, stream=True)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ModelUnreachable(
                UNREACHABLE_HINT.format(url=self.url, model=self.model)) from e

        if r.status_code >= 400:
            raise ModelError(f"{self.url} returned {r.status_code}: {_body_snippet(r)}")

        text_parts = []
        tool_frags = {}
        usage = {}

        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            usage = payload.get("usage") or usage
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            piece = delta.get("content")
            if piece:
                if ttft is None:
                    ttft = time.time() - started
                text_parts.append(piece)
                if on_token:
                    try:
                        on_token(piece)
                    except Exception:
                        pass

            for tc in delta.get("tool_calls") or []:
                if ttft is None:
                    ttft = time.time() - started
                idx = tc.get("index", 0)
                slot = tool_frags.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]

        latency = time.time() - started
        calls = []
        for idx in sorted(tool_frags):
            slot = tool_frags[idx]
            if not slot["name"]:
                continue
            call = _as_tool_call({"name": slot["name"],
                                  "arguments": slot["args"] or "{}"},
                                 recovered=False)
            if call is not None:
                call.id = slot["id"]
                calls.append(call)

        text = "".join(text_parts)
        if not calls and text:
            recovered, leftover = recover_tool_calls_from_text(text)
            if recovered:
                calls, text = recovered, leftover

        out = ModelResponse(text=text.strip(), tool_calls=calls, raw={"usage": usage},
                            latency_s=latency,
                            tokens={"prompt": usage.get("prompt_tokens"),
                                    "completion": usage.get("completion_tokens"),
                                    "total": usage.get("total_tokens")})
        out.ttft_s = ttft
        return out

    def _chat_blocking(self, messages, tools=None):
        body = self.build_body(messages, tools)
        started = time.time()
        last_exc = None

        for attempt in (1, 2):                     # one retry, then give up
            try:
                r = self.session.post(self.url, headers=self.headers(),
                                      json=body, timeout=self.timeout)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt == 2:
                    raise ModelUnreachable(
                        UNREACHABLE_HINT.format(url=self.url, model=self.model)
                    ) from last_exc
                time.sleep(0.5)

        latency = time.time() - started

        if r.status_code >= 400:
            raise ModelError(
                f"{self.url} returned {r.status_code}: {_body_snippet(r)}")

        try:
            payload = r.json()
        except Exception as e:
            raise ModelError(
                f"{self.url} returned non-JSON: {_body_snippet(r)}") from e

        return self.parse(payload, latency)

    # -- response shaping ----------------------------------------------------

    def parse(self, payload, latency_s=0.0):
        if not isinstance(payload, dict):
            raise ModelError(f"expected a JSON object, got {type(payload).__name__}")
        if "error" in payload and not payload.get("choices"):
            raise ModelError(f"model returned an error: {payload['error']}")

        choices = payload.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}

        text = message.get("content") or ""
        calls = []
        for raw in message.get("tool_calls") or []:
            call = _as_tool_call(raw, recovered=False)
            if call is not None:
                call.id = raw.get("id", "") or ""
                calls.append(call)

        # Only go looking in the prose when the proper channel came back empty;
        # a model that used tool_calls correctly may still legitimately mention
        # JSON in its text.
        if not calls and text:
            recovered, leftover = recover_tool_calls_from_text(text)
            if recovered:
                calls = recovered
                text = leftover

        usage = payload.get("usage") or {}
        tokens = {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }

        return ModelResponse(text=text.strip(), tool_calls=calls, raw=payload,
                             latency_s=latency_s, tokens=tokens)

    # -- diagnostics ---------------------------------------------------------

    def reachable(self, timeout=2.0):
        """Cheap liveness check for the UI. Never raises."""
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=timeout,
                                 headers=self.headers())
            return r.status_code < 500
        except Exception:
            return False


def _body_snippet(response, limit=300):
    try:
        return response.text[:limit]
    except Exception:
        return "<unreadable body>"

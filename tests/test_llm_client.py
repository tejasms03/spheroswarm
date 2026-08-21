import json

import pytest
import requests

from llm.client import (ModelClient, ModelError, ModelUnreachable, ToolCall,
                        available_presets, client_for, load_config,
                        recover_tool_calls_from_text, to_openai_tools)


# -- a fake transport ------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Records what was posted and replays queued responses."""

    def __init__(self, responses=None, raise_times=0, exc=requests.ConnectionError):
        self.responses = list(responses or [])
        self.raise_times = raise_times
        self.exc = exc
        self.posts = []
        self.gets = []

    def post(self, url, headers=None, json=None, timeout=None, stream=False):
        self.posts.append({"url": url, "headers": headers, "body": json,
                            "timeout": timeout})
        if self.raise_times > 0:
            self.raise_times -= 1
            raise self.exc("boom")
        return self.responses.pop(0) if self.responses else FakeResponse(reply())

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return FakeResponse({"data": []})


def reply(content="", tool_calls=None, usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5,
                                "total_tokens": 15}}


def openai_call(name, args, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def make(**kw):
    session = kw.pop("session", None) or FakeSession()
    client = ModelClient(model="qwen3.5:9b", session=session, **kw)
    return client, session


# -- request shaping --------------------------------------------------------

def test_num_ctx_is_passed_through():
    client, session = make(num_ctx=16384)
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["options"]["num_ctx"] == 16384


def test_num_ctx_zero_omits_the_options_block():
    """A frontier endpoint should not be sent Ollama-only keys."""
    client, session = make(num_ctx=0)
    client.chat([{"role": "user", "content": "hi"}])
    assert "options" not in session.posts[0]["body"]


def test_thinking_toggle_chat_template_style():
    client, session = make(thinking=True, thinking_style="chat_template")
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["chat_template_kwargs"]["enable_thinking"] is True

    client, session = make(thinking=False, thinking_style="chat_template")
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_thinking_toggle_other_styles():
    client, session = make(thinking=True, thinking_style="ollama_think")
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["think"] is True

    client, session = make(thinking=False, thinking_style="reasoning_effort")
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["reasoning_effort"] == "none"

    client, session = make(thinking=True, thinking_style="none")
    client.chat([{"role": "user", "content": "hi"}])
    body = session.posts[0]["body"]
    assert "think" not in body and "chat_template_kwargs" not in body


def test_tool_schemas_are_passed_through_in_openai_shape():
    from tools import schemas

    client, session = make()
    client.chat([{"role": "user", "content": "hi"}], tools=schemas())
    sent = session.posts[0]["body"]["tools"]
    assert len(sent) == len(schemas())
    assert all(t["type"] == "function" for t in sent)
    names = {t["function"]["name"] for t in sent}
    assert {"move_to", "compute_points", "wait_until_settled"} <= names
    move = next(t for t in sent if t["function"]["name"] == "move_to")
    assert move["function"]["parameters"]["properties"]["points"]["type"] == "array"


def test_to_openai_tools_is_idempotent():
    from tools import schemas

    once = to_openai_tools(schemas())
    assert to_openai_tools(once) == once


def test_api_key_becomes_a_bearer_header():
    client, session = make(api_key="sk-test")
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_no_api_key_sends_no_auth_header():
    client, session = make(api_key=None)
    client.chat([{"role": "user", "content": "hi"}])
    assert "Authorization" not in session.posts[0]["headers"]


def test_base_url_trailing_slash_is_tolerated():
    client, _ = make(base_url="http://localhost:11434/v1/")
    assert client.url == "http://localhost:11434/v1/chat/completions"


# -- response parsing -------------------------------------------------------

def test_parses_a_proper_tool_call():
    session = FakeSession([FakeResponse(
        reply(tool_calls=[openai_call("move_to", {"points": [[10, 10]]})]))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "go"}])
    assert len(r.tool_calls) == 1
    call = r.tool_calls[0]
    assert call.name == "move_to"
    assert call.arguments == {"points": [[10, 10]]}
    assert call.recovered_from_text is False
    assert r.recovered_count == 0


def test_parses_plain_text_reply():
    session = FakeSession([FakeResponse(reply(content="I know wedge."))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "?"}])
    assert r.text == "I know wedge."
    assert r.tool_calls == []


def test_tokens_and_latency_are_reported():
    session = FakeSession([FakeResponse(reply(
        content="ok", usage={"prompt_tokens": 120, "completion_tokens": 8,
                              "total_tokens": 128}))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "?"}])
    assert r.tokens == {"prompt": 120, "completion": 8, "total": 128}
    assert r.latency_s >= 0.0
    assert r.raw["choices"]


def test_arguments_may_arrive_as_a_dict_not_a_string():
    session = FakeSession([FakeResponse(reply(tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "stop", "arguments": {}}}]))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "stop"}])
    assert r.tool_calls[0].name == "stop"
    assert r.tool_calls[0].arguments == {}


# -- the tool-call-as-text failure -----------------------------------------

def test_recovers_tool_call_tag_shape():
    text = ('Sure, placing them now.\n'
            '<tool_call>{"name": "move_to", "arguments": {"points": [[60, 40]]}}</tool_call>')
    calls, leftover = recover_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "move_to"
    assert calls[0].arguments == {"points": [[60, 40]]}
    assert calls[0].recovered_from_text is True
    assert "tool_call" not in leftover
    assert leftover.startswith("Sure")


def test_recovers_fenced_json_shape():
    text = 'Here you go:\n```json\n{"name": "stop", "arguments": {}}\n```'
    calls, leftover = recover_tool_calls_from_text(text)
    assert len(calls) == 1 and calls[0].name == "stop"
    assert "```" not in leftover


def test_recovers_bare_json_object_shape():
    text = '{"name": "list_formations", "arguments": {}}'
    calls, leftover = recover_tool_calls_from_text(text)
    assert len(calls) == 1 and calls[0].name == "list_formations"
    assert leftover == ""


def test_recovery_handles_nested_argument_objects():
    """Brace-counting, not regex: `assign` is itself an object."""
    text = ('<tool_call>{"name": "move_to", "arguments": '
            '{"assign": {"SSMK": [100, 80], "CRXS": [140, 80]}}}</tool_call>')
    calls, _ = recover_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].arguments["assign"]["SSMK"] == [100, 80]


def test_recovery_handles_stringified_arguments():
    text = '<tool_call>{"name": "stop", "arguments": "{}"}</tool_call>'
    calls, _ = recover_tool_calls_from_text(text)
    assert len(calls) == 1 and calls[0].arguments == {}


def test_recovery_handles_braces_inside_strings():
    text = '{"name": "save_formation", "arguments": {"name": "a}b", "description": "{x"}}'
    calls, _ = recover_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].arguments["name"] == "a}b"


def test_recovery_surfaces_through_chat_with_the_flag_set():
    session = FakeSession([FakeResponse(reply(
        content='<tool_call>{"name": "stop", "arguments": {}}</tool_call>'))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "stop"}])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].recovered_from_text is True
    assert r.recovered_count == 1


def test_multiple_recovered_calls_in_one_message():
    text = ('<tool_call>{"name": "move_to", "arguments": {"points": [[10, 10]]}}</tool_call>'
            '<tool_call>{"name": "wait_until_settled", "arguments": {}}</tool_call>')
    calls, _ = recover_tool_calls_from_text(text)
    assert [c.name for c in calls] == ["move_to", "wait_until_settled"]


def test_prose_mentioning_json_is_not_mistaken_for_a_call():
    calls, leftover = recover_tool_calls_from_text(
        "I would use a JSON object with a name field, but I need more detail.")
    assert calls == []
    assert leftover.startswith("I would use")


def test_proper_tool_calls_suppress_text_scanning():
    """A real tool call plus prose that happens to contain JSON must not double up."""
    session = FakeSession([FakeResponse(reply(
        content='thinking about {"name": "stop", "arguments": {}}',
        tool_calls=[openai_call("move_to", {"points": [[10, 10]]})]))])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "go"}])
    assert [c.name for c in r.tool_calls] == ["move_to"]
    assert r.recovered_count == 0


def test_malformed_json_in_a_tag_does_not_raise():
    calls, leftover = recover_tool_calls_from_text('<tool_call>{not json}</tool_call>')
    assert calls == []
    assert isinstance(leftover, str)


def test_object_without_a_name_is_not_a_call():
    calls, _ = recover_tool_calls_from_text('{"arguments": {"points": []}}')
    assert calls == []


# -- failure modes -----------------------------------------------------------

def test_retries_once_then_raises_a_named_error():
    session = FakeSession(raise_times=5)
    client, _ = make(session=session)
    with pytest.raises(ModelUnreachable) as e:
        client.chat([{"role": "user", "content": "hi"}])
    assert len(session.posts) == 2, "should attempt exactly twice"
    assert "ollama serve" in str(e.value)
    assert "qwen3.5:9b" in str(e.value)


def test_recovers_if_the_retry_succeeds():
    session = FakeSession([FakeResponse(reply(content="hello"))], raise_times=1)
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r.text == "hello"
    assert len(session.posts) == 2


def test_timeout_is_treated_as_unreachable():
    session = FakeSession(raise_times=5, exc=requests.Timeout)
    client, _ = make(session=session)
    with pytest.raises(ModelUnreachable):
        client.chat([{"role": "user", "content": "hi"}])


def test_http_error_names_the_status_and_body():
    session = FakeSession([FakeResponse({"error": "no such model"},
                                         status_code=404, text="no such model")])
    client, _ = make(session=session)
    with pytest.raises(ModelError) as e:
        client.chat([{"role": "user", "content": "hi"}])
    assert "404" in str(e.value) and "no such model" in str(e.value)


def test_non_json_body_raises_model_error():
    session = FakeSession([FakeResponse(None, text="<html>gateway</html>")])
    client, _ = make(session=session)
    with pytest.raises(ModelError):
        client.chat([{"role": "user", "content": "hi"}])


def test_error_payload_raises():
    session = FakeSession([FakeResponse({"error": {"message": "context overflow"}})])
    client, _ = make(session=session)
    with pytest.raises(ModelError) as e:
        client.chat([{"role": "user", "content": "hi"}])
    assert "context overflow" in str(e.value)


def test_empty_choices_is_an_empty_response_not_a_crash():
    session = FakeSession([FakeResponse({"choices": []})])
    client, _ = make(session=session)
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r.text == "" and r.tool_calls == []


def test_reachable_never_raises():
    class Dead:
        def get(self, *a, **k):
            raise requests.ConnectionError("nope")

    client = ModelClient(model="m", session=Dead())
    assert client.reachable() is False


# -- presets -----------------------------------------------------------------

def test_all_required_presets_exist():
    assert set(available_presets()) >= {"qwen3.5:9b", "qwen3.5:4b", "qwen3:8b",
                                        "glm-4.7-flash", "frontier"}


def test_preset_builds_a_client_with_its_settings():
    c = client_for("qwen3.5:9b")
    assert c.model == "qwen3.5:9b"
    assert c.num_ctx == 16384
    assert c.base_url == "http://localhost:11434/v1"
    # interactive presets ship with thinking off: measured 4.6x fewer tokens
    # and 2.3x lower latency on this workload for the same result
    assert c.thinking is False


def test_frontier_preset_reads_the_environment(monkeypatch):
    monkeypatch.setenv("FRONTIER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("FRONTIER_MODEL", "some-big-model")
    monkeypatch.setenv("FRONTIER_API_KEY", "sk-secret")
    c = client_for("frontier")
    assert c.base_url == "https://example.test/v1"
    assert c.model == "some-big-model"
    assert c.api_key == "sk-secret"
    assert c.num_ctx == 0


def test_frontier_preset_has_defaults_without_the_environment(monkeypatch):
    for k in ("FRONTIER_BASE_URL", "FRONTIER_MODEL", "FRONTIER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    c = client_for("frontier")
    assert c.base_url.startswith("http")
    assert c.api_key is None


def test_unknown_preset_names_the_alternatives():
    with pytest.raises(ModelError) as e:
        client_for("gpt-9-ultra")
    assert "qwen3.5:9b" in str(e.value)


def test_overrides_beat_the_preset():
    c = client_for("qwen3.5:9b", temperature=0.0, thinking=False)
    assert c.temperature == 0.0 and c.thinking is False


def test_config_notes_are_not_passed_to_the_constructor():
    """models.yaml carries human notes; they must not reach ModelClient."""
    assert "notes" in load_config()["qwen3.5:9b"]
    client_for("qwen3.5:9b")     # would TypeError if notes leaked through


# -- streaming -----------------------------------------------------------------

class FakeStream:
    """An SSE response, the shape Ollama's /v1 returns when stream=true."""

    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code
        self.text = ""

    def iter_lines(self, decode_unicode=True):
        for c in self._chunks:
            yield "data: " + json.dumps(c) if isinstance(c, dict) else c


def sse_text(pieces, usage=None):
    out = [{"choices": [{"delta": {"content": p}}]} for p in pieces]
    if usage:
        out.append({"choices": [], "usage": usage})
    out.append("data: [DONE]")
    return out


def test_streaming_assembles_text_and_reports_ttft():
    session = FakeSession([FakeStream(sse_text(["Formed ", "a ", "circle."],
                                                usage={"prompt_tokens": 2400,
                                                       "completion_tokens": 9,
                                                       "total_tokens": 2409}))])
    client, _ = make(session=session, stream=True)
    seen = []
    r = client.chat([{"role": "user", "content": "go"}], on_token=seen.append)
    assert r.text == "Formed a circle."
    assert seen == ["Formed ", "a ", "circle."]
    assert r.ttft_s is not None and r.ttft_s <= r.latency_s
    assert r.tokens["prompt"] == 2400


def test_streaming_reassembles_a_tool_call_split_across_chunks():
    """Arguments arrive as fragments; a naive parse loses the call entirely."""
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "move_to", "arguments": '{"po'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ints": [[10, 2'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '0]]}'}}]}}]},
        "data: [DONE]",
    ]
    session = FakeSession([FakeStream(chunks)])
    client, _ = make(session=session, stream=True)
    r = client.chat([{"role": "user", "content": "go"}])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "move_to"
    assert r.tool_calls[0].arguments == {"points": [[10, 20]]}


def test_streaming_handles_two_tool_calls_interleaved():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "stop", "arguments": "{}"}},
            {"index": 1, "id": "b", "function": {"name": "get_state", "arguments": "{}"}}]}}]},
        "data: [DONE]",
    ]
    session = FakeSession([FakeStream(chunks)])
    client, _ = make(session=session, stream=True)
    r = client.chat([{"role": "user", "content": "x"}])
    assert [c.name for c in r.tool_calls] == ["stop", "get_state"]


def test_streaming_still_recovers_a_tool_call_written_as_text():
    session = FakeSession([FakeStream(sse_text(
        ['<tool_call>{"name": "stop", ', '"arguments": {}}</tool_call>']))])
    client, _ = make(session=session, stream=True)
    r = client.chat([{"role": "user", "content": "stop"}])
    assert len(r.tool_calls) == 1 and r.tool_calls[0].recovered_from_text


def test_a_broken_token_callback_does_not_kill_the_stream():
    def bad(_):
        raise RuntimeError("UI exploded")

    session = FakeSession([FakeStream(sse_text(["a", "b"]))])
    client, _ = make(session=session, stream=True)
    assert client.chat([{"role": "user", "content": "x"}], on_token=bad).text == "ab"


def test_streaming_is_off_by_default():
    client, session = make()
    client.chat([{"role": "user", "content": "hi"}])
    assert session.posts[0]["body"]["stream"] is False

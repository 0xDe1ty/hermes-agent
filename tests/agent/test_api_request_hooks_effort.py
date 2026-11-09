"""Exercise the real pre-hook, sanitizer, plugin correlation and card footer."""

from types import ModuleType
import sys

import pytest

from agent.api_request_hooks import ApiRequestHooksMixin
from agent.turn_api_request import _fire_pre_api_request_hook
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def build_event(monkeypatch, fields, *, large=True):
    from hermes_cli import lifecycle

    body = {"model": "example", "input": [{"role": "user", "content": "x" * 2000} for _ in range(80)] if large else []}
    body.update(fields)
    captured = []
    monkeypatch.setenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "50000")
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_api_request")
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kwargs: captured.append(kwargs))
    conversation = ModuleType("agent.conversation_loop")
    conversation._system_prompt_for_hooks = lambda *_args: None
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", conversation)
    agent = ApiRequestHooksMixin()
    agent.session_id, agent.platform, agent.model, agent.provider = "session", "feishu", "example", "test"
    agent.base_url, agent.api_mode, agent.tools, agent.max_tokens = "", "codex_responses", [], 1024
    _fire_pre_api_request_hook(
        agent, body, [], [], messages=[], original_user_message="test", approx_tokens=0,
        total_chars=0, retry_count=0, api_call_count=1, api_request_id="request", api_start_time=0,
        effective_task_id="task", turn_id="turn",
    )
    assert len(captured) == 1
    return captured[0]


def card_footer(monkeypatch, tmp_path, event):
    hooks = pytest.importorskip("hermes_lark_streaming.hooks")
    builder = pytest.importorskip("hermes_lark_streaming.cardkit.builder")
    observed = []
    monkeypatch.setattr(hooks, "_dispatch", lambda kind, payload: observed.append(payload))
    home = set_hermes_home_override(tmp_path / "saber")
    session = set_session_vars(platform="feishu", message_id="incoming", profile="saber")
    hooks._requests.clear()
    try:
        hooks.observe_request(**event)
        hooks.observe_usage(api_request_id="request", model="example")
    finally:
        clear_session_vars(session)
        reset_hermes_home_override(home)
        hooks._requests.clear()
    return builder._render_footer_field("model", observed[0], False, False, False)[0]


@pytest.mark.parametrize(("fields", "expected"), [
    ({"reasoning": {"effort": "high"}}, "high"),
    ({"reasoning_effort": "low", "extra_body": {"reasoning_effort": "high"}}, "high"),
    ({"extra_body": {"reasoning": {"effort": "max"}}}, "max"),
    ({"reasoning": {"enabled": False}}, "none"),
    ({}, None),
    ({"reasoning_effort": "invalid"}, None),
    ({"reasoning_effort": "low", "reasoning": {"effort": "high"}}, None),
])
def test_large_request_preserves_only_unambiguous_effort(monkeypatch, fields, expected):
    event = build_event(monkeypatch, fields)
    assert event["request"].get("_truncated") is True
    assert "reasoning_effort" in event["request"]
    assert event["request"]["reasoning_effort"] == expected


def test_large_request_effort_reaches_card_footer(monkeypatch, tmp_path):
    event = build_event(monkeypatch, {"reasoning": {"effort": "high"}})
    assert event["request"].get("_truncated") is True
    assert card_footer(monkeypatch, tmp_path, event) == "example (high)"


def test_short_request_still_shows_effort(monkeypatch, tmp_path):
    event = build_event(monkeypatch, {"reasoning": {"effort": "high"}}, large=False)
    assert not event["request"].get("_truncated")
    assert card_footer(monkeypatch, tmp_path, event) == "example (high)"


def test_host_unknown_does_not_fall_back_to_conflicting_body(monkeypatch, tmp_path):
    event = build_event(monkeypatch, {"reasoning_effort": "low", "reasoning": {"effort": "high"}}, large=False)
    assert card_footer(monkeypatch, tmp_path, event) == "example"

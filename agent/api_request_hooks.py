"""Lifecycle-hook payloads for ``AIAgent`` API requests.

JSON-safe coercion, secret-key redaction, size caps, and the ``api_request_error`` hook dispatch.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import json
import os
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Dict, Optional

from agent.usage_pricing import normalize_usage

_SENSITIVE_HOOK_KEYS = {"api_key", "authorization", "proxy_authorization", "cookie", "set_cookie"}


def _request_reasoning_effort(api_kwargs: Optional[dict[str, Any]]) -> Optional[str]:
    """Return a recognized, unambiguous request effort without copying request content."""
    from agent.reasoning_effort import EFFORT_LADDER

    if not isinstance(api_kwargs, dict):
        return None
    extra = api_kwargs.get("extra_body")
    if extra is not None and not isinstance(extra, dict):
        return None

    def value(key: str) -> Any:
        # The OpenAI SDK applies extra_body as a shallow right-hand override.
        return extra[key] if isinstance(extra, dict) and key in extra else api_kwargs.get(key)

    reasoning = value("reasoning")
    output_config = value("output_config")
    if reasoning is not None and not isinstance(reasoning, dict):
        return None
    if output_config is not None and not isinstance(output_config, dict):
        return None
    levels = [value("reasoning_effort")]
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            return None
        if enabled is False:
            levels.append("none")
        levels.append(reasoning.get("effort"))
        if enabled is True and any(isinstance(x, str) and x.strip().lower() == "none" for x in levels):
            return None
    if isinstance(output_config, dict):
        levels.append(output_config.get("effort"))
    levels = [level for level in levels if level is not None and level != ""]
    if any(not isinstance(level, str) or level.strip().lower() not in EFFORT_LADDER for level in levels):
        return None
    normalized = {level.strip().lower() for level in levels}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _model_dump(value: Any) -> Any:
    """``value.model_dump(mode="json")`` with graceful degradation for older pydantic signatures.

    warnings=False: pydantic UserWarnings on generic-union SDK models would leak to the terminal.
    """
    try:
        return value.model_dump(mode="json", warnings=False)
    except TypeError:
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()


class ApiRequestHooksMixin:
    """Hook payload sanitising + ``api_request_error`` dispatch (see module docstring)."""

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[dict[str, Any]]:
        """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    @staticmethod
    def _hook_payload_max_chars() -> int:
        raw = os.getenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "50000")
        try:
            return max(1000, int(raw))
        except (TypeError, ValueError):
            return 50000

    @staticmethod
    def _is_sensitive_hook_key(key: Any) -> bool:
        if not isinstance(key, str):
            return False
        lowered = key.lower().replace("-", "_")
        return lowered in _SENSITIVE_HOOK_KEYS or lowered.endswith("_api_key")

    @classmethod
    def _hook_jsonable(
        cls, value: Any, *, depth: int = 0, max_depth: int = 8, max_string: int = 8000,
        max_sequence: int = 200,
    ) -> Any:
        if depth > max_depth:
            return f"<{type(value).__name__} depth limit>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) > max_string:
                return value[:max_string] + f"...[truncated {len(value) - max_string} chars]"
            return value
        if isinstance(value, (bytes, bytearray)):
            return f"<{len(value)} bytes>"

        def recurse(item):
            return cls._hook_jsonable(
                item, depth=depth + 1, max_depth=max_depth, max_string=max_string,
                max_sequence=max_sequence,
            )

        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= max_sequence:
                    out["_truncated_items"] = len(value) - max_sequence
                    break
                str_key = str(key)
                out[str_key] = "<redacted>" if cls._is_sensitive_hook_key(str_key) else recurse(item)
            return out
        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            out = [recurse(item) for item in seq[:max_sequence]]
            if len(seq) > max_sequence:
                out.append({"_truncated_items": len(seq) - max_sequence})
            return out
        with suppress(Exception):
            if hasattr(value, "model_dump"):
                return recurse(_model_dump(value))
        with suppress(Exception):
            from dataclasses import asdict, is_dataclass
            if is_dataclass(value):
                return recurse(asdict(value))
        if isinstance(value, SimpleNamespace):
            return recurse(vars(value))
        if hasattr(value, "__dict__"):
            with suppress(Exception):
                return recurse({k: v for k, v in vars(value).items() if not str(k).startswith("_")})
        return str(value)[:max_string]

    @classmethod
    def _sanitize_hook_payload(cls, value: Any) -> Any:
        """JSON-able payload under the size cap: full → reduced caps → truncated preview."""
        limit = cls._hook_payload_max_chars()
        encoded = ""
        for caps in ({}, {"max_string": 1000, "max_sequence": 50}):
            payload = cls._hook_jsonable(value, **caps)
            try:
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                return str(payload)[:limit]
            if len(encoded) <= limit:
                return payload
        return {
            "_truncated": True, "original_type": type(value).__name__, "preview": encoded[:limit]
        }

    def _api_request_payload_for_hook(self, api_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
        body = {
            key: value
            for key, value in (api_kwargs or {}).items()
            if key not in {"timeout", "http_client"}
        }
        payload = self._sanitize_hook_payload({"method": "POST", "body": body})
        if isinstance(payload, dict):
            # This tiny scalar survives the payload cap; None means do not guess.
            payload["reasoning_effort"] = _request_reasoning_effort(api_kwargs)
        return payload

    def _api_response_payload_for_hook(
        self, response: Any, assistant_message: Any, *, finish_reason: Optional[str]
    ) -> dict[str, Any]:
        # Raw provider SDK tool_call objects are handed to the sanitizer on purpose; `_hook_jsonable` must
        # keep normalising them (model_dump / __dict__ / dataclass) or subscribers get str() blobs.
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        return self._sanitize_hook_payload(
            {
                "model": getattr(response, "model", None),
                # Downstream that served the call (relays re-roll it per request; #90216).
                "upstream_provider": getattr(response, "provider", None),
                "finish_reason": finish_reason,
                "assistant_message": {
                    "role": getattr(assistant_message, "role", "assistant"),
                    "content": getattr(assistant_message, "content", None),
                    "tool_calls": tool_calls,
                },
                "usage": self._usage_summary_for_api_request_hook(response),
            }
        )

    def _invoke_api_request_error_hook(
        self, *, task_id: str, turn_id: str, api_request_id: str, api_call_count: int,
        api_start_time: float, api_kwargs: Optional[dict[str, Any]], error_type: str,
        error_message: str, status_code: Optional[int] = None, retry_count: Optional[int] = None,
        max_retries: Optional[int] = None, retryable: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> None:
        # Lazy module import (not from-import) so tests can replace lifecycle dispatch at this call site.
        with suppress(Exception):
            from hermes_cli import lifecycle as _lifecycle
            if not _lifecycle.has_hook("api_request_error"):
                return
            ended_at = time.time()
            _lifecycle.invoke_hook(
                "api_request_error",
                task_id=task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=self.session_id or "",
                platform=self.platform or "",
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                api_mode=self.api_mode,
                api_call_count=api_call_count,
                api_duration=ended_at - api_start_time,
                started_at=api_start_time,
                ended_at=ended_at,
                status_code=status_code,
                retry_count=retry_count,
                max_retries=max_retries,
                retryable=retryable,
                reason=reason,
                error={"type": error_type, "message": error_message},
                request=self._api_request_payload_for_hook(api_kwargs),
            )

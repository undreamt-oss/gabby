# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026-present Gabby Contributors.

"""Pre-provider and post-provider model payload filtering contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol

from .models import ModelResponse

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)")
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|above)\s+"
    r"(?:instructions|rules|policies)\b|\b(?:reveal|print|expose)\s+(?:the\s+)?"
    r"(?:system|developer)\s+(?:prompt|message|instructions)\b"
    r"|\b(?:ignore|disregard|override|forget|abandon|change|break|drop)\s+"
    r"(?:your\s+)?(?:assigned\s+)?(?:persona|character|tone|role)\b",
    re.IGNORECASE,
)


class PayloadFilterError(ValueError):
    """A configured payload filter rejected an outbound or inbound model payload."""


class PayloadFilter(Protocol):
    """Pluggable validator or transformer at the model-provider boundary."""

    def filter_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        purpose: str,
    ) -> (
        tuple[list[dict[str, Any]], list[dict[str, Any]]]
        | Awaitable[tuple[list[dict[str, Any]], list[dict[str, Any]]]]
    ):
        """Return a sanitized request or raise PayloadFilterError before provider transmission."""

    def filter_response(
        self,
        *,
        response: ModelResponse,
        purpose: str,
    ) -> ModelResponse | Awaitable[ModelResponse]:
        """Return a sanitized response or raise PayloadFilterError before runtime handling."""


def _transform_strings(value: Any, transform: Any) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [_transform_strings(child, transform) for child in value]
    if isinstance(value, tuple):
        return [_transform_strings(child, transform) for child in value]
    if isinstance(value, Mapping):
        return {key: _transform_strings(child, transform) for key, child in value.items()}
    return value


def redact_sensitive_text(value: str) -> str:
    """Redact common email, phone, cloud-key, and token patterns from model payload text."""
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _PHONE.sub("[REDACTED_PHONE]", value)
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED_SECRET]", value)
    return value


class DefaultPayloadFilter:
    """Redact common sensitive strings and reject direct prompt-override attempts.

    This lightweight filter is a guardrail, not a semantic prompt-injection detector or a
    substitute for host data classification and provider controls.
    """

    def filter_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        purpose: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Sanitize outbound model input and reject direct instruction overrides."""
        del model, purpose
        for message in messages:
            role = message.get("role")
            if role in {"user", "tool"}:
                content = message.get("content")
                if isinstance(content, str) and _INJECTION.search(content):
                    raise PayloadFilterError("Request contains a blocked instruction override")
        return (
            _transform_strings(messages, redact_sensitive_text),
            _transform_strings(tools, redact_sensitive_text),
        )

    def filter_response(self, *, response: ModelResponse, purpose: str) -> ModelResponse:
        """Redact common sensitive patterns from returned text and raw provider data."""
        del purpose
        content = response.content
        if content is not None:
            try:
                structured = json.loads(content)
            except (json.JSONDecodeError, RecursionError):
                content = redact_sensitive_text(content)
            else:
                # Preserve valid JSON as JSON while redacting string values. Leave object keys
                # untouched because changing a key can alter the caller's structured contract.
                def redact_json_values(value: Any) -> Any:
                    if isinstance(value, str):
                        return redact_sensitive_text(value)
                    if isinstance(value, list):
                        return [redact_json_values(item) for item in value]
                    if isinstance(value, dict):
                        return {key: redact_json_values(item) for key, item in value.items()}
                    return value

                content = json.dumps(
                    redact_json_values(structured),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        return ModelResponse(
            content=content,
            # Tool arguments are structured operations; changing them after the model returns can
            # silently alter a requested action. Hosts can add a filter that validates or rejects
            # tool arguments without rewriting them.
            tool_calls=response.tool_calls,
            usage=response.usage,
            raw=_transform_strings(response.raw, redact_sensitive_text),
        )

    def filter_stream_delta(self, *, content_delta: str, purpose: str) -> str:
        """Redact one text fragment without treating it as a complete JSON document.

        Stream fragments are not response boundaries. Parsing each fragment as JSON can rewrite
        whitespace or object keys, so structured JSON is passed through this text-only redactor.
        Patterns split across fragments remain subject to the documented streaming limitation.
        """
        del purpose
        return redact_sensitive_text(content_delta)


def validate_payload_filters(filters: Sequence[PayloadFilter] | None) -> tuple[PayloadFilter, ...]:
    """Snapshot host-supplied filters and validate their callable contracts."""
    if filters is None:
        return (DefaultPayloadFilter(),)
    if isinstance(filters, (str, bytes)) or not isinstance(filters, Sequence):
        raise TypeError("payload_filters must be a sequence of PayloadFilter implementations")
    selected: tuple[PayloadFilter, ...] = (DefaultPayloadFilter(), *filters)
    if any(
        not callable(getattr(item, "filter_request", None))
        or not callable(getattr(item, "filter_response", None))
        for item in selected
    ):
        raise TypeError("each payload filter must implement filter_request() and filter_response()")
    return selected

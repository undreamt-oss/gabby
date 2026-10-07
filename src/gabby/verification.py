# Copyright 2026-present Gabby Contributors.
#
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
"""Typed contracts for run verification extensions."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .runtime import RunRequest
    from .tracing import ExecutionTrace


@dataclass(frozen=True)
class VerificationResult:
    """Structured outcome from one configured verifier."""

    passed: bool
    method: str
    details: Mapping[str, object] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid values before a result reaches the API or trace."""
        if not isinstance(self.passed, bool):
            raise TypeError("VerificationResult.passed must be a boolean")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("VerificationResult.method must be a non-empty string")
        if not isinstance(self.details, Mapping) or not all(
            isinstance(key, str) for key in self.details
        ):
            raise TypeError("VerificationResult.details must be a mapping with string keys")
        try:
            snapshot = json.loads(json.dumps(dict(self.details), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "VerificationResult.details must contain JSON-compatible values"
            ) from exc
        object.__setattr__(self, "details", snapshot)
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, str) for item in self.evidence
        ):
            raise TypeError("VerificationResult.evidence must be a tuple of strings")

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible public representation of this result."""
        return {
            "passed": self.passed,
            "method": self.method,
            "details": dict(self.details),
            "evidence": list(self.evidence),
        }


class Verifier(Protocol):
    """Async verifier contract; synchronous implementations are also supported at runtime."""

    def verify(
        self, *, request: RunRequest, output: str, trace: ExecutionTrace
    ) -> VerificationResult | Awaitable[VerificationResult]:
        """Check an output and return a structured, evidence-bearing outcome."""

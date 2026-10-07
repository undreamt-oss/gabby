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
"""Shared bounded serialization helpers for approval audit backends."""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_MAX_AUDIT_ARGUMENT_BYTES = 1024 * 1024
MAX_AUDIT_QUERY_LIMIT = 1000
MAX_AUDIT_ID_BYTES = 512
MAX_AUDIT_NAME_BYTES = 256
MAX_AUDIT_SUBJECT_BYTES = 4096


class ApprovalAuditError(RuntimeError):
    """Sanitized failure to durably record or read an approval audit entry."""


class ArgumentSizeError(ValueError):
    """The canonical JSON representation is larger than the configured audit bound."""


def canonical_argument_digest(arguments: Any, *, max_bytes: int) -> str:
    """Hash canonical JSON arguments without materializing one serialized copy."""
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in encoder.iterencode(arguments):
            for offset in range(0, len(chunk), 4096):
                encoded = chunk[offset : offset + 4096].encode("utf-8")
                size += len(encoded)
                if size > max_bytes:
                    raise ArgumentSizeError("Tool arguments exceed the audit size limit")
                digest.update(encoded)
    except ArgumentSizeError:
        raise
    except Exception:
        raise ValueError("Tool arguments cannot be safely audited") from None
    return digest.hexdigest()

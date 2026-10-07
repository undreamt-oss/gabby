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
"""Loading and validating agent and skill definitions."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import yaml
from jsonschema import Draft202012Validator, SchemaError

from .knowledge import MAX_RETRIEVAL_DOCUMENTS


class ConfigError(ValueError):
    """Raised when an agent or skill definition is invalid."""


DEFAULT_MAX_MODEL_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_MODEL_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_KNOWLEDGE_CONTEXT_BYTES = 1024 * 1024
_MAX_AGENT_CONFIG_BYTES = 10 * 1024 * 1024
_MAX_SKILL_MANIFEST_BYTES = 1024 * 1024
_MAX_SKILL_TEXT_BYTES = 10 * 1024 * 1024
_MAX_SKILL_METADATA_BYTES = 1024 * 1024
_MAX_SKILL_METADATA_VALUES = 10_000
_MAX_RESOLVED_SKILLS = 256
_MAX_AGENT_TEXT_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_VALUE_DEPTH = 128
_MAX_CONFIG_VALUE_NODES = 100_000
DEFAULT_MAX_TOOL_CALLS = 64
MAX_TOOL_CALLS = 1024
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 1
MAX_PARALLEL_TOOL_CALLS = 32
MAX_MODEL_RETRIES = 3
MAX_REPLANS = 3
MAX_KNOWLEDGE_TOP_K = MAX_RETRIEVAL_DOCUMENTS
_MODEL_CREDENTIAL_MARKERS = (
    "key",
    "auth",
    "token",
    "secret",
    "credential",
    "password",
    "passwd",
    "cookie",
    "bearer",
    "signature",
)
_BUILTIN_MODEL_PROVIDERS = frozenset(
    {
        "openai",
        "openai_compatible",
        "ollama",
        "huggingface",
        "anthropic",
        "gemini",
        "transformers",
    }
)
_SEMVER_PATTERN = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)(?:\."
    r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
DEFAULT_SANDBOX_USER = "65532:65532"
_MAX_SANDBOX_ID = 2**32 - 1


def validate_model_credentials(model: Mapping[str, Any], *, provider: str | None = None) -> None:
    """Reject model credentials embedded in agent configuration mappings."""
    selected_provider = provider or model.get("provider", "openai_compatible")
    if selected_provider == "anthropic" and "max_tokens" in model:
        max_tokens = model["max_tokens"]
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 200_000
        ):
            raise ConfigError("'model.max_tokens' must be an integer from 1 through 200000")
    if selected_provider == "gemini" and "max_output_tokens" in model:
        max_output_tokens = model["max_output_tokens"]
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= 200_000
        ):
            raise ConfigError("'model.max_output_tokens' must be an integer from 1 through 200000")
    if "api_key" in model:
        raise ConfigError(
            "'model.api_key' cannot be stored in an agent definition; "
            "use model.api_key_env or inject a ModelProvider"
        )
    api_key_env = model.get("api_key_env")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env)
    ):
        raise ConfigError("'model.api_key_env' must be an environment variable name")
    headers = model.get("headers", {})
    if not isinstance(headers, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
    ):
        raise ConfigError("'model.headers' must be a mapping of string names to string values")
    if any(_contains_model_credential_marker(key) for key in headers):
        raise ConfigError(
            "Credential-bearing model headers cannot be stored in an agent definition; "
            "use model.api_key_env or inject a ModelProvider"
        )
    base_url = model.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str):
            raise ConfigError("'model.base_url' must be a string")
        try:
            parsed_url = urlsplit(base_url)
            query_keys = (key for key, _ in parse_qsl(parsed_url.query, keep_blank_values=True))
            has_url_credentials = parsed_url.username is not None or parsed_url.password is not None
            has_query_credentials = any(
                _contains_model_credential_marker(key) for key in query_keys
            )
        except ValueError as exc:
            raise ConfigError("'model.base_url' is not a valid URL") from exc
        if has_url_credentials or has_query_credentials:
            raise ConfigError(
                "Credential-bearing provider URLs cannot be stored in an agent definition; "
                "use model.api_key_env or inject a ModelProvider"
            )
        if selected_provider in _BUILTIN_MODEL_PROVIDERS:
            validate_model_endpoint(base_url, provider=selected_provider)


def validate_model_endpoint(base_url: str, *, provider: str) -> None:
    """Require TLS for remote built-in model endpoints; permit cleartext loopback only."""
    try:
        parsed_url = urlsplit(base_url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
    except ValueError as exc:
        raise ConfigError("'model.base_url' is not a valid URL") from exc
    if parsed_url.scheme == "https" and hostname:
        return
    if parsed_url.scheme == "http" and hostname and _is_loopback_host(hostname):
        return
    raise ConfigError(
        f"Built-in model provider {provider!r} requires HTTPS for remote endpoints; "
        "HTTP is allowed only for loopback local inference"
    )


def validate_sandbox_api_endpoint(base_url: str) -> None:
    """Require HTTPS for remote container APIs and permit HTTP only on loopback."""
    try:
        parsed_url = urlsplit(base_url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
        query_keys = (key for key, _ in parse_qsl(parsed_url.query, keep_blank_values=True))
        has_url_credentials = parsed_url.username is not None or parsed_url.password is not None
        has_query_credentials = any(_contains_model_credential_marker(key) for key in query_keys)
    except ValueError as exc:
        raise ConfigError("sandbox.api.base_url must be a valid HTTP(S) URL") from exc
    if has_url_credentials or has_query_credentials:
        raise ConfigError(
            "sandbox.api.base_url cannot contain credentials; inject an authenticated engine client"
        )
    if (parsed_url.scheme == "https" and hostname) or (
        parsed_url.scheme == "http" and hostname and _is_loopback_host(hostname)
    ):
        return
    raise ConfigError("Remote container APIs must use HTTPS; HTTP is allowed only for loopback")


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return isinstance(address, ipaddress.IPv6Address) and (
        address.ipv4_mapped is not None and address.ipv4_mapped.is_loopback
    )


def _contains_model_credential_marker(value: str) -> bool:
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    parts = set(re.split(r"[^a-z0-9]+", value.casefold()))
    return (
        "authorization" in normalized
        or "apikey" in normalized
        or bool(parts.intersection(_MODEL_CREDENTIAL_MARKERS))
        or normalized.endswith(("accesskey", "clientkey", "subscriptionkey"))
        or "secret" in normalized
        or "bearer" in normalized
    )


@dataclass(frozen=True)
class WorkspaceMount:
    """A host directory mounted into a per-run sandbox container."""

    host_path: Path
    access: str
    container_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.host_path, Path) or not self.host_path.is_dir():
            raise ConfigError("sandbox.workspace.path must be an existing directory")
        object.__setattr__(self, "host_path", self.host_path.resolve())
        if self.access not in ("read_only", "read_write"):
            raise ConfigError("sandbox.workspace.access must be 'read_only' or 'read_write'")
        if self.container_path is not None and not isinstance(self.container_path, str):
            raise ConfigError("sandbox.workspace.container_path must be a string")
        if self.container_path is not None:
            windows_absolute = (
                len(self.container_path) >= 3
                and self.container_path[1] == ":"
                and self.container_path[2] in ("\\", "/")
            )
            if not self.container_path.startswith("/") and not windows_absolute:
                raise ConfigError("sandbox.workspace.container_path must be absolute")
            if ".." in self.container_path.replace("\\", "/").split("/"):
                raise ConfigError("sandbox.workspace.container_path cannot contain '..'")


@dataclass(frozen=True)
class SandboxDefinition:
    """Container engine and per-run isolation configuration."""

    engine: str
    image: str
    keepalive_argv: tuple[str, ...]
    adapter: str = "cli"
    workspace: WorkspaceMount | None = None
    cpus: float = 2.0
    memory_bytes: int = 2 * 1024**3
    process_limit: int | None = 256
    require_image_digest: bool = False
    api_base_url: str | None = None
    api_unix_socket: Path | None = None
    api_named_pipe: str | None = None
    user: str = DEFAULT_SANDBOX_USER

    def __post_init__(self) -> None:
        if self.engine not in ("docker", "podman"):
            raise ConfigError("sandbox.engine must be 'docker' or 'podman'")
        if not isinstance(self.image, str) or not self.image.strip():
            raise ConfigError("sandbox.image must be a non-empty image reference")
        image = self.image.strip()
        if image.startswith("-") or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in image
        ):
            raise ConfigError(
                "sandbox.image must not begin with an engine option or contain "
                "whitespace/control characters"
            )
        object.__setattr__(self, "image", image)
        if not isinstance(self.require_image_digest, bool):
            raise ConfigError("sandbox.require_image_digest must be a boolean")
        if self.require_image_digest and not re.search(r"@sha256:[0-9a-fA-F]{64}$", self.image):
            raise ConfigError(
                "sandbox.image must end with @sha256:<64 hexadecimal characters> "
                "when require_image_digest is true"
            )
        if self.adapter not in ("cli", "api"):
            raise ConfigError("sandbox.adapter must be 'cli' or 'api'")
        if self.workspace is not None and not isinstance(self.workspace, WorkspaceMount):
            raise ConfigError("sandbox.workspace must be a WorkspaceMount or None")
        if (
            not isinstance(self.keepalive_argv, tuple)
            or not self.keepalive_argv
            or any(not isinstance(value, str) or not value for value in self.keepalive_argv)
        ):
            raise ConfigError(
                "sandbox.keepalive_argv must be a non-empty list of non-empty strings"
            )
        if (
            isinstance(self.cpus, bool)
            or not isinstance(self.cpus, (int, float))
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
        ):
            raise ConfigError("sandbox.resources.cpus must be a finite positive number")
        if isinstance(self.memory_bytes, bool) or not isinstance(self.memory_bytes, int):
            raise ConfigError("sandbox.resources.memory_bytes must be a positive integer")
        if self.memory_bytes < 1:
            raise ConfigError("sandbox.resources.memory_bytes must be a positive integer")
        if self.process_limit is not None and (
            isinstance(self.process_limit, bool) or not isinstance(self.process_limit, int)
        ):
            raise ConfigError("sandbox.resources.process_limit must be a positive integer or null")
        if self.process_limit is not None and self.process_limit < 1:
            raise ConfigError("sandbox.resources.process_limit must be a positive integer")
        if not isinstance(self.user, str) or not re.fullmatch(
            r"[1-9][0-9]*:[1-9][0-9]*", self.user
        ):
            raise ConfigError(
                "sandbox.user must be a numeric non-root UID:GID pair in the range 1..4294967295"
            )
        uid, gid = (int(value) for value in self.user.split(":"))
        if uid > _MAX_SANDBOX_ID or gid > _MAX_SANDBOX_ID:
            raise ConfigError(
                "sandbox.user must be a numeric non-root UID:GID pair in the range 1..4294967295"
            )
        endpoints = (self.api_base_url, self.api_unix_socket, self.api_named_pipe)
        if self.adapter == "api" and not any(endpoints):
            raise ConfigError(
                "sandbox.api requires base_url, unix_socket, or named_pipe when adapter is 'api'"
            )
        if sum(endpoint is not None for endpoint in endpoints) > 1:
            raise ConfigError("sandbox.api must configure exactly one API endpoint")
        if self.api_base_url is not None:
            if not isinstance(self.api_base_url, str):
                raise ConfigError("sandbox.api.base_url must be an HTTP(S) URL")
            try:
                parsed_url = urlsplit(self.api_base_url)
                hostname = parsed_url.hostname
                _ = parsed_url.port
            except ValueError as exc:
                raise ConfigError("sandbox.api.base_url must be a valid HTTP(S) URL") from exc
            if parsed_url.scheme not in ("http", "https") or not hostname:
                raise ConfigError("sandbox.api.base_url must be a valid HTTP(S) URL")
        if self.api_unix_socket is not None:
            if not isinstance(self.api_unix_socket, Path):
                raise ConfigError("sandbox.api.unix_socket must be a pathlib Path")
            object.__setattr__(self, "api_unix_socket", self.api_unix_socket.expanduser().resolve())
        if self.api_named_pipe is not None and (
            not isinstance(self.api_named_pipe, str)
            or not re.fullmatch(r"\\\\\.\\pipe\\[A-Za-z0-9._-]+", self.api_named_pipe)
        ):
            raise ConfigError(
                r"sandbox.api.named_pipe must be a local Windows pipe such as "
                r"\\.\pipe\docker_engine"
            )


@dataclass
class AgentDefinition:
    """Mutable source configuration used to construct an immutable resolved agent."""

    name: str
    model: dict[str, Any]
    description: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    skill_paths: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    knowledge: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    policies: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    sandbox: SandboxDefinition | None = None
    source_path: Path | None = None
    output_schema: dict[str, Any] | None = None


@dataclass
class SkillDefinition:
    """Reusable capability definition that can be attached to one or more agents."""

    name: str
    description: str = ""
    instructions: str = ""
    examples: str = ""
    tools: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    source_path: Path | None = None
    version: str = "0.1.0"
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedSkillDefinition:
    """Immutable skill snapshot attached to one constructed agent."""

    name: str
    description: str
    instructions: str
    examples: str
    tools: tuple[str, ...]
    knowledge: tuple[str, ...]
    constraints: tuple[str, ...]
    dependencies: tuple[str, ...]
    verification: tuple[str, ...]
    triggers: tuple[str, ...]
    source_path: Path | None
    version: str = "0.1.0"
    input_schema: Mapping[str, Any] | None = None
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    """Immutable configuration snapshot used by one agent's runtime."""

    name: str
    model: Mapping[str, Any]
    description: str
    environment: Mapping[str, Any]
    skills: tuple[str, ...]
    skill_paths: tuple[str, ...]
    tools: tuple[str, ...]
    knowledge: Mapping[str, Any]
    instructions: str
    policies: Mapping[str, Any]
    verification: Mapping[str, Any]
    sandbox: SandboxDefinition | None
    source_path: Path | None
    skill_definitions: tuple[ResolvedSkillDefinition, ...]
    output_schema: Mapping[str, Any] | None = None


def _freeze_config(value: Any) -> Any:
    """Validate and freeze bounded JSON configuration values without retaining mutable aliases."""
    active: set[int] = set()
    node_count = 0
    byte_count = 0

    def account(size: int, path: str) -> None:
        nonlocal byte_count
        byte_count += size
        if byte_count > _MAX_AGENT_CONFIG_BYTES:
            raise ConfigError(
                f"{path} exceeds the {_MAX_AGENT_CONFIG_BYTES}-byte configuration limit"
            )

    def freeze(item: Any, path: str, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        account(1, path)
        if node_count > _MAX_CONFIG_VALUE_NODES:
            raise ConfigError(
                f"{path} exceeds the {_MAX_CONFIG_VALUE_NODES}-node configuration limit"
            )
        if depth > _MAX_CONFIG_VALUE_DEPTH:
            raise ConfigError(
                f"{path} exceeds the {_MAX_CONFIG_VALUE_DEPTH}-level configuration depth limit"
            )
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ConfigError(f"{path} contains a cyclic configuration value")
            active.add(identity)
            frozen: dict[str, Any] = {}
            try:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ConfigError(f"{path} contains a non-string mapping key")
                    try:
                        account(len(key.encode("utf-8")) + 2, path)
                    except UnicodeEncodeError as exc:
                        raise ConfigError(f"{path} contains invalid Unicode") from exc
                    frozen[key] = freeze(child, f"{path}.{key}", depth + 1)
            finally:
                active.remove(identity)
            return MappingProxyType(frozen)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ConfigError(f"{path} contains a cyclic configuration value")
            active.add(identity)
            try:
                return tuple(
                    freeze(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item)
                )
            finally:
                active.remove(identity)
        if item is None or type(item) in (bool, str):
            if type(item) is str:
                try:
                    account(len(item.encode("utf-8")) + 2, path)
                except UnicodeEncodeError as exc:
                    raise ConfigError(f"{path} contains invalid Unicode") from exc
            return item
        if type(item) is int:
            # Estimate JSON's decimal representation without converting an arbitrarily large int.
            account((item.bit_length() * 30_103) // 100_000 + 2, path)
            return item
        if type(item) is float and math.isfinite(item):
            account(len(repr(item)), path)
            return item
        raise ConfigError(f"{path} must contain only finite JSON-compatible values")

    return freeze(value, "Agent configuration", 0)


def _resolved_skill(skill: SkillDefinition) -> ResolvedSkillDefinition:
    return ResolvedSkillDefinition(
        name=skill.name,
        description=skill.description,
        instructions=skill.instructions,
        examples=skill.examples,
        tools=tuple(skill.tools),
        knowledge=tuple(skill.knowledge),
        constraints=tuple(skill.constraints),
        dependencies=tuple(_split_skill_reference(value)[0] for value in skill.dependencies),
        verification=tuple(skill.verification),
        triggers=tuple(skill.triggers),
        source_path=skill.source_path,
        version=skill.version,
        input_schema=(
            _freeze_config(skill.input_schema) if skill.input_schema is not None else None
        ),
        output_schema=(
            _freeze_config(skill.output_schema) if skill.output_schema is not None else None
        ),
    )


def _validate_skill_id(name: str) -> None:
    """Reject IDs that can escape a configured registry or collide with version syntax."""
    windows_path = PureWindowsPath(name)
    if (
        not name
        or "@" in name
        or "\\" in name
        or name.startswith("/")
        or windows_path.drive
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ConfigError(f"Invalid skill ID {name!r}: expected a safe relative skill ID")


def _split_skill_reference(reference: str) -> tuple[str, str | None]:
    """Parse an unpinned skill ID or an exact ``skill-id@semver`` reference."""
    if not isinstance(reference, str):
        raise ConfigError("Skill references must be strings")
    if "@" in reference:
        name, separator, version = reference.rpartition("@")
        if not separator or not _SEMVER_PATTERN.fullmatch(version):
            raise ConfigError(
                f"Invalid skill reference {reference!r}: expected skill-id@MAJOR.MINOR.PATCH"
            )
    else:
        name, version = reference, None
    _validate_skill_id(name)
    return name, version


def validate_skill_definition(skill: SkillDefinition) -> None:
    """Apply the same structural checks to loaded and programmatic skill packages."""
    if not isinstance(skill, SkillDefinition):
        raise ConfigError("skill definition must be a SkillDefinition")
    name = skill.name
    if type(name) is not str or not name or name != name.strip():
        raise ConfigError("skill.name must be a non-empty canonical skill ID")
    _validate_skill_id(name)
    if type(skill.version) is not str or not _SEMVER_PATTERN.fullmatch(skill.version):
        raise ConfigError("skill.version must be a valid semantic version")
    for field_name in ("description", "instructions", "examples"):
        text_value = getattr(skill, field_name)
        if type(text_value) is not str:
            raise ConfigError(f"skill.{field_name} must be a string")
        _validate_text_bytes(f"skill.{field_name}", text_value, _MAX_SKILL_TEXT_BYTES)
    for field_name in (
        "tools",
        "knowledge",
        "constraints",
        "dependencies",
        "verification",
        "triggers",
    ):
        values = getattr(skill, field_name)
        if not isinstance(values, list) or any(
            type(value) is not str or not value for value in values
        ):
            raise ConfigError(f"skill.{field_name} must be a list of non-empty strings")
    metadata_values = 0
    metadata_bytes = 0
    for field_name in (
        "tools",
        "knowledge",
        "constraints",
        "dependencies",
        "verification",
        "triggers",
    ):
        for value in getattr(skill, field_name):
            metadata_values += 1
            if len(value) > _MAX_SKILL_METADATA_BYTES:
                raise ConfigError(
                    f"Skill metadata exceeds the {_MAX_SKILL_METADATA_BYTES}-byte limit"
                )
            try:
                metadata_bytes += len(value.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ConfigError(f"skill.{field_name} must contain valid Unicode") from exc
            if metadata_values > _MAX_SKILL_METADATA_VALUES:
                raise ConfigError(
                    f"Skill metadata exceeds the {_MAX_SKILL_METADATA_VALUES}-value limit"
                )
            if metadata_bytes > _MAX_SKILL_METADATA_BYTES:
                raise ConfigError(
                    f"Skill metadata exceeds the {_MAX_SKILL_METADATA_BYTES}-byte limit"
                )
    for dependency in skill.dependencies:
        _split_skill_reference(dependency)
    if skill.source_path is not None and not isinstance(skill.source_path, Path):
        raise ConfigError("skill.source_path must be a pathlib Path or None")
    schema_bytes = 0
    for field_name in ("input_schema", "output_schema"):
        schema = getattr(skill, field_name)
        _validate_output_schema(schema, field_name=field_name)
        if schema is not None:
            try:
                schema_bytes += len(
                    json.dumps(
                        schema,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            except (TypeError, ValueError, UnicodeEncodeError):
                raise ConfigError(f"skill.{field_name} must be bounded JSON") from None
            if schema_bytes > _MAX_SKILL_METADATA_BYTES:
                raise ConfigError(
                    f"Skill schemas exceed the {_MAX_SKILL_METADATA_BYTES}-byte limit"
                )


def resolve_agent_definition(
    definition: AgentDefinition, skills: list[SkillDefinition]
) -> ResolvedAgentDefinition:
    """Compile mutable source definitions into an immutable execution snapshot."""
    for skill in skills:
        validate_skill_definition(skill)
    sandbox = definition.sandbox
    sandbox_config = None
    if sandbox is not None:
        workspace = sandbox.workspace
        sandbox_config = {
            "engine": sandbox.engine,
            "image": sandbox.image,
            "adapter": sandbox.adapter,
            "keepalive_argv": sandbox.keepalive_argv,
            "workspace": (
                None
                if workspace is None
                else {
                    "host_path": str(workspace.host_path),
                    "access": workspace.access,
                    "container_path": workspace.container_path,
                }
            ),
            "cpus": sandbox.cpus,
            "memory_bytes": sandbox.memory_bytes,
            "process_limit": sandbox.process_limit,
            "require_image_digest": sandbox.require_image_digest,
            "api_base_url": sandbox.api_base_url,
            "api_unix_socket": (
                str(sandbox.api_unix_socket) if sandbox.api_unix_socket is not None else None
            ),
            "api_named_pipe": sandbox.api_named_pipe,
            "user": sandbox.user,
        }
    frozen_config = _freeze_config(
        {
            "name": definition.name,
            "description": definition.description,
            "model": definition.model,
            "environment": definition.environment,
            "skills": definition.skills,
            "skill_paths": definition.skill_paths,
            "tools": definition.tools,
            "knowledge": definition.knowledge,
            "instructions": definition.instructions,
            "policies": definition.policies,
            "verification": definition.verification,
            "output_schema": definition.output_schema,
            "sandbox": sandbox_config,
        }
    )
    return ResolvedAgentDefinition(
        name=frozen_config["name"],
        model=frozen_config["model"],
        description=frozen_config["description"],
        environment=frozen_config["environment"],
        skills=frozen_config["skills"],
        skill_paths=frozen_config["skill_paths"],
        tools=frozen_config["tools"],
        knowledge=frozen_config["knowledge"],
        instructions=frozen_config["instructions"],
        policies=frozen_config["policies"],
        verification=frozen_config["verification"],
        output_schema=frozen_config["output_schema"],
        sandbox=definition.sandbox,
        source_path=definition.source_path,
        skill_definitions=tuple(_resolved_skill(skill) for skill in skills),
    )


def _read_yaml(path: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    byte_limit = _MAX_AGENT_CONFIG_BYTES if max_bytes is None else max_bytes
    try:
        with path.open("rb") as stream:
            contents = stream.read(byte_limit + 1)
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    if len(contents) > byte_limit:
        raise ConfigError(f"Configuration file exceeds the {byte_limit}-byte limit: {path}")
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Configuration file is not valid UTF-8: {path}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _validate_text_bytes(field_name: str, value: str, max_bytes: int) -> None:
    """Reject configuration text whose UTF-8 representation exceeds its byte budget."""
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ConfigError(f"{field_name} must contain valid Unicode text") from exc
    if size > max_bytes:
        raise ConfigError(f"{field_name} exceeds the {max_bytes}-byte limit")


def _strings(value: Any, field_name: str, *, allow_none: bool = True) -> list[str]:
    if value is None and allow_none:
        return []
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise ConfigError(f"{field_name} must be a list of non-empty strings")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], field_name: str, allowed: frozenset[str]
) -> None:
    """Fail closed when a Gabby-owned configuration mapping has unknown keys."""
    unknown = [key for key in value if key not in allowed]
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ConfigError(f"{field_name} contains unknown field(s): {names}")


def _string(value: Any, field_name: str, *, default: str = "") -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    return value if value else default


def _validate_execution_config(
    environment: dict[str, Any],
    knowledge: dict[str, Any],
    policies: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    _reject_unknown_fields(
        environment,
        "environment",
        frozenset({"type", "description", "capabilities", "allowed_tools", "resources"}),
    )
    _reject_unknown_fields(
        knowledge, "knowledge", frozenset({"sources", "top_k", "max_context_bytes"})
    )
    _reject_unknown_fields(
        policies,
        "policies",
        frozenset(
            {
                "max_steps",
                "max_tool_calls",
                "max_parallel_tool_calls",
                "max_model_retries",
                "max_replans",
                "timeout_seconds",
                "require_sandbox",
                "allowed_tools",
                "allowed_permissions",
                "max_model_request_bytes",
                "max_model_response_bytes",
            }
        ),
    )
    _reject_unknown_fields(verification, "verification", frozenset({"enabled"}))
    for key in ("type", "description"):
        if key in environment and not isinstance(environment[key], str):
            raise ConfigError(f"environment.{key} must be a string")
    if "capabilities" in environment:
        _strings(environment["capabilities"], "environment.capabilities", allow_none=False)
    if "allowed_tools" in environment:
        _strings(environment["allowed_tools"], "environment.allowed_tools", allow_none=False)
    if "resources" in environment and not isinstance(environment["resources"], dict):
        raise ConfigError("environment.resources must be a mapping")

    if "max_steps" in policies:
        max_steps = policies["max_steps"]
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ConfigError("policies.max_steps must be a positive integer")
    if "max_tool_calls" in policies:
        max_tool_calls = policies["max_tool_calls"]
        if (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or not 1 <= max_tool_calls <= MAX_TOOL_CALLS
        ):
            raise ConfigError(
                f"policies.max_tool_calls must be an integer from 1 through {MAX_TOOL_CALLS}"
            )
    if "max_parallel_tool_calls" in policies:
        parallel_calls = policies["max_parallel_tool_calls"]
        if (
            isinstance(parallel_calls, bool)
            or not isinstance(parallel_calls, int)
            or not DEFAULT_MAX_PARALLEL_TOOL_CALLS <= parallel_calls <= MAX_PARALLEL_TOOL_CALLS
        ):
            raise ConfigError(
                "policies.max_parallel_tool_calls must be an integer from 1 through "
                f"{MAX_PARALLEL_TOOL_CALLS}"
            )
    if "max_model_retries" in policies:
        retries = policies["max_model_retries"]
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= MAX_MODEL_RETRIES
        ):
            raise ConfigError(
                f"policies.max_model_retries must be an integer from 0 through {MAX_MODEL_RETRIES}"
            )
    if "max_replans" in policies:
        max_replans = policies["max_replans"]
        if (
            isinstance(max_replans, bool)
            or not isinstance(max_replans, int)
            or not 0 <= max_replans <= MAX_REPLANS
        ):
            raise ConfigError(
                f"policies.max_replans must be an integer from 0 through {MAX_REPLANS}"
            )
    if "max_model_request_bytes" in policies:
        request_bytes = policies["max_model_request_bytes"]
        if (
            isinstance(request_bytes, bool)
            or not isinstance(request_bytes, int)
            or request_bytes < 1
        ):
            raise ConfigError("policies.max_model_request_bytes must be a positive integer")
    if "max_model_response_bytes" in policies:
        response_bytes = policies["max_model_response_bytes"]
        if (
            isinstance(response_bytes, bool)
            or not isinstance(response_bytes, int)
            or response_bytes < 1
        ):
            raise ConfigError("policies.max_model_response_bytes must be a positive integer")
    if "timeout_seconds" in policies:
        timeout = policies["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigError("policies.timeout_seconds must be a finite positive number")
    if "require_sandbox" in policies and not isinstance(policies["require_sandbox"], bool):
        raise ConfigError("policies.require_sandbox must be a boolean")
    for key in ("allowed_tools", "allowed_permissions"):
        if key in policies:
            _strings(policies[key], f"policies.{key}", allow_none=False)

    if "top_k" in knowledge:
        top_k = knowledge["top_k"]
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= MAX_KNOWLEDGE_TOP_K
        ):
            raise ConfigError(
                f"knowledge.top_k must be an integer from 1 through {MAX_KNOWLEDGE_TOP_K}"
            )
    if "max_context_bytes" in knowledge:
        context_bytes = knowledge["max_context_bytes"]
        if (
            isinstance(context_bytes, bool)
            or not isinstance(context_bytes, int)
            or context_bytes < 1
        ):
            raise ConfigError("knowledge.max_context_bytes must be a positive integer")
    if "enabled" in verification and not isinstance(verification["enabled"], bool):
        raise ConfigError("verification.enabled must be a boolean")


def validate_agent_definition(definition: AgentDefinition) -> None:
    """Validate an agent definition regardless of whether it came from YAML or Python."""
    if not isinstance(definition, AgentDefinition):
        raise ConfigError("Expected an AgentDefinition")
    if type(definition.name) is not str or not definition.name.strip():
        raise ConfigError("Agent definition requires a non-empty 'name'")
    model = definition.model
    if (
        not isinstance(model, dict)
        or type(model.get("model")) is not str
        or not model.get("model")
        or type(model.get("provider")) is not str
        or not model.get("provider")
    ):
        raise ConfigError("Agent definition requires model.provider and model.model")
    validate_model_credentials(model, provider=model.get("provider"))

    for field_name, text_value in (
        ("description", definition.description),
        ("instructions", definition.instructions),
    ):
        if type(text_value) is not str:
            raise ConfigError(f"'{field_name}' must be a string")
        _validate_text_bytes(field_name, text_value, _MAX_AGENT_CONFIG_BYTES)
    for field_name, mapping_value in (
        ("environment", definition.environment),
        ("knowledge", definition.knowledge),
        ("policies", definition.policies),
        ("verification", definition.verification),
    ):
        if not isinstance(mapping_value, dict):
            raise ConfigError(f"'{field_name}' must be a mapping")
    _validate_output_schema(definition.output_schema)
    skill_references = _strings(definition.skills, "skills", allow_none=False)
    if len(skill_references) > _MAX_RESOLVED_SKILLS:
        raise ConfigError(f"skills may reference at most {_MAX_RESOLVED_SKILLS} skills")
    _strings(definition.skill_paths, "skill_paths", allow_none=False)
    _strings(definition.tools, "tools", allow_none=False)
    if definition.source_path is not None and not isinstance(definition.source_path, Path):
        raise ConfigError("source_path must be a pathlib Path or None")
    if definition.sandbox is not None and not isinstance(definition.sandbox, SandboxDefinition):
        raise ConfigError("sandbox must be a SandboxDefinition or None")
    if definition.sandbox is not None and definition.sandbox.api_base_url is not None:
        validate_sandbox_api_endpoint(definition.sandbox.api_base_url)
    _validate_execution_config(
        definition.environment,
        definition.knowledge,
        definition.policies,
        definition.verification,
    )


def _validate_output_schema(
    schema: dict[str, Any] | None, *, field_name: str = "output_schema"
) -> None:
    """Validate and bound an optional local JSON Schema contract."""
    if schema is None:
        return
    if not isinstance(schema, dict):
        raise ConfigError(f"{field_name} must be a JSON Schema object or None")
    # Reuse the config snapshot limits and cycle checks before jsonschema traverses developer input.
    _freeze_config(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, TypeError, ValueError):
        raise ConfigError(
            f"{field_name} must be a valid JSON Schema Draft 2020-12 object"
        ) from None

    map_schema_keywords = {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
    list_schema_keywords = {"allOf", "anyOf", "oneOf", "prefixItems"}
    single_schema_keywords = {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
    pending: list[Any] = [schema]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            reference = item.get("$ref", item.get("$dynamicRef"))
            if reference is not None and (
                not isinstance(reference, str) or not reference.startswith("#")
            ):
                raise ConfigError(f"{field_name} references must be local JSON fragments")
            identifier_value = item.get("$id")
            if isinstance(identifier_value, str):
                identifier = urlsplit(identifier_value)
                if identifier.scheme or identifier.netloc:
                    raise ConfigError(f"{field_name} identifiers must not reference remote URIs")
            for keyword in map_schema_keywords:
                children = item.get(keyword)
                if isinstance(children, dict):
                    pending.extend(children.values())
            for keyword in list_schema_keywords:
                children = item.get(keyword)
                if isinstance(children, list):
                    pending.extend(children)
            for keyword in single_schema_keywords:
                child = item.get(keyword)
                if isinstance(child, (dict, bool)):
                    pending.append(child)


def _sandbox_config(value: Any, source: Path) -> SandboxDefinition | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("'sandbox' must be a mapping")
    _reject_unknown_fields(
        value,
        "sandbox",
        frozenset(
            {
                "engine",
                "image",
                "adapter",
                "keepalive_argv",
                "workspace",
                "resources",
                "require_image_digest",
                "api",
                "user",
            }
        ),
    )

    engine = value.get("engine")
    image = value.get("image")
    adapter = value.get("adapter", "cli")
    keepalive = value.get("keepalive_argv")
    if engine not in ("docker", "podman"):
        raise ConfigError("sandbox.engine must be 'docker' or 'podman'")
    if not isinstance(image, str) or not image.strip():
        raise ConfigError("sandbox.image must be a non-empty image reference")
    if adapter not in ("cli", "api"):
        raise ConfigError("sandbox.adapter must be 'cli' or 'api'")
    if (
        not isinstance(keepalive, list)
        or not keepalive
        or any(not isinstance(item, str) or not item for item in keepalive)
    ):
        raise ConfigError("sandbox.keepalive_argv must be a non-empty list of non-empty strings")

    workspace_value = value.get("workspace")
    workspace: WorkspaceMount | None = None
    if workspace_value is not None:
        if not isinstance(workspace_value, dict):
            raise ConfigError("sandbox.workspace must be a mapping")
        _reject_unknown_fields(
            workspace_value, "sandbox.workspace", frozenset({"path", "access", "container_path"})
        )
        host_path_value = workspace_value.get("path")
        if not isinstance(host_path_value, str) or not host_path_value.strip():
            raise ConfigError("sandbox.workspace.path must be a non-empty path")
        host_path = Path(host_path_value).expanduser()
        if not host_path.is_absolute():
            host_path = source.parent / host_path
        host_path = host_path.resolve()
        if not host_path.is_dir():
            raise ConfigError(f"sandbox.workspace.path must be an existing directory: {host_path}")
        access = workspace_value.get("access", "read_only")
        if access not in ("read_only", "read_write"):
            raise ConfigError("sandbox.workspace.access must be 'read_only' or 'read_write'")
        container_path = workspace_value.get("container_path")
        if container_path is not None:
            if not isinstance(container_path, str) or not container_path:
                raise ConfigError("sandbox.workspace.container_path must be a non-empty path")
            windows_absolute = (
                len(container_path) >= 3
                and container_path[1] == ":"
                and container_path[2] in ("\\", "/")
            )
            if not container_path.startswith("/") and not windows_absolute:
                raise ConfigError("sandbox.workspace.container_path must be absolute")
            if ".." in container_path.replace("\\", "/").split("/"):
                raise ConfigError("sandbox.workspace.container_path cannot contain '..'")
        workspace = WorkspaceMount(host_path, access, container_path)

    resources = value.get("resources", {})
    if not isinstance(resources, dict):
        raise ConfigError("sandbox.resources must be a mapping")
    _reject_unknown_fields(
        resources, "sandbox.resources", frozenset({"cpus", "memory_bytes", "process_limit"})
    )
    cpus = resources.get("cpus", 2)
    if (
        isinstance(cpus, bool)
        or not isinstance(cpus, (int, float))
        or not math.isfinite(cpus)
        or cpus <= 0
    ):
        raise ConfigError("sandbox.resources.cpus must be a finite positive number")
    memory_bytes = resources.get("memory_bytes", 2 * 1024**3)
    process_limit = resources.get("process_limit", 256)
    require_image_digest = value.get("require_image_digest", False)
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes < 1:
        raise ConfigError("sandbox.resources.memory_bytes must be a positive integer")
    if process_limit is not None and (
        isinstance(process_limit, bool) or not isinstance(process_limit, int) or process_limit < 1
    ):
        raise ConfigError("sandbox.resources.process_limit must be a positive integer or null")
    if not isinstance(require_image_digest, bool):
        raise ConfigError("sandbox.require_image_digest must be a boolean")

    api_value = value.get("api", {})
    if not isinstance(api_value, dict):
        raise ConfigError("sandbox.api must be a mapping")
    _reject_unknown_fields(
        api_value, "sandbox.api", frozenset({"base_url", "unix_socket", "named_pipe"})
    )
    api_base_url = api_value.get("base_url")
    api_unix_socket_value = api_value.get("unix_socket")
    api_named_pipe = api_value.get("named_pipe")
    endpoints = (api_base_url, api_unix_socket_value, api_named_pipe)
    if adapter == "api" and not any(endpoints):
        raise ConfigError(
            "sandbox.api requires base_url, unix_socket, or named_pipe when adapter is 'api'"
        )
    if sum(endpoint is not None for endpoint in endpoints) > 1:
        raise ConfigError("sandbox.api must configure exactly one API endpoint")
    api_unix_socket = None
    if api_unix_socket_value is not None:
        if not isinstance(api_unix_socket_value, str) or not api_unix_socket_value:
            raise ConfigError("sandbox.api.unix_socket must be a non-empty path")
        api_unix_socket = Path(api_unix_socket_value).expanduser().resolve()
    if api_named_pipe is not None and not isinstance(api_named_pipe, str):
        raise ConfigError("sandbox.api.named_pipe must be a local Windows named-pipe path")

    return SandboxDefinition(
        engine=engine,
        image=image.strip(),
        keepalive_argv=tuple(keepalive),
        adapter=adapter,
        workspace=workspace,
        cpus=float(cpus),
        memory_bytes=memory_bytes,
        process_limit=process_limit,
        require_image_digest=require_image_digest,
        api_base_url=api_base_url,
        api_unix_socket=api_unix_socket,
        api_named_pipe=api_named_pipe,
        user=value.get("user", DEFAULT_SANDBOX_USER),
    )


def load_agent(path: str | Path) -> AgentDefinition:
    source = Path(path).expanduser().resolve()
    data = _read_yaml(source)
    name = data.get("name")
    model = data.get("model")
    if not isinstance(name, str):
        raise ConfigError("Agent definition requires a non-empty 'name'")
    if not isinstance(model, dict):
        raise ConfigError("Agent definition requires model.provider and model.model")
    _reject_unknown_fields(
        data,
        "agent",
        frozenset(
            {
                "name",
                "description",
                "model",
                "environment",
                "skills",
                "skill_paths",
                "tools",
                "knowledge",
                "instructions",
                "policies",
                "verification",
                "output_schema",
                "sandbox",
            }
        ),
    )
    definition = AgentDefinition(
        name=name,
        description=data.get("description", ""),
        model=model,
        environment=data.get("environment", {}),
        skills=_strings(data.get("skills", []), "skills"),
        skill_paths=_strings(data.get("skill_paths", []), "skill_paths"),
        tools=_strings(data.get("tools", []), "tools"),
        knowledge=data.get("knowledge", {}),
        instructions=data.get("instructions", ""),
        policies=data.get("policies", {}),
        verification=data.get("verification", {}),
        output_schema=data.get("output_schema"),
        sandbox=_sandbox_config(data.get("sandbox"), source),
        source_path=source,
    )
    validate_agent_definition(definition)
    definition.name = definition.name.strip()
    return definition


def load_skill(path: str | Path) -> SkillDefinition:
    source = Path(path).expanduser().resolve()
    data = _read_yaml(source, max_bytes=_MAX_SKILL_MANIFEST_BYTES)
    _reject_unknown_fields(
        data,
        "skill",
        frozenset(
            {
                "name",
                "version",
                "description",
                "instructions",
                "instructions_file",
                "examples",
                "examples_file",
                "tools",
                "knowledge",
                "constraints",
                "dependencies",
                "verification",
                "triggers",
                "input_schema",
                "output_schema",
            }
        ),
    )
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"Skill definition {source} requires a non-empty 'name'")
    description = _string(data.get("description", ""), "skill.description")
    instructions = _load_skill_instructions(data, source)
    examples = _load_skill_examples(data, source)
    definition = SkillDefinition(
        name=name.strip(),
        version=_string(data.get("version", "0.1.0"), "skill.version"),
        description=description,
        instructions=instructions,
        examples=examples,
        tools=_strings(data.get("tools", []), "skill.tools"),
        knowledge=_strings(data.get("knowledge", []), "skill.knowledge"),
        constraints=_strings(data.get("constraints", []), "skill.constraints"),
        dependencies=_strings(data.get("dependencies", []), "skill.dependencies"),
        verification=_strings(data.get("verification", []), "skill.verification"),
        triggers=_strings(data.get("triggers", []), "skill.triggers"),
        input_schema=data.get("input_schema"),
        output_schema=data.get("output_schema"),
        source_path=source,
    )
    validate_skill_definition(definition)
    return definition


def _load_skill_instructions(data: dict[str, Any], source: Path) -> str:
    return _load_skill_text_resource(
        data,
        source,
        inline_field="instructions",
        file_field="instructions_file",
        conventional_file="instructions.md",
    )


def _load_skill_examples(data: dict[str, Any], source: Path) -> str:
    return _load_skill_text_resource(
        data,
        source,
        inline_field="examples",
        file_field="examples_file",
        conventional_file="examples.md",
    )


def _load_skill_text_resource(
    data: dict[str, Any],
    source: Path,
    *,
    inline_field: str,
    file_field: str,
    conventional_file: str,
) -> str:
    inline = _string(data.get(inline_field, ""), f"skill.{inline_field}")
    resource_file = data.get(file_field)
    if resource_file is None and not inline.strip():
        conventional_path = source.parent / conventional_file
        if conventional_path.is_file():
            resource_file = conventional_file
    if resource_file is None:
        _validate_text_bytes(f"skill.{inline_field}", inline, _MAX_SKILL_TEXT_BYTES)
        return inline
    if not isinstance(resource_file, str) or not resource_file.strip():
        raise ConfigError(f"skill.{file_field} must be a non-empty relative path")
    if inline.strip():
        raise ConfigError(f"Specify either skill.{inline_field} or skill.{file_field}, not both")
    windows_path = PureWindowsPath(resource_file)
    relative_path = Path(resource_file)
    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "\\" in resource_file
        or ".." in relative_path.parts
    ):
        raise ConfigError(f"skill.{file_field} must stay inside the skill package")
    skill_root = source.parent.resolve()
    resolved_file = (skill_root / relative_path).resolve()
    try:
        resolved_file.relative_to(skill_root)
    except ValueError as exc:
        raise ConfigError(f"skill.{file_field} cannot resolve outside the skill package") from exc
    if not resolved_file.is_file():
        raise ConfigError(f"Skill {file_field} file was not found: {resource_file}")
    try:
        with resolved_file.open("rb") as stream:
            contents = stream.read(_MAX_SKILL_TEXT_BYTES + 1)
    except OSError as exc:
        raise ConfigError(f"Skill {file_field} file could not be read: {resource_file}") from exc
    if len(contents) > _MAX_SKILL_TEXT_BYTES:
        raise ConfigError(f"Skill {file_field} exceeds the {_MAX_SKILL_TEXT_BYTES}-byte limit")
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Skill {file_field} file could not be read as UTF-8: {resource_file}"
        ) from exc

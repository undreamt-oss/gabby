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
"""Agent construction from reusable definitions and runtime components."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from ._sync_runner import SyncLoopBridge
from .approval import ApprovalHandler
from .auth import Principal
from .config import (
    _MAX_AGENT_TEXT_BYTES,
    _MAX_RESOLVED_SKILLS,
    DEFAULT_MAX_MODEL_REQUEST_BYTES,
    DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    AgentDefinition,
    ConfigError,
    ResolvedAgentDefinition,
    ResolvedSkillDefinition,
    SkillDefinition,
    _split_skill_reference,
    _validate_text_bytes,
    load_agent,
    load_skill,
    resolve_agent_definition,
    validate_agent_definition,
    validate_skill_definition,
)
from .environment import Environment, ResolvedEnvironment
from .knowledge import Retriever
from .models import (
    AnthropicProvider,
    GeminiProvider,
    HuggingFaceInferenceProvider,
    ModelProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    TransformersProvider,
)
from .planning import ModelPlanner, Planner
from .runtime import AgentStreamEvent, ExecutionResult, RunRequest, Runtime
from .sandbox import EngineAdapter
from .sandbox_tools import sandbox_tools
from .skill_packages import SkillPackageError
from .skill_selection import ConfiguredSkillSelector, ModelSkillSelector, SkillSelector
from .skill_trust import (
    SkillIntegrityError,
    SkillRevocationChecker,
    SkillRevocationUnavailable,
    SkillRevokedError,
    SkillTrustPolicy,
)
from .tools import DefaultPolicyEngineFactory, PolicyEngineFactory, ToolRegistry
from .tracing import Tracer
from .verification import Verifier


def _skill_files_under(base: Path, name: str) -> list[tuple[str | None, Path]]:
    """List legacy and versioned package manifests confined to one registry root."""
    root = base.resolve()
    package_root = (root / name).resolve()
    try:
        package_root.relative_to(root)
    except ValueError as exc:
        raise ConfigError(
            f"Skill {name!r} resolves outside its configured registry directory {root}"
        ) from exc
    if not package_root.is_dir():
        return []

    candidates: list[tuple[str | None, Path]] = []
    legacy = package_root / "skill.yaml"
    if legacy.is_file():
        resolved_legacy = legacy.resolve()
        try:
            resolved_legacy.relative_to(root)
        except ValueError as exc:
            raise ConfigError(
                f"Skill {name!r} resolves outside its configured registry directory {root}"
            ) from exc
        candidates.append((None, resolved_legacy))

    try:
        entries = sorted(package_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ConfigError(
            f"Skill registry directory could not be inspected: {package_root}"
        ) from exc
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            entry_name, version = _split_skill_reference(f"{name}@{entry.name}")
        except ConfigError:
            continue
        if entry_name != name or version is None:
            continue
        manifest = (entry / "skill.yaml").resolve()
        try:
            manifest.relative_to(root)
        except ValueError as exc:
            raise ConfigError(
                f"Skill {name!r} resolves outside its configured registry directory {root}"
            ) from exc
        if manifest.is_file():
            candidates.append((version, manifest))
    return candidates


class Agent:
    """Resolved stateless agent with owned model resources and injected capabilities."""

    def __init__(
        self,
        definition: AgentDefinition,
        *,
        model: ModelProvider | None = None,
        tools: ToolRegistry | None = None,
        retriever: Retriever | None = None,
        verifier: Verifier | None = None,
        environment: Environment | None = None,
        policy_engine_factory: PolicyEngineFactory | None = None,
        sandbox_adapter: EngineAdapter | None = None,
        skill_registry: Mapping[str, SkillDefinition] | None = None,
        skill_trust_policy: SkillTrustPolicy | None = None,
        skill_revocation_checker: SkillRevocationChecker | None = None,
        skill_revocation_timeout_seconds: float = 5.0,
        skill_revocation_poll_interval_seconds: float = 1.0,
        skill_selector: SkillSelector | None = None,
        planner: Planner | None = None,
        tracer: Tracer | None = None,
        approval_handler: ApprovalHandler | None = None,
        global_instructions: str = "",
    ) -> None:
        validate_agent_definition(definition)
        selected_policy_engine_factory = (
            DefaultPolicyEngineFactory() if policy_engine_factory is None else policy_engine_factory
        )
        if not callable(getattr(selected_policy_engine_factory, "create", None)):
            raise ConfigError("policy_engine_factory must provide create()")
        self._uses_default_policy_engine_factory = policy_engine_factory is None
        self.policy_engine_factory = selected_policy_engine_factory
        if not isinstance(global_instructions, str):
            raise ConfigError("global_instructions must be a string")
        _validate_text_bytes("global_instructions", global_instructions, _MAX_AGENT_TEXT_BYTES)
        try:
            valid_revocation_timeout = (
                not isinstance(skill_revocation_timeout_seconds, bool)
                and isinstance(skill_revocation_timeout_seconds, (int, float))
                and math.isfinite(skill_revocation_timeout_seconds)
                and 0 < skill_revocation_timeout_seconds <= 60
            )
        except OverflowError:
            valid_revocation_timeout = False
        if not valid_revocation_timeout:
            raise ConfigError(
                "skill_revocation_timeout_seconds must be greater than 0 and at most 60"
            )
        try:
            valid_revocation_poll_interval = (
                not isinstance(skill_revocation_poll_interval_seconds, bool)
                and isinstance(skill_revocation_poll_interval_seconds, (int, float))
                and math.isfinite(skill_revocation_poll_interval_seconds)
                and 0.1 <= skill_revocation_poll_interval_seconds <= 60
            )
        except OverflowError:
            valid_revocation_poll_interval = False
        if not valid_revocation_poll_interval:
            raise ConfigError("skill_revocation_poll_interval_seconds must be from 0.1 through 60")
        if skill_trust_policy is not None and not isinstance(skill_trust_policy, SkillTrustPolicy):
            raise ConfigError("skill_trust_policy must be a SkillTrustPolicy")
        if skill_revocation_checker is not None:
            if skill_trust_policy is None:
                raise ConfigError("skill_revocation_checker requires skill_trust_policy")
            if not callable(getattr(skill_revocation_checker, "check_not_revoked", None)):
                raise ConfigError("skill_revocation_checker must provide async check_not_revoked()")
        self._global_instructions = global_instructions
        loaded_skills = self._load_skills(
            definition, skill_registry or {}, global_instructions=global_instructions
        )
        trusted_skill_signers: set[str] = set()
        trusted_skill_provenance: list[tuple[SkillDefinition, str]] = []
        if skill_trust_policy is not None:
            for source_skill in loaded_skills:
                try:
                    signer = skill_trust_policy.verify(source_skill)
                    trusted_skill_signers.add(signer)
                    trusted_skill_provenance.append((source_skill, signer))
                except SkillPackageError as exc:
                    raise ConfigError(
                        f"Skill {source_skill.name!r}@{source_skill.version} "
                        f"failed the host trust policy: {exc}"
                    ) from exc
        self.skill_trust_policy = skill_trust_policy
        self.skill_revocation_checker = skill_revocation_checker
        self.skill_revocation_timeout_seconds = float(skill_revocation_timeout_seconds)
        self.skill_revocation_poll_interval_seconds = float(skill_revocation_poll_interval_seconds)
        self._trusted_skill_signers = frozenset(trusted_skill_signers)
        self._trusted_skill_provenance = tuple(trusted_skill_provenance)
        self.definition: ResolvedAgentDefinition = resolve_agent_definition(
            definition, loaded_skills
        )
        self._output_schema_json: str | None = None
        self._output_validator: Draft202012Validator | None = None
        self._skill_input_validators: dict[str, Draft202012Validator] = {}
        self._skill_output_validators: dict[str, Draft202012Validator] = {}
        self._skill_input_schema_json: dict[str, str] = {}
        self._skill_output_schema_json: dict[str, str] = {}

        def thaw_json(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: thaw_json(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return [thaw_json(child) for child in value]
            return value

        def schema_json(schema: Mapping[str, Any]) -> str:
            return json.dumps(
                thaw_json(schema),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        if self.definition.output_schema is not None:
            self._output_schema_json = json.dumps(
                thaw_json(self.definition.output_schema),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._output_validator = Draft202012Validator(json.loads(self._output_schema_json))
        for skill in self.definition.skill_definitions:
            if skill.input_schema is not None:
                serialized = schema_json(skill.input_schema)
                self._skill_input_schema_json[skill.name] = serialized
                self._skill_input_validators[skill.name] = Draft202012Validator(
                    json.loads(serialized)
                )
            if skill.output_schema is not None:
                serialized = schema_json(skill.output_schema)
                self._skill_output_schema_json[skill.name] = serialized
                self._skill_output_validators[skill.name] = Draft202012Validator(
                    json.loads(serialized)
                )
        source_environment = environment or Environment.from_config(dict(definition.environment))
        agent_tools = ToolRegistry()
        for name in source_environment.tools.names():
            agent_tools.register(source_environment.tools.get(name).snapshot())
        if tools is not None and tools is not source_environment.tools:
            for name in tools.names():
                agent_tools.register(tools.get(name).snapshot())
        if self.definition.sandbox is not None:
            for tool in sandbox_tools():
                if tool.name in agent_tools.names():
                    raise ConfigError(
                        f"Built-in sandbox tool name is already registered: {tool.name}"
                    )
                agent_tools.register(tool)
        declared_tools = set(self.definition.tools)
        for skill in self.definition.skill_definitions:
            declared_tools.update(skill.tools)
        missing_tools = sorted(set(declared_tools) - set(agent_tools.names()))
        if missing_tools:
            raise ConfigError("Agent references unregistered tool(s): " + ", ".join(missing_tools))
        if self.definition.sandbox is None:
            sandbox_required = sorted(
                name for name in declared_tools if agent_tools.get(name).sandbox_action is not None
            )
            if sandbox_required:
                raise ConfigError(
                    "Agent tools require a sandbox configuration: " + ", ".join(sandbox_required)
                )
        for name in agent_tools.names():
            tool = agent_tools.get(name)
            missing_resources = tuple(
                resource_name
                for resource_name in tool.context_resources
                if resource_name not in source_environment.resources
            )
            if missing_resources:
                raise ConfigError(
                    f"Tool {name!r} requires unavailable environment resources: "
                    + ", ".join(repr(resource_name) for resource_name in missing_resources)
                )
        agent_tools.freeze()
        self.tools = agent_tools
        self.environment = ResolvedEnvironment.from_environment(source_environment, agent_tools)
        if self.definition.knowledge.get("sources") and retriever is None:
            raise ConfigError("Knowledge sources are configured but no Retriever was supplied")
        if self.definition.verification.get("enabled", False) and verifier is None:
            raise ConfigError("Verification is enabled but no verifier was supplied")
        self._owns_model = model is None
        self.model = model or self._provider_from_definition()
        self.retriever = retriever
        self.verifier = verifier
        self.tracer = tracer
        self.approval_handler = approval_handler
        self.sandbox_adapter = sandbox_adapter
        max_model_request_bytes = self.definition.policies.get(
            "max_model_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES
        )
        if (
            isinstance(max_model_request_bytes, bool)
            or not isinstance(max_model_request_bytes, int)
            or max_model_request_bytes < 1
        ):
            raise ConfigError("policies.max_model_request_bytes must be a positive integer")
        self.max_model_request_bytes = max_model_request_bytes
        max_model_response_bytes = self.definition.policies.get(
            "max_model_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES
        )
        if (
            isinstance(max_model_response_bytes, bool)
            or not isinstance(max_model_response_bytes, int)
            or max_model_response_bytes < 1
        ):
            raise ConfigError("policies.max_model_response_bytes must be a positive integer")
        self.max_model_response_bytes = max_model_response_bytes
        selected_skill_selector = (
            skill_selector if skill_selector is not None else ConfiguredSkillSelector()
        )
        self.skill_selector = (
            replace(
                selected_skill_selector,
                max_model_request_bytes=max_model_request_bytes,
                max_model_response_bytes=max_model_response_bytes,
                global_instructions=self.global_instructions,
                agent_instructions=self.definition.instructions,
            )
            if isinstance(selected_skill_selector, ModelSkillSelector)
            else selected_skill_selector
        )
        self.planner = (
            replace(
                planner,
                max_model_request_bytes=max_model_request_bytes,
                max_model_response_bytes=max_model_response_bytes,
                global_instructions=self.global_instructions,
                agent_instructions=self.definition.instructions,
            )
            if isinstance(planner, ModelPlanner)
            else planner
        )
        if self.definition.policies.get("max_replans", 0) and self.planner is None:
            raise ConfigError("policies.max_replans requires an injected planner")
        self._sync_runner: SyncLoopBridge | None = None
        self._execution_mode: str | None = None
        self._closed = False
        self._closing = False
        self._active_runs = 0
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._runs_drained: asyncio.Event | None = None
        self._aclose_lock: asyncio.Lock | None = None

    def _bind_async_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._async_loop is not None and self._async_loop is not loop:
            raise RuntimeError("An async Agent must be used and closed on the same event loop")
        if self._async_loop is None:
            self._async_loop = loop
            self._runs_drained = asyncio.Event()
            self._runs_drained.set()
            self._aclose_lock = asyncio.Lock()
        return loop

    def _begin_async_run(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("Agent is closing or closed")
        if self._execution_mode == "sync":
            raise RuntimeError("This Agent uses the synchronous API; use run() consistently")
        self._bind_async_loop()
        self._execution_mode = "async"
        self._active_runs += 1
        assert self._runs_drained is not None
        self._runs_drained.clear()

    def _finish_async_run(self) -> None:
        self._active_runs -= 1
        if self._active_runs == 0:
            assert self._runs_drained is not None
            self._runs_drained.set()

    async def _check_skill_revocations(self, *, deadline: float) -> float | None:
        """Check current signer revocations within the execution's remaining deadline."""
        checker = self.skill_revocation_checker
        if checker is None or not self._trusted_skill_signers:
            return None
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise SkillRevocationUnavailable("Skill revocation check exceeded the run deadline")
        timeout_seconds = min(remaining, self.skill_revocation_timeout_seconds)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await cast(Any, checker.check_not_revoked)(self._trusted_skill_signers)
        except SkillRevokedError:
            raise
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise SkillRevocationUnavailable("Skill revocation check timed out") from None
        except Exception:
            raise SkillRevocationUnavailable("Skill revocation check failed") from None
        if result is not None:
            raise SkillRevocationUnavailable("Skill revocation checker returned an invalid result")
        return (time.perf_counter() - started) * 1000

    async def _check_skill_integrity(self, *, deadline: float) -> float | None:
        """Revalidate every trusted installed skill before using this agent's snapshot."""
        policy = self.skill_trust_policy
        if policy is None or not self._trusted_skill_provenance:
            return None
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise SkillIntegrityError("Trusted skill integrity check exceeded the run deadline")
        started = time.perf_counter()

        try:
            # Package verification is bounded by the skill package file, byte, and entry limits.
            # Keep it on the run's event loop: cancelling an executor future cannot stop its
            # filesystem scan, which could otherwise continue consuming resources after timeout.
            for skill, expected_signer in self._trusted_skill_provenance:
                signer = policy.verify(skill)
                if signer != expected_signer:
                    raise SkillIntegrityError(
                        "Trusted skill signer changed after agent construction"
                    )
                if time.perf_counter() >= deadline:
                    raise SkillIntegrityError(
                        "Trusted skill integrity check exceeded the run deadline"
                    )
        except SkillIntegrityError:
            raise
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise SkillIntegrityError(
                "Trusted skill integrity check exceeded the run deadline"
            ) from None
        except Exception:
            raise SkillIntegrityError("Trusted skill integrity check failed") from None
        return (time.perf_counter() - started) * 1000

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> Agent:
        """Load a YAML definition and construct an agent with optional injected services."""
        return cls(load_agent(path), **kwargs)

    def _provider_from_definition(self) -> ModelProvider:
        config = self.definition.model
        provider = config.get("provider", "openai_compatible")
        if provider in ("openai", "openai_compatible"):
            return OpenAICompatibleProvider.from_config(config)
        if provider == "ollama":
            return OllamaProvider.from_config(config)
        if provider == "huggingface":
            return HuggingFaceInferenceProvider.from_config(config)
        if provider == "anthropic":
            return AnthropicProvider.from_config(config)
        if provider == "gemini":
            return GeminiProvider.from_config(config)
        if provider == "transformers":
            return TransformersProvider.from_config(config)
        raise ConfigError(
            f"No built-in model provider named {provider!r}; pass a ModelProvider instance"
        )

    @property
    def skills(self) -> tuple[ResolvedSkillDefinition, ...]:
        """Skills resolved and frozen when this agent was constructed."""
        return self.definition.skill_definitions

    @property
    def global_instructions(self) -> str:
        """Shared instructions captured when this agent was constructed."""
        return self._global_instructions

    def _load_skills(
        self,
        definition: AgentDefinition,
        registry: Mapping[str, SkillDefinition],
        *,
        global_instructions: str,
    ) -> list[SkillDefinition]:
        if definition.source_path is None and definition.skills and not registry:
            raise ConfigError(
                "Agent skills require a file-backed definition; in-memory skill registries "
                "must be passed through skill_registry"
            )
        search_roots: list[Path] = []
        if definition.source_path is not None:
            root = definition.source_path.parent
            search_roots = [root / "skills"] + [root / value for value in definition.skill_paths]
        loaded: dict[str, SkillDefinition] = {}
        visiting: set[str] = set()
        request_budget = definition.policies.get(
            "max_model_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES
        )
        text_budget = min(request_budget, _MAX_AGENT_TEXT_BYTES)
        resolved_text_bytes = sum(
            len(value.encode("utf-8"))
            for value in (definition.description, definition.instructions, global_instructions)
        )
        if resolved_text_bytes > text_budget:
            raise ConfigError(
                "Agent descriptions and instructions exceed the configured text budget of "
                f"{text_budget} bytes"
            )

        def resolve(reference: str) -> None:
            nonlocal resolved_text_bytes
            name, requested_version = _split_skill_reference(reference)
            if name in loaded:
                resolved = loaded[name]
                if requested_version is not None and resolved.version != requested_version:
                    raise ConfigError(
                        f"Skill {name!r} is already resolved at version {resolved.version}, "
                        f"not requested version {requested_version}"
                    )
                return
            if name in visiting:
                raise ConfigError(f"Circular skill dependency involving {name!r}")
            if len(loaded) + len(visiting) >= _MAX_RESOLVED_SKILLS:
                raise ConfigError(f"An agent may resolve at most {_MAX_RESOLVED_SKILLS} skills")
            visiting.add(name)

            registry_key = (
                reference if reference in registry else name if name in registry else None
            )
            if registry_key is None and requested_version is None:
                matching_entries = [
                    key
                    for key, candidate in registry.items()
                    if isinstance(candidate, SkillDefinition) and candidate.name == name
                ]
                if len(matching_entries) == 1:
                    registry_key = matching_entries[0]
                elif len(matching_entries) > 1:
                    raise ConfigError(
                        f"Skill {name!r} has multiple registry versions; pin it as "
                        "skill-id@MAJOR.MINOR.PATCH"
                    )

            if registry_key is not None:
                skill = registry[registry_key]
                if not isinstance(skill, SkillDefinition):
                    raise ConfigError(
                        f"Skill registry entry {registry_key!r} is not a SkillDefinition"
                    )
                key_name, key_version = _split_skill_reference(registry_key)
                if key_name != skill.name or (
                    key_version is not None and key_version != skill.version
                ):
                    raise ConfigError(
                        f"Skill registry key {registry_key!r} does not match package "
                        f"{skill.name}@{skill.version}"
                    )
                file_path = None
            else:
                skill = None
                file_path = None
                mismatched_versions: list[str] = []
                for base in search_roots:
                    candidates = _skill_files_under(base, name)
                    legacy = next(
                        ((version, path) for version, path in candidates if version is None),
                        None,
                    )
                    versioned = [(version, path) for version, path in candidates if version]

                    if requested_version is not None:
                        exact = next(
                            (path for version, path in versioned if version == requested_version),
                            None,
                        )
                        if exact is not None:
                            skill = load_skill(exact)
                            if skill.version != requested_version:
                                raise ConfigError(
                                    f"Versioned skill path {exact.parent} declares "
                                    f"{skill.version}, not directory version {requested_version}"
                                )
                            file_path = exact
                            break
                        if legacy is not None:
                            legacy_skill = load_skill(legacy[1])
                            if legacy_skill.version == requested_version:
                                skill = legacy_skill
                                file_path = legacy[1]
                                break
                            mismatched_versions.append(legacy_skill.version)
                        mismatched_versions.extend(
                            version for version, _ in versioned if version is not None
                        )
                        continue

                    if legacy is not None:
                        skill = load_skill(legacy[1])
                        file_path = legacy[1]
                        break
                    if len(versioned) > 1:
                        versions = ", ".join(version for version, _ in versioned if version)
                        raise ConfigError(
                            f"Skill {name!r} has multiple directory versions ({versions}); "
                            "pin it as skill-id@MAJOR.MINOR.PATCH"
                        )
                    if versioned:
                        file_path = versioned[0][1]
                        skill = load_skill(file_path)
                        if skill.version != versioned[0][0]:
                            raise ConfigError(
                                f"Versioned skill path {file_path.parent} declares "
                                f"{skill.version}, not directory version {versioned[0][0]}"
                            )
                        break

                if skill is None or file_path is None:
                    if requested_version is not None and mismatched_versions:
                        available_versions = sorted(set(mismatched_versions))
                        if len(available_versions) == 1:
                            raise ConfigError(
                                f"Skill {name!r} is version {available_versions[0]}, "
                                f"not requested version {requested_version}"
                            )
                        available = ", ".join(available_versions)
                        raise ConfigError(
                            f"Skill {name!r} is available at version(s) {available}, "
                            f"not requested version {requested_version}"
                        )
                    searched_paths = [base / name / "skill.yaml" for base in search_roots]
                    looked_in = ", ".join(map(str, searched_paths)) or (
                        "the in-memory skill registry"
                    )
                    if requested_version is not None:
                        looked_in = (
                            ", ".join(
                                map(
                                    str,
                                    (
                                        base / name / requested_version / "skill.yaml"
                                        for base in search_roots
                                    ),
                                )
                            )
                            or "the in-memory skill registry"
                        )
                    raise ConfigError(f"Skill {reference!r} was not found (looked in {looked_in})")

            if skill is None:
                raise ConfigError(f"Skill {reference!r} could not be resolved")
            validate_skill_definition(skill)
            if skill.name != name:
                raise ConfigError(
                    f"Requested skill {reference!r}, but {skill.source_path} declares "
                    f"{skill.name!r}"
                )
            if requested_version is not None and skill.version != requested_version:
                raise ConfigError(
                    f"Skill {name!r} is version {skill.version}, not requested version "
                    f"{requested_version}"
                )
            skill_text_bytes = sum(
                len(value.encode("utf-8"))
                for value in (skill.description, skill.instructions, skill.examples)
            )
            for schema in (skill.input_schema, skill.output_schema):
                if schema is not None:
                    skill_text_bytes += len(
                        json.dumps(
                            schema,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
            if resolved_text_bytes + skill_text_bytes > text_budget:
                raise ConfigError(
                    "Resolved agent and skill text exceeds the configured text budget of "
                    f"{text_budget} bytes"
                )
            resolved_text_bytes += skill_text_bytes
            for dependency in skill.dependencies:
                resolve(dependency)
            visiting.remove(name)
            loaded[name] = skill

        for skill_name in definition.skills:
            resolve(skill_name)
        # Dependencies precede the skill that depends on them.
        ordered: list[SkillDefinition] = []
        seen: set[str] = set()

        def append_tree(reference: str) -> None:
            name, _ = _split_skill_reference(reference)
            if name in seen:
                return
            skill = loaded[name]
            for dependency in skill.dependencies:
                append_tree(dependency)
            seen.add(name)
            ordered.append(skill)

        for skill_name in definition.skills:
            append_tree(skill_name)
        return ordered

    async def arun(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> ExecutionResult:
        """Execute one asynchronous run without retaining its request state."""
        self._begin_async_run()
        try:
            return await self._execute(
                input, context=context, memory=memory, metadata=metadata, principal=principal
            )
        finally:
            self._finish_async_run()

    async def _execute(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        principal: Principal | None = None,
        event_sink: Callable[[AgentStreamEvent], Awaitable[None]] | None = None,
    ) -> ExecutionResult:
        if principal is not None and not isinstance(principal, Principal):
            raise TypeError("principal must be a gabby.Principal or None")
        request = RunRequest(
            input=input,
            context={} if context is None else context,
            memory={} if memory is None else memory,
            metadata={} if metadata is None else metadata,
            principal=principal,
            max_model_request_bytes=self.max_model_request_bytes,
        )
        return await Runtime(
            self,
            verifier=self.verifier,
            tracer=self.tracer,
            approval_handler=self.approval_handler,
        ).run(request, event_sink=event_sink)

    async def astream(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Stream progress and text events while executing one stateless run."""
        self._begin_async_run()
        queue: asyncio.Queue[AgentStreamEvent | None] = asyncio.Queue(maxsize=64)

        async def emit(event: AgentStreamEvent) -> None:
            await queue.put(event)

        async def execute() -> ExecutionResult:
            try:
                return await self._execute(
                    input,
                    context=context,
                    memory=memory,
                    metadata=metadata,
                    principal=principal,
                    event_sink=emit,
                )
            finally:
                current = asyncio.current_task()
                if current is None or current.cancelling() == 0:
                    if queue.full():
                        await queue.put(None)
                    else:
                        queue.put_nowait(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                # The execution task can finish with an exception after its own
                # cancellation was translated (for example, a skill revocation).
                # In that case `execute` intentionally cannot enqueue the normal
                # sentinel, so wait for both queue data and task completion.
                if queue.empty() and task.done():
                    break
                pending_event = asyncio.create_task(queue.get())
                try:
                    done, _ = await asyncio.wait(
                        {pending_event, task}, return_when=asyncio.FIRST_COMPLETED
                    )
                except BaseException:
                    pending_event.cancel()
                    await asyncio.gather(pending_event, return_exceptions=True)
                    raise
                if pending_event not in done:
                    pending_event.cancel()
                    await asyncio.gather(pending_event, return_exceptions=True)
                    break
                event = pending_event.result()
                if event is None:
                    break
                yield event
            await task
        except BaseException:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._finish_async_run()

    def run(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> ExecutionResult:
        """Synchronous convenience wrapper. Use ``await agent.arun(...)`` in async code."""
        if self._closed:
            raise RuntimeError("Agent is closed")
        if self._execution_mode == "async":
            raise RuntimeError(
                "This Agent uses the asynchronous API; use await arun() consistently"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._sync_runner is None:
                self._sync_runner = SyncLoopBridge()
            self._execution_mode = "sync"
            return self._sync_runner.run(
                self._execute(
                    input,
                    context=context,
                    memory=memory,
                    metadata=metadata,
                    principal=principal,
                )
            )
        raise RuntimeError(
            "Agent.run() cannot run inside an active event loop; await Agent.arun() instead"
        )

    async def aclose(self) -> None:
        """Drain active asynchronous runs, then close owned resources."""
        if self._closed:
            return
        runner = self._sync_runner
        if runner is not None and runner.is_running:
            raise RuntimeError(
                "This Agent uses the synchronous API; call close() outside an event loop"
            )
        self._bind_async_loop()
        self._execution_mode = "async"
        self._closing = True
        assert self._aclose_lock is not None
        assert self._runs_drained is not None
        async with self._aclose_lock:
            if self._closed:
                return
            await self._runs_drained.wait()
            await self._aclose_owned_model()
            self._closed = True

    async def _aclose_owned_model(self) -> None:
        if self._owns_model:
            close = getattr(self.model, "aclose", None)
            if close is not None:
                await close()

    def close(self) -> None:
        """Close the agent and its persistent synchronous event-loop bridge."""
        if self._closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Agent.close() cannot run inside an active event loop; await aclose()"
            )

        if self._execution_mode == "async":
            raise RuntimeError("This Agent uses the asynchronous API; await aclose()")

        runner = self._sync_runner
        if runner is None:
            asyncio.run(self._aclose_owned_model())
        else:
            runner.run(self._aclose_owned_model())
            runner.stop()
        self._closed = True

    def __enter__(self) -> Agent:
        """Return this agent for use as a synchronous context manager."""
        if self._closed:
            raise RuntimeError("Agent is closed")
        return self

    def __exit__(self, *_: object) -> None:
        """Close owned resources on synchronous context exit."""
        self.close()

    async def __aenter__(self) -> Agent:
        """Return this agent for use as an asynchronous context manager."""
        if self._closed:
            raise RuntimeError("Agent is closed")
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close owned resources on asynchronous context exit."""
        await self.aclose()

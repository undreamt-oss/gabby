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
"""Agent composition, sync wrapper, and resource ownership behavior."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import gabby.agent as agent_module
import gabby.config as config_module
from gabby.agent import Agent
from gabby.config import AgentDefinition, ConfigError, SandboxDefinition, SkillDefinition
from gabby.environment import Environment
from gabby.models import (
    AnthropicProvider,
    HuggingFaceInferenceProvider,
    ModelResponse,
    ModelStreamDelta,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from gabby.skill_packages import install_skill, pack_skill
from gabby.skill_signing import sign_skill_package
from gabby.skill_trust import (
    SkillIntegrityError,
    SkillRevocationUnavailable,
    SkillRevokedError,
    SkillTrustPolicy,
)
from gabby.tools import Tool, ToolRegistry


class ReusableModel:
    name = "reusable"

    def __init__(self) -> None:
        self.responses = ["first", "second"]
        self.loop_ids: list[int] = []

    async def complete(self, **_: Any) -> ModelResponse:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return ModelResponse(content=self.responses.pop(0))


def _definition() -> AgentDefinition:
    return AgentDefinition(
        name="agent-test",
        model={"provider": "openai_compatible", "model": "test"},
        policies={"max_steps": 1, "timeout_seconds": 1},
    )


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        handler=lambda: None,
        output_schema={"type": "null"},
    )


def test_sync_wrapper_supports_repeated_stateless_runs_on_one_loop() -> None:
    model = ReusableModel()
    agent = Agent(_definition(), model=model)

    with agent:
        first = agent.run("first request")
        second = agent.run("second request")

    assert first.output == "first"
    assert second.output == "second"
    assert first.trace.trace_id != second.trace.trace_id
    assert model.loop_ids[0] == model.loop_ids[1]

    with pytest.raises(RuntimeError, match="Agent is closed"):
        agent.run("after close")


def test_global_instructions_must_be_a_string() -> None:
    with pytest.raises(ConfigError, match="global_instructions must be a string"):
        Agent(_definition(), model=ReusableModel(), global_instructions=["bad type"])  # type: ignore[arg-type]


def test_agent_requires_typed_skill_trust_policy() -> None:
    with pytest.raises(ConfigError, match="must be a SkillTrustPolicy"):
        Agent(_definition(), model=ReusableModel(), skill_trust_policy=object())  # type: ignore[arg-type]


def test_agent_validates_policy_factory_and_revocation_checker_contracts() -> None:
    class InvalidPolicyFactory:
        pass

    class InvalidChecker:
        pass

    with pytest.raises(ConfigError, match="policy_engine_factory must provide create"):
        Agent(
            _definition(),
            model=ReusableModel(),
            policy_engine_factory=InvalidPolicyFactory(),  # type: ignore[arg-type]
        )

    with pytest.raises(ConfigError, match="requires skill_trust_policy"):
        Agent(
            _definition(),
            model=ReusableModel(),
            skill_revocation_checker=InvalidChecker(),  # type: ignore[arg-type]
        )

    policy = SkillTrustPolicy({"publisher": b"p" * 32})
    with pytest.raises(ConfigError, match="must provide async check_not_revoked"):
        Agent(
            _definition(),
            model=ReusableModel(),
            skill_trust_policy=policy,
            skill_revocation_checker=InvalidChecker(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("timeout", [0, 10**400])
def test_agent_rejects_invalid_revocation_deadline_options(timeout: object) -> None:
    with pytest.raises(ConfigError, match="skill_revocation_timeout_seconds"):
        Agent(
            _definition(),
            model=ReusableModel(),
            skill_revocation_timeout_seconds=timeout,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("policy", "value", "message"),
    [
        ("max_model_request_bytes", True, "max_model_request_bytes"),
        ("max_model_response_bytes", 0, "max_model_response_bytes"),
        ("max_replans", 1, "max_replans requires an injected planner"),
    ],
)
def test_agent_rejects_invalid_runtime_policy_dependencies(
    policy: str, value: object, message: str
) -> None:
    definition = _definition()
    definition.policies[policy] = value
    with pytest.raises(ConfigError, match=message):
        Agent(definition, model=ReusableModel())


@pytest.mark.asyncio
async def test_skill_revocation_checker_errors_and_timeouts_fail_closed() -> None:
    class FailingChecker:
        async def check_not_revoked(self, _key_ids: frozenset[str]) -> None:
            raise RuntimeError("private database detail")

    class HangingChecker:
        async def check_not_revoked(self, _key_ids: frozenset[str]) -> None:
            await asyncio.Event().wait()

    agent = Agent(_definition(), model=ReusableModel())
    agent._trusted_skill_signers = frozenset({"publisher"})
    agent.skill_revocation_checker = FailingChecker()
    with pytest.raises(SkillRevocationUnavailable, match="check failed") as failure:
        await agent._check_skill_revocations(deadline=time.perf_counter() + 1)
    assert "private database detail" not in str(failure.value)

    agent.skill_revocation_checker = HangingChecker()
    with pytest.raises(SkillRevocationUnavailable, match="timed out"):
        await agent._check_skill_revocations(deadline=time.perf_counter() + 0.01)
    await agent.aclose()


@pytest.mark.parametrize("interval", [0, 0.09, 61, float("nan"), True, 10**400])
def test_agent_rejects_invalid_skill_revocation_poll_intervals(interval: object) -> None:
    with pytest.raises(ConfigError, match="skill_revocation_poll_interval_seconds"):
        Agent(
            _definition(),
            model=ReusableModel(),
            skill_revocation_poll_interval_seconds=interval,  # type: ignore[arg-type]
        )


def test_global_instructions_cannot_change_after_construction() -> None:
    agent = Agent(
        _definition(), model=ReusableModel(), global_instructions="Use concise responses."
    )

    with pytest.raises(AttributeError):
        agent.global_instructions = "Changed after construction"  # type: ignore[misc]

    assert agent.global_instructions == "Use concise responses."


@pytest.mark.asyncio
async def test_agent_trust_policy_verifies_installed_skills_at_construction(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_key = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    source = tmp_path / "skill-source"
    source.mkdir()
    (source / "skill.yaml").write_text(
        "name: trust/review\nversion: 1.2.3\ndescription: Review trusted changes\n"
        "instructions_file: instructions.md\n",
        encoding="utf-8",
    )
    (source / "instructions.md").write_text("Review only verified changes.\n", encoding="utf-8")
    package = tmp_path / "review.gabskill"
    pack_skill(source, package)
    sign_skill_package(package, key_id="release-key", private_key=private_key)
    registry = tmp_path / "skills"
    installed = install_skill(
        package,
        registry,
        require_signature=True,
        trusted_keys={"release-key": public_key},
    )

    definition = _definition()
    definition.policies["timeout_seconds"] = 10
    definition.skills = ["trust/review@1.2.3"]
    definition.source_path = tmp_path / "agent.yaml"
    original_trust = {"release-key": public_key}
    policy = SkillTrustPolicy(original_trust)
    original_trust.clear()

    class Checker:
        def __init__(self) -> None:
            self.revoked = False
            self.checks: list[frozenset[str]] = []

        async def check_not_revoked(self, key_ids: frozenset[str]) -> None:
            self.checks.append(key_ids)
            if self.revoked and "release-key" in key_ids:
                raise SkillRevokedError("publisher revoked")
            return None

    checker = Checker()

    agent = Agent(
        definition,
        model=ReusableModel(),
        skill_trust_policy=policy,
        skill_revocation_checker=checker,
    )
    assert [skill.name for skill in agent.skills] == ["trust/review"]
    result = await agent.arun("Review the change")
    assert any(event.kind == "skill_integrity_check" for event in result.trace.events)
    assert any(event.kind == "skill_revocation_check" for event in result.trace.events)
    checker.revoked = True
    with pytest.raises(SkillRevokedError):
        await agent.arun("Review another change")
    with pytest.raises(SkillRevokedError):
        _ = [event async for event in agent.astream("Review a streamed change")]
    assert checker.checks == [
        frozenset({"release-key"}),
        frozenset({"release-key"}),
        frozenset({"release-key"}),
    ]
    await agent.aclose()

    class BlockingModel:
        name = "blocking-revocation-model"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def complete(self, **_: Any) -> ModelResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ModelResponse(content="unexpected")

    blocking_model = BlockingModel()
    active_checker = Checker()
    active_agent = Agent(
        definition,
        model=blocking_model,
        skill_trust_policy=policy,
        skill_revocation_checker=active_checker,
        skill_revocation_poll_interval_seconds=0.1,
    )
    active_run = asyncio.create_task(active_agent.arun("Review during key revocation"))
    await asyncio.wait_for(blocking_model.started.wait(), timeout=1)
    active_checker.revoked = True
    with pytest.raises(SkillRevokedError):
        await asyncio.wait_for(active_run, timeout=1)
    assert blocking_model.cancelled.is_set()
    assert active_checker.checks == [frozenset({"release-key"})] * 2

    active_checker.revoked = False
    blocking_model.started.clear()
    blocking_model.cancelled.clear()

    async def consume_stream() -> list[Any]:
        return [
            event async for event in active_agent.astream("Review during streamed key revocation")
        ]

    active_stream = asyncio.create_task(consume_stream())
    await asyncio.wait_for(blocking_model.started.wait(), timeout=1)
    active_checker.revoked = True
    with pytest.raises(SkillRevokedError):
        await asyncio.wait_for(active_stream, timeout=1)
    assert blocking_model.cancelled.is_set()
    assert active_checker.checks == [frozenset({"release-key"})] * 4

    active_checker.revoked = False
    blocking_model.started.clear()
    blocking_model.cancelled.clear()
    cancelled_run = asyncio.create_task(active_agent.arun("Cancel independently"))
    await asyncio.wait_for(blocking_model.started.wait(), timeout=1)
    cancelled_run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_run
    assert blocking_model.cancelled.is_set()
    await active_agent.aclose()

    with pytest.raises(ConfigError, match="signer is revoked"):
        Agent(
            definition,
            model=ReusableModel(),
            skill_trust_policy=SkillTrustPolicy(
                {"release-key": public_key}, revoked_key_ids=frozenset({"release-key"})
            ),
        )

    integrity_agent = Agent(
        definition,
        model=ReusableModel(),
        skill_trust_policy=policy,
    )
    (installed.path / "instructions.md").write_text("Changed after audit.\n", encoding="utf-8")
    with pytest.raises(SkillIntegrityError, match="integrity check failed"):
        await integrity_agent.arun("This must fail before model use")
    with pytest.raises(SkillIntegrityError, match="integrity check failed"):
        _ = [event async for event in integrity_agent.astream("This must fail before model use")]
    await integrity_agent.aclose()

    with pytest.raises(ConfigError, match="does not match its signed manifest"):
        Agent(definition, model=ReusableModel(), skill_trust_policy=policy)

    unsigned_source = tmp_path / "skills" / "trust" / "unsigned" / "skill.yaml"
    unsigned_source.parent.mkdir(parents=True)
    unsigned_source.write_text(
        "name: trust/unsigned\nversion: 1.0.0\ndescription: Unsigned skill\n",
        encoding="utf-8",
    )
    definition.skills = ["trust/unsigned@1.0.0"]
    with pytest.raises(ConfigError, match="trust policy"):
        Agent(definition, model=ReusableModel(), skill_trust_policy=policy)


@pytest.mark.asyncio
async def test_agent_closes_only_a_model_it_constructed() -> None:
    agent = Agent(_definition())
    model = agent.model
    assert isinstance(model, OpenAICompatibleProvider)
    assert model._client is None
    await agent.aclose()
    assert model._client is None


@pytest.mark.asyncio
async def test_aclose_drains_active_streams_and_closes_owned_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingModel:
        name = "blocking-stream"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.close_count = 0

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="completed")

        async def stream(self, **_: Any) -> Any:
            self.started.set()
            await self.release.wait()
            yield ModelStreamDelta(content_delta="completed")

        async def aclose(self) -> None:
            self.close_count += 1

    agent = Agent(_definition())
    model = BlockingModel()
    monkeypatch.setattr(agent, "model", model)

    async def consume() -> list[Any]:
        return [event async for event in agent.astream("active request")]

    run = asyncio.create_task(consume())
    await model.started.wait()
    first_close = asyncio.create_task(agent.aclose())
    second_close = asyncio.create_task(agent.aclose())
    await asyncio.sleep(0)

    assert first_close.done() is False
    assert second_close.done() is False
    assert model.close_count == 0
    with pytest.raises(RuntimeError, match="closing or closed"):
        await agent.arun("new work")

    model.release.set()
    events = await run
    await asyncio.gather(first_close, second_close)

    assert any(event.type == "completed" for event in events)
    assert model.close_count == 1
    with pytest.raises(RuntimeError, match="Agent is closing or closed"):
        await agent.arun("after shutdown")


def test_agent_merges_tool_registries_without_mutating_the_callers() -> None:
    environment_tools = ToolRegistry()
    environment_tools.register(_tool("environment_tool"))
    injected_tools = ToolRegistry()
    injected_tools.register(_tool("application_tool"))
    environment = Environment(type="test", tools=environment_tools)

    agent = Agent(
        _definition(),
        model=ReusableModel(),
        tools=injected_tools,
        environment=environment,
    )

    assert agent.tools.names() == ["application_tool", "environment_tool"]
    assert environment.tools.names() == ["environment_tool"]
    assert injected_tools.names() == ["application_tool"]
    assert id(agent.environment) != id(environment)
    assert agent.environment.resources is not environment.resources


@pytest.mark.parametrize("declaration", ["agent", "skill"])
def test_agent_rejects_unregistered_tools_during_construction(declaration: str) -> None:
    definition = _definition()
    definition.model["provider"] = "unavailable-provider"
    skill_registry: dict[str, SkillDefinition] = {}
    if declaration == "agent":
        definition.tools = ["missing_tool"]
    else:
        definition.skills = ["diagnostics"]
        skill_registry["diagnostics"] = SkillDefinition(name="diagnostics", tools=["missing_tool"])

    with pytest.raises(ConfigError, match="unregistered tool.*missing_tool"):
        Agent(definition, skill_registry=skill_registry)


@pytest.mark.parametrize(
    ("dependency", "expected_error"),
    [
        ("knowledge", "no Retriever was supplied"),
        ("verification", "no verifier was supplied"),
    ],
)
def test_agent_rejects_missing_runtime_dependencies_during_construction(
    dependency: str, expected_error: str
) -> None:
    definition = _definition()
    definition.model["provider"] = "unavailable-provider"
    if dependency == "knowledge":
        definition.knowledge = {"sources": ["./docs"]}
    else:
        definition.verification = {"enabled": True}

    with pytest.raises(ConfigError, match=expected_error):
        Agent(definition)


@pytest.mark.parametrize(
    ("policy", "value"),
    [
        ("max_model_request_bytes", True),
        ("max_model_request_bytes", 0),
        ("max_model_response_bytes", False),
        ("max_model_response_bytes", -1),
    ],
)
def test_agent_rejects_invalid_model_io_budgets(policy: str, value: object) -> None:
    definition = _definition()
    definition.policies[policy] = value

    with pytest.raises(ConfigError, match=f"policies\\.{policy} must be a positive integer"):
        Agent(definition, model=ReusableModel())


def test_agent_rejects_sandbox_tool_name_collisions() -> None:
    definition = _definition()
    definition.sandbox = SandboxDefinition(
        engine="docker", image="python:3.14", keepalive_argv=("sleep", "1")
    )
    tools = ToolRegistry()
    tools.register(_tool("shell_run"))

    with pytest.raises(ConfigError, match="Built-in sandbox tool name is already registered"):
        Agent(definition, model=ReusableModel(), tools=tools)


def test_agent_rejects_tools_with_missing_environment_resources() -> None:
    tool = Tool(
        name="resource_reader",
        description="Read an injected resource",
        parameters={"type": "object", "properties": {}},
        handler=lambda: None,
        output_schema={"type": "null"},
        context_parameter="context",
        context_resources=("catalog",),
    )
    tools = ToolRegistry()
    tools.register(tool)
    definition = _definition()
    definition.tools = ["resource_reader"]

    with pytest.raises(ConfigError, match="requires unavailable environment resources: 'catalog'"):
        Agent(definition, model=ReusableModel(), tools=tools)


def test_agent_resolves_exact_file_backed_skill_versions(tmp_path: Path) -> None:
    definition = _definition()
    definition.source_path = tmp_path / "agent.yaml"
    definition.skills = ["research@1.2.0"]
    manifest = tmp_path / "skills" / "research" / "1.2.0" / "skill.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "name: research\nversion: 1.2.0\ndescription: Gather reliable information\n",
        encoding="utf-8",
    )

    agent = Agent(definition, model=ReusableModel())

    assert [(skill.name, skill.version) for skill in agent.skills] == [("research", "1.2.0")]


def test_agent_requires_version_pins_for_multiple_file_backed_skills(tmp_path: Path) -> None:
    definition = _definition()
    definition.source_path = tmp_path / "agent.yaml"
    definition.skills = ["research"]
    for version in ("1.0.0", "1.2.0"):
        manifest = tmp_path / "skills" / "research" / version / "skill.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            f"name: research\nversion: {version}\ndescription: Research capability\n",
            encoding="utf-8",
        )

    with pytest.raises(ConfigError, match="multiple directory versions"):
        Agent(definition, model=ReusableModel())


def test_agent_reports_file_backed_skill_version_mismatch(tmp_path: Path) -> None:
    definition = _definition()
    definition.source_path = tmp_path / "agent.yaml"
    definition.skills = ["research@1.0.0"]
    manifest = tmp_path / "skills" / "research" / "1.0.0" / "skill.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "name: research\nversion: 1.2.0\ndescription: Research capability\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="declares 1.2.0, not directory version 1.0.0"):
        Agent(definition, model=ReusableModel())


def test_agent_resolves_unique_versioned_registry_entries_without_a_pin() -> None:
    definition = _definition()
    definition.skills = ["research"]
    skill = SkillDefinition(name="research", version="1.2.0")

    agent = Agent(
        definition,
        model=ReusableModel(),
        skill_registry={"research@1.2.0": skill},
    )

    assert [(item.name, item.version) for item in agent.skills] == [("research", "1.2.0")]


def test_agent_rejects_ambiguous_and_mismatched_skill_registry_entries() -> None:
    definition = _definition()
    definition.skills = ["research"]
    with pytest.raises(ConfigError, match="multiple registry versions"):
        Agent(
            definition,
            model=ReusableModel(),
            skill_registry={
                "research@1.0.0": SkillDefinition(name="research", version="1.0.0"),
                "research@1.2.0": SkillDefinition(name="research", version="1.2.0"),
            },
        )

    definition.skills = ["research@1.0.0"]
    with pytest.raises(ConfigError, match="does not match package"):
        Agent(
            definition,
            model=ReusableModel(),
            skill_registry={"research@1.0.0": SkillDefinition(name="research", version="1.2.0")},
        )


def test_agent_reports_file_backed_skill_unavailable_and_version_mismatch(
    tmp_path: Path,
) -> None:
    definition = _definition()
    definition.source_path = tmp_path / "agent.yaml"
    definition.skills = ["research@1.0.0"]
    with pytest.raises(ConfigError, match="was not found"):
        Agent(definition, model=ReusableModel())

    manifest = tmp_path / "skills" / "research" / "1.2.0" / "skill.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "name: research\nversion: 1.2.0\ndescription: Research capability\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="is version 1.2.0, not requested version 1.0.0"):
        Agent(definition, model=ReusableModel())


def test_agent_supports_legacy_skill_files_when_the_exact_version_matches(
    tmp_path: Path,
) -> None:
    definition = _definition()
    definition.source_path = tmp_path / "agent.yaml"
    definition.skills = ["research@1.2.0"]
    manifest = tmp_path / "skills" / "research" / "skill.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "name: research\nversion: 1.2.0\ndescription: Research capability\n",
        encoding="utf-8",
    )

    agent = Agent(definition, model=ReusableModel())

    assert agent.skills[0].version == "1.2.0"


def test_agent_rejects_conflicting_skill_versions_and_dependency_cycles() -> None:
    definition = _definition()
    definition.skills = ["research@1.0.0", "research@1.2.0"]
    with pytest.raises(ConfigError, match="already resolved at version 1.0.0"):
        Agent(
            definition,
            model=ReusableModel(),
            skill_registry={
                "research@1.0.0": SkillDefinition(name="research", version="1.0.0"),
                "research@1.2.0": SkillDefinition(name="research", version="1.2.0"),
            },
        )

    definition.skills = ["first"]
    with pytest.raises(ConfigError, match="Circular skill dependency"):
        Agent(
            definition,
            model=ReusableModel(),
            skill_registry={
                "first": SkillDefinition(name="first", dependencies=["second"]),
                "second": SkillDefinition(name="second", dependencies=["first"]),
            },
        )


def test_agents_using_one_environment_get_separate_tool_registries() -> None:
    environment_tools = ToolRegistry()
    environment_tools.register(_tool("environment_tool"))
    environment = Environment(type="shared", tools=environment_tools)
    injected_tools = ToolRegistry()
    injected_tools.register(_tool("agent_only_tool"))

    first = Agent(
        _definition(), model=ReusableModel(), environment=environment, tools=injected_tools
    )
    second = Agent(_definition(), model=ReusableModel(), environment=environment)

    assert first.tools.names() == ["agent_only_tool", "environment_tool"]
    assert second.tools.names() == ["environment_tool"]
    assert environment.tools.names() == ["environment_tool"]
    assert injected_tools.names() == ["agent_only_tool"]


def test_agent_resolves_immutable_definition_skills_and_tool_registry() -> None:
    definition = _definition()
    definition.model["headers"] = {"X-Mode": "original"}
    definition.policies = {"allowed_tools": ["lookup"]}
    definition.skills = ["review"]
    skill = SkillDefinition(name="review", instructions="original", triggers=["review"])
    source_registry = ToolRegistry()
    source_tool = _tool("lookup")
    source_registry.register(source_tool)
    agent = Agent(
        definition,
        model=ReusableModel(),
        tools=source_registry,
        skill_registry={"review": skill},
    )

    definition.model["headers"]["X-Mode"] = "changed"
    definition.policies["allowed_tools"].append("later")
    skill.instructions = "changed"
    skill.triggers.append("changed")
    source_registry.register(_tool("later"))

    assert agent.definition.model["headers"]["X-Mode"] == "original"
    assert agent.definition.policies["allowed_tools"] == ("lookup",)
    assert agent.skills[0].instructions == "original"
    assert agent.skills[0].triggers == ("review",)
    assert agent.tools.names() == ["lookup"]
    assert agent.tools.get("lookup") is not source_tool
    with pytest.raises(TypeError):
        agent.definition.model["model"] = "mutated"  # type: ignore[index]
    with pytest.raises(AttributeError):
        agent.skills[0].instructions = "mutated"  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="registry is frozen"):
        agent.tools.register(_tool("after_construction"))


@pytest.mark.parametrize(
    ("model_value", "message"),
    [
        (bytearray(b"mutable"), "finite JSON-compatible"),
        ({"provider_option": {1: "non-string key"}}, "non-string mapping key"),
    ],
)
def test_agent_rejects_non_json_model_options(model_value: object, message: str) -> None:
    definition = _definition()
    definition.model["provider_options"] = model_value

    with pytest.raises(ConfigError, match=message):
        Agent(definition, model=ReusableModel())


def test_agent_rejects_cyclic_model_options() -> None:
    definition = _definition()
    cyclic: list[object] = []
    cyclic.append(cyclic)
    definition.model["provider_options"] = cyclic

    with pytest.raises(ConfigError, match="cyclic configuration"):
        Agent(definition, model=ReusableModel())


def test_agent_rejects_mutable_python_subclasses_of_json_scalars() -> None:
    class MutableString(str):
        mutable_state: list[str]

    class MutableInteger(int):
        mutable_state: list[str]

    text_value = MutableString("provider-option")
    text_value.mutable_state = []
    integer_value = MutableInteger(3)
    integer_value.mutable_state = []
    definition = _definition()
    definition.model["provider_options"] = {
        "text": text_value,
        "count": integer_value,
    }

    with pytest.raises(ConfigError, match="finite JSON-compatible values"):
        Agent(definition, model=ReusableModel())


def test_agent_bounds_nested_model_configuration() -> None:
    definition = _definition()
    nested: dict[str, object] = {"leaf": "value"}
    for _ in range(130):
        nested = {"child": nested}
    definition.model["provider_options"] = nested

    with pytest.raises(ConfigError, match="configuration depth limit"):
        Agent(definition, model=ReusableModel())


def test_agent_bounds_programmatic_model_configuration_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "_MAX_AGENT_CONFIG_BYTES", 256)
    definition = _definition()
    definition.model["provider_options"] = {"payload": "x" * 80}
    definition.environment["resources"] = {"payload": "x" * 80}
    definition.knowledge["sources"] = ["x" * 80]

    with pytest.raises(ConfigError, match="configuration limit"):
        Agent(definition, model=ReusableModel())


def test_agent_bounds_aggregate_resolved_skill_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "_MAX_AGENT_TEXT_BYTES", 16)
    definition = _definition()
    definition.skills = ["analysis", "review"]
    registry = {
        "analysis": SkillDefinition(name="analysis", instructions="x" * 9),
        "review": SkillDefinition(name="review", instructions="y" * 9),
    }

    with pytest.raises(ConfigError, match="Resolved agent and skill text exceeds"):
        Agent(definition, model=ReusableModel(), skill_registry=registry)


def test_agent_bounds_dependency_expansion_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "_MAX_RESOLVED_SKILLS", 2)
    definition = _definition()
    definition.skills = ["root"]
    registry = {
        "root": SkillDefinition(name="root", dependencies=["one", "two"]),
        "one": SkillDefinition(name="one"),
        "two": SkillDefinition(name="two"),
    }

    with pytest.raises(ConfigError, match="may resolve at most 2 skills"):
        Agent(definition, model=ReusableModel(), skill_registry=registry)


def test_agent_snapshots_injected_environment_declarations() -> None:
    allowed_tools = ["lookup"]
    environment = Environment(
        type="research",
        description="original description",
        capabilities=["search"],
        resources={"endpoint": "original"},
        allowed_tools=allowed_tools,
    )
    agent = Agent(_definition(), model=ReusableModel(), environment=environment)

    environment.description = "changed description"
    environment.capabilities.append("filesystem")
    environment.resources["endpoint"] = "changed"
    allowed_tools.append("shell")

    assert agent.environment.description == "original description"
    assert agent.environment.capabilities == ("search",)
    assert agent.environment.resources["endpoint"] == "original"
    assert agent.environment.allowed_tools == ("lookup",)
    with pytest.raises(AttributeError):
        agent.environment.description = "mutated"  # type: ignore[misc]


def test_sandbox_tools_are_registered_only_on_the_agent_local_registry() -> None:
    definition = _definition()
    definition.sandbox = SandboxDefinition(
        engine="docker", image="python:3.14", keepalive_argv=("sleep", "1")
    )
    tools = ToolRegistry()
    tools.register(_tool("caller_tool"))

    agent = Agent(definition, model=ReusableModel(), tools=tools)

    assert tools.names() == ["caller_tool"]
    assert "shell_run" in agent.tools.names()
    assert "filesystem_read_file" in agent.tools.names()


def test_agent_rejects_unregistered_model_provider(tmp_path: Path) -> None:
    definition = _definition()
    definition.model["provider"] = "unknown"

    with pytest.raises(ConfigError, match="No built-in model provider"):
        Agent(definition)


def test_agent_constructs_ollama_provider_with_local_defaults() -> None:
    definition = _definition()
    definition.model = {"provider": "ollama", "model": "qwen3:8b"}

    agent = Agent(definition)

    assert isinstance(agent.model, OllamaProvider)
    assert agent.model.name == "ollama"
    assert agent.model.base_url == "http://localhost:11434/v1"
    assert agent.model.api_key_env == "GABBY_OLLAMA_API_KEY"


def test_agent_constructs_huggingface_inference_provider_with_hosted_defaults() -> None:
    definition = _definition()
    definition.model = {"provider": "huggingface", "model": "Qwen/Qwen3-8B"}

    agent = Agent(definition)

    assert isinstance(agent.model, HuggingFaceInferenceProvider)
    assert agent.model.name == "huggingface"
    assert agent.model.base_url == "https://router.huggingface.co/v1"
    assert agent.model.api_key_env == "HF_TOKEN"


def test_agent_constructs_anthropic_provider_with_native_defaults() -> None:
    definition = _definition()
    definition.model = {"provider": "anthropic", "model": "claude-test", "max_tokens": 4096}

    agent = Agent(definition)

    assert isinstance(agent.model, AnthropicProvider)
    assert agent.model.name == "anthropic"
    assert agent.model.base_url == "https://api.anthropic.com/v1"
    assert agent.model.api_key_env == "ANTHROPIC_API_KEY"
    assert agent.model.max_tokens == 4096


def test_from_file_constructs_an_agent(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: loaded\nmodel:\n  provider: openai_compatible\n  model: fake\n",
        encoding="utf-8",
    )

    agent = Agent.from_file(path, model=ReusableModel())

    assert agent.definition.name == "loaded"

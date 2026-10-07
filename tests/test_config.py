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
"""Agent and skill YAML loading failures and defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gabby.config as config_module
from gabby.config import (
    AgentDefinition,
    ConfigError,
    SandboxDefinition,
    WorkspaceMount,
    load_agent,
    load_skill,
    validate_agent_definition,
    validate_model_credentials,
    validate_model_endpoint,
    validate_sandbox_api_endpoint,
)


def _valid_agent_yaml(**fields: str) -> str:
    values = ["name: helper", "model:", "  provider: openai_compatible", "  model: small"]
    values.extend(f"{key}: {value}" for key, value in fields.items())
    return "\n".join(values) + "\n"


def test_load_agent_resolves_path_and_applies_optional_defaults(tmp_path: Path) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(_valid_agent_yaml(instructions="Answer clearly"), encoding="utf-8")

    definition = load_agent(source)

    assert definition.name == "helper"
    assert definition.source_path == source.resolve()
    assert definition.instructions == "Answer clearly"
    assert definition.skills == []
    assert definition.policies == {}


def test_load_agent_accepts_output_schema_and_programmatic_validation_matches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(
        "name: helper\nmodel:\n  provider: openai\n  model: x\n"
        "output_schema:\n  type: object\n  required: [result]\n"
        "  properties:\n    result:\n      type: string\n",
        encoding="utf-8",
    )
    loaded = load_agent(source)
    assert loaded.output_schema == {
        "type": "object",
        "required": ["result"],
        "properties": {"result": {"type": "string"}},
    }
    validate_agent_definition(
        AgentDefinition(
            name="helper",
            model={"provider": "openai", "model": "x"},
            output_schema=loaded.output_schema,
        )
    )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "not-a-json-schema-type"}, "valid JSON Schema"),
        ({"$ref": "https://example.test/schema.json"}, "references must be local"),
        ({"$id": "https://example.test/schema.json", "type": "object"}, "identifiers must not"),
    ],
)
def test_agent_output_schema_rejects_invalid_or_remote_references(
    schema: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_agent_definition(
            AgentDefinition(
                name="helper",
                model={"provider": "openai", "model": "x"},
                output_schema=schema,
            )
        )


def test_agent_output_schema_does_not_treat_instance_data_as_schema_references() -> None:
    validate_agent_definition(
        AgentDefinition(
            name="helper",
            model={"provider": "openai", "model": "x"},
            output_schema={
                "type": "object",
                "properties": {"payload": {"type": "object", "default": {"$ref": "value"}}},
            },
        )
    )


def test_load_skill_accepts_and_validates_input_and_output_schemas(tmp_path: Path) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text(
        "name: triage\n"
        "input_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    task: {type: string}\n"
        "  required: [task]\n"
        "output_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    category: {type: string}\n"
        "  required: [category]\n",
        encoding="utf-8",
    )

    skill = load_skill(source)

    assert skill.input_schema == {
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    }
    assert skill.output_schema == {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
    }


@pytest.mark.parametrize(
    ("schema_field", "schema", "message"),
    [
        ("input_schema", {"type": "invalid"}, "valid JSON Schema"),
        ("output_schema", {"$ref": "https://example.test/schema.json"}, "local JSON fragments"),
    ],
)
def test_skill_schemas_reject_invalid_and_remote_references(
    tmp_path: Path,
    schema_field: str,
    schema: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text(
        f"name: triage\n{schema_field}: {json.dumps(schema)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_skill(source)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("- not a mapping\n", "must contain a YAML mapping"),
        ("name: [unterminated\n", "Invalid YAML"),
        ("name: helper\nmodel: []\n", "model.provider and model.model"),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\nenvironment: []\n",
            "environment.*must be a mapping",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\ninstructions: 3\n",
            "instructions.*must be a string",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\ndescription: 3\n",
            "description.*must be a string",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n  api_key: secret-value\n",
            "model.api_key.*cannot be stored",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "  headers: {Authorization: 'Bearer secret-value'}\n",
            "Credential-bearing model headers.*cannot be stored",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n  headers: []\n",
            "model.headers.*must be a mapping",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\npolicies:\n  max_steps: true\n",
            "policies.max_steps.*positive integer",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_tool_calls: 1025\n",
            "policies.max_tool_calls.*1 through 1024",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_parallel_tool_calls: 0\n",
            "policies.max_parallel_tool_calls.*1 through 32",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_parallel_tool_calls: 33\n",
            "policies.max_parallel_tool_calls.*1 through 32",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_parallel_tool_calls: true\n",
            "policies.max_parallel_tool_calls.*1 through 32",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_model_retries: -1\n",
            "policies.max_model_retries.*0 through 3",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_model_retries: 4\n",
            "policies.max_model_retries.*0 through 3",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_model_retries: true\n",
            "policies.max_model_retries.*0 through 3",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\npolicies:\n  max_replans: 4\n",
            "policies.max_replans.*0 through 3",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  max_replans: true\n",
            "policies.max_replans.*0 through 3",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  timeout_seconds: .inf\n",
            "policies.timeout_seconds.*finite positive",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\nknowledge:\n  top_k: false\n",
            "knowledge.top_k.*1 through 100",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\nknowledge:\n  top_k: 101\n",
            "knowledge.top_k.*1 through 100",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "knowledge:\n  max_context_bytes: false\n",
            "knowledge.max_context_bytes.*positive integer",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "verification:\n  enabled: 'false'\n",
            "verification.enabled.*boolean",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "policies:\n  require_sandbox: 'true'\n",
            "policies.require_sandbox.*boolean",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "environment:\n  capabilities: read-only\n",
            "environment.capabilities.*list",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\nskills: [review, 3]\n",
            "skills must be a list of non-empty strings",
        ),
    ],
)
def test_load_agent_rejects_invalid_documents(tmp_path: Path, contents: str, message: str) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_agent(source)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\ntoolz: []\n",
            "agent contains unknown field",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\npolicies:\n  max_step: 8\n",
            "policies contains unknown field",
        ),
        (
            "name: helper\nmodel:\n  provider: openai\n  model: x\n"
            "sandbox:\n  engine: docker\n  image: python:3.12\n"
            "  keepalive_argv: [sleep]\n  workspace:\n    path: .\n    acess: read_write\n",
            "sandbox.workspace contains unknown field",
        ),
    ],
)
def test_load_agent_rejects_unknown_fields(tmp_path: Path, contents: str, message: str) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_agent(source)


def test_load_agent_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_agent(tmp_path / "missing.yaml")


def test_load_agent_bounds_yaml_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_MAX_AGENT_CONFIG_BYTES", 32)
    source = tmp_path / "agent.yaml"
    source.write_bytes(b"x" * 33)

    with pytest.raises(ConfigError, match="exceeds the 32-byte limit"):
        load_agent(source)


def test_load_skill_bounds_manifest_and_text_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_MAX_SKILL_MANIFEST_BYTES", 32)
    manifest = tmp_path / "large-skill.yaml"
    manifest.write_bytes(b"x" * 33)
    with pytest.raises(ConfigError, match="exceeds the 32-byte limit"):
        load_skill(manifest)

    monkeypatch.setattr(config_module, "_MAX_SKILL_MANIFEST_BYTES", 1024)
    monkeypatch.setattr(config_module, "_MAX_SKILL_TEXT_BYTES", 16)
    source = tmp_path / "skill.yaml"
    source.write_text("name: review\ninstructions_file: instructions.md\n", encoding="utf-8")
    (tmp_path / "instructions.md").write_bytes(b"x" * 17)
    with pytest.raises(ConfigError, match="instructions_file exceeds the 16-byte limit"):
        load_skill(source)


def test_programmatic_definitions_share_text_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "_MAX_AGENT_CONFIG_BYTES", 16)
    definition = AgentDefinition(
        name="helper",
        model={"provider": "openai", "model": "small"},
        instructions="€" * 6,
    )
    with pytest.raises(ConfigError, match="instructions exceeds the 16-byte limit"):
        validate_agent_definition(definition)

    monkeypatch.setattr(config_module, "_MAX_SKILL_TEXT_BYTES", 16)
    with pytest.raises(ConfigError, match="skill.instructions exceeds the 16-byte limit"):
        config_module.validate_skill_definition(
            config_module.SkillDefinition(name="review", instructions="€" * 6)
        )


def test_programmatic_skill_metadata_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "_MAX_SKILL_METADATA_BYTES", 8)
    with pytest.raises(ConfigError, match="Skill metadata exceeds the 8-byte limit"):
        config_module.validate_skill_definition(
            config_module.SkillDefinition(name="review", constraints=["12345"], triggers=["6789"])
        )

    monkeypatch.setattr(config_module, "_MAX_SKILL_METADATA_BYTES", 1024)
    monkeypatch.setattr(config_module, "_MAX_SKILL_METADATA_VALUES", 1)
    with pytest.raises(ConfigError, match="Skill metadata exceeds the 1-value limit"):
        config_module.validate_skill_definition(
            config_module.SkillDefinition(name="review", constraints=["one"], triggers=["two"])
        )


def test_skill_definition_rejects_str_subclasses() -> None:
    class MutableString(str):
        mutable_state: list[str]

    skill_name = MutableString("review")
    skill_name.mutable_state = []

    with pytest.raises(ConfigError, match="canonical skill ID"):
        config_module.validate_skill_definition(config_module.SkillDefinition(name=skill_name))


def test_agent_definition_bounds_configured_skill_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_MAX_RESOLVED_SKILLS", 2)
    definition = AgentDefinition(
        name="helper",
        model={"provider": "openai", "model": "small"},
        skills=["one", "two", "three"],
    )

    with pytest.raises(ConfigError, match="at most 2 skills"):
        validate_agent_definition(definition)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://models.example/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
        "http://[::ffff:127.0.0.1]:8080/v1",
    ],
)
def test_builtin_model_endpoint_accepts_tls_or_loopback_http(endpoint: str) -> None:
    validate_model_endpoint(endpoint, provider="test")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://models.example/v1",
        "ftp://localhost/model",
        "https:///missing-host",
        "https://models.example:bad-port/v1",
    ],
)
def test_builtin_model_endpoint_rejects_insecure_or_malformed_urls(endpoint: str) -> None:
    with pytest.raises(ConfigError):
        validate_model_endpoint(endpoint, provider="test")


@pytest.mark.parametrize(
    "model",
    [
        {"api_key": "secret"},
        {"api_key_env": "bad-name"},
        {"headers": {"Authorization": "Bearer secret"}},
        {"headers": {"x-api-key": "secret"}},
        {"headers": {"X-Request-ID": 1}},
        {"base_url": "https://user:password@models.example/v1"},
        {"base_url": "https://models.example/v1?api_key=secret"},
        {"base_url": "https://models.example:invalid/v1"},
    ],
)
def test_model_credentials_and_malformed_urls_are_rejected(model: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        validate_model_credentials(model)


def test_model_credentials_allow_safe_environment_names_and_custom_http_transport() -> None:
    validate_model_credentials(
        {"api_key_env": "GABBY_MODEL_KEY", "base_url": "http://custom.internal/model"},
        provider="custom_provider",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://containers.example/engine",
        "http://localhost:2375",
        "http://127.0.0.1:2375",
        "http://[::1]:2375",
    ],
)
def test_sandbox_api_endpoint_accepts_tls_or_loopback_http(endpoint: str) -> None:
    validate_sandbox_api_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://containers.example/engine",
        "ftp://localhost/engine",
        "https://user:secret@containers.example/engine",
        "https://containers.example/engine?access_token=secret",
        "https://containers.example:bad-port/engine",
    ],
)
def test_sandbox_api_endpoint_rejects_remote_http_and_credentials(endpoint: str) -> None:
    with pytest.raises(ConfigError):
        validate_sandbox_api_endpoint(endpoint)


def test_in_memory_agent_definition_uses_the_same_validation_contract() -> None:
    valid = AgentDefinition(name="helper", model={"provider": "local", "model": "small"})
    validate_agent_definition(valid)

    invalid = AgentDefinition(
        name="helper",
        model={"provider": "openai_compatible", "model": "small"},
        policies={"allowed_tools": ["shell", ""]},
    )
    with pytest.raises(ConfigError, match="allowed_tools.*non-empty strings"):
        validate_agent_definition(invalid)


def test_load_agent_parses_sandbox_config_and_relative_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "agent.yaml"
    source.write_text(
        "name: helper\nmodel:\n  provider: openai\n  model: x\n"
        "sandbox:\n  engine: docker\n  image: python:3.12-slim\n"
        "  keepalive_argv: [python, -c, 'import time; time.sleep(3600)']\n"
        "  user: '1001:1002'\n"
        "  workspace:\n    path: ./workspace\n    access: read_write\n"
        "  resources:\n    cpus: 1.5\n    memory_bytes: 1073741824\n    process_limit: 128\n",
        encoding="utf-8",
    )

    definition = load_agent(source)

    assert definition.sandbox is not None
    assert definition.sandbox.engine == "docker"
    assert definition.sandbox.image == "python:3.12-slim"
    assert definition.sandbox.keepalive_argv[0] == "python"
    assert definition.sandbox.workspace is not None
    assert definition.sandbox.workspace.host_path == workspace.resolve()
    assert definition.sandbox.workspace.access == "read_write"
    assert definition.sandbox.cpus == 1.5
    assert definition.sandbox.memory_bytes == 1073741824
    assert definition.sandbox.process_limit == 128
    assert definition.sandbox.user == "1001:1002"


@pytest.mark.parametrize(
    ("sandbox", "message"),
    [
        ("[]", "sandbox.*must be a mapping"),
        ("{engine: unknown, image: x, keepalive_argv: [sleep]}", "sandbox.engine"),
        ("{engine: docker, keepalive_argv: [sleep]}", "sandbox.image"),
        (
            "{engine: docker, image: --privileged, keepalive_argv: [sleep]}",
            "sandbox.image.*engine option",
        ),
        ("{engine: docker, image: x}", "keepalive_argv"),
        ("{engine: docker, image: x, keepalive_argv: [sleep], user: '0:0'}", "sandbox.user"),
        (
            "{engine: podman, image: x, keepalive_argv: [sleep], adapter: api}",
            "sandbox.api requires",
        ),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], resources: {cpus: 0}}",
            "resources.cpus",
        ),
    ],
)
def test_load_agent_validates_sandbox_fields(tmp_path: Path, sandbox: str, message: str) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(
        f"name: helper\nmodel:\n  provider: openai\n  model: x\nsandbox: {sandbox}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=message):
        load_agent(source)


def test_load_skill_validates_name_and_string_lists(tmp_path: Path) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text(
        "name: review\ndescription: Review changes\n"
        "dependencies: [analysis]\ntriggers: [review, inspect]\n",
        encoding="utf-8",
    )

    skill = load_skill(source)

    assert skill.name == "review"
    assert skill.version == "0.1.0"
    assert skill.dependencies == ["analysis"]
    assert skill.triggers == ["review", "inspect"]
    assert skill.source_path == source.resolve()


def test_load_skill_rejects_missing_name(tmp_path: Path) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text("description: nameless\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="requires a non-empty 'name'"):
        load_skill(source)


def test_load_skill_rejects_unknown_fields(tmp_path: Path) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text("name: review\ndependecies: [base]\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="skill contains unknown field.*dependecies"):
        load_skill(source)


def test_load_skill_rejects_invalid_semantic_version(tmp_path: Path) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text("name: review\nversion: latest\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="skill.version must be a valid semantic version"):
        load_skill(source)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("description", "skill.description.*must be a string"),
        ("instructions", "skill.instructions.*must be a string"),
    ],
)
def test_load_skill_rejects_non_string_text_fields(
    tmp_path: Path, field: str, message: str
) -> None:
    source = tmp_path / "skill.yaml"
    source.write_text(f"name: review\n{field}: [not, a, string]\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_skill(source)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine": "invalid"}, "sandbox.engine"),
        ({"image": 3}, "sandbox.image"),
        ({"image": "  "}, "sandbox.image"),
        ({"image": "--privileged"}, "sandbox.image.*engine option"),
        ({"image": "python:3.12\n--privileged"}, "sandbox.image.*whitespace/control"),
        ({"image": "python:3.12\x7f"}, "sandbox.image.*whitespace/control"),
        ({"adapter": "bad"}, "sandbox.adapter"),
        ({"workspace": "./workspace"}, "sandbox.workspace"),
        ({"user": "0:0"}, "sandbox.user"),
        ({"user": 1001}, "sandbox.user"),
        ({"user": "1001"}, "sandbox.user"),
        ({"user": "4294967296:1001"}, "sandbox.user"),
        ({"keepalive_argv": ("",)}, "keepalive_argv"),
        ({"keepalive_argv": ("sleep", 1)}, "keepalive_argv"),
        ({"keepalive_argv": ["sleep"]}, "keepalive_argv"),
        ({"cpus": True}, "finite positive"),
        ({"cpus": "2"}, "finite positive"),
        ({"cpus": float("inf")}, "finite positive"),
        ({"memory_bytes": True}, "memory_bytes"),
        ({"memory_bytes": 0}, "memory_bytes"),
        ({"memory_bytes": 2.5}, "memory_bytes"),
        ({"process_limit": True}, "process_limit"),
        ({"process_limit": 0}, "process_limit"),
        ({"process_limit": -1}, "process_limit"),
        ({"require_image_digest": True}, "@sha256:"),
        ({"require_image_digest": "yes"}, "require_image_digest"),
        ({"adapter": "api", "api_base_url": None}, "requires base_url"),
        ({"adapter": "api", "api_base_url": "ftp://engine.example.test"}, r"HTTP\(S\) URL"),
        (
            {"adapter": "api", "api_base_url": "https://engine.example.test:bad"},
            r"valid HTTP\(S\) URL",
        ),
        (
            {
                "adapter": "api",
                "api_base_url": "http://localhost",
                "api_unix_socket": Path("/tmp/docker.sock"),
            },
            "exactly one API endpoint",
        ),
        ({"adapter": "api", "api_unix_socket": "/tmp/docker.sock"}, "pathlib Path"),
        ({"adapter": "api", "api_named_pipe": 1}, "named_pipe"),
    ],
)
def test_sandbox_definition_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "engine": "docker",
        "image": "python:3.12",
        "keepalive_argv": ("sleep", "infinity"),
    }
    values.update(kwargs)
    with pytest.raises(ConfigError, match=message):
        SandboxDefinition(**values)  # type: ignore[arg-type]


def test_sandbox_definition_accepts_local_windows_api_named_pipe() -> None:
    definition = SandboxDefinition(
        engine="docker",
        image="windows/servercore:ltsc2022",
        keepalive_argv=("powershell.exe", "-Command", "Start-Sleep -Seconds 60"),
        adapter="api",
        api_named_pipe=r"\\.\pipe\docker_engine",
    )
    assert definition.api_named_pipe == r"\\.\pipe\docker_engine"


def test_load_agent_accepts_windows_api_named_pipe(tmp_path: Path) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(
        r"""name: windows
model:
  provider: openai_compatible
  model: small
sandbox:
  engine: docker
  image: mcr.microsoft.com/windows/servercore:ltsc2022
  keepalive_argv: [powershell.exe, -Command, Start-Sleep]
  adapter: api
  api:
    named_pipe: '\\.\pipe\docker_engine'
""",
        encoding="utf-8",
    )

    definition = load_agent(source)

    assert definition.sandbox is not None
    assert definition.sandbox.api_named_pipe == r"\\.\pipe\docker_engine"


@pytest.mark.parametrize("named_pipe", ["docker_engine", r"\\.\pipe\..\docker_engine"])
def test_sandbox_definition_rejects_nonlocal_or_invalid_named_pipe(named_pipe: str) -> None:
    with pytest.raises(ConfigError, match="named_pipe"):
        SandboxDefinition(
            engine="docker",
            image="windows/servercore:ltsc2022",
            keepalive_argv=("powershell.exe",),
            adapter="api",
            api_named_pipe=named_pipe,
        )


def test_sandbox_definition_defaults_to_unprivileged_linux_identity() -> None:
    definition = SandboxDefinition(
        engine="docker",
        image="python:3.12",
        keepalive_argv=("sleep", "infinity"),
    )

    assert definition.user == "65532:65532"


def test_sandbox_definition_can_require_an_immutable_image_digest() -> None:
    definition = SandboxDefinition(
        engine="docker",
        image="registry.example.test/team/agent@sha256:" + "a" * 64,
        keepalive_argv=("sleep", "infinity"),
        require_image_digest=True,
    )

    assert definition.require_image_digest


def test_workspace_mount_requires_existing_directory_and_safe_container_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ConfigError, match="existing directory"):
        WorkspaceMount(missing, "read_only")
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ConfigError, match="access"):
        WorkspaceMount(tmp_path, "write")
    with pytest.raises(ConfigError, match="absolute"):
        WorkspaceMount(tmp_path, "read_only", "relative/path")
    with pytest.raises(ConfigError, match="cannot contain"):
        WorkspaceMount(tmp_path, "read_only", "/workspace/../escape")
    with pytest.raises(ConfigError, match="container_path must be a string"):
        WorkspaceMount(tmp_path, "read_only", 7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sandbox", "message"),
    [
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], resources: []}",
            "resources.*mapping",
        ),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], resources: {memory_bytes: 1.5}}",
            "memory_bytes",
        ),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], resources: {process_limit: 0}}",
            "process_limit",
        ),
        ("{engine: docker, image: x, keepalive_argv: [sleep], api: []}", "sandbox.api.*mapping"),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], api: {base_url: ftp://host}}",
            r"HTTP\(S\)",
        ),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], workspace: {path: missing}}",
            "existing directory",
        ),
        (
            "{engine: docker, image: x, keepalive_argv: [sleep], "
            "workspace: {path: ., access: write}}",
            "workspace.access",
        ),
    ],
)
def test_load_agent_rejects_malformed_sandbox_subsections(
    tmp_path: Path, sandbox: str, message: str
) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(
        f"name: helper\nmodel:\n  provider: openai\n  model: x\nsandbox: {sandbox}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=message):
        load_agent(source)

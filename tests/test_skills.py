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
"""Skill resolution and registry boundary contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gabby.agent import Agent
from gabby.config import AgentDefinition, ConfigError, SkillDefinition


def _definition(source: Path, skills: list[str]) -> AgentDefinition:
    return AgentDefinition(
        name="skill-test",
        model={"provider": "fake", "model": "test"},
        skills=skills,
        source_path=source,
    )


def _write_skill(
    root: Path,
    name: str,
    *,
    dependencies: list[str] | None = None,
    version: str = "0.1.0",
) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    dep_yaml = "\n".join(f"  - {dependency}" for dependency in dependencies or [])
    skill_dir.joinpath("skill.yaml").write_text(
        f"name: {name}\nversion: {version}\ndependencies:\n{dep_yaml or '  []'}\n",
        encoding="utf-8",
    )


def _write_versioned_skill(
    root: Path, name: str, version: str, *, dependencies: list[str] | None = None
) -> None:
    skill_dir = root / "skills" / name / version
    skill_dir.mkdir(parents=True)
    dep_yaml = "\n".join(f"  - {dependency}" for dependency in dependencies or [])
    skill_dir.joinpath("skill.yaml").write_text(
        f"name: {name}\nversion: {version}\ndependencies:\n{dep_yaml or '  []'}\n",
        encoding="utf-8",
    )


def test_skill_dependencies_resolve_once_in_dependency_order(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "base")
    _write_skill(tmp_path, "review", dependencies=["base"])

    agent = Agent(_definition(agent_file, ["review", "base"]), model=object())  # type: ignore[arg-type]

    assert [skill.name for skill in agent.skills] == ["base", "review"]


def test_skill_reference_pins_package_and_dependency_versions(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "base", version="1.4.0")
    _write_skill(
        tmp_path,
        "review",
        dependencies=["base@1.4.0"],
        version="2.1.0",
    )

    agent = Agent(
        _definition(agent_file, ["review@2.1.0"]),
        model=object(),  # type: ignore[arg-type]
    )

    assert [(skill.name, skill.version, skill.dependencies) for skill in agent.skills] == [
        ("base", "1.4.0", ()),
        ("review", "2.1.0", ("base",)),
    ]


def test_skill_registry_can_resolve_multiple_versions_by_exact_reference() -> None:
    definition = AgentDefinition(
        name="skill-test",
        model={"provider": "fake", "model": "test"},
        skills=["review@2.0.0"],
    )
    registry = {
        "review@1.0.0": SkillDefinition(name="review", version="1.0.0"),
        "review@2.0.0": SkillDefinition(name="review", version="2.0.0"),
    }

    agent = Agent(definition, model=object(), skill_registry=registry)  # type: ignore[arg-type]

    assert [(skill.name, skill.version) for skill in agent.skills] == [("review", "2.0.0")]


def test_filesystem_registry_resolves_exact_and_unique_versions(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_versioned_skill(tmp_path, "review", "1.0.0")
    _write_versioned_skill(tmp_path, "review", "2.0.0")

    pinned = Agent(
        _definition(agent_file, ["review@2.0.0"]),
        model=object(),  # type: ignore[arg-type]
    )
    assert [(skill.name, skill.version) for skill in pinned.skills] == [("review", "2.0.0")]

    with pytest.raises(ConfigError, match="multiple directory versions"):
        Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]

    unique_root = tmp_path / "unique"
    unique_agent_file = unique_root / "agent.yaml"
    unique_agent_file.parent.mkdir()
    unique_agent_file.touch()
    _write_versioned_skill(unique_root, "review", "1.0.0")
    unpinned = Agent(
        _definition(unique_agent_file, ["review"]),
        model=object(),  # type: ignore[arg-type]
    )
    assert [(skill.name, skill.version) for skill in unpinned.skills] == [("review", "1.0.0")]


def test_versioned_directory_must_match_manifest_version(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_versioned_skill(tmp_path, "review", "2.0.0")
    manifest = tmp_path / "skills" / "review" / "2.0.0" / "skill.yaml"
    manifest.write_text("name: review\nversion: 1.0.0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="declares 1.0.0, not directory version 2.0.0"):
        Agent(
            _definition(agent_file, ["review@2.0.0"]),
            model=object(),  # type: ignore[arg-type]
        )


def test_skill_reference_rejects_manifest_version_mismatch(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "review", version="1.0.0")

    with pytest.raises(ConfigError, match="is version 1.0.0, not requested version 2.0.0"):
        Agent(
            _definition(agent_file, ["review@2.0.0"]),
            model=object(),  # type: ignore[arg-type]
        )


def test_skill_references_cannot_escape_the_registry(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_file = project / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "private")

    with pytest.raises(ConfigError, match="safe relative skill ID"):
        Agent(_definition(agent_file, ["../private"]), model=object())  # type: ignore[arg-type]


def test_skill_symlink_cannot_escape_the_registry(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent_file = project / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "private")
    registry = project / "skills"
    registry.mkdir()
    (registry / "linked").symlink_to(tmp_path / "skills" / "private", target_is_directory=True)

    with pytest.raises(ConfigError, match="outside its configured registry"):
        Agent(_definition(agent_file, ["linked"]), model=object())  # type: ignore[arg-type]


def test_declared_skills_cannot_be_silently_ignored_without_a_definition_path() -> None:
    definition = AgentDefinition(
        name="skill-test",
        model={"provider": "fake", "model": "test"},
        skills=["review"],
    )

    with pytest.raises(ConfigError, match="must be passed through skill_registry"):
        Agent(definition, model=object())  # type: ignore[arg-type]


def test_programmatic_skill_registry_uses_shared_structural_validation() -> None:
    definition = AgentDefinition(
        name="skill-test",
        model={"provider": "fake", "model": "test"},
        skills=["review"],
    )

    with pytest.raises(ConfigError, match="skill.tools must be a list of non-empty strings"):
        Agent(
            definition,
            model=object(),  # type: ignore[arg-type]
            skill_registry={"review": SkillDefinition(name="review", tools=[""])},
        )


def test_missing_and_mismatched_skill_files_fail_during_construction(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()

    with pytest.raises(ConfigError, match="Skill 'missing' was not found"):
        Agent(_definition(agent_file, ["missing"]), model=object())  # type: ignore[arg-type]

    _write_skill(tmp_path, "requested")
    skill_file = tmp_path / "skills" / "requested" / "skill.yaml"
    skill_file.write_text("name: different\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="declares 'different'"):
        Agent(_definition(agent_file, ["requested"]), model=object())  # type: ignore[arg-type]


def test_skill_package_loads_conventional_markdown_instructions(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "research")
    (tmp_path / "skills" / "research" / "instructions.md").write_text(
        "Keep source citations with each finding.\n", encoding="utf-8"
    )

    agent = Agent(_definition(agent_file, ["research"]), model=object())  # type: ignore[arg-type]

    assert agent.skills[0].instructions == "Keep source citations with each finding.\n"


def test_skill_package_loads_a_declared_nested_instruction_file(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "review")
    skill_dir = tmp_path / "skills" / "review"
    (skill_dir / "procedures").mkdir()
    (skill_dir / "procedures" / "review.md").write_text("Review evidence first.", encoding="utf-8")
    (skill_dir / "skill.yaml").write_text(
        "name: review\ninstructions_file: procedures/review.md\n", encoding="utf-8"
    )

    agent = Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]

    assert agent.skills[0].instructions == "Review evidence first."


def test_skill_package_loads_examples_and_rejects_path_escape(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "research")
    skill_dir = tmp_path / "skills" / "research"
    (skill_dir / "examples.md").write_text(
        "Request: compare two sources.\nExpected: cite both sources.", encoding="utf-8"
    )

    agent = Agent(_definition(agent_file, ["research"]), model=object())  # type: ignore[arg-type]
    assert agent.skills[0].examples == (
        "Request: compare two sources.\nExpected: cite both sources."
    )

    (skill_dir / "skill.yaml").write_text(
        "name: research\nexamples_file: ../private.md\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="examples_file must stay inside"):
        Agent(_definition(agent_file, ["research"]), model=object())  # type: ignore[arg-type]


@pytest.mark.parametrize("instruction_file", ["../outside.md", "/tmp/outside.md", "C:\\outside.md"])
def test_skill_instruction_file_cannot_escape_its_package(
    tmp_path: Path, instruction_file: str
) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "review")
    skill_file = tmp_path / "skills" / "review" / "skill.yaml"
    skill_file.write_text(
        f"name: review\ninstructions_file: '{instruction_file}'\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="stay inside the skill package"):
        Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]


def test_skill_instruction_symlink_cannot_escape_its_package(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "review")
    outside = tmp_path / "private.md"
    outside.write_text("private procedure", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "review"
    (skill_dir / "instructions.md").symlink_to(outside)

    with pytest.raises(ConfigError, match="cannot resolve outside"):
        Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]


def test_skill_instruction_file_rejects_missing_and_conflicting_sources(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "review")
    skill_file = tmp_path / "skills" / "review" / "skill.yaml"
    skill_file.write_text("name: review\ninstructions_file: missing.md\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="was not found"):
        Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]

    skill_file.write_text(
        "name: review\ninstructions: inline\ninstructions_file: instructions.md\n",
        encoding="utf-8",
    )
    (skill_file.parent / "instructions.md").write_text("file", encoding="utf-8")
    with pytest.raises(ConfigError, match="Specify either"):
        Agent(_definition(agent_file, ["review"]), model=object())  # type: ignore[arg-type]


def test_circular_skill_dependencies_fail_during_construction(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.touch()
    _write_skill(tmp_path, "first", dependencies=["second"])
    _write_skill(tmp_path, "second", dependencies=["first"])

    with pytest.raises(ConfigError, match="Circular skill dependency"):
        Agent(_definition(agent_file, ["first"]), model=object())  # type: ignore[arg-type]

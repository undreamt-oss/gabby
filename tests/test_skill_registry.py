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
"""Tests for signed remote skill registry discovery and installation."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import gabby.cli as cli_module
import gabby.skill_registry as registry_module
from gabby import SkillRegistryClient, SkillRegistryError, pack_skill, sign_skill_package
from gabby.skill_packages import SkillPackageInfo

_CATALOG: dict[str, Any] = {
    "format": "gabby-skill-catalog",
    "format_version": 1,
    "skills": [
        {
            "name": "support/triage",
            "description": "Sort incoming support cases",
            "versions": ["1.2.0", "1.1.0"],
        },
        {"name": "research", "description": "Find reliable sources", "versions": ["2.0.0"]},
    ],
}


def _signed_archive(tmp_path: Path, *, name: str = "support/triage") -> tuple[bytes, bytes, bytes]:
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
    source = tmp_path / "skill"
    source.mkdir()
    (source / "skill.yaml").write_text(
        f"name: {name}\nversion: 1.2.0\ndescription: A test skill\n", encoding="utf-8"
    )
    package = tmp_path / "skill.gabskill"
    pack_skill(source, package)
    signature = sign_skill_package(package, key_id="publisher", private_key=private_key)
    return package.read_bytes(), signature.read_bytes(), public_key


def _registry_signing_key() -> tuple[bytes, bytes]:
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    return (
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ),
        key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ),
    )


def test_catalog_parser_rejects_invalid_structure_and_version_contracts() -> None:
    invalid_documents: list[Any] = []
    invalid_documents.extend((None, [], {"format": "other", "format_version": 1, "skills": []}))
    invalid_documents.extend(
        (
            {"format": "gabby-skill-catalog", "format_version": True, "skills": []},
            {"format": "gabby-skill-catalog", "format_version": 1, "skills": [], "extra": 1},
            {"format": "gabby-skill-catalog", "format_version": 1, "skills": "not-a-list"},
            {**deepcopy(_CATALOG), "skills": [{}]},
            {
                **deepcopy(_CATALOG),
                "skills": [{"name": 3, "description": "x", "versions": ["1.0.0"]}],
            },
            {
                **deepcopy(_CATALOG),
                "skills": [{"name": "ok", "description": 3, "versions": ["1.0.0"]}],
            },
            {
                **deepcopy(_CATALOG),
                "skills": [{"name": "../bad", "description": "x", "versions": ["1.0.0"]}],
            },
            {**deepcopy(_CATALOG), "skills": [{"name": "ok", "description": "x", "versions": []}]},
            {
                **deepcopy(_CATALOG),
                "skills": [{"name": "ok", "description": "x", "versions": ["1.0.0", "1.0.0"]}],
            },
            {**deepcopy(_CATALOG), "skills": [{"name": "ok", "description": "x", "versions": [1]}]},
            {
                **deepcopy(_CATALOG),
                "skills": [{"name": "ok", "description": "x", "versions": ["latest"]}],
            },
        )
    )
    for document in invalid_documents:
        with pytest.raises(SkillRegistryError):
            registry_module._parse_catalog(document)


def test_catalog_parser_checks_freshness_timestamp_and_utf8_bounds() -> None:
    current = datetime(2026, 1, 1, tzinfo=UTC)
    for timestamp in (3, "short", "not-a-timestamp", "2026-01-01T00:00:00"):
        document = {**deepcopy(_CATALOG), "generated_at": timestamp}
        with pytest.raises(SkillRegistryError):
            registry_module._parse_catalog(document, now=current)
    with pytest.raises(SkillRegistryError, match="no freshness"):
        registry_module._parse_catalog(_CATALOG, max_age_seconds=60, now=current)
    future = {**deepcopy(_CATALOG), "generated_at": "2026-01-02T00:00:00Z"}
    with pytest.raises(SkillRegistryError, match="future"):
        registry_module._parse_catalog(future, now=current)
    stale = {**deepcopy(_CATALOG), "generated_at": "2025-12-31T23:00:00Z"}
    with pytest.raises(SkillRegistryError, match="stale"):
        registry_module._parse_catalog(stale, max_age_seconds=60, now=current)
    with pytest.raises(ValueError, match="timezone"):
        registry_module._parse_catalog(stale, now=datetime(2026, 1, 1), max_age_seconds=60)
    oversized = {
        **deepcopy(_CATALOG),
        "skills": [{"name": "ok", "description": "x" * 16385, "versions": ["1.0.0"]}],
    }
    with pytest.raises(SkillRegistryError, match="description exceeds"):
        registry_module._parse_catalog(oversized)


def test_catalog_parser_rejects_duplicate_names_and_json_members() -> None:
    duplicate = deepcopy(_CATALOG)
    duplicate["skills"].append(deepcopy(duplicate["skills"][0]))
    with pytest.raises(SkillRegistryError, match="duplicate skill IDs"):
        registry_module._parse_catalog(duplicate)
    with pytest.raises(SkillRegistryError, match="duplicate JSON keys"):
        registry_module._unique_json_object([("format", "one"), ("format", "two")])


def test_skill_registry_url_and_http_status_boundaries() -> None:
    assert registry_module._validate_base_url("https://registry.example.test/prefix/") == (
        "https://registry.example.test/prefix"
    )
    assert registry_module._validate_base_url("http://localhost:8080") == "http://localhost:8080"
    assert registry_module._validate_base_url("http://[::1]/skills") == "http://[::1]/skills"
    for address in (
        "ftp://registry.example.test",
        "https:///missing-host",
        "https://user:password@registry.example.test",
        "https://registry.example.test?token=secret",
        "https://registry.example.test/#fragment",
        "http://registry.example.test",
        "https://registry.example.test:99999",
        "x" * 2049,
    ):
        with pytest.raises(ValueError):
            registry_module._validate_base_url(address)
    assert registry_module._is_loopback("localhost")
    assert registry_module._is_loopback("127.0.0.1")
    assert not registry_module._is_loopback("registry.example.test")
    registry_module._ensure_success(httpx.Response(200))
    with pytest.raises(SkillRegistryError, match="redirects"):
        registry_module._ensure_success(httpx.Response(302))
    with pytest.raises(SkillRegistryError, match="HTTP status 503"):
        registry_module._ensure_success(httpx.Response(503))


def _make_signed_skill(
    root: Path,
    *,
    name: str,
    version: str,
    dependencies: tuple[str, ...],
    private_key: bytes,
) -> tuple[bytes, bytes]:
    source = root / f"source-{name.replace('/', '-')}-{version}"
    source.mkdir()
    dependency_yaml = (
        "dependencies: []\n"
        if not dependencies
        else ("dependencies:\n" + "".join(f"  - '{dependency}'\n" for dependency in dependencies))
    )
    (source / "skill.yaml").write_text(
        f"name: {name}\nversion: {version}\ndescription: {name} capability\n{dependency_yaml}",
        encoding="utf-8",
    )
    archive = root / f"{name.replace('/', '-')}-{version}.gabskill"
    pack_skill(source, archive)
    signature = sign_skill_package(archive, key_id="publisher", private_key=private_key)
    return archive.read_bytes(), signature.read_bytes()


def test_remote_registry_search_and_exact_versions() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer registry-secret"
        assert request.url.path == "/prefix/v1/catalog.json"
        return httpx.Response(200, json=_CATALOG)

    async def scenario() -> None:
        async with SkillRegistryClient(
            "https://registry.example.test/prefix/",
            headers={"Authorization": "Bearer registry-secret"},
            transport=httpx.MockTransport(handler),
        ) as client:
            results = await client.search("SUPPORT")
            assert [entry.name for entry in results] == ["support/triage"]
            assert await client.versions("support/triage") == ("1.2.0", "1.1.0")

    import asyncio

    asyncio.run(scenario())
    assert len(requests) == 2


def test_static_registry_builder_creates_client_compatible_layout(tmp_path: Path) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path)
    package = tmp_path / "publisher-package.gabskill"
    package.write_bytes(package_bytes)
    Path(f"{package}.sig").write_bytes(signature_bytes)

    output = tmp_path / "static-registry"
    result = registry_module.build_static_skill_registry(
        [package], output, trusted_keys={"publisher": public_key}
    )

    assert result.path == output
    assert result.skill_count == 1
    assert result.package_count == 1
    assert len(result.catalog_sha256) == 64
    catalog_path = output / "v1" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert registry_module._parse_catalog(catalog) == (
        registry_module.SkillCatalogEntry("support/triage", "A test skill", ("1.2.0",)),
    )
    artifact = output / "v1" / "skills" / "support" / "triage" / "versions" / "1.2.0"
    assert (artifact / "package.gabskill").read_bytes() == package_bytes
    assert (artifact / "package.gabskill.sig").read_bytes() == signature_bytes

    async def serve_static_file(request: httpx.Request) -> httpx.Response:
        path = output / request.url.path.lstrip("/")
        try:
            return httpx.Response(200, content=path.read_bytes())
        except OSError:
            return httpx.Response(404)

    async def consume_registry() -> None:
        async with SkillRegistryClient(
            "https://registry.example.test",
            transport=httpx.MockTransport(serve_static_file),
        ) as client:
            assert await client.versions("support/triage") == ("1.2.0",)
            installed = await client.install(
                "support/triage",
                "1.2.0",
                tmp_path / "consumer-skills",
                trusted_keys={"publisher": public_key},
            )
            assert installed.name == "support/triage"
            assert installed.version == "1.2.0"

    import asyncio

    asyncio.run(consume_registry())


def test_static_registry_builder_preserves_existing_signed_versions(tmp_path: Path) -> None:
    private_key, public_key = _registry_signing_key()
    older_archive, older_signature = _make_signed_skill(
        tmp_path,
        name="research/brief",
        version="1.0.0",
        dependencies=(),
        private_key=private_key,
    )
    older = tmp_path / "brief-1.0.0.gabskill"
    older.write_bytes(older_archive)
    Path(f"{older}.sig").write_bytes(older_signature)
    initial = registry_module.build_static_skill_registry(
        [older],
        tmp_path / "registry-v1",
        trusted_keys={"publisher": public_key},
    )

    newer_archive, newer_signature = _make_signed_skill(
        tmp_path,
        name="research/brief",
        version="1.1.0",
        dependencies=(),
        private_key=private_key,
    )
    newer = tmp_path / "brief-1.1.0.gabskill"
    newer.write_bytes(newer_archive)
    Path(f"{newer}.sig").write_bytes(newer_signature)
    updated = registry_module.build_static_skill_registry(
        [newer],
        tmp_path / "registry-v2",
        trusted_keys={"publisher": public_key},
        existing_registry=initial.path,
    )

    assert updated.package_count == 2
    catalog = json.loads((updated.path / "v1" / "catalog.json").read_text(encoding="utf-8"))
    assert registry_module._parse_catalog(catalog) == (
        registry_module.SkillCatalogEntry(
            "research/brief", "research/brief capability", ("1.0.0", "1.1.0")
        ),
    )
    older_artifact = updated.path / "v1" / "skills" / "research" / "brief" / "versions" / "1.0.0"
    assert (older_artifact / "package.gabskill").read_bytes() == older_archive
    assert (older_artifact / "package.gabskill.sig").read_bytes() == older_signature


def test_static_registry_update_rejects_tampered_existing_artifact(tmp_path: Path) -> None:
    private_key, public_key = _registry_signing_key()
    archive, signature = _make_signed_skill(
        tmp_path,
        name="research/brief",
        version="1.0.0",
        dependencies=(),
        private_key=private_key,
    )
    package = tmp_path / "brief-1.0.0.gabskill"
    package.write_bytes(archive)
    Path(f"{package}.sig").write_bytes(signature)
    initial = registry_module.build_static_skill_registry(
        [package], tmp_path / "registry-v1", trusted_keys={"publisher": public_key}
    )
    old_artifact = (
        initial.path
        / "v1"
        / "skills"
        / "research"
        / "brief"
        / "versions"
        / "1.0.0"
        / "package.gabskill"
    )
    old_artifact.write_bytes(old_artifact.read_bytes() + b"tampered")

    new_archive, new_signature = _make_signed_skill(
        tmp_path,
        name="research/brief",
        version="1.1.0",
        dependencies=(),
        private_key=private_key,
    )
    new_package = tmp_path / "brief-1.1.0.gabskill"
    new_package.write_bytes(new_archive)
    Path(f"{new_package}.sig").write_bytes(new_signature)
    output = tmp_path / "registry-v2"
    with pytest.raises(SkillRegistryError, match="identity or content validation"):
        registry_module.build_static_skill_registry(
            [new_package],
            output,
            trusted_keys={"publisher": public_key},
            existing_registry=initial.path,
        )
    assert not output.exists()


def test_static_registry_builder_requires_trust_and_never_overwrites_output(
    tmp_path: Path,
) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path)
    package = tmp_path / "publisher-package.gabskill"
    package.write_bytes(package_bytes)
    Path(f"{package}.sig").write_bytes(signature_bytes)

    with pytest.raises(SkillRegistryError, match="Trusted signing keys"):
        registry_module.build_static_skill_registry(
            [package], tmp_path / "untrusted", trusted_keys={}
        )
    assert not (tmp_path / "untrusted").exists()

    existing = tmp_path / "existing-registry"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(SkillRegistryError, match="new directory"):
        registry_module.build_static_skill_registry(
            [package], existing, trusted_keys={"publisher": public_key}
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_static_registry_builder_rejects_invalid_inputs_and_duplicate_versions(
    tmp_path: Path,
) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path)
    package = tmp_path / "publisher-package.gabskill"
    package.write_bytes(package_bytes)
    Path(f"{package}.sig").write_bytes(signature_bytes)

    with pytest.raises(SkillRegistryError, match="keys are invalid"):
        registry_module.build_static_skill_registry(
            [package], tmp_path / "bad-key", trusted_keys={"publisher": b"short"}
        )
    with pytest.raises(SkillRegistryError, match="sequence"):
        registry_module.build_static_skill_registry(
            str(package), tmp_path / "string-packages", trusted_keys={"publisher": public_key}
        )
    with pytest.raises(SkillRegistryError, match="iterable"):
        registry_module.build_static_skill_registry(
            7,
            tmp_path / "non-iterable",
            trusted_keys={"publisher": public_key},  # type: ignore[arg-type]
        )
    with pytest.raises(SkillRegistryError, match="between 1 and"):
        registry_module.build_static_skill_registry(
            [], tmp_path / "empty-packages", trusted_keys={"publisher": public_key}
        )
    with pytest.raises(SkillRegistryError, match="parent directory is unavailable"):
        registry_module.build_static_skill_registry(
            [package], tmp_path / "missing" / "registry", trusted_keys={"publisher": public_key}
        )

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file", encoding="utf-8")
    with pytest.raises(SkillRegistryError, match="parent must be a directory"):
        registry_module.build_static_skill_registry(
            [package], parent_file / "registry", trusted_keys={"publisher": public_key}
        )
    with pytest.raises(SkillRegistryError, match="duplicate skill version"):
        registry_module.build_static_skill_registry(
            [package, package], tmp_path / "duplicate", trusted_keys={"publisher": public_key}
        )
    assert not (tmp_path / "duplicate").exists()


def test_registry_path_and_bounded_file_contracts(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    (root / "v1" / "skills").mkdir(parents=True)
    ordinary = root / "v1" / "catalog.json"
    ordinary.write_bytes(b"{}")

    assert registry_module._read_bounded_registry_file(ordinary, limit=2) == b"{}"
    with pytest.raises(SkillRegistryError, match="bounded regular file"):
        registry_module._read_bounded_registry_file(ordinary, limit=1)
    with pytest.raises(SkillRegistryError, match="escaped its root"):
        registry_module._ensure_existing_registry_path(
            root, tmp_path / "outside", expect_directory=False
        )
    with pytest.raises(SkillRegistryError, match="regular file"):
        registry_module._ensure_existing_registry_path(root, root / "v1", expect_directory=False)
    registry_module._ensure_existing_registry_path(root, root / "v1", expect_directory=True)

    with pytest.raises(SkillRegistryError, match="unavailable"):
        registry_module._existing_static_registry_packages(
            tmp_path / "missing-registry", remaining_package_capacity=10
        )
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SkillRegistryError, match="must be a directory"):
        registry_module._existing_static_registry_packages(file_root, remaining_package_capacity=10)
    with pytest.raises(SkillRegistryError, match="invalid structure"):
        registry_module._existing_static_registry_packages(root, remaining_package_capacity=10)
    if hasattr(Path, "symlink_to"):
        root_link = tmp_path / "registry-link"
        root_link.symlink_to(root, target_is_directory=True)
        with pytest.raises(SkillRegistryError, match="must not be a symlink"):
            registry_module._existing_static_registry_packages(
                root_link, remaining_package_capacity=10
            )

    missing_parent = root / "missing"
    with pytest.raises(SkillRegistryError, match="unavailable"):
        registry_module._ensure_existing_registry_path(
            root, missing_parent / "artifact", expect_directory=False
        )
    with pytest.raises(SkillRegistryError, match="symlink or file"):
        registry_module._ensure_existing_registry_path(
            root, ordinary / "child", expect_directory=False
        )
    with pytest.raises(SkillRegistryError, match="unavailable"):
        registry_module._read_bounded_registry_file(root / "absent", limit=10)

    semver = registry_module._semver_sort_key
    assert semver("1.0.0-alpha.2") < semver("1.0.0-alpha.10")
    assert semver("1.0.0-alpha.10") < semver("1.0.0-beta")
    assert semver("1.0.0-beta") < semver("1.0.0")


def test_remote_install_with_dependencies_verifies_graph_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    import asyncio

    private_key, public_key = _registry_signing_key()
    package_artifacts = {
        "data/load": _make_signed_skill(
            tmp_path,
            name="data/load",
            version="2.1.0",
            dependencies=(),
            private_key=private_key,
        ),
        "analysis/report": _make_signed_skill(
            tmp_path,
            name="analysis/report",
            version="1.0.0",
            dependencies=("data/load@2.1.0",),
            private_key=private_key,
        ),
    }
    catalog = {
        "format": "gabby-skill-catalog",
        "format_version": 1,
        "skills": [
            {"name": "analysis/report", "description": "Report", "versions": ["1.0.0"]},
            {"name": "data/load", "description": "Load data", "versions": ["2.1.0"]},
        ],
    }
    artifact_responses: dict[str, bytes] = {"/v1/catalog.json": json.dumps(catalog).encode()}
    for skill_name, (archive, signature) in package_artifacts.items():
        encoded_name = "/".join(skill_name.split("/"))
        base = f"/v1/skills/{encoded_name}/versions/"
        version = "2.1.0" if skill_name == "data/load" else "1.0.0"
        package_path = f"{base}{version}/package.gabskill"
        artifact_responses[package_path] = archive
        artifact_responses[f"{package_path}.sig"] = signature

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = artifact_responses.get(request.url.path)
        return httpx.Response(200, content=payload) if payload is not None else httpx.Response(404)

    async def scenario() -> None:
        registry = tmp_path / "local-skill-registry"
        async with SkillRegistryClient(
            "https://registry.example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            installed = await client.install_with_dependencies(
                "analysis/report",
                "1.0.0",
                registry,
                trusted_keys={"publisher": public_key},
            )
            assert [(item.name, item.version) for item in installed] == [
                ("data/load", "2.1.0"),
                ("analysis/report", "1.0.0"),
            ]
            resumed = await client.install_with_dependencies(
                "analysis/report",
                "1.0.0",
                registry,
                trusted_keys={"publisher": public_key},
            )
            assert [item.sha256 for item in resumed] == [item.sha256 for item in installed]

    asyncio.run(scenario())


def test_remote_dependency_fetch_rejects_unpinned_reference_before_install(
    tmp_path: Path,
) -> None:
    import asyncio

    private_key, public_key = _registry_signing_key()
    package_bytes, signature_bytes = _make_signed_skill(
        tmp_path,
        name="analysis/report",
        version="1.0.0",
        dependencies=("data/load",),
        private_key=private_key,
    )
    catalog = {
        "format": "gabby-skill-catalog",
        "format_version": 1,
        "skills": [
            {"name": "analysis/report", "description": "Report", "versions": ["1.0.0"]},
            {"name": "data/load", "description": "Load data", "versions": ["1.0.0", "2.0.0"]},
        ],
    }
    package_path = "/v1/skills/analysis/report/versions/1.0.0/package.gabskill"
    responses = {
        "/v1/catalog.json": json.dumps(catalog).encode(),
        package_path: package_bytes,
        f"{package_path}.sig": signature_bytes,
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = responses.get(request.url.path)
        return httpx.Response(200, content=payload) if payload is not None else httpx.Response(404)

    async def scenario() -> None:
        registry = tmp_path / "must-stay-empty"
        async with SkillRegistryClient(
            "https://registry.example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(SkillRegistryError, match="exact skill-id@version pins"):
                await client.install_with_dependencies(
                    "analysis/report",
                    "1.0.0",
                    registry,
                    trusted_keys={"publisher": public_key},
                )
        assert not registry.exists()

    asyncio.run(scenario())


def test_skill_catalog_build_cli_uses_host_trusted_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path)
    package = tmp_path / "publisher-package.gabskill"
    package.write_bytes(package_bytes)
    Path(f"{package}.sig").write_bytes(signature_bytes)
    output = tmp_path / "generated-site"
    monkeypatch.setattr(
        cli_module,
        "_resolve_trusted_skill_keys",
        lambda _entries, _directory, required: {"publisher": public_key} if required else None,
    )

    result = cli_module.main(
        [
            "skill",
            "catalog",
            "build",
            "--package",
            str(package),
            "--trusted-key-dir",
            str(tmp_path / "publisher-keys"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert (output / "v1" / "catalog.json").is_file()
    assert "Built static registry with 1 skill and 1 version" in capsys.readouterr().out


def test_remote_registry_install_requires_signature_and_identity_match(tmp_path: Path) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path)
    catalog_bytes = json.dumps(_CATALOG).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("catalog.json"):
            return httpx.Response(200, content=catalog_bytes)
        if request.url.path.endswith("package.gabskill.sig"):
            return httpx.Response(200, content=signature_bytes)
        assert request.url.path.endswith("package.gabskill")
        assert request.url.path == "/v1/skills/support/triage/versions/1.2.0/package.gabskill"
        return httpx.Response(200, content=package_bytes)

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(handler)
        )
        try:
            installed = await client.install(
                "support/triage",
                "1.2.0",
                tmp_path / "installed",
                trusted_keys={"publisher": public_key},
            )
        finally:
            await client.aclose()
        assert installed.name == "support/triage"
        assert installed.version == "1.2.0"
        assert installed.signing_key_id == "publisher"
        assert (installed.path / "skill.yaml").is_file()

    import asyncio

    asyncio.run(scenario())


def test_remote_registry_rejects_untrusted_or_mismatched_packages(tmp_path: Path) -> None:
    package_bytes, signature_bytes, public_key = _signed_archive(tmp_path, name="other-skill")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("package.gabskill.sig"):
            return httpx.Response(200, content=signature_bytes)
        return httpx.Response(200, content=package_bytes)

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(handler)
        )
        local_registry = tmp_path / "rejected-install"
        try:
            with pytest.raises(SkillRegistryError, match="identity does not match"):
                await client.install(
                    "support/triage",
                    "1.2.0",
                    local_registry,
                    trusted_keys={"publisher": public_key},
                )
        finally:
            await client.aclose()
        assert not local_registry.exists()

    import asyncio

    asyncio.run(scenario())


def test_remote_registry_detects_package_change_between_validation_and_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    local_registry = tmp_path / "local-registry"
    unexpected = local_registry / "unexpected" / "1.0.0"

    def install_stub(
        _package: Path,
        _registry: str | Path,
        *,
        require_signature: bool,
        trusted_keys: object,
    ) -> SkillPackageInfo:
        nonlocal calls
        assert require_signature
        assert trusted_keys == {"publisher": b"k" * 32}
        calls += 1
        if calls == 1:
            return SkillPackageInfo("requested", "1.0.0", tmp_path / "scratch", "same", "publisher")
        unexpected.mkdir(parents=True)
        return SkillPackageInfo("unexpected", "1.0.0", unexpected, "changed", "publisher")

    monkeypatch.setattr(registry_module, "install_skill", install_stub)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = b"signature" if request.url.path.endswith(".sig") else b"archive"
        return httpx.Response(200, content=body)

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(SkillRegistryError, match="changed during installation"):
                await client.install(
                    "requested",
                    "1.0.0",
                    local_registry,
                    trusted_keys={"publisher": b"k" * 32},
                )
        finally:
            await client.aclose()

    import asyncio

    asyncio.run(scenario())
    assert not unexpected.exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://registry.example.test",
        "https://user:password@registry.example.test",
        "https://registry.example.test/?token=secret",
        "https://registry.example.test/#fragment",
    ],
)
def test_registry_rejects_insecure_or_credential_bearing_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS|registry base URL"):
        SkillRegistryClient(base_url)
    assert SkillRegistryClient("http://127.0.0.1:8000").base_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize("timeout", [0, -1, 301, float("nan"), True])
def test_registry_rejects_invalid_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        SkillRegistryClient("https://registry.example.test", timeout_seconds=timeout)


def test_registry_rejects_invalid_headers_and_proxy_policy() -> None:
    with pytest.raises(ValueError, match="string-to-string"):
        SkillRegistryClient("https://registry.example.test", headers={"Authorization": 42})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="trust_env"):
        SkillRegistryClient("https://registry.example.test", trust_env=1)  # type: ignore[arg-type]


def test_registry_rejects_redirects_duplicate_catalog_keys_and_oversize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def redirect_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://elsewhere.example.test/"})

    async def redirect_scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(redirect_handler)
        )
        try:
            with pytest.raises(SkillRegistryError, match="redirects are not allowed"):
                await client.catalog()
        finally:
            await client.aclose()

    asyncio.run(redirect_scenario())

    async def duplicate_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"format":"gabby-skill-catalog","format":"bad",'
            b'"format_version":1,"skills":[]}',
        )

    async def duplicate_scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(duplicate_handler)
        )
        try:
            with pytest.raises(SkillRegistryError, match="invalid JSON"):
                await client.catalog()
        finally:
            await client.aclose()

    asyncio.run(duplicate_scenario())

    monkeypatch.setattr(registry_module, "_MAX_CATALOG_BYTES", 4)

    async def oversized_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"12345")

    async def oversized_scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(oversized_handler)
        )
        try:
            with pytest.raises(SkillRegistryError, match="exceeded its size limit"):
                await client.catalog()
        finally:
            await client.aclose()

    asyncio.run(oversized_scenario())


@pytest.mark.parametrize(
    "catalog",
    [
        [],
        {"format": "wrong", "format_version": 1, "skills": []},
        {"format": "gabby-skill-catalog", "format_version": True, "skills": []},
        {
            "format": "gabby-skill-catalog",
            "format_version": 1,
            "skills": [{"name": "../bad", "description": "bad", "versions": ["1.0.0"]}],
        },
        {
            "format": "gabby-skill-catalog",
            "format_version": 1,
            "skills": [
                {"name": "repeat", "description": "one", "versions": ["1.0.0"]},
                {"name": "repeat", "description": "two", "versions": ["2.0.0"]},
            ],
        },
        {
            "format": "gabby-skill-catalog",
            "format_version": 1,
            "skills": [{"name": "skill", "description": "x", "versions": ["1.0.0", "1.0.0"]}],
        },
        {
            "format": "gabby-skill-catalog",
            "format_version": 1,
            "skills": [{"name": "skill", "description": "x", "versions": ["1.0.0-alpha.01"]}],
        },
    ],
)
def test_registry_catalog_schema_rejects_unsafe_entries(catalog: object) -> None:
    with pytest.raises(SkillRegistryError):
        registry_module._parse_catalog(catalog)


def test_registry_catalog_optional_freshness_validation() -> None:
    now = datetime(2026, 10, 3, 12, tzinfo=UTC)
    fresh = {
        **_CATALOG,
        "generated_at": (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z"),
    }
    assert registry_module._parse_catalog(fresh, max_age_seconds=60, now=now)

    stale = {
        **fresh,
        "generated_at": (now - timedelta(seconds=61)).isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(SkillRegistryError, match="stale"):
        registry_module._parse_catalog(stale, max_age_seconds=60, now=now)

    future = {
        **fresh,
        "generated_at": (now + timedelta(minutes=6)).isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(SkillRegistryError, match="in the future"):
        registry_module._parse_catalog(future, now=now)

    with pytest.raises(SkillRegistryError, match="no freshness timestamp"):
        registry_module._parse_catalog(_CATALOG, max_age_seconds=60, now=now)

    naive = {**fresh, "generated_at": "2026-10-03T12:00:00"}
    with pytest.raises(SkillRegistryError, match="include a timezone"):
        registry_module._parse_catalog(naive, now=now)


def test_registry_client_rejects_stale_catalog() -> None:
    import asyncio

    catalog = {
        **_CATALOG,
        "generated_at": "2026-01-01T00:00:00Z",
    }

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=catalog)

    async def scenario() -> None:
        async with SkillRegistryClient(
            "https://registry.example.test",
            max_catalog_age_seconds=60,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(SkillRegistryError, match="stale"):
                await client.catalog()

    asyncio.run(scenario())


def test_registry_catalog_description_and_version_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "_MAX_DESCRIPTION_BYTES", 2)
    with pytest.raises(SkillRegistryError, match="description exceeds"):
        registry_module._parse_catalog(
            {
                "format": "gabby-skill-catalog",
                "format_version": 1,
                "skills": [{"name": "skill", "description": "long", "versions": ["1.0.0"]}],
            }
        )


def test_registry_methods_reject_invalid_queries_and_missing_skill() -> None:
    import asyncio

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_CATALOG)

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(ValueError, match="query"):
                await client.search("   ")
            with pytest.raises(ValueError, match="SemVer"):
                await client.versions("../escape")
            with pytest.raises(SkillRegistryError, match="not found"):
                await client.versions("absent")
            with pytest.raises(ValueError, match="SemVer"):
                await client.install("support/triage", "latest", "/tmp/unused", trusted_keys={})
            with pytest.raises(SkillRegistryError, match="Trusted signing keys"):
                await client.install("support/triage", "1.2.0", "/tmp/unused", trusted_keys={})
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_registry_request_total_deadline_and_http_error_are_sanitized() -> None:
    import asyncio

    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_CATALOG)

    async def timeout_scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test",
            timeout_seconds=0.01,
            transport=httpx.MockTransport(slow_handler),
        )
        try:
            with pytest.raises(SkillRegistryError, match="deadline"):
                await client.catalog()
        finally:
            await client.aclose()

    asyncio.run(timeout_scenario())

    async def error_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"secret backend detail")

    async def error_scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(error_handler)
        )
        try:
            with pytest.raises(SkillRegistryError, match="HTTP status 404") as exc_info:
                await client.catalog()
            assert "secret" not in str(exc_info.value)
        finally:
            await client.aclose()

    asyncio.run(error_scenario())


def test_registry_download_has_a_total_deadline(tmp_path: Path) -> None:
    import asyncio

    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=b"artifact")

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test",
            timeout_seconds=0.01,
            transport=httpx.MockTransport(slow_handler),
        )
        destination = tmp_path / "partial-artifact"
        try:
            with pytest.raises(SkillRegistryError, match="download exceeded its deadline"):
                await client._download("/artifact", destination, limit=100)
        finally:
            await client.aclose()
        assert not destination.exists()

    asyncio.run(scenario())


def test_registry_download_does_not_overwrite_existing_destination(tmp_path: Path) -> None:
    import asyncio

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"untrusted replacement")

    async def scenario() -> None:
        client = SkillRegistryClient(
            "https://registry.example.test", transport=httpx.MockTransport(handler)
        )
        destination = tmp_path / "existing-artifact"
        destination.write_bytes(b"trusted prior contents")
        try:
            with pytest.raises(SkillRegistryError, match="download failed"):
                await client._download("/artifact", destination, limit=100)
        finally:
            await client.aclose()
        assert destination.read_bytes() == b"trusted prior contents"

    asyncio.run(scenario())

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
"""Bounded local skill package contract tests."""

from __future__ import annotations

import base64
import builtins
import hashlib
import io
import json
import os
import shutil
import stat
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import gabby.cli as cli_module
import gabby.skill_packages as package_module
import gabby.skill_signing as signing_module
from gabby import (
    SkillPackageError,
    audit_skill_registry,
    inspect_skill_package,
    install_skill,
    pack_skill,
    sign_skill_package,
    uninstall_skill,
    verify_skill_package_signature,
)
from gabby.config import load_skill


def _rewrite_package_manifest(
    source: Path,
    destination: Path,
    transform: Callable[[Any], Any],
) -> None:
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in original.infolist():
            data = original.read(entry.filename)
            if entry.filename == ".gabby-skill.json":
                data = json.dumps(transform(json.loads(data))).encode("utf-8")
            changed.writestr(entry.filename, data)


def _skill_directory(root: Path, *, name: str = "support/triage") -> Path:
    skill = root / "skill-source"
    (skill / "examples").mkdir(parents=True)
    (skill / "skill.yaml").write_text(
        f"name: {name}\n"
        "version: 1.2.3\n"
        "description: Triage support requests\n"
        "instructions_file: instructions.md\n"
        "dependencies: []\n",
        encoding="utf-8",
    )
    (skill / "instructions.md").write_text(
        "Classify the issue and cite evidence.\n", encoding="utf-8"
    )
    (skill / "examples" / "sample.txt").write_text("Example resource\n", encoding="utf-8")
    return skill


def _ed25519_keypair() -> tuple[bytes, bytes]:
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def test_pack_is_deterministic_and_install_preserves_versioned_package(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path)
    output = tmp_path / "packages"
    output.mkdir()
    first_archive = output / "first.gabskill"
    second_archive = output / "second.gabskill"

    first = pack_skill(source, first_archive)
    second = pack_skill(source, second_archive)
    assert first.name == "support/triage"
    assert first.version == "1.2.3"
    assert first.sha256 == second.sha256
    assert first_archive.read_bytes() == second_archive.read_bytes()

    registry = tmp_path / "registry"
    installed = install_skill(first_archive, registry)
    assert installed.path == registry / "support" / "triage" / "1.2.3"
    assert installed.sha256 == first.sha256
    loaded = load_skill(installed.path / "skill.yaml")
    assert loaded.name == "support/triage"
    assert loaded.version == "1.2.3"
    assert loaded.instructions == "Classify the issue and cite evidence.\n"
    assert (installed.path / "examples" / "sample.txt").read_text(encoding="utf-8") == (
        "Example resource\n"
    )


def test_skill_schemas_survive_package_inspection_and_installation(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="support/triage")
    manifest = source / "skill.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + "input_schema:\n  type: object\n  required: [task]\n"
        + "output_schema:\n  type: object\n  required: [category]\n",
        encoding="utf-8",
    )
    expected_input = {"type": "object", "required": ["task"]}
    expected_output = {"type": "object", "required": ["category"]}
    archive = tmp_path / "triage.gabskill"
    pack_skill(source, archive)

    inspection = inspect_skill_package(archive)
    installed = install_skill(archive, tmp_path / "registry")
    loaded = load_skill(installed.path / "skill.yaml")

    assert inspection.input_schema == expected_input
    assert inspection.output_schema == expected_output
    assert loaded.input_schema == expected_input
    assert loaded.output_schema == expected_output


def test_inspect_skill_package_reports_capabilities_and_signature_state(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="research/source-check")
    (source / "skill.yaml").write_text(
        "name: research/source-check\n"
        "version: 1.2.3\n"
        "description: Verify primary sources\n"
        "instructions_file: instructions.md\n"
        "tools: [search_web]\n"
        "knowledge: [reports]\n"
        "dependencies: []\n"
        "verification: [citations]\n",
        encoding="utf-8",
    )
    archive = tmp_path / "source-check.gabskill"
    package = pack_skill(source, archive)

    unsigned = inspect_skill_package(archive)
    assert unsigned.name == "research/source-check"
    assert unsigned.version == "1.2.3"
    assert unsigned.tools == ("search_web",)
    assert unsigned.knowledge == ("reports",)
    assert unsigned.verification == ("citations",)
    assert {entry.path for entry in unsigned.files} >= {"skill.yaml", "instructions.md"}
    assert unsigned.signature_present is False
    assert unsigned.signature_verified is False

    private_key, public_key = _ed25519_keypair()
    sign_skill_package(archive, key_id="research-publisher", private_key=private_key)
    signed = inspect_skill_package(
        archive,
        trusted_keys={"research-publisher": public_key},
        require_signature=True,
    )
    assert signed.archive_sha256 == package.sha256
    assert signed.signature_present is True
    assert signed.signature_verified is True
    assert signed.signing_key_id == "research-publisher"
    assert not (tmp_path / "research" / "source-check").exists()


def test_inspect_skill_package_rejects_untrusted_and_tampered_archives(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    archive = tmp_path / "review.gabskill"
    pack_skill(source, archive)
    private_key, public_key = _ed25519_keypair()
    sign_skill_package(archive, key_id="publisher", private_key=private_key)

    with pytest.raises(SkillPackageError, match="not trusted"):
        inspect_skill_package(
            archive,
            trusted_keys={"different-publisher": public_key},
            require_signature=True,
        )

    tampered = tmp_path / "tampered.gabskill"
    with (
        zipfile.ZipFile(archive) as original,
        zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "instructions.md":
                data = data.replace(b"Classify", b"Altered!")
            changed.writestr(item.filename, data)
    with pytest.raises(SkillPackageError, match="checksum mismatch"):
        inspect_skill_package(tampered)


def test_skill_inspect_cli_emits_safe_json_and_can_require_trust(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _skill_directory(tmp_path, name="support/review")
    archive = tmp_path / "review.gabskill"
    pack_skill(source, archive)
    private_key, public_key = _ed25519_keypair()
    sign_skill_package(archive, key_id="support-publisher", private_key=private_key)
    trust_directory = tmp_path / "trusted"
    trust_directory.mkdir()
    (trust_directory / "support-publisher.pub").write_bytes(public_key)

    exit_code = cli_module.main(
        [
            "skill",
            "inspect",
            str(archive),
            "--trusted-key-dir",
            str(trust_directory),
            "--require-signature",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["name"] == "support/review"
    assert result["signature_verified"] is True
    assert result["signing_key_id"] == "support-publisher"


def test_install_rejects_modified_package_content_without_partial_install(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    archive_path = tmp_path / "review.gabskill"
    pack_skill(source, archive_path)
    tampered = tmp_path / "tampered.gabskill"
    with (
        zipfile.ZipFile(archive_path) as original,
        zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "instructions.md":
                data = data.replace(b"Classify", b"Changed!")
            changed.writestr(item.filename, data)

    registry = tmp_path / "registry"
    with pytest.raises(SkillPackageError, match="checksum mismatch"):
        install_skill(tampered, registry)
    assert list(registry.iterdir()) == []


def test_pack_and_install_reject_paths_that_collide_on_other_filesystems(
    tmp_path: Path,
) -> None:
    source = _skill_directory(tmp_path, name="review")
    (source / "Readme.md").write_text("first", encoding="utf-8")
    (source / "README.md").write_text("second", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="case-insensitive or Unicode-normalizing"):
        pack_skill(source, tmp_path / "case-collision.gabskill")

    contents = {
        "skill.yaml": (source / "skill.yaml").read_bytes(),
        "instructions.md": (source / "instructions.md").read_bytes(),
        "Readme.md": b"first",
        "README.md": b"second",
    }
    manifest = {
        "format": package_module._PACKAGE_FORMAT,
        "format_version": package_module._PACKAGE_FORMAT_VERSION,
        "skill": {"name": "review", "version": "1.2.3"},
        "files": [
            {
                "path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in contents.items()
        ],
    }
    malicious = tmp_path / "portable-collision.gabskill"
    with zipfile.ZipFile(malicious, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            package_module._PACKAGE_MANIFEST,
            json.dumps(manifest, separators=(",", ":")),
        )
        for name, data in contents.items():
            archive.writestr(name, data)

    registry = tmp_path / "collision-registry"
    with pytest.raises(SkillPackageError, match="case-insensitive or Unicode-normalizing"):
        install_skill(malicious, registry)
    assert list(registry.iterdir()) == []


def test_portable_path_collision_validation_normalizes_unicode_and_hierarchy() -> None:
    with pytest.raises(SkillPackageError, match="case-insensitive or Unicode-normalizing"):
        package_module._validate_portable_path_collisions(["café.md", "cafe\u0301.md"])
    with pytest.raises(SkillPackageError, match="case-insensitive or Unicode-normalizing"):
        package_module._validate_portable_path_collisions(["Readme", "README/child.txt"])


def test_ed25519_skill_package_signature_and_required_install(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)

    signature = sign_skill_package(
        archive,
        key_id="gabby-review-publisher",
        private_key=private_key,
    )
    assert signature == Path(f"{archive}.sig")
    assert (
        verify_skill_package_signature(
            archive,
            trusted_keys={"gabby-review-publisher": public_key},
        )
        == "gabby-review-publisher"
    )

    registry = tmp_path / "trusted-registry"
    installed = install_skill(
        archive,
        registry,
        require_signature=True,
        trusted_keys={"gabby-review-publisher": public_key},
    )
    assert installed.signing_key_id == "gabby-review-publisher"
    assert (installed.path / "skill.yaml").is_file()
    provenance = json.loads((installed.path / ".gabby-install.json").read_text(encoding="utf-8"))
    assert provenance["archive_sha256"] == installed.sha256
    assert provenance["signing_key_id"] == "gabby-review-publisher"
    assert provenance["signature_verified"] is True
    assert provenance["format_version"] == 2
    assert provenance["signature"]["format_version"] == 2


def test_skill_registry_audit_reports_signed_unsigned_legacy_and_revoked_installs(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    registry = tmp_path / "registry"
    signed_source = _skill_directory(tmp_path, name="support/signed")
    signed_archive = tmp_path / "signed.gabskill"
    pack_skill(signed_source, signed_archive)
    sign_skill_package(signed_archive, key_id="publisher-one", private_key=private_key)
    signed = install_skill(
        signed_archive,
        registry,
        require_signature=True,
        trusted_keys={"publisher-one": public_key},
    )

    unsigned_source = _skill_directory(tmp_path / "unsigned", name="support/unsigned")
    unsigned_archive = tmp_path / "unsigned.gabskill"
    pack_skill(unsigned_source, unsigned_archive)
    unsigned = install_skill(unsigned_archive, registry)

    initial = {record.name: record for record in audit_skill_registry(registry)}
    assert initial["support/signed"].status == "recorded-signed"
    assert initial["support/unsigned"].status == "unsigned"

    revoked = audit_skill_registry(registry, revoked_key_ids={"publisher-one"})
    assert {record.name: record.status for record in revoked} == {
        "support/signed": "revoked",
        "support/unsigned": "unsigned",
    }

    (signed.path / ".gabby-install.json").unlink()
    legacy = {record.name: record for record in audit_skill_registry(registry)}
    assert legacy["support/signed"].status == "unknown"
    assert legacy["support/unsigned"].status == "unsigned"
    assert unsigned.signing_key_id is None


def test_skill_registry_audit_verifies_installed_content_and_detects_tampering(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="support/audited")
    archive = tmp_path / "audited.gabskill"
    pack_skill(source, archive)
    sign_skill_package(archive, key_id="publisher-one", private_key=private_key)
    installed = install_skill(
        archive,
        tmp_path / "registry",
        require_signature=True,
        trusted_keys={"publisher-one": public_key},
    )

    records = audit_skill_registry(
        tmp_path / "registry", trusted_keys={"publisher-one": public_key}
    )
    assert records[0].status == "verified"

    (installed.path / "instructions.md").write_text("Tampered instructions\n", encoding="utf-8")
    records = audit_skill_registry(
        tmp_path / "registry", trusted_keys={"publisher-one": public_key}
    )
    assert records[0].status == "invalid"


def test_skill_registry_audit_reports_v2_signature_without_trusted_key_as_unknown(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="support/untrusted-audit")
    archive = tmp_path / "untrusted-audit.gabskill"
    pack_skill(source, archive)
    sign_skill_package(archive, key_id="publisher-one", private_key=private_key)
    install_skill(
        archive,
        tmp_path / "registry",
        require_signature=True,
        trusted_keys={"publisher-one": public_key},
    )

    records = audit_skill_registry(tmp_path / "registry", trusted_keys={"other": public_key})
    assert records[0].status == "unknown"


def test_v1_package_signature_remains_installable_and_marked_install_time_only(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="support/legacy-signature")
    archive = tmp_path / "legacy-signature.gabskill"
    pack_skill(source, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(
        signing_module._signature_message("publisher-one", digest)
    )
    Path(f"{archive}.sig").write_text(
        json.dumps(
            {
                "format": "gabby-skill-signature",
                "format_version": 1,
                "key_id": "publisher-one",
                "archive_sha256": digest,
                "signature": base64.b64encode(signature).decode("ascii"),
            },
            separators=(",", ":"),
        ),
        encoding="ascii",
    )

    installed = install_skill(
        archive,
        tmp_path / "registry",
        require_signature=True,
        trusted_keys={"publisher-one": public_key},
    )
    records = audit_skill_registry(
        tmp_path / "registry", trusted_keys={"publisher-one": public_key}
    )
    assert installed.signing_key_id == "publisher-one"
    assert records[0].status == "recorded-signed"


def test_skill_registry_audit_fails_closed_on_invalid_provenance(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="support/audited")
    archive = tmp_path / "audited.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, tmp_path / "registry")
    (installed.path / ".gabby-install.json").write_text(
        '{"format":"gabby-skill-install","format_version":1,'
        '"archive_sha256":"bad","signing_key_id":null,"signature_verified":false}',
        encoding="utf-8",
    )

    records = audit_skill_registry(tmp_path / "registry")
    assert len(records) == 1
    assert records[0].status == "invalid"


def test_skill_registry_audit_reports_legacy_layout_as_unknown(tmp_path: Path) -> None:
    legacy = tmp_path / "registry" / "legacy-skill"
    legacy.mkdir(parents=True)
    (legacy / "skill.yaml").write_text(
        "name: legacy-skill\nversion: 1.0.0\ndescription: Existing skill\n",
        encoding="utf-8",
    )

    records = audit_skill_registry(tmp_path / "registry")
    assert len(records) == 1
    assert records[0].name == "legacy-skill"
    assert records[0].status == "unknown"


@pytest.mark.parametrize("encoded", [b"[]", b"{", b'{"format":"bad"}'])
def test_skill_registry_audit_marks_malformed_provenance_invalid(
    tmp_path: Path, encoded: bytes
) -> None:
    source = _skill_directory(tmp_path, name="support/malformed")
    archive = tmp_path / "malformed.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, tmp_path / "registry")
    (installed.path / ".gabby-install.json").write_bytes(encoded)

    records = audit_skill_registry(tmp_path / "registry")
    assert records[0].status == "invalid"


def test_skill_registry_audit_handles_symlinks_and_ignores_nested_skill_resources(
    tmp_path: Path,
) -> None:
    source = _skill_directory(tmp_path, name="support/audited")
    archive = tmp_path / "audited.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, tmp_path / "registry")
    nested = installed.path / "examples" / "nested"
    nested.mkdir()
    (nested / "skill.yaml").write_text(
        "name: nested-example\nversion: 1.0.0\ndescription: Nested resource\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "registry" / "linked").symlink_to(outside, target_is_directory=True)

    records = audit_skill_registry(tmp_path / "registry")
    assert [(record.name, record.status) for record in records] == [
        ("linked", "invalid"),
        ("support/audited", "unsigned"),
    ]


def test_skill_registry_audit_rejects_bad_revoked_ids_roots_and_entry_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SkillPackageError, match="key IDs"):
        audit_skill_registry(tmp_path, revoked_key_ids={"bad key"})
    with pytest.raises(SkillPackageError, match="unavailable"):
        audit_skill_registry(tmp_path / "missing")

    real_registry = tmp_path / "registry"
    real_registry.mkdir()
    (real_registry / "entry").touch()
    linked_registry = tmp_path / "registry-link"
    linked_registry.symlink_to(real_registry, target_is_directory=True)
    with pytest.raises(SkillPackageError, match="must not be a symlink"):
        audit_skill_registry(linked_registry)

    monkeypatch.setattr(package_module, "_MAX_REGISTRY_AUDIT_ENTRIES", 0)
    with pytest.raises(SkillPackageError, match="entry limit"):
        audit_skill_registry(real_registry)


@pytest.mark.parametrize("use_symlink", [False, True])
def test_skill_registry_audit_rejects_unbounded_or_symlink_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_symlink: bool
) -> None:
    source = _skill_directory(tmp_path, name="support/bounded")
    archive = tmp_path / "bounded.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, tmp_path / "registry")
    provenance = installed.path / ".gabby-install.json"
    if use_symlink:
        replacement = tmp_path / "provenance-copy"
        replacement.write_bytes(provenance.read_bytes())
        provenance.unlink()
        provenance.symlink_to(replacement)
    else:
        monkeypatch.setattr(package_module, "_MAX_INSTALL_PROVENANCE_BYTES", 8)

    records = audit_skill_registry(tmp_path / "registry")
    assert records[0].status == "invalid"


@pytest.mark.parametrize("reserved_name", [".gabby-install.json", ".GABBY-INSTALL.JSON"])
def test_skill_package_cannot_claim_reserved_install_provenance(
    tmp_path: Path, reserved_name: str
) -> None:
    source = _skill_directory(tmp_path, name="support/reserved")
    (source / reserved_name).write_text("{}", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="reserved file name|collide"):
        pack_skill(source, tmp_path / "reserved.gabskill")


def test_uninstall_skill_removes_only_the_requested_version(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="support/versioned")
    first_archive = tmp_path / "first.gabskill"
    pack_skill(source, first_archive)
    registry = tmp_path / "registry"
    first = install_skill(first_archive, registry)

    manifest = source / "skill.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("1.2.3", "1.2.4"), encoding="utf-8"
    )
    second_archive = tmp_path / "second.gabskill"
    pack_skill(source, second_archive)
    second = install_skill(second_archive, registry)

    uninstall_skill(registry, "support/versioned", "1.2.3")
    assert not first.path.exists()
    assert (second.path / "skill.yaml").is_file()


def test_uninstall_skill_rejects_invalid_or_mismatched_targets(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    source = _skill_directory(tmp_path, name="support/installed")
    archive = tmp_path / "installed.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, registry)

    with pytest.raises(SkillPackageError, match="name is invalid"):
        uninstall_skill(registry, "../installed", "1.2.3")
    with pytest.raises(SkillPackageError, match="exact semantic version"):
        uninstall_skill(registry, "support/installed", "latest")

    mismatched = registry / "support" / "wrong" / "1.0.0"
    mismatched.mkdir(parents=True)
    (mismatched / "skill.yaml").write_text(
        "name: support/other\nversion: 1.0.0\ndescription: Mismatched\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillPackageError, match="identity does not match"):
        uninstall_skill(registry, "support/wrong", "1.0.0")
    assert (installed.path / "skill.yaml").is_file()


def test_uninstall_skill_rejects_symlinked_registry_paths(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (registry / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(SkillPackageError, match="must not contain symlinks"):
        uninstall_skill(registry, "linked", "1.0.0")


def test_uninstall_skill_rejects_symlinked_registry_root(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    registry_link = tmp_path / "registry-link"
    try:
        registry_link.symlink_to(registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(SkillPackageError, match="registry must not be a symlink"):
        uninstall_skill(registry_link, "support/skill", "1.0.0")


@pytest.mark.parametrize(
    ("root_kind", "expected_error"),
    [("missing", "unavailable"), ("file", "must be a directory")],
)
def test_uninstall_skill_rejects_unavailable_or_non_directory_registry(
    tmp_path: Path, root_kind: str, expected_error: str
) -> None:
    registry = tmp_path / "registry"
    if root_kind == "file":
        registry.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SkillPackageError, match=expected_error):
        uninstall_skill(registry, "support/skill", "1.0.0")


@pytest.mark.parametrize(
    ("target_kind", "expected_error"),
    [
        ("file", "path is invalid"),
        ("empty", "manifest is unavailable"),
        ("invalid-manifest", "could not be validated"),
    ],
)
def test_uninstall_skill_rejects_invalid_installation_targets(
    tmp_path: Path, target_kind: str, expected_error: str
) -> None:
    registry = tmp_path / "registry"
    target = registry / "support" / "invalid" / "1.0.0"
    target.parent.mkdir(parents=True)
    if target_kind == "file":
        target.write_text("not a directory", encoding="utf-8")
    else:
        target.mkdir()
        if target_kind == "invalid-manifest":
            (target / "skill.yaml").write_text("[invalid", encoding="utf-8")

    with pytest.raises(SkillPackageError, match=expected_error):
        uninstall_skill(registry, "support/invalid", "1.0.0")


def test_uninstall_skill_reports_removal_errors_without_hiding_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="support/remove-error")
    archive = tmp_path / "remove-error.gabskill"
    pack_skill(source, archive)
    installed = install_skill(archive, tmp_path / "registry")

    def fail_remove(_path: Path) -> None:
        raise OSError("private filesystem detail")

    monkeypatch.setattr(shutil, "rmtree", fail_remove)
    with pytest.raises(SkillPackageError, match="Could not remove") as exc_info:
        uninstall_skill(tmp_path / "registry", "support/remove-error", "1.2.3")
    assert "private filesystem detail" not in str(exc_info.value)
    assert (installed.path / "skill.yaml").is_file()


def test_skill_signature_rejects_unknown_wrong_and_changed_packages(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    _, other_public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    signature = sign_skill_package(
        archive,
        key_id="publisher-a",
        private_key=private_key,
    )

    with pytest.raises(SkillPackageError, match="not trusted"):
        verify_skill_package_signature(archive, trusted_keys={"publisher-b": public_key})
    with pytest.raises(SkillPackageError, match="verification failed"):
        verify_skill_package_signature(archive, trusted_keys={"publisher-a": other_public_key})
    changed = tmp_path / "changed.gabskill"
    changed.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(SkillPackageError, match="does not match the package digest"):
        verify_skill_package_signature(
            changed,
            trusted_keys={"publisher-a": public_key},
            signature_path=signature,
        )

    unsigned = tmp_path / "unsigned.gabskill"
    pack_skill(source, unsigned)
    registry = tmp_path / "unsigned-registry"
    with pytest.raises(SkillPackageError, match="signature"):
        install_skill(
            unsigned, registry, require_signature=True, trusted_keys={"publisher-a": public_key}
        )
    assert not registry.exists()


def test_skill_signature_rejects_invalid_trust_sources_and_sidecars(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    signature = sign_skill_package(archive, key_id="publisher", private_key=private_key)

    with pytest.raises(SkillPackageError, match="At least one trusted"):
        verify_skill_package_signature(archive, trusted_keys={})
    with pytest.raises(SkillPackageError, match="unavailable"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=tmp_path / "absent.sig"
        )

    sidecar_link = tmp_path / "linked.sig"
    sidecar_link.symlink_to(signature)
    with pytest.raises(SkillPackageError, match="must not be a symlink"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=sidecar_link
        )

    malformed = tmp_path / "malformed.sig"
    malformed.write_text("not JSON", encoding="ascii")
    with pytest.raises(SkillPackageError, match="document is invalid"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=malformed
        )

    invalid_signature = tmp_path / "invalid.sig"
    document = json.loads(signature.read_text(encoding="ascii"))
    document["signature"] = base64.b64encode(bytes(64)).decode("ascii")
    invalid_signature.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(SkillPackageError, match="verification failed"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=invalid_signature
        )


def test_skill_signature_rejects_oversized_and_replaced_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    signature = sign_skill_package(archive, key_id="publisher", private_key=private_key)

    oversized = tmp_path / "oversized.sig"
    oversized.write_bytes(b"x" * (signing_module._MAX_SIGNATURE_BYTES + 1))
    with pytest.raises(SkillPackageError, match="bounded regular file"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=oversized
        )

    replaceable = tmp_path / "replaceable.sig"
    replaceable.write_bytes(signature.read_bytes())
    replacement = tmp_path / "replacement.sig"
    original_open = os.open

    def replace_sidecar_before_open(
        path: str | os.PathLike[str],
        flags: int,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if Path(path) == replaceable:
            replacement.write_bytes(replaceable.read_bytes())
            replacement.replace(replaceable)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_sidecar_before_open)
    with pytest.raises(SkillPackageError, match="changed while it was opened"):
        verify_skill_package_signature(
            archive, trusted_keys={"publisher": public_key}, signature_path=replaceable
        )


def test_skill_signature_reports_missing_optional_crypto_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    signature = sign_skill_package(archive, key_id="publisher", private_key=private_key)
    original_import = builtins.__import__

    def import_without_crypto(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cryptography"):
            raise ImportError("optional test dependency disabled")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_crypto)
    with pytest.raises(SkillPackageError, match=r"\[skill-signing\].*verify"):
        verify_skill_package_signature(archive, trusted_keys={"publisher": public_key})
    with pytest.raises(SkillPackageError, match=r"\[skill-signing\].*sign"):
        sign_skill_package(
            archive,
            key_id="another-publisher",
            private_key=private_key,
            signature_path=tmp_path / "second.sig",
        )
    assert signature.is_file()


def test_skill_signature_writer_cleans_temporary_file_on_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, _ = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)

    def fail_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes], destination: Any
    ) -> None:
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(SkillPackageError, match="Could not write skill signature"):
        sign_skill_package(archive, key_id="publisher", private_key=private_key)
    assert list(tmp_path.glob(".gabby-signature-*.tmp")) == []


def test_skill_signature_writer_handles_relative_paths_and_atomic_name_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, _ = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    monkeypatch.chdir(tmp_path)
    relative_signature = sign_skill_package(
        archive,
        key_id="publisher",
        private_key=private_key,
        signature_path="relative.sig",
    )
    assert relative_signature == tmp_path / "relative.sig"

    destination = tmp_path / "raced.sig"
    original_link = os.link

    def occupy_destination_before_link(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        Path(destination).write_bytes(b"host-created file")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", occupy_destination_before_link)
    with pytest.raises(SkillPackageError, match="already exists"):
        sign_skill_package(
            archive,
            key_id="publisher",
            private_key=private_key,
            signature_path=destination,
        )
    assert destination.read_bytes() == b"host-created file"
    assert list(tmp_path.glob(".gabby-signature-*.tmp")) == []


@pytest.mark.parametrize(
    "document",
    [
        {"format": "wrong", "format_version": 2},
        {"format": "gabby-skill-signature", "format_version": 3},
        {"format": "gabby-skill-signature", "format_version": True},
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "bad key",
            "archive_sha256": "0" * 64,
            "content_sha256": "0" * 64,
            "signature": base64.b64encode(bytes(64)).decode("ascii"),
        },
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "publisher",
            "archive_sha256": "z" * 64,
            "content_sha256": "0" * 64,
            "signature": base64.b64encode(bytes(64)).decode("ascii"),
        },
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "publisher",
            "archive_sha256": "0" * 64,
            "content_sha256": "z" * 64,
            "signature": base64.b64encode(bytes(64)).decode("ascii"),
        },
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "publisher",
            "archive_sha256": "0" * 64,
            "content_sha256": "0" * 64,
            "signature": "x",
        },
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "publisher",
            "archive_sha256": "0" * 64,
            "content_sha256": "0" * 64,
            "signature": "!" * 88,
        },
        {
            "format": "gabby-skill-signature",
            "format_version": 2,
            "key_id": "publisher",
            "archive_sha256": "0" * 64,
            "content_sha256": "0" * 64,
            "signature": base64.b64encode(bytes(65)).decode("ascii"),
        },
    ],
)
def test_skill_signature_parser_rejects_invalid_fields(document: dict[str, Any]) -> None:
    with pytest.raises(SkillPackageError):
        signing_module._parse_signature(json.dumps(document).encode("ascii"))


def test_skill_signature_parser_rejects_duplicate_fields() -> None:
    encoded = (
        b'{"format":"gabby-skill-signature","format":"gabby-skill-signature","format_version":2}'
    )
    with pytest.raises(SkillPackageError, match="duplicate JSON keys"):
        signing_module._parse_signature(encoded)


def test_skill_package_content_digest_rejects_missing_oversized_and_invalid_manifests(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest.gabskill"
    with zipfile.ZipFile(missing_manifest, "w") as archive:
        archive.writestr("skill.yaml", "name: example\n")
    with pytest.raises(SkillPackageError, match="manifest is missing or oversized"):
        signing_module._package_content_digest(missing_manifest)

    oversized_manifest = tmp_path / "oversized-manifest.gabskill"
    with zipfile.ZipFile(oversized_manifest, "w") as archive:
        archive.writestr(".gabby-skill.json", b"x" * (1024 * 1024 + 1))
    with pytest.raises(SkillPackageError, match="manifest is missing or oversized"):
        signing_module._package_content_digest(oversized_manifest)

    invalid_archive = tmp_path / "invalid-archive.gabskill"
    invalid_archive.write_bytes(b"not a zip archive")
    with pytest.raises(SkillPackageError, match="manifest is invalid"):
        signing_module._package_content_digest(invalid_archive)


def test_skill_signature_rejects_manifest_content_changed_after_signing(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    original_archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, original_archive)
    signature = sign_skill_package(
        package=original_archive, key_id="publisher", private_key=private_key
    )

    changed_archive = tmp_path / "changed.gabskill"
    _rewrite_package_manifest(
        original_archive,
        changed_archive,
        lambda manifest: {
            **manifest,
            "skill": {**manifest["skill"], "version": "9.9.9"},
        },
    )
    document = json.loads(signature.read_text(encoding="ascii"))
    document["archive_sha256"] = hashlib.sha256(changed_archive.read_bytes()).hexdigest()
    changed_signature = tmp_path / "changed.sig"
    changed_signature.write_text(json.dumps(document), encoding="ascii")

    with pytest.raises(SkillPackageError, match="does not match the package content"):
        verify_skill_package_signature(
            changed_archive,
            trusted_keys={"publisher": public_key},
            signature_path=changed_signature,
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"{",
        b"[]",
        b'{"format":"wrong"}',
        b'{"format":"gabby-skill-signature","format_version":true}',
    ],
)
def test_skill_signature_document_parser_rejects_invalid_shapes(encoded: bytes) -> None:
    with pytest.raises(SkillPackageError):
        signing_module._parse_signature(encoded)


def test_skill_signing_rejects_invalid_key_and_existing_output(tmp_path: Path) -> None:
    private_key, _ = _ed25519_keypair()
    source = _skill_directory(tmp_path, name="signed-review")
    archive = tmp_path / "signed-review.gabskill"
    pack_skill(source, archive)
    with pytest.raises(SkillPackageError, match="exactly 32 raw bytes"):
        sign_skill_package(archive, key_id="publisher", private_key=b"short")
    with pytest.raises(SkillPackageError, match="key IDs"):
        sign_skill_package(archive, key_id="bad key", private_key=private_key)
    with pytest.raises(SkillPackageError, match="output directory"):
        sign_skill_package(
            archive,
            key_id="publisher",
            private_key=private_key,
            signature_path=tmp_path / "missing" / "signature.sig",
        )
    signature = sign_skill_package(archive, key_id="publisher", private_key=private_key)
    with pytest.raises(SkillPackageError, match="already exists"):
        sign_skill_package(archive, key_id="publisher", private_key=private_key)
    assert signature.is_file()


def test_install_rejects_traversal_and_leaves_registry_untouched(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    archive_path = tmp_path / "review.gabskill"
    pack_skill(source, archive_path)
    malicious = tmp_path / "traversal.gabskill"
    with (
        zipfile.ZipFile(archive_path) as original,
        zipfile.ZipFile(malicious, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            changed.writestr(item.filename, original.read(item.filename))
        changed.writestr("../escape.txt", b"escaped")

    registry = tmp_path / "registry"
    with pytest.raises(SkillPackageError, match="safe relative paths"):
        install_skill(malicious, registry)
    assert not (tmp_path / "escape.txt").exists()
    assert list(registry.iterdir()) == []


@pytest.mark.parametrize(
    "unsafe_path",
    ["/absolute.txt", "C:/drive.txt", "folder\\windows.txt", "CON.txt"],
)
def test_install_rejects_nonportable_archive_paths(tmp_path: Path, unsafe_path: str) -> None:
    source = _skill_directory(tmp_path, name="review")
    original_path = tmp_path / "review.gabskill"
    pack_skill(source, original_path)
    malicious_path = tmp_path / "unsafe.gabskill"
    with (
        zipfile.ZipFile(original_path) as original,
        zipfile.ZipFile(malicious_path, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            changed.writestr(item.filename, original.read(item.filename))
        changed.writestr(unsafe_path, b"unsafe")

    with pytest.raises(SkillPackageError, match="invalid file path|safe relative paths|portable"):
        install_skill(malicious_path, tmp_path / "registry")


def test_install_rejects_zip_symlink_entries(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    original_path = tmp_path / "review.gabskill"
    pack_skill(source, original_path)
    malicious_path = tmp_path / "symlink.gabskill"
    with (
        zipfile.ZipFile(original_path) as original,
        zipfile.ZipFile(malicious_path, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            changed.writestr(item.filename, original.read(item.filename))
        symlink = zipfile.ZipInfo("linked.txt")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        changed.writestr(symlink, b"target.txt")

    with pytest.raises(SkillPackageError, match="regular files only"):
        install_skill(malicious_path, tmp_path / "registry")


def test_install_rejects_duplicate_zip_paths(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    original_path = tmp_path / "review.gabskill"
    pack_skill(source, original_path)
    malicious_path = tmp_path / "duplicate.gabskill"
    with (
        zipfile.ZipFile(original_path) as original,
        zipfile.ZipFile(malicious_path, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            changed.writestr(item.filename, original.read(item.filename))
        with pytest.warns(UserWarning, match="Duplicate name"):
            changed.writestr("skill.yaml", b"name: altered\nversion: 1.2.3\n")

    with pytest.raises(SkillPackageError, match="duplicate paths"):
        install_skill(malicious_path, tmp_path / "registry")


def test_install_rejects_manifest_boolean_format_version(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    original_path = tmp_path / "review.gabskill"
    pack_skill(source, original_path)
    malicious_path = tmp_path / "bad-version.gabskill"
    with (
        zipfile.ZipFile(original_path) as original,
        zipfile.ZipFile(malicious_path, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == ".gabby-skill.json":
                manifest = json.loads(data)
                manifest["format_version"] = True
                data = json.dumps(manifest).encode()
            changed.writestr(item.filename, data)

    with pytest.raises(SkillPackageError, match="Unsupported skill package format"):
        install_skill(malicious_path, tmp_path / "registry")


def test_install_rejects_existing_versions(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    archive_path = tmp_path / "review.gabskill"
    pack_skill(source, archive_path)
    registry = tmp_path / "registry"

    install_skill(archive_path, registry)
    with pytest.raises(SkillPackageError, match="already installed"):
        install_skill(archive_path, registry)


def test_pack_rejects_source_symlinks_and_reserved_metadata(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    (source / ".gabby-skill.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="reserved file name"):
        pack_skill(source, tmp_path / "bad.gabskill")

    (source / ".gabby-skill.json").unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (source / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    with pytest.raises(SkillPackageError, match="must not contain symlinks"):
        pack_skill(source, tmp_path / "symlink.gabskill")


def test_package_manifest_checksums_describe_all_skill_files(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    archive_path = tmp_path / "review.gabskill"
    pack_skill(source, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(".gabby-skill.json"))
        assert manifest["format"] == "gabby-skill-package"
        assert manifest["format_version"] == 1
        assert manifest["skill"] == {"name": "review", "version": "1.2.3"}
        assert {record["path"] for record in manifest["files"]} == {
            "skill.yaml",
            "instructions.md",
            "examples/sample.txt",
        }
        assert all(len(record["sha256"]) == 64 for record in manifest["files"])


def test_pack_rejects_missing_file_and_invalid_skill_sources(tmp_path: Path) -> None:
    with pytest.raises(SkillPackageError, match="source directory is unavailable"):
        pack_skill(tmp_path / "missing", tmp_path / "missing.gabskill")

    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("not a skill", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="must be a directory"):
        pack_skill(regular_file, tmp_path / "file.gabskill")

    invalid_skill = tmp_path / "invalid-skill"
    invalid_skill.mkdir()
    (invalid_skill / "skill.yaml").write_text("name: [", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="valid skill.yaml"):
        pack_skill(invalid_skill, tmp_path / "invalid.gabskill")


def test_pack_rejects_symlink_root_and_output_location_errors(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    linked_source = tmp_path / "linked-source"
    try:
        linked_source.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    with pytest.raises(SkillPackageError, match="source must not be a symlink"):
        pack_skill(linked_source, tmp_path / "linked.gabskill")

    with pytest.raises(SkillPackageError, match="outside the skill source"):
        pack_skill(source, source / "inside.gabskill")
    with pytest.raises(SkillPackageError, match="Output directory must already exist"):
        pack_skill(source, tmp_path / "missing-dir" / "skill.gabskill")
    existing = tmp_path / "existing.gabskill"
    existing.write_text("occupied", encoding="utf-8")
    with pytest.raises(SkillPackageError, match="already exists"):
        pack_skill(source, existing)


def test_pack_supports_relative_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="relative")
    monkeypatch.chdir(tmp_path)
    result = pack_skill(source, "relative.gabskill")
    assert result.path == tmp_path / "relative.gabskill"


def test_pack_rejects_non_regular_and_oversized_source_files(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    fifo = source / "special.pipe"
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this host")
    try:
        os.mkfifo(fifo)
    except OSError:
        pytest.skip("FIFO creation is unavailable on this host")
    with pytest.raises(SkillPackageError, match="only regular files"):
        pack_skill(source, tmp_path / "fifo.gabskill")
    fifo.unlink()

    (source / "too-large.bin").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(SkillPackageError, match="limited to 10 MiB"):
        pack_skill(source, tmp_path / "oversized.gabskill")


def test_install_rejects_missing_invalid_and_non_file_archives(tmp_path: Path) -> None:
    with pytest.raises(SkillPackageError, match="archive is unavailable"):
        install_skill(tmp_path / "missing.gabskill", tmp_path / "registry")

    non_file = tmp_path / "directory.gabskill"
    non_file.mkdir()
    with pytest.raises(SkillPackageError, match="not a bounded regular file"):
        install_skill(non_file, tmp_path / "registry")

    invalid = tmp_path / "invalid.gabskill"
    invalid.write_bytes(b"not a zip archive")
    with pytest.raises(SkillPackageError, match="Could not install skill package"):
        install_skill(invalid, tmp_path / "registry")


def test_install_rejects_manifest_identity_mismatch_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)

    mismatch = tmp_path / "mismatch.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(mismatch, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in source_zip.infolist():
            data = source_zip.read(entry.filename)
            if entry.filename == ".gabby-skill.json":
                manifest = json.loads(data)
                manifest["skill"]["version"] = "2.0.0"
                data = json.dumps(manifest).encode()
            changed.writestr(entry.filename, data)
    with pytest.raises(SkillPackageError, match="identity does not match"):
        install_skill(mismatch, tmp_path / "mismatch-registry")

    duplicated_key = tmp_path / "duplicate-key.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(duplicated_key, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in source_zip.infolist():
            data = source_zip.read(entry.filename)
            if entry.filename == ".gabby-skill.json":
                data = data.replace(b'"format_version":1', b'"format_version":1,"format_version":1')
            changed.writestr(entry.filename, data)
    with pytest.raises(SkillPackageError, match="duplicate JSON keys"):
        install_skill(duplicated_key, tmp_path / "duplicate-key-registry")


def test_install_rejects_registry_symlink_and_identity_mismatch(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="nested/review")
    archive = tmp_path / "review.gabskill"
    pack_skill(source, archive)
    registry = tmp_path / "registry"
    registry.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (registry / "nested").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    with pytest.raises(SkillPackageError, match="path contains a symlink"):
        install_skill(archive, registry)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("not-object", "invalid structure"),
        ("unknown-version", "Unsupported skill package format"),
        ("bad-skill-container", "identity is invalid"),
        ("bad-skill-name", "identity is invalid"),
        ("bad-file-list", "invalid file list"),
        ("bad-file-record", "file record is invalid"),
        ("reserved-path", "duplicate or reserved paths"),
        ("bad-file-size", "file size is invalid"),
        ("bad-checksum", "file checksum is invalid"),
        ("missing-skill-manifest", "omits skill.yaml"),
    ],
)
def test_install_validates_package_manifest_schema(
    tmp_path: Path, variant: str, message: str
) -> None:
    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)
    malformed = tmp_path / f"{variant}.gabskill"

    def transform(manifest: Any) -> Any:
        if variant == "not-object":
            return []
        if variant == "unknown-version":
            manifest["format_version"] = 2
        elif variant == "bad-skill-container":
            manifest["skill"] = []
        elif variant == "bad-skill-name":
            manifest["skill"]["name"] = 3
        elif variant == "bad-file-list":
            manifest["files"] = {}
        elif variant == "bad-file-record":
            manifest["files"][0] = None
        elif variant == "reserved-path":
            manifest["files"][0]["path"] = ".gabby-skill.json"
        elif variant == "bad-file-size":
            manifest["files"][0]["size"] = True
        elif variant == "bad-checksum":
            manifest["files"][0]["sha256"] = "not-a-checksum"
        elif variant == "missing-skill-manifest":
            manifest["files"] = [
                entry for entry in manifest["files"] if entry["path"] != "skill.yaml"
            ]
        return manifest

    _rewrite_package_manifest(original, malformed, transform)
    with pytest.raises(SkillPackageError, match=message):
        install_skill(malformed, tmp_path / "registry")


def test_install_rejects_missing_or_oversized_package_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)

    missing = tmp_path / "missing-manifest.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(missing, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in source_zip.infolist():
            if entry.filename != ".gabby-skill.json":
                changed.writestr(entry.filename, source_zip.read(entry.filename))
    with pytest.raises(SkillPackageError, match="manifest is missing or oversized"):
        install_skill(missing, tmp_path / "missing-registry")

    monkeypatch.setattr("gabby.skill_packages._MAX_PACKAGE_MANIFEST_BYTES", 1)
    with pytest.raises(SkillPackageError, match="manifest is missing or oversized"):
        install_skill(original, tmp_path / "oversized-registry")


def test_pack_enforces_tree_manifest_content_and_archive_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TREE_ENTRIES", 1)
    with pytest.raises(SkillPackageError, match="tree exceeds"):
        pack_skill(source, tmp_path / "tree.gabskill")

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TREE_ENTRIES", 10_000)
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TOTAL_BYTES", 1)
    with pytest.raises(SkillPackageError, match="total limit"):
        pack_skill(source, tmp_path / "total.gabskill")

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TOTAL_BYTES", 100 * 1024 * 1024)
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_FILES", 1)
    with pytest.raises(SkillPackageError, match="limited to 1000 files"):
        pack_skill(source, tmp_path / "files.gabskill")

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_FILES", 1000)
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_MANIFEST_BYTES", 1)
    with pytest.raises(SkillPackageError, match="manifest exceeds"):
        pack_skill(source, tmp_path / "manifest.gabskill")


def test_pack_cleans_temporary_archives_after_io_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    monkeypatch.setattr(
        package_module.os,  # type: ignore[attr-defined]
        "link",
        lambda *_: (_ for _ in ()).throw(PermissionError()),
    )
    with pytest.raises(SkillPackageError, match="Could not create skill package archive"):
        pack_skill(source, tmp_path / "permission.gabskill")
    assert not (tmp_path / "permission.gabskill").exists()
    assert not list(tmp_path.glob(".gabby-skill-*.tmp"))


def test_pack_aborts_if_source_changes_after_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    instructions = source / "instructions.md"
    write_entry = package_module._write_source_entry

    def mutate_before_write(archive: zipfile.ZipFile, name: str, path: Path, digest: str) -> None:
        if path == instructions:
            instructions.write_text("Changed after hashing", encoding="utf-8")
        write_entry(archive, name, path, digest)

    monkeypatch.setattr(package_module, "_write_source_entry", mutate_before_write)
    with pytest.raises(SkillPackageError, match="source changed during packaging"):
        pack_skill(source, tmp_path / "changed.gabskill")
    assert not (tmp_path / "changed.gabskill").exists()


def test_install_rejects_empty_archives_unsupported_compression_and_extra_members(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.gabskill"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(SkillPackageError, match="invalid file count"):
        install_skill(empty, tmp_path / "empty-registry")

    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)
    bzip2 = tmp_path / "bzip2.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(bzip2, "w", compression=zipfile.ZIP_BZIP2) as changed,
    ):
        for entry in source_zip.infolist():
            changed.writestr(entry.filename, source_zip.read(entry.filename))
    with pytest.raises(SkillPackageError, match="unsupported compression"):
        install_skill(bzip2, tmp_path / "bzip2-registry")

    extra = tmp_path / "extra-file.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(extra, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in source_zip.infolist():
            changed.writestr(entry.filename, source_zip.read(entry.filename))
        changed.writestr("unlisted.txt", b"not in manifest")
    with pytest.raises(SkillPackageError, match="manifest does not match"):
        install_skill(extra, tmp_path / "extra-registry")


def test_pack_rejects_archive_size_limit_and_temporary_file_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_ARCHIVE_BYTES", 1)
    with pytest.raises(SkillPackageError, match="archive exceeds"):
        pack_skill(source, tmp_path / "too-large.gabskill")
    assert not list(tmp_path.glob(".gabby-skill-*.tmp"))

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_ARCHIVE_BYTES", 100 * 1024 * 1024)
    monkeypatch.setattr(
        package_module.tempfile,  # type: ignore[attr-defined]
        "NamedTemporaryFile",
        lambda **_: (_ for _ in ()).throw(OSError("unavailable")),
    )
    with pytest.raises(SkillPackageError, match="Could not create skill package archive"):
        pack_skill(source, tmp_path / "temp-failure.gabskill")


def test_installer_enforces_uncompressed_and_manifest_declared_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)
    with zipfile.ZipFile(original) as archive:
        manifest_size = archive.getinfo(".gabby-skill.json").file_size

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_FILE_BYTES", 1)
    with pytest.raises(SkillPackageError, match="files are limited to 10 MiB"):
        install_skill(original, tmp_path / "file-limit")

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_FILE_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TOTAL_BYTES", manifest_size)
    with pytest.raises(SkillPackageError, match="total limit"):
        install_skill(original, tmp_path / "total-limit")

    monkeypatch.setattr(package_module, "_MAX_PACKAGE_TOTAL_BYTES", 100 * 1024 * 1024)
    mismatch = tmp_path / "size-mismatch.gabskill"

    def increment_instruction_size(manifest: Any) -> Any:
        record = next(entry for entry in manifest["files"] if entry["path"] == "instructions.md")
        record["size"] += 1
        return manifest

    _rewrite_package_manifest(original, mismatch, increment_instruction_size)
    with pytest.raises(SkillPackageError, match="file size mismatch"):
        install_skill(mismatch, tmp_path / "size-mismatch-registry")


def test_installer_rejects_invalid_json_manifest(tmp_path: Path) -> None:
    source = _skill_directory(tmp_path, name="review")
    original = tmp_path / "original.gabskill"
    pack_skill(source, original)
    malformed = tmp_path / "invalid-json.gabskill"
    with (
        zipfile.ZipFile(original) as source_zip,
        zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as changed,
    ):
        for entry in source_zip.infolist():
            data = source_zip.read(entry.filename)
            if entry.filename == ".gabby-skill.json":
                data = b"{"
            changed.writestr(entry.filename, data)
    with pytest.raises(SkillPackageError, match="manifest is invalid"):
        install_skill(malformed, tmp_path / "registry")


def test_installer_rejects_encryption_and_actual_size_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "format": "gabby-skill-package",
        "format_version": 1,
        "skill": {"name": "review", "version": "1.0.0"},
        "files": [{"path": "skill.yaml", "size": 0, "sha256": "e3b0c442" + "0" * 56}],
    }
    manifest_bytes = json.dumps(manifest).encode()
    manifest_info = zipfile.ZipInfo(".gabby-skill.json")
    manifest_info.file_size = len(manifest_bytes)
    manifest_info.external_attr = (stat.S_IFREG | 0o600) << 16
    skill_info = zipfile.ZipInfo("skill.yaml")
    skill_info.file_size = 0
    skill_info.external_attr = (stat.S_IFREG | 0o600) << 16

    class FakeArchive:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def infolist(self) -> list[zipfile.ZipInfo]:
            return [manifest_info, skill_info]

        def read(self, _info: zipfile.ZipInfo) -> bytes:
            return manifest_bytes

        def open(self, _info: zipfile.ZipInfo, _mode: str) -> io.BytesIO:
            return io.BytesIO(b"x")

    monkeypatch.setattr(package_module.zipfile, "ZipFile", FakeArchive)  # type: ignore[attr-defined]
    skill_info.flag_bits = 1
    with pytest.raises(SkillPackageError, match="Encrypted skill packages"):
        package_module._extract_and_verify(tmp_path / "fake.gabskill", tmp_path)

    skill_info.flag_bits = 0
    with pytest.raises(SkillPackageError, match="exceeded its declared size"):
        package_module._extract_and_verify(tmp_path / "fake.gabskill", tmp_path)

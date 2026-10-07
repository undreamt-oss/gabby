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
"""Host-owned skill trust policy validation tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from gabby.config import ConfigError, SkillDefinition
from gabby.skill_packages import SkillPackageError
from gabby.skill_trust import SkillTrustPolicy, load_skill_trust_keys


@pytest.mark.parametrize(
    ("trusted_keys", "revoked_key_ids", "message"),
    [
        ({}, frozenset(), "at least one trusted public key"),
        ({"bad key": bytes(32)}, frozenset(), "valid key IDs"),
        ({"publisher": b"short"}, frozenset(), "32-byte public keys"),
        ({"publisher": bytes(32)}, "publisher", "collection of IDs"),
        ({"publisher": bytes(32)}, None, "iterable of IDs"),
        ({"publisher": bytes(32)}, frozenset({"bad key"}), "invalid format"),
    ],
)
def test_skill_trust_policy_rejects_invalid_host_configuration(
    trusted_keys: object,
    revoked_key_ids: object,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        SkillTrustPolicy(trusted_keys, revoked_key_ids)  # type: ignore[arg-type]


def test_skill_trust_policy_rejects_in_memory_skill_without_provenance() -> None:
    policy = SkillTrustPolicy({"publisher": bytes(32)})
    with pytest.raises(SkillPackageError, match="no installed package provenance"):
        policy.verify(SkillDefinition(name="local-skill"))


def test_skill_trust_policy_loads_a_bounded_key_directory(tmp_path: Path) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    (directory / "publisher.pub").write_bytes(b"p" * 32)

    assert load_skill_trust_keys(directory) == {"publisher": b"p" * 32}
    policy = SkillTrustPolicy.from_directory(directory, revoked_key_ids=frozenset({"old"}))
    assert dict(policy.trusted_keys) == {"publisher": b"p" * 32}
    assert policy.revoked_key_ids == frozenset({"old"})


@pytest.mark.parametrize(
    ("entry_name", "contents", "message"),
    [
        ("notes.txt", b"ignored?", "only KEY_ID.pub files"),
        ("bad key.pub", b"p" * 32, "filenames must use KEY_ID.pub"),
        ("short.pub", b"short", "32-byte file"),
    ],
)
def test_skill_trust_policy_rejects_invalid_directory_contents(
    tmp_path: Path, entry_name: str, contents: bytes, message: str
) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    (directory / entry_name).write_bytes(contents)

    with pytest.raises(ConfigError, match=message):
        load_skill_trust_keys(directory)


def test_skill_trust_policy_rejects_symlink_directory_and_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ConfigError, match="must not be a symlink"):
        load_skill_trust_keys(symlink)

    with pytest.raises(ConfigError, match="at least one KEY_ID.pub"):
        load_skill_trust_keys(target)


def test_skill_trust_policy_rejects_file_and_unavailable_directory_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_bytes(b"not a directory")
    with pytest.raises(ConfigError, match="must be a directory"):
        load_skill_trust_keys(file_path)
    with pytest.raises(ConfigError, match="directory is unavailable"):
        load_skill_trust_keys(tmp_path / "missing")


def test_skill_trust_policy_rejects_symlink_entries_and_subdirectories(tmp_path: Path) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    target = tmp_path / "publisher.pub"
    target.write_bytes(b"p" * 32)
    try:
        (directory / "publisher.pub").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    with pytest.raises(ConfigError, match="entries must not be symlinks"):
        load_skill_trust_keys(directory)

    (directory / "publisher.pub").unlink()
    (directory / "nested.pub").mkdir()
    with pytest.raises(ConfigError, match="only KEY_ID.pub files"):
        load_skill_trust_keys(directory)


def test_skill_trust_policy_enforces_key_count_limit(tmp_path: Path) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    for index in range(257):
        (directory / f"publisher-{index:03d}.pub").write_bytes(bytes(32))

    with pytest.raises(ConfigError, match="256-key limit"):
        load_skill_trust_keys(directory)


def test_skill_trust_policy_rejects_key_path_replacement_between_scan_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    key_path = directory / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    replacement = tmp_path / "replacement.pub"
    replacement.write_bytes(b"p" * 32)
    original_open = os.open

    def replace_key_before_open(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == key_path:
            replacement.replace(key_path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_key_before_open)
    with pytest.raises(ConfigError, match="changed while loading"):
        load_skill_trust_keys(directory)


def test_skill_trust_policy_rejects_key_growing_during_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    key_path = directory / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    original_open = os.open

    def grow_key_before_open(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == key_path:
            with key_path.open("ab") as output:
                output.write(b"x")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", grow_key_before_open)
    with pytest.raises(ConfigError, match="must contain 32 raw bytes"):
        load_skill_trust_keys(directory)


def test_skill_trust_policy_rejects_key_replaced_by_fifo_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFO creation and nonblocking file opens are unavailable")
    directory = tmp_path / "trust"
    directory.mkdir()
    key_path = directory / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    original_open = os.open

    def replace_key_with_fifo(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == key_path:
            key_path.unlink()
            os.mkfifo(key_path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_key_with_fifo)
    with pytest.raises(ConfigError, match="changed while loading"):
        load_skill_trust_keys(directory)

#!/usr/bin/env python3
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
#
# Copyright 2026-present Gabby Contributors.

"""Build and sign the reviewed starter-skill packages for a Gabby release."""

from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
from pathlib import Path

from gabby.skill_packages import SkillPackageError, inspect_skill_package, pack_skill
from gabby.skill_signing import sign_skill_package, verify_skill_package_signature

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bundles" / "starter-packs"
STARTER_NAMES = (
    "customer-support-triage",
    "read-only-sql-analyst",
    "sandboxed-git-assistant",
)


def _signing_material() -> tuple[str, bytes, bytes]:
    key_id = os.environ.get("GABBY_STARTER_SIGNING_KEY_ID", "")
    encoded_key = os.environ.get("GABBY_STARTER_SIGNING_KEY", "")
    if not key_id or not encoded_key:
        raise SkillPackageError(
            "Set GABBY_STARTER_SIGNING_KEY_ID and GABBY_STARTER_SIGNING_KEY in the signing "
            "environment; private signing keys are never stored in the repository"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", key_id, flags=re.ASCII) is None:
        raise SkillPackageError("Starter signing key ID must use letters, numbers, dot, '_' or '-'")
    if len(encoded_key) > 128:
        raise SkillPackageError("Starter signing key value exceeds its encoded size limit")
    try:
        private_key = base64.b64decode(encoded_key, validate=True)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        signer = Ed25519PrivateKey.from_private_bytes(private_key)
        public_key = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    except ImportError as exc:
        raise SkillPackageError("Install gabby-ai[skill-signing] to sign starter bundles") from exc
    except (ValueError, TypeError) as exc:
        raise SkillPackageError(
            "Starter bundle signing key must be base64-encoded raw Ed25519"
        ) from exc
    if len(private_key) != 32:
        raise SkillPackageError("Starter bundle signing key must decode to 32 bytes")
    return key_id, private_key, public_key


def build(output: Path) -> None:
    key_id, private_key, public_key = _signing_material()
    destination = output.expanduser().resolve()
    if destination.exists():
        raise SkillPackageError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)
    trust_dir = destination / "trusted-publishers"
    trust_dir.mkdir()
    (trust_dir / f"{key_id}.pub").write_bytes(public_key)

    for bundle_name in STARTER_NAMES:
        bundle = SOURCE / bundle_name
        skill_dir = bundle / "skills" / bundle_name
        archive = destination / f"{bundle_name}-1.0.0.gabskill"
        pack_skill(skill_dir, archive)
        sign_skill_package(archive, key_id=key_id, private_key=private_key)
        verify_skill_package_signature(archive, trusted_keys={key_id: public_key})
        inspection = inspect_skill_package(archive, trusted_keys={key_id: public_key})
        if not inspection.signature_verified or inspection.name != bundle_name:
            raise SkillPackageError(f"Signed starter bundle audit failed for {bundle_name}")
        shutil.copyfile(bundle / "agent.yaml", destination / f"{bundle_name}.agent.yaml")
        shutil.copyfile(bundle / "AUDIT.md", destination / f"{bundle_name}.AUDIT.md")
        workspace = bundle / "workspace"
        if workspace.is_dir():
            shutil.copytree(workspace, destination / "workspace", dirs_exist_ok=True)

    shutil.copyfile(ROOT / "bundles" / "README.md", destination / "README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        build(args.output)
    except (OSError, SkillPackageError) as exc:
        parser.error(str(exc))
    print(f"Built and verified {len(STARTER_NAMES)} signed starter bundles in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

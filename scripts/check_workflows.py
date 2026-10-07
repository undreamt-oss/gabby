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
"""Check workflow action pinning, platform coverage, and release publishing gates."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION = re.compile(r"^\s*(?:-\s+)?uses: [^@\s]+@([a-f0-9]{40})(?:\s+#.*)?$")
LOCAL_WORKFLOW = re.compile(
    r"^\s*(?:-\s+)?uses: (\./\.github/workflows/[A-Za-z0-9_.-]+\.yml)(?:\s+#.*)?$"
)


def main() -> int:
    workflow_dir = ROOT / ".github" / "workflows"
    failures: list[str] = []
    workflows = sorted(workflow_dir.glob("*.yml"))
    if not workflows:
        failures.append("No GitHub Actions workflows were found.")
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "uses:" not in line:
                continue
            local_workflow = LOCAL_WORKFLOW.fullmatch(line)
            if local_workflow is not None:
                if not (ROOT / local_workflow.group(1)).is_file():
                    failures.append(
                        f"{path.relative_to(ROOT)} references a missing local workflow: "
                        f"{local_workflow.group(1)}"
                    )
            elif not ACTION.fullmatch(line):
                failures.append(f"{path.relative_to(ROOT)} has an unpinned action: {line.strip()}")

    ci_path = workflow_dir / "ci.yml"
    if ci_path.is_file():
        ci = ci_path.read_text(encoding="utf-8")
        postgres_workflow = workflow_dir / "postgres-validation.yml"
        if not postgres_workflow.is_file():
            failures.append("CI is missing the reusable PostgreSQL acceptance workflow.")
        else:
            postgres_checks = postgres_workflow.read_text(encoding="utf-8")
            for required in (
                "workflow_call:",
                "image: postgres:17-alpine",
                "GABBY_POSTGRES_DSN:",
                "tests/integration/test_postgres_indexing_live.py",
                "tests/integration/test_postgres_knowledge_live.py",
                "tests/integration/test_postgres_stream_journal_live.py",
                "image: pgvector/pgvector:0.8.6-pg17-trixie",
                "tests/integration/test_postgres_vector_live.py",
            ):
                if required not in postgres_checks:
                    failures.append(f"PostgreSQL acceptance workflow must include {required!r}.")
        if "uses: ./.github/workflows/postgres-validation.yml" not in ci:
            failures.append("CI must call the reusable PostgreSQL acceptance workflow.")
        ci_platform = ci.partition("  platform:\n")[2].partition("  quality:\n")[0]
        for version in ("3.11", "3.12", "3.13", "3.14"):
            if f'"{version}"' not in ci:
                failures.append(f"CI does not test supported Python {version}.")
            if f'"{version}"' not in ci_platform:
                failures.append(f"CI platform job does not test Python {version}.")
        if "permissions:\n  contents: read" not in ci:
            failures.append("CI must declare read-only default repository permissions.")
        for runner in ("ubuntu-latest", "macos-latest", "windows-latest"):
            if runner not in ci:
                failures.append(f"CI does not include the {runner} platform runner.")
            if runner not in ci_platform:
                failures.append(f"CI platform job does not include the {runner} runner.")
        if "python-version: ${{ matrix.python-version }}" not in ci_platform:
            failures.append("CI platform job must run its supported Python version matrix.")
        ci_sandbox = ci.partition("  sandbox-live:\n")[2].partition("  quality:\n")[0]
        for required in (
            "runs-on: ubuntu-24.04",
            "sudo systemctl start docker",
            'GABBY_RUN_DOCKER_INTEGRATION: "1"',
            'GABBY_RUN_PODMAN_INTEGRATION: "1"',
            "tests/integration/test_docker_sandbox_live.py",
            "tests/integration/test_podman_sandbox_live.py",
        ):
            if required not in ci_sandbox:
                failures.append(f"CI sandbox-live job must include {required!r}.")
        ci_deployment = ci.partition("  deployment-container:\n")[2].partition("  quality:\n")[0]
        reusable_workflow = workflow_dir / "container-validation.yml"
        if "uses: ./.github/workflows/container-validation.yml" not in ci_deployment:
            failures.append("CI must call the reusable hosted-container validation workflow.")
        container_checks = reusable_workflow.read_text(encoding="utf-8")
        compose_config = (ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")
        nginx_config = (ROOT / "deploy" / "nginx.conf").read_text(encoding="utf-8")
        for required in (
            "UV_IMAGE: ${GABBY_UV_IMAGE:-",
            "RUNTIME_IMAGE: ${GABBY_RUNTIME_IMAGE:-",
        ):
            if required not in compose_config:
                failures.append(
                    f"Compose deployment must allow reviewed base image overrides: {required!r}."
                )
        for required in (
            "workflow_call:",
            "runs-on: ubuntu-24.04",
            "docker compose -f deploy/compose.yaml config --quiet",
            "up --build --detach --wait",
            "{{.Config.User}}",
            "{{.HostConfig.ReadonlyRootfs}}",
            "{{.HostConfig.PidsLimit}}",
            "{{.HostConfig.Memory}}",
            "{{.HostConfig.NanoCpus}}",
            "{{json .HostConfig.CapDrop}}",
            "{{json .HostConfig.SecurityOpt}}",
            "http://127.0.0.1:8787/health",
            "http://127.0.0.1:8787/v1/agents/research-synthesizer/run",
            'test "$status" = "401"',
            "python scripts/check_nginx_ingress.py",
            "if: always()",
            "down --volumes --remove-orphans",
        ):
            if required not in container_checks:
                failures.append(
                    f"Reusable container validation workflow must include {required!r}."
                )
        for required in (
            "limit_req_zone $binary_remote_addr zone=gabby_api:10m rate=10r/s;",
            "client_max_body_size 1000000;",
            "limit_req_status 429;",
            "proxy_buffering off;",
            "server 127.0.0.1:8787;",
        ):
            if required not in nginx_config:
                failures.append(f"Nginx deployment reference must include {required!r}.")
        if nginx_config.count('proxy_set_header Connection "";') != 3:
            failures.append("Every Nginx upstream route must preserve HTTP/1.1 keepalive.")
        if "--extra server --extra auth" not in ci_platform:
            failures.append(
                "CI platform job must install optional server and JWT auth dependencies."
            )
        if "--extra mcp" not in ci_platform:
            failures.append("CI platform job must test the optional MCP adapter.")
        if (
            "if: runner.os == 'Windows'" not in ci_platform
            or "--extra windows-sandbox" not in ci_platform
        ):
            failures.append(
                "CI Windows platform job must install the optional Windows sandbox dependency."
            )
        ci_quality = ci.partition("  quality:\n")[2]
        if "--extra server --extra auth" not in ci_quality:
            failures.append(
                "CI quality job must install optional server and JWT auth dependencies."
            )
        if "--extra windows-sandbox" not in ci_quality:
            failures.append("CI quality job must audit optional Windows sandbox dependencies.")
        if "--extra transformers" not in ci_quality:
            failures.append("CI quality job must audit the optional Transformers provider.")
        if "--extra mcp" not in ci_quality:
            failures.append("CI quality job must audit the optional MCP adapter.")
        for command in (
            "ruff check src tests scripts examples",
            "ruff format --check src tests scripts examples",
            "mypy src/gabby tests scripts examples",
        ):
            if command not in ci_quality:
                failures.append(f"CI quality job must include examples in {command.split()[0]}.")
        package_check = "uv run --no-sync python scripts/check_package.py dist"
        if "uv build --out-dir dist" not in ci_quality or package_check not in ci_quality:
            failures.append("CI quality job must build and inspect wheel and sdist contents.")
        elif ci_quality.index("uv build --out-dir dist") > ci_quality.index(package_check):
            failures.append("CI must inspect the package archives after building them.")

    release_path = workflow_dir / "release.yml"
    if release_path.is_file():
        release = release_path.read_text(encoding="utf-8")
        if 'tags: ["v*"]' not in release:
            failures.append("Release workflow must run only for v-prefixed version tags.")
        if (
            "  build:\n    needs: [platform, quality, sandbox-live, deployment-container, "
            "postgres-live]" not in release
        ):
            failures.append(
                "Release distribution build must wait for platform, quality, sandbox, container, "
                "and PostgreSQL acceptance gates."
            )
        if "uses: ./.github/workflows/postgres-validation.yml" not in release:
            failures.append("Release workflow must run PostgreSQL acceptance before publishing.")
        release_deployment = release.partition("  deployment-container:\n")[2].partition(
            "  quality:\n"
        )[0]
        if "uses: ./.github/workflows/container-validation.yml" not in release_deployment:
            failures.append(
                "Tagged releases must call the reusable hosted-container validation workflow."
            )
        if "  publish:\n    needs: build" not in release:
            failures.append("PyPI publishing must wait for the distribution build.")
        build_job = release.partition("  build:\n")[2].partition("  publish:\n")[0]
        for permission in (
            "      contents: read",
            "      id-token: write",
            "      attestations: write",
            "      artifact-metadata: write",
        ):
            if permission not in build_job:
                failures.append(
                    f"Release build job must grant {permission.strip()} for provenance."
                )
        if (
            "Attest release distributions" not in build_job
            or "subject-path: dist/*" not in build_job
        ):
            failures.append(
                "Release build job must attest the built wheel and source distribution."
            )
        for required in (
            "uv export --frozen --no-dev --extra server --extra auth "
            "--preview-features sbom-export",
            "--format cyclonedx1.5 --output-file sbom/gabby-runtime-sbom.cdx.json",
            "Attest wheel with runtime SBOM",
            "Attest source distribution with runtime SBOM",
            "sbom-path: sbom/gabby-runtime-sbom.cdx.json",
            "name: gabby-runtime-sbom",
            "retention-days: 90",
            "path: sbom/gabby-runtime-sbom.cdx.json",
        ):
            if required not in build_job:
                failures.append(
                    f"Release build job must generate, attest, and retain its SBOM ({required})."
                )
        if build_job.count("sbom-path: sbom/gabby-runtime-sbom.cdx.json") != 2:
            failures.append(
                "Release build job must attest both distributions with the runtime SBOM."
            )
        wheel_attestation = build_job.partition("- name: Attest wheel with runtime SBOM")[
            2
        ].partition("- name: Attest source distribution with runtime SBOM")[0]
        sdist_attestation = build_job.partition(
            "- name: Attest source distribution with runtime SBOM"
        )[2]
        if "subject-path: dist/*.whl" not in wheel_attestation:
            failures.append("Runtime SBOM attestation must name the wheel as its subject.")
        if "subject-path: dist/*.tar.gz" not in sdist_attestation:
            failures.append(
                "Runtime SBOM attestation must name the source distribution as its subject."
            )
        publish_job = release.partition("  publish:\n")[2]
        if "name: gabby-runtime-sbom" in publish_job:
            failures.append("The PyPI publisher must not receive the SBOM as a distribution file.")
        if "      name: pypi" not in publish_job:
            failures.append("PyPI publishing must use the protected 'pypi' environment.")
        if "      id-token: write" not in publish_job or release.count("id-token: write") != 2:
            failures.append("Only the release build and PyPI publish jobs may receive OIDC grants.")
        if 'scripts/check_release.py "$GITHUB_REF_NAME"' not in release:
            failures.append("Release workflow must verify tag and package version alignment.")
        if "Confirm tag commit is on main" not in release:
            failures.append("Release workflow must confirm the tag commit is reachable from main.")
        release_platform = release.partition("  platform:\n")[2].partition("  quality:\n")[0]
        for version in ("3.11", "3.12", "3.13", "3.14"):
            if f'"{version}"' not in release_platform:
                failures.append(f"Release platform job does not test Python {version}.")
        for runner in ("ubuntu-latest", "macos-latest", "windows-latest"):
            if runner not in release_platform:
                failures.append(f"Release platform job does not include the {runner} runner.")
        if "python-version: ${{ matrix.python-version }}" not in release_platform:
            failures.append("Release platform job must run its supported Python version matrix.")
        if "--extra server --extra auth" not in release_platform:
            failures.append(
                "Release platform job must install optional server and JWT auth dependencies."
            )
        if "--extra mcp" not in release_platform:
            failures.append("Release platform job must test the optional MCP adapter.")
        if (
            "if: runner.os == 'Windows'" not in release_platform
            or "--extra windows-sandbox" not in release_platform
        ):
            failures.append(
                "Release Windows platform job must install the optional Windows sandbox dependency."
            )
        release_quality = release.partition("  quality:\n")[2].partition("  build:\n")[0]
        if "--extra server --extra auth" not in release_quality:
            failures.append(
                "Release quality job must install optional server and JWT auth dependencies."
            )
        if "--extra windows-sandbox" not in release_quality:
            failures.append("Release quality job must audit optional Windows sandbox dependencies.")
        if "--extra transformers" not in release_quality:
            failures.append("Release quality job must audit the optional Transformers provider.")
        if "--extra mcp" not in release_quality:
            failures.append("Release quality job must audit the optional MCP adapter.")
        release_sandbox = release.partition("  sandbox-live:\n")[2].partition("  quality:\n")[0]
        for required in (
            "runs-on: ubuntu-24.04",
            "sudo systemctl start docker",
            'GABBY_RUN_DOCKER_INTEGRATION: "1"',
            'GABBY_RUN_PODMAN_INTEGRATION: "1"',
            "tests/integration/test_docker_sandbox_live.py",
            "tests/integration/test_podman_sandbox_live.py",
        ):
            if required not in release_sandbox:
                failures.append(f"Release sandbox-live job must include {required!r}.")
        for command in (
            "ruff check src tests scripts examples",
            "ruff format --check src tests scripts examples",
            "mypy src/gabby tests scripts examples",
        ):
            if command not in release_quality:
                failures.append(
                    f"Release quality job must include examples in {command.split()[0]}."
                )
        package_check = "uv run --no-sync python scripts/check_package.py dist"
        if "uv build --out-dir dist" not in release_quality or package_check not in release_quality:
            failures.append("Release quality job must build and inspect wheel and sdist contents.")
        elif release_quality.index("uv build --out-dir dist") > release_quality.index(
            package_check
        ):
            failures.append(
                "Release quality must inspect the package archives after building them."
            )
        release_archive_check = "python scripts/check_package.py dist"
        if release_archive_check not in build_job:
            failures.append("Release build job must inspect package archives before attestation.")
        elif build_job.index(release_archive_check) > build_job.index(
            "Attest release distributions"
        ):
            failures.append("Release archives must pass package inspection before attestation.")

    security_path = workflow_dir / "security.yml"
    if not security_path.is_file():
        failures.append("An OpenSSF Scorecard security workflow is required.")
    else:
        security = security_path.read_text(encoding="utf-8")
        for required in (
            "permissions: read-all",
            "ossf/scorecard-action@55891bbd73f2425e97637d96e306fc9d491d0b21",
            "github/codeql-action/upload-sarif@",
            "publish_results: true",
        ):
            if required not in security:
                failures.append(f"OpenSSF Scorecard workflow must include {required!r}.")

    if failures:
        for failure in failures:
            print(failure)
        return 1
    print(
        "Workflow actions are pinned; CI and release platform gates cover Python 3.11–3.14 "
        "on Linux/macOS/Windows with Linux Docker/Podman sandbox acceptance, plus reusable "
        "PostgreSQL, Compose, and Nginx ingress gates in CI and tagged releases; "
        "tagged PyPI publication is gated by verification and a protected environment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
